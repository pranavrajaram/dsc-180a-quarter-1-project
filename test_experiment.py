"""Run with python3 -B -m unittest -v. Includes actual OS-isolation checks."""
import copy
import csv
import inspect
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import experiment as ex


# Independent reference implementations check the hand-authored expected answers.
def merge_intervals(intervals):
    output = []
    for start, end in sorted(intervals):
        if output and start <= output[-1][1]:
            output[-1][1] = max(end, output[-1][1])
        else:
            output.append([start, end])
    return output


def compare_versions(a, b):
    pairs = itertools.zip_longest(map(int, a.split('.')), map(int, b.split('.')), fillvalue=0)
    return next(((x > y) - (x < y) for x, y in pairs if x != y), 0)


def split_escaped(text):
    fields, field, index = [], '', 0
    while index < len(text):
        char = text[index]
        if char == '\\' and index + 1 < len(text):
            index += 1
            field += text[index]
        elif char == ',':
            fields.append(field)
            field = ''
        else:
            field += char
        index += 1
    return fields + [field]


def topological_order(n, edges):
    # Brute-force oracle differs from the expected heap/indegree implementation.
    for order in itertools.permutations(range(n)):
        positions = {node: i for i, node in enumerate(order)}
        if all(positions[u] < positions[v] for u, v in edges):
            return list(order)
    return []


def lru_results(capacity, operations):
    cache, recency, output = {}, [], []
    for operation in operations:
        kind, key, *value = operation
        if kind == 'get':
            output.append(cache.get(key, -1))
            if key not in cache:
                continue
        elif capacity == 0:
            continue
        else:
            if key not in cache and len(cache) == capacity:
                del cache[recency.pop(0)]
            cache[key] = value[0]
        if key in recency:
            recency.remove(key)
        recency.append(key)
    return output


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.config = ex.read_json(ex.ROOT / 'config.json')
        self.tasks = ex.read_json(ex.ROOT / 'tasks.json')

    def test_dataset_oracles(self):
        ex.validate(self.config, self.tasks)
        for task in self.tasks:
            for case in task['tests']:
                with self.subTest(task=task['task_id'], args=case['args']):
                    self.assertEqual(globals()[task['entry_point']](*copy.deepcopy(case['args'])), case['expected'])

    def test_paired_design_and_identical_task_bytes(self):
        trials = ex.plan_trials(self.config, self.tasks)
        self.assertEqual(trials, ex.plan_trials(self.config, self.tasks))
        self.assertEqual(len(trials), 30)
        self.assertEqual(len(set(trials)), 30)
        for task in self.tasks:
            for tone in self.config['tones']:
                self.assertEqual(ex.build_prompt(task, tone, self.config).split('\n\n', 1)[1], task['prompt'])
                self.assertNotIn('expected', ex.build_prompt(task, tone, self.config))

    def test_extraction(self):
        response = '```text\nlong irrelevant explanation\n```\n```python\ndef f(): return 1\n```'
        self.assertEqual(ex.extract_code(response), ('def f(): return 1', 'fenced'))
        self.assertEqual(ex.extract_code('def f(): pass'), ('def f(): pass', 'raw'))
        self.assertEqual(ex.extract_code('```\nx\n```\n```\nyyyy\n```')[0], 'yyyy')

    def test_bootstrap_and_incomplete_run(self):
        self.assertEqual(ex.bootstrap([0.25] * 5, 100, 42), (0.25, 0.25))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ex.append_record(path / 'raw.jsonl', {'kind': 'metadata', 'config': {}, 'planned_generations': 30})
            with self.assertRaisesRegex(ValueError, 'Incomplete run'):
                ex.analyze(path)

    def test_analysis_preserves_pairing_and_is_deterministic(self):
        config = {**self.config, 'bootstrap_samples': 100}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {'kind': 'metadata', 'config': config, 'tasks': self.tasks[:2],
                        'planned_generations': 12, 'model': {'digest': 'fixture'},
                        'dataset_sha256': 'fixture', 'source_sha256': 'fixture',
                        'ollama': {}, 'python': 'fixture', 'platform': 'fixture'}
            ex.append_record(path / 'raw.jsonl', metadata)
            for trial_id, (task_index, tone, rep) in enumerate(itertools.product(range(2), config['tones'], range(2))):
                value = [0.25, 0.75][task_index] + {'neutral': 0, 'friendly': 0.25, 'adversarial': -0.25}[tone]
                ex.append_record(path / 'raw.jsonl', {'kind': 'generation', 'trial_id': trial_id,
                    'task_id': self.tasks[task_index]['task_id'], 'tone': tone, 'repetition': rep,
                    'response_tokens': 100})
                ex.append_record(path / 'raw.jsonl', {'kind': 'evaluation', 'trial_id': trial_id,
                    'pass_rate': value, 'fully_correct': value == 1, 'runtime_error': False,
                    'timeout': False, 'compile_success': True})
            ex.analyze(path)
            with (path / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            friendly = next(row for row in rows if row['tone'] == 'friendly')
            self.assertEqual(float(friendly['difference_from_neutral']), 0.25)
            self.assertEqual(float(friendly['difference_ci_low']), 0.25)
            self.assertEqual(float(friendly['difference_ci_high']), 0.25)
            first = (path / 'report.html').read_bytes()
            ex.analyze(path)
            self.assertEqual(first, (path / 'report.html').read_bytes())


class EvaluatorTests(unittest.TestCase):
    task = {'entry_point': 'solve', 'tests': [{'args': [], 'expected': 3}]}

    def test_success_syntax_error_runtime_and_timeout(self):
        self.assertTrue(ex.evaluate('def solve(): return 3', self.task, 3)['fully_correct'])
        broken = ex.evaluate('def !!!', self.task, 3)
        self.assertFalse(broken['compile_success'])
        self.assertEqual(broken['pass_rate'], 0)
        self.assertTrue(ex.evaluate('def solve(): return 1/0', self.task, 3)['runtime_error'])
        self.assertTrue(ex.evaluate('def solve():\n while True: pass', self.task, 0.3)['timeout'])

    def test_stdout_and_input_mutation(self):
        self.assertTrue(ex.evaluate('def solve():\n print("noise")\n return 3', self.task, 3)['fully_correct'])
        task = {'entry_point': 'solve', 'check_no_mutation': True, 'tests': [{'args': [[1]], 'expected': 3}]}
        self.assertEqual(ex.evaluate('def solve(x):\n x.clear()\n return 3', task, 3)['pass_rate'], 0)

    def test_reference_solutions_pass_all_hidden_tests(self):
        for task in ex.read_json(ex.ROOT / 'tasks.json'):
            code = 'import itertools\n' + inspect.getsource(globals()[task['entry_point']])
            with self.subTest(task=task['task_id']):
                self.assertTrue(ex.evaluate(code, task, 3)['fully_correct'])

    def test_os_isolation(self):
        # Success means the forbidden action raised PermissionError.
        actions = [f'open({str(ex.ROOT / "config.json")!r})',
                   'open("/etc/passwd")', 'open("/tmp/tone-escape-probe", "w")',
                   '__import__("socket").create_connection(("127.0.0.1", 11434), timeout=1)',
                   '__import__("subprocess").run(["/bin/echo", "child"])']
        for action in actions:
            code = f'def solve():\n try:\n  {action}\n except (PermissionError, BlockingIOError):\n  return 3\n return 0'
            with self.subTest(action=action):
                self.assertTrue(ex.evaluate(code, self.task, 3)['fully_correct'])


if __name__ == '__main__':
    unittest.main()
