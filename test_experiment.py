"""Run with python3 -B -m unittest -v. Includes actual OS-isolation checks (macOS)."""
import copy
import csv
import inspect
import itertools
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import build_coding
import build_math
import experiment as ex


# Independent reference implementations for the five pilot tasks.
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


PILOT = {'merge_intervals': merge_intervals, 'compare_versions': compare_versions, 'split_escaped': split_escaped,
         'topological_order': topological_order, 'lru_results': lru_results}


def reference_source(task):
    """Source code of a reference solution, used to check the evaluator end to end."""
    if task['entry_point'] in PILOT:
        return 'import itertools\n' + inspect.getsource(PILOT[task['entry_point']])
    imports = 'import bisect\nimport copy\nimport heapq\nimport itertools\nimport re\nfrom collections import deque\n'
    return imports + inspect.getsource(getattr(build_coding, task['entry_point']))


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.config = ex.read_json(ex.ROOT / 'config.json')
        self.tasks = ex.load_tasks(self.config)

    def test_datasets_match_independent_oracles(self):
        ex.validate(self.config, self.tasks)
        coding = [t for t in self.tasks if t['domain'] == 'coding']
        math = [t for t in self.tasks if t['domain'] == 'math']
        self.assertEqual((len(coding), len(math)), (20, 20))
        brutes = {name: brute for name, _, _, _, brute, _, _ in build_coding.NEW_TASKS}
        for task in coding:
            oracle = PILOT.get(task['entry_point']) or brutes[task['task_id']]
            for case in task['tests']:
                with self.subTest(task=task['task_id'], args=case['args']):
                    self.assertEqual(oracle(*copy.deepcopy(case['args'])), case['expected'])
        # Math answers and hints are recomputed from scratch by brute force.
        self.assertEqual(build_math.build(), math)
        for task in math:
            self.assertTrue(3 <= len(task['parts']) <= 4)
            self.assertNotEqual(Fraction(task['hint']), Fraction(task['parts'][-1]['answer']))

    def test_paired_design_hints_and_identical_task_bytes(self):
        trials = ex.plan_trials(self.config, self.tasks)
        self.assertEqual(trials, ex.plan_trials(self.config, self.tasks))
        self.assertEqual(len(trials), 40 * 7 * 2)
        self.assertEqual(len(set(trials)), len(trials))
        for task in self.tasks:
            for tone in self.config['tones']:
                for rep in range(self.config['repetitions']):
                    hint = ex.uses_hint(task, rep, self.config)
                    self.assertEqual(hint, task['domain'] == 'math' and rep == 1)
                    prompt = ex.build_prompt(task, tone, self.config, hint)
                    prefix = self.config['tones'][tone]
                    expected = (prefix + '\n\n' if prefix else '') + task['prompt']
                    if hint:
                        expected += '\n\n' + self.config['hint_template'].format(hint=task['hint'])
                    self.assertEqual(prompt, expected)
                    if task['domain'] == 'coding':  # hidden test answers never reach the model
                        self.assertNotIn('expected', prompt)

    def test_tone_prefixes_similar_length(self):
        lengths = [len(p.split()) for t, p in self.config['tones'].items() if t != 'none']
        self.assertLessEqual(max(lengths) - min(lengths), 2)

    def test_dry_run_subset(self):
        small = ex.subset(self.tasks, 2)
        self.assertEqual([t['domain'] for t in small], ['coding', 'coding', 'math', 'math'])
        self.assertEqual(len(ex.plan_trials(self.config, small)), 4 * 7 * 2)

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

    def test_analysis_groups_pairing_and_determinism(self):
        config = {**self.config, 'bootstrap_samples': 100}
        tasks = [t for t in self.tasks if t['domain'] == 'coding'][:2] + [t for t in self.tasks if t['domain'] == 'math'][:2]
        shift = {tone: 0 for tone in config['tones']}
        shift.update({'very_polite': 0.25, 'very_rude': -0.25})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ex.append_record(path / 'raw.jsonl', {'kind': 'metadata', 'config': config, 'tasks': tasks,
                             'planned_generations': 4 * 7 * 2, 'model': {'digest': 'fixture'},
                             'dataset_sha256': 'fixture', 'source_sha256': 'fixture',
                             'ollama': {}, 'python': 'fixture', 'platform': 'fixture'})
            trials = itertools.product(range(4), config['tones'], range(2))
            for trial_id, (i, tone, rep) in enumerate(trials):
                hint = ex.uses_hint(tasks[i], rep, config)
                value = [0.25, 0.5, 0.5, 0.75][i] + shift[tone]
                ex.append_record(path / 'raw.jsonl', {'kind': 'generation', 'trial_id': trial_id,
                    'task_id': tasks[i]['task_id'], 'domain': tasks[i]['domain'], 'tone': tone,
                    'repetition': rep, 'hint': hint, 'response_tokens': 100, 'truncated': False})
                ex.append_record(path / 'raw.jsonl', {'kind': 'evaluation', 'trial_id': trial_id,
                    'pass_rate': value, 'fully_correct': value == 1, 'runtime_error': False, 'timeout': False,
                    'compile_success': True, 'hint_adopted': (tone == 'very_rude') if hint else None})
            ex.analyze(path)
            with (path / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['group'] for r in rows}, {'all_no_hint', 'coding', 'math_no_hint', 'math_hint'})
            def get(group, tone, field):
                return next(float(r[field]) for r in rows if r['group'] == group and r['tone'] == tone)
            for group in ('all_no_hint', 'coding', 'math_no_hint', 'math_hint'):
                self.assertEqual(get(group, 'very_polite', 'difference_from_baseline'), 0.25)
                self.assertEqual(get(group, 'very_polite', 'difference_ci_low'), 0.25)
                self.assertEqual(get(group, 'very_rude', 'difference_ci_high'), -0.25)
            self.assertEqual(get('math_hint', 'very_rude', 'hint_adoption_rate'), 1.0)
            self.assertEqual(get('math_hint', 'control', 'hint_adoption_rate'), 0.0)
            self.assertEqual(get('all_no_hint', 'control', 'n_generations'), 2 * 2 + 2 * 1)
            first = (path / 'report.html').read_bytes()
            ex.analyze(path)
            self.assertEqual(first, (path / 'report.html').read_bytes())


class MathGradingTests(unittest.TestCase):
    def setUp(self):
        tasks = ex.read_json(ex.ROOT / 'tasks_math.json')
        self.task = next(t for t in tasks if t['task_id'] == 'matching_socks')  # 190, 10, 1/19, 3/19; hint 16/19

    def score(self, response, hint=True):
        result = ex.evaluate_math(response, self.task, hint)
        return result['tests_passed'], result['hint_adopted']

    def test_lenient_formats(self):
        formats = [
            'Work...\nAnswer (a): 190\nAnswer (b): 10\nAnswer (c): 1/19\nAnswer (d): 3/19',
            '(a) C(20,2) = 190.\n(b) There are 10.\n(c) 10/190 = 1/19 ≈ 0.0526\n(d) 1 - 16/19 = 3/19',
            '### Part (a)\n$190$\n### Part (b)\n10\n### Part (c)\n$\\boxed{\\frac{1}{19}}$\n### Part (d)\n$\\boxed{\\dfrac{3}{19}}$',
            'a) 190\nb) 10\nc) 5.26%\nd) 0.1579',
            '**(a)** 190 ways\n**(b)** 10\n**(c)** The probability is 1/19.\n**(d)** 3/19.',
            'Part A: 190\nPart B: 10\nPart C: 1/19\nPart D: 3/19',
            '(a) 190\n(b) 10\n(c) Using part (a), 10/190 = 1/19\n(d) 3/19',
            'The answers are 190, 10, 1/19 and 3/19.',
        ]
        for response in formats:
            with self.subTest(response=response):
                self.assertEqual(self.score(response), (4, False))

    def test_partial_credit_rounding_and_hint(self):
        self.assertEqual(self.score('(a) 190\n(b) 10\n(c) 1/19\n(d) 16/19'), (3, True))
        self.assertEqual(self.score('(a) 190\n(b) 10\n(c) 0.05\n(d) 0.16'), (2, False))
        self.assertEqual(self.score('(a) 380\n(b) 20\n(c) 1/19\n(d) 3/19', hint=False), (2, None))
        self.assertEqual(self.score('I cannot solve this.'), (0, False))

    def test_number_parsing(self):
        self.assertEqual(ex.parse_value('total 1,326 hands'), (Fraction(1326), True))
        self.assertEqual(ex.parse_value('$79.20'), (Fraction('79.2'), False))
        self.assertEqual(ex.parse_value('\\frac{6}{4165}'), (Fraction(6, 4165), True))
        self.assertTrue(ex.same_value(ex.parse_value('0.0526'), Fraction(1, 19)))
        self.assertFalse(ex.same_value(ex.parse_value('0.05'), Fraction(1, 19)))
        self.assertFalse(ex.same_value(ex.parse_value('19'), Fraction(1, 19)))


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
        for task in ex.read_json(ex.ROOT / 'tasks_coding.json'):
            with self.subTest(task=task['task_id']):
                self.assertTrue(ex.evaluate(reference_source(task), task, 3)['fully_correct'])

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
