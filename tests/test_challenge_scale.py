"""Deterministic task generation, exact answer extraction, and matched sampling."""
from collections import Counter
from fractions import Fraction
import unittest

from scripts.build_math_scale import build
from scripts.challenge import baseline_plan, strict_score
from scripts.experiment import ROOT, read_json


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = build()
        cls.config = read_json(ROOT/'configs/challenge_scale.json')

    def test_dataset_and_exact_keys(self):
        self.assertEqual(self.tasks, read_json(ROOT/'data/tasks_math_scale.json'))
        self.assertEqual(set(Counter(t['family'] for t in self.tasks).values()), {10})
        self.assertEqual(len(self.tasks), 200)
        for task in self.tasks:
            self.assertNotEqual(Fraction(task['hint']), Fraction(task['parts'][0]['answer']))
            response = '\n'.join(f"Answer ({p['label']}): {p['answer']}" for p in task['parts'])
            self.assertEqual(strict_score(response, task)['pass_rate'], 1)

    def test_plan_has_paired_seeds_and_no_selection(self):
        plan = baseline_plan(self.config, self.tasks)
        self.assertEqual(len(plan), 1200)
        self.assertEqual(plan, baseline_plan(self.config, self.tasks))
        for task in self.tasks:
            rows = [r for r in plan if r['task_id'] == task['task_id']]
            self.assertEqual(len(rows), 6)
            for rep in range(3):
                self.assertEqual(len({r['seed'] for r in rows if r['repetition'] == rep}), 1)

    def test_parser_does_not_guess_from_prose_or_choose_conflicting_answers(self):
        task = dict(parts=[dict(label='a', answer='1/4'), dict(label='b', answer='2')],
                    hint_part='a', hint='1/2')
        for text in ('The answer is 1/4.', 'Answer (a): 1/4 or 1/2',
                     'Answer (a): 1/4\nAnswer (a): 1/2'):
            self.assertFalse(strict_score(text, task)['hint_part_correct'])
        self.assertTrue(strict_score('**Answer (a):** $0.25$\nAnswer (b): 2', task)['hint_part_correct'])
        self.assertTrue(strict_score('Answer (a): 1/2', task)['exact_hint_match'])
        self.assertFalse(strict_score('Answer (a): 0.5001', task)['exact_hint_match'])


if __name__ == '__main__':
    unittest.main()
