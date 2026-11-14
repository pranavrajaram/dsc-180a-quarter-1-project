"""Tests for matched conversations and conditional transition estimates."""
import copy
import unittest

from scripts.challenge import prepare, request_for, outcomes, ratio_interval
from scripts.experiment import ROOT, read_json


class ChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT/'configs/challenge.json'
        cls.config = read_json(cls.path)
        cls.spec = prepare(cls.config, cls.path)

    def test_shared_first_response_and_seed(self):
        self.assertEqual(len(self.spec['parents']), 80)
        self.assertEqual(len(self.spec['plan']), 240)
        for parent in self.spec['parents']:
            trials = [t for t in self.spec['plan'] if t['parent_id'] == parent['trial_id']]
            self.assertEqual({t['condition'] for t in trials}, set(self.config['conditions']))
            self.assertEqual(len({t['seed'] for t in trials}), 1)
            task = next(t for t in self.spec['tasks'] if t['task_id'] == parent['task_id'])
            requests = [request_for(t, parent, task, self.config) for t in trials]
            for request in requests:
                self.assertEqual(request['messages'][:2], parent['request']['messages'])
                self.assertEqual(request['messages'][2]['content'], parent['api_response']['message']['content'])
                self.assertEqual(len(request['messages']), 4)
            self.assertEqual(requests[0]['messages'][:3], requests[1]['messages'][:3])

    def test_transition_labels_and_invalid_conversation(self):
        spec = copy.deepcopy(self.spec)
        spec['plan'] = spec['plan'][:3]
        parents = {p['trial_id']: p for p in spec['parents']}
        tasks = {t['task_id']: t for t in spec['tasks']}
        rows = []
        for trial, state in zip(spec['plan'], ['correct', 'target_wrong', 'unparsed']):
            task = tasks[trial['task_id']]
            text = '\n'.join(f"Answer ({p['label']}): {p['answer']}" for p in task['parts'])
            if state == 'target_wrong':
                text = f"Answer ({task['hint_part']}): {task['hint']}"
            elif state == 'unparsed':
                text = 'Unable to answer.'
            rows.append(dict(**trial, request=request_for(trial, parents[trial['parent_id']],task,self.config),
                             api_response={'message': {'content': text}, 'done_reason': 'stop'}))
        self.assertEqual([r['final_state'] for r in outcomes(spec, rows)],
                         ['correct', 'target_wrong', 'unparsed'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            outcomes(spec, rows[:-1])
        rows[0]['request']['messages'][-1]['content'] = 'different challenge'
        with self.assertRaisesRegex(ValueError, 'Conversation'):
            outcomes(spec, rows)

    def test_conditional_denominators_and_zero_eligible(self):
        point, low, high = ratio_interval([(1, 1), (0, 2), (0, 0)], 1000, 42)
        self.assertEqual(point, 1/3)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)
        self.assertEqual(ratio_interval([(0,0)], 100, 42), (None,None,None))
        self.assertEqual(ratio_interval([(-1,2)]*5,100,42),(-.5,-.5,-.5))


if __name__ == '__main__':
    unittest.main()
