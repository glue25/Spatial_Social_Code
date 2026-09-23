import copy
import unittest

from experiments.plot_habitat_memory_progress import padded_values


def rollout(values, terminated=True):
    goals = ['g0', 'g1']
    return {'terminal_turn': len(values), 'terminated': terminated,
            'final_completion': values[-1], 'goal_ids': goals,
            'measurements': [
                {'turn': t, 'measurement_valid': True, 'completion': value,
                 'completed_goals': int(2 * value), 'total_goals': 2,
                 'goal_checks': [{'goal_id': g, 'satisfied': i < int(2 * value)}
                                 for i, g in enumerate(goals)]}
                for t, value in enumerate(values, 1)]}


class CurveTests(unittest.TestCase):
    def test_only_terminal_padding_and_retain_failure(self):
        values = padded_values(rollout([0, .5]))
        self.assertEqual(values, [0] + [.5] * 19)

    def test_preserve_physical_regression_not_cumulative_maximum(self):
        self.assertEqual(padded_values(rollout([0, 1, .5]))[:4], [0, 1, .5, .5])

    def test_reject_interior_gap_or_duplicate(self):
        r = rollout([0, .5, 1])
        for turn in (2, 4):
            modified = copy.deepcopy(r); modified['measurements'][2]['turn'] = turn
            with self.assertRaises(AssertionError): padded_values(modified)

    def test_reject_interrupted_episode(self):
        with self.assertRaises(AssertionError): padded_values(rollout([0, .5], False))

    def test_reject_invalid_measurement_and_mismatched_endpoint(self):
        r = rollout([0, .5]); r['measurements'][0]['measurement_valid'] = False
        with self.assertRaises(AssertionError): padded_values(r)
        r = rollout([0, .5]); r['final_completion'] = 1
        with self.assertRaises(AssertionError): padded_values(r)


if __name__ == '__main__':
    unittest.main()
