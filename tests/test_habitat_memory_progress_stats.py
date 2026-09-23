import unittest
from experiments.plot_habitat_memory_progress_stats import moments, compute


class StatisticsTests(unittest.TestCase):
    def test_ddof_one_and_unit_conversion(self):
        result = moments([0, 1])
        self.assertEqual(result['mean_percent'], 50)
        self.assertEqual(result['sample_variance_fraction_squared'], .5)
        self.assertEqual(result['sample_variance_pp_squared'], 5000)
        self.assertAlmostEqual(result['sample_sd_pp'], 70.71067811865476)
        self.assertGreater(result['mean_plus_sd_percent'], 100)

    def test_constant_values(self):
        self.assertEqual(moments([.5] * 12)['sample_sd_pp'], 0)

    def test_pairing_preserves_covariance(self):
        rows = []
        for i in range(12):
            for c in ('dynamic_ssc', 'nl_dynamic_v2'):
                value = i % 2
                rows.append({'task_id': str(i), 'condition': c, 'seed': 891011,
                    'terminal_turn': 1, 'terminated': True, 'final_completion': value,
                    'goal_ids': ['g'], 'measurements': [{'turn': 1, 'measurement_valid': True,
                        'completion': value, 'completed_goals': value, 'total_goals': 1,
                        'goal_checks': [{'goal_id': 'g', 'satisfied': bool(value)}]}]})
        means, differences = compute(rows)
        self.assertGreater(means[0]['sample_variance_pp_squared'], 0)
        self.assertTrue(all(r['sample_variance_difference_pp_squared'] == 0 for r in differences))

    def test_reject_singleton_statistics(self):
        with self.assertRaises(AssertionError): moments([1])


if __name__ == '__main__':
    unittest.main()
