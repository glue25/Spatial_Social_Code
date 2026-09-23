import unittest
import numpy as np

from experiments.plot_habitat_memory_progress import ROOT, read, SOURCE, padded_values
from experiments.plot_habitat_memory_subgroups import INPUT, metadata, analyze


class SubgroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = read(INPUT)
        config = SOURCE / 'frozen_config.json'
        cls.meta = (metadata(cls.r, read(config)) if config.exists() else
                    {r['task_id']: r for r in read(ROOT / 'reports/habitat_memory_subgroups_20260923/task_metadata.json')})
        cls.summary, cls.turns = analyze(cls.r, cls.meta)

    def test_original_splits_partition_complete_cohort(self):
        groups = [{t for t, m in self.meta.items() if m['split'] == s}
                  for s in ('hard', 'standard')]
        self.assertEqual([len(g) for g in groups], [6, 6])
        self.assertFalse(groups[0] & groups[1])
        self.assertEqual(groups[0] | groups[1], set(self.meta))
        hard_ids = {self.meta[t]['episode_id'] for t in groups[0]}
        self.assertEqual(hard_ids, {'1781', '1013', '623', '744', '1166', '772'})

    def test_every_moment_matches_numpy(self):
        for row in self.turns:
            values = [padded_values(r)[row['turn'] - 1] for r in self.r
                      if r['condition'] == row['condition'] and
                      self.meta[r['task_id']][row['field']] == row['value']]
            self.assertEqual(len(values), row['n_tasks'])
            self.assertAlmostEqual(100 * np.mean(values), row['mean_percent'])
            self.assertAlmostEqual(10000 * np.var(values, ddof=1),
                                   row['sample_variance_pp_squared'])

    def test_hard_does_not_drop_counterexample(self):
        hard = [r for r in self.summary if r['field'] == 'split' and r['value'] == 'hard']
        self.assertEqual([r['successes'] for r in hard], [3, 4])
        self.assertAlmostEqual(hard[0]['ssc_minus_nl_final_pp'], 100 / 9)
        self.assertLess(hard[0]['paired_bootstrap_95_low_pp'], 0)
        self.assertGreater(hard[0]['paired_bootstrap_95_high_pp'], 0)
        losing = [r for r in self.r if self.meta[r['task_id']]['episode_id'] == '744']
        self.assertTrue(all(r['task_id'] in hard[0]['task_ids'] for r in losing))
        self.assertEqual([r['final_success'] for r in losing], [True, False])

    def test_service_affected_run_retained(self):
        row, = [r for r in self.summary if r['field'] == 'split' and
                r['value'] == 'standard' and r['condition'] == 'nl_dynamic_v2']
        self.assertEqual(row['service_affected_jobs'], [35])
        self.assertEqual(row['n_tasks'], 6)


if __name__ == '__main__':
    unittest.main()
