import unittest
import numpy as np

from experiments.habitat_figure_options import (
    INPUT, SOURCE, OUT, read, metadata, views, summarize, predicate_curve,
)


class FigureOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = read(INPUT)
        config = SOURCE / 'frozen_config.json'
        cls.meta = (metadata(cls.rs, read(config)) if config.exists() else
                    {r['task_id']: r for r in read(OUT.parent / 'habitat_memory_subgroups_20260923/task_metadata.json')})
        cls.data = views(cls.rs, cls.meta)
        cls.summaries, cls.rows = summarize(cls.data)

    def test_categories_partition_every_task(self):
        a, b = set(self.data['cooperative']), set(self.data['mixed'])
        self.assertEqual((len(a), len(b)), (10, 14))
        self.assertFalse(a & b)
        self.assertEqual(a | b, set(self.data['all']))
        ids = {self.meta[t]['episode_id'] for t, c in a}
        self.assertEqual(ids, {'1781', '772', '588', '618', '619'})

    def test_predicates_do_not_reduce_task_coverage(self):
        self.assertEqual(set(self.data['placement']), set(self.data['all']))
        self.assertEqual(set(self.data['opening']), set(self.data['all']))

    def test_predicate_curves_reconstruct_full_scores(self):
        for r in self.rs:
            p = predicate_curve(r, 'is_on_top')
            o = predicate_curve(r, 'is_open')
            goals = r['goal_ids']
            np_ = sum(g.startswith('is_on_top:') for g in goals)
            no = sum(g.startswith('is_open:') for g in goals)
            self.assertEqual(np_ + no, len(goals))
            for i, v in enumerate(self.data['all'][r['task_id'], r['condition']]):
                self.assertAlmostEqual((np_ * p[i] + no * o[i]) / len(goals), v)

    def test_all_exported_moments_match_numpy(self):
        for row in self.rows:
            vals = [v[row['turn']-1] for (t, c), v in self.data[row['view']].items()
                    if c == row['condition']]
            self.assertAlmostEqual(row['mean_percent'], 100 * np.mean(vals))
            self.assertAlmostEqual(row['sample_variance_pp_squared'], 10000 * np.var(vals, ddof=1))

    def test_unfavorable_temporal_results_retained(self):
        delta = {r['view']:r['delta_temporal_mean_pp'] for r in self.summaries}
        self.assertLess(delta['all'], 0)
        self.assertLess(delta['opening'], 0)
        self.assertLess(delta['mixed'], 0)
        self.assertAlmostEqual(delta['cooperative'], 49 / 6)

    def test_replication_snapshot_not_a_complete_seed(self):
        rows = read(OUT / 'additional_seed_coverage.json')
        completed = [r for r in rows if r['status'] == 'completed']
        self.assertEqual(len(completed), 8)
        self.assertEqual({r['seed'] for r in completed}, {891013})
        self.assertEqual(len({r['task_id'] for r in completed}), 4)
        aborted, = [r for r in rows if r['status'] == 'started_without_result']
        self.assertEqual(aborted['job_index'], 60)
        self.assertEqual(aborted['last_observed_turn'], 8)
        extra = read(OUT / 'additional_seed_measurements.json')
        tea = {r['condition']:r['final_completion'] for r in extra if r['job_index'] in (120, 122)}
        self.assertAlmostEqual(tea['nl_dynamic_v2'], 2/3)
        self.assertEqual(tea['dynamic_ssc'], 0)


if __name__ == '__main__':
    unittest.main()
