"""Verify/re-render published Habitat figures without private workdirs or APIs.

This checks the reduced measurement export, not a replay of simulator physics.
Run from repository root: python -m experiments.verify_habitat_figure_exports
Add --render to recreate plots from the published data (requires matplotlib).
"""
import argparse
import csv
import math

from experiments.plot_habitat_memory_progress import ROOT, read, padded_values, aggregate
from experiments.plot_habitat_memory_progress_stats import compute
from experiments.habitat_figure_options import views, summarize, plot
from experiments.plot_habitat_memory_subgroups import analyze


def compare_csv(path, expected, keys):
    with path.open(newline='') as f:
        actual = list(csv.DictReader(f))
    def key(row):
        return tuple(str(row[k]) for k in keys)
    lookup = {key(r): r for r in actual}
    assert len(lookup) == len(actual) == len(expected), path
    assert set(lookup) == {key(r) for r in expected}, path
    for r in expected:
        a = lookup[key(r)]
        for field, value in r.items():
            if isinstance(value, (int, float)):
                assert math.isclose(float(a[field]), value, rel_tol=1e-10, abs_tol=1e-9), (path, key(r), field)
            else:
                assert a[field] == value, (path, key(r), field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    base = ROOT / 'reports'
    original = base / 'habitat_memory_progress_20260923'
    stats = base / 'habitat_memory_progress_stats_20260923'
    subgroups = base / 'habitat_memory_subgroups_20260923'
    options = base / 'habitat_figure_options_20260923'
    rollouts = read(original / 'per_rollout_measurements.json')
    assert len(rollouts) == 24 and {r['seed'] for r in rollouts} == {891011}
    assert rollouts == read(stats / 'per_rollout_measurements.json')
    for r in rollouts:
        padded_values(r)
    meta = {r['task_id']:r for r in read(subgroups / 'task_metadata.json')}
    assert len(meta) == 12
    all_means = aggregate(rollouts)
    compare_csv(original / 'mean_completion_by_turn.csv', all_means, ('turn',))
    moments, paired = compute(rollouts)
    compare_csv(stats / 'mean_variance_by_turn.csv', moments, ('turn', 'condition'))
    compare_csv(stats / 'paired_difference_stats_by_turn.csv', paired, ('turn',))
    _, subgroup_turns = analyze(rollouts, meta)
    compare_csv(subgroups / 'subgroup_mean_variance_by_turn.csv', subgroup_turns,
                ('field', 'value', 'condition', 'turn'))
    summary, turns = summarize(views(rollouts, meta))
    compare_csv(options / 'figure_options.csv', summary, ('view', 'condition'))
    compare_csv(options / 'all_views_by_turn.csv', turns, ('view', 'condition', 'turn'))
    extra = read(options / 'additional_seed_measurements.json')
    assert len(extra) == 8 and {r['seed'] for r in extra} == {891013}
    for r in extra:
        padded_values(r)
    assert len({r['task_id'] for r in extra}) == 4
    if args.render:
        from experiments.plot_habitat_memory_progress import plot as plot_mean
        from experiments.plot_habitat_memory_progress_stats import plot as plot_stats
        from experiments.plot_habitat_memory_subgroups import plot as plot_subgroups
        plot_mean(all_means, original)
        plot_stats(moments, stats)
        plot_subgroups(rollouts, meta, subgroup_turns, subgroups)
        plot(turns, [('cooperative', 'Cooperative (all 5 tasks)'), ('mixed', 'Mixed (all 7 tasks)')],
             options / 'category_mean_sd', 'Original task categories')
        plot(turns, [('placement', 'Object placement (all 12 tasks)'), ('opening', 'Opening goals (all 12 tasks)')],
             options / 'goal_type_mean_sd', 'Physical goal types (task-macro averages)')
        plot(turns, [('cooperative', 'Cooperative (all 5 tasks)')],
             options / 'panel_d_cooperative_mean_sd', '(d) Goal completion over interaction')
    print('PASS: 24 complete-cohort traces + 8 partial-seed traces; published means, variances and subgroup statistics.')
    print('No API calls, private run directory, credentials or simulator required.')


if __name__ == '__main__':
    main()
