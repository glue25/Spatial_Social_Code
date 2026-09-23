"""Offline, post-hoc subgroup audit using original frozen task metadata.

No outcome-defined subsets. Show both difficulty splits, and export every
inspected metadata partition. Positive individual cases are explicitly selected
after outcomes; they do not constitute an evaluation subset.
"""
from pathlib import Path
from statistics import mean

from experiments.plot_habitat_memory_progress import (
    ROOT, SOURCE, CONDITIONS, HORIZON, SEED, read, write, sha, csv_write,
    padded_values,
)
from experiments.plot_habitat_memory_progress_stats import moments

OUT = ROOT / 'reports/habitat_memory_subgroups_20260923'
INPUT = ROOT / 'reports/habitat_memory_progress_20260923/per_rollout_measurements.json'
FIELDS = ('split', 'category', 'tom_level', 'num_agents')


def metadata(rollouts, cfg):
    result = {}
    for r in rollouts:
        job = cfg['jobs'][r['job_index']]
        assert job['seed'] == r['seed'] == SEED
        assert job['condition'] == r['condition']
        path = Path(job['task'])
        task = read(path)
        assert task['task_id'] == r['task_id'] == job['task_id']
        row = {k: task[k] for k in ('split', 'category', 'tom_level')}
        row.update(num_agents=task.get('num_agents', len(task['agent_spawns'])),
                   task_id=task['task_id'], title=task['title'],
                   episode_id=str(task['episode_id']),
                   active_mechanics=task['active_mechanics'],
                   tom_level_method=task.get('tom_level_method'),
                   task_source=str(path.relative_to(ROOT)), task_sha256=sha(path))
        if r['task_id'] in result:
            assert result[r['task_id']] == row
        result[r['task_id']] = row
    assert len(result) == 12
    assert {(r['task_id'], r['condition']) for r in rollouts} == {
        (t, c) for t in result for c in CONDITIONS}
    return result


def analyze(rollouts, meta):
    import numpy as np
    lookup = {(r['task_id'], r['condition']): r for r in rollouts}
    curves = {key: padded_values(r) for key, r in lookup.items()}
    summaries, turns = [], []
    for field in FIELDS:
        for value in sorted({m[field] for m in meta.values()}):
            tasks = sorted(t for t in meta if meta[t][field] == value)
            n = len(tasks)
            assert n >= 2
            diffs = np.array([100 * (curves[t, 'dynamic_ssc'][-1] -
                                    curves[t, 'nl_dynamic_v2'][-1]) for t in tasks])
            # Paired task bootstrap; exploratory, pointwise, not multiplicity-adjusted.
            rng = np.random.default_rng(20260923)
            samples = rng.choice(diffs, size=(50000, n), replace=True).mean(axis=1)
            ci = np.quantile(samples, [.025, .975]).tolist()
            for c in CONDITIONS:
                vals = [curves[t, c][-1] for t in tasks]
                summaries.append({'field': field, 'value': value, 'condition': c,
                    **moments(vals),
                    'successes': sum(lookup[t, c]['final_success'] for t in tasks),
                    'success_percent': 100 * mean(lookup[t, c]['final_success'] for t in tasks),
                    'mean_completion_over_20_rounds_percent':
                        100 * mean(mean(curves[t, c]) for t in tasks),
                    'ssc_minus_nl_final_pp': float(diffs.mean()),
                    'paired_bootstrap_95_low_pp': ci[0],
                    'paired_bootstrap_95_high_pp': ci[1],
                    'service_affected_jobs': [lookup[t, c]['job_index'] for t in tasks
                                              if lookup[t, c]['service_affected']],
                    'task_ids': tasks})
                for turn in range(HORIZON):
                    turns.append({'field': field, 'value': value, 'condition': c,
                                  'turn': turn + 1,
                                  **moments([curves[t, c][turn] for t in tasks])})
    return summaries, turns


def plot(rollouts, meta, turns, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    styles = [('dynamic_ssc', 'Dynamic SSC', '#2166ac', '-'),
              ('nl_dynamic_v2', 'Dynamic NL (matched readout)', '#c86b16', '--')]

    def finish(fig, axes, name, note):
        for ax in axes:
            ax.set(xlim=(1, 20), ylim=(0, 103), xticks=[1, 5, 10, 15, 20],
                   yticks=[0, 25, 50, 75, 100], xlabel='Interaction round')
            ax.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Physical goal completion (%)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(.5, .92))
        fig.text(.5, .025, note, ha='center', fontsize=9, color='#555555')
        fig.tight_layout(rect=(0, .10, 1, .83))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True)
    fig.suptitle('Original task difficulty splits: exploratory paired analysis',
                 y=.99, fontweight='bold', fontsize=13)
    for ax, split in zip(axes, ('hard', 'standard')):
        ax.set_title(f'{split.capitalize()} tasks (n = 6)', loc='left', fontweight='bold')
        for c, label, color, style in styles:
            rows = [r for r in turns if r['field'] == 'split' and
                    r['value'] == split and r['condition'] == c]
            assert len(rows) == HORIZON
            x = [r['turn'] for r in rows]
            ax.fill_between(x, [r['mean_minus_sd_percent'] for r in rows],
                            [r['mean_plus_sd_percent'] for r in rows], color=color, alpha=.12)
            ax.plot(x, [r['mean_percent'] for r in rows], color=color,
                    linestyle=style, label=label, linewidth=2.2)
    finish(fig, axes, 'difficulty_mean_sd',
           'Mean ± between-task sample SD; one seed (891011). All tasks in each split retained.\n'
           'Terminal scores carried forward; failures included. Bands clipped at plotting bounds.')

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True)
    fig.suptitle('Illustrative paired trajectories (selected after outcomes)',
                 y=.99, fontweight='bold', fontsize=13)
    for ax, episode, title in zip(axes, ('1781', '1166'),
                                  ('Tea reset: 3/3 vs 1/3 goals',
                                   'Guest reset: completion at round 8 vs 13')):
        ax.set_title(title, loc='left', fontweight='bold')
        for c, label, color, style in styles:
            r, = [r for r in rollouts if meta[r['task_id']]['episode_id'] == episode
                  and r['condition'] == c]
            ax.step(range(1, 21), [100 * x for x in padded_values(r)], where='post',
                    color=color, linestyle=style, label=label, linewidth=2.2)
    finish(fig, axes, 'illustrative_cases',
           'Individual runs, seed 891011: no variance estimate. Not an aggregate evaluation subset.\n'
           'Natural-language condition renders the same revisable state store.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts = read(INPUT)
    assert len(rollouts) == 24
    meta = metadata(rollouts, read(SOURCE / 'frozen_config.json'))
    summaries, turns = analyze(rollouts, meta)
    write(OUT / 'task_metadata.json', list(meta.values()))
    write(OUT / 'subgroup_summary.json', summaries)
    csv_write(OUT / 'subgroup_mean_variance_by_turn.csv', turns)
    ledger = []
    for r in rollouts:
        ledger.append({k: v for k, v in {**meta[r['task_id']], **r,
            'mean_completion_over_20_rounds': mean(padded_values(r))}.items()
            if k != 'measurements'})
    write(OUT / 'all_task_ledger.json', ledger)
    write(OUT / 'provenance.json', {'source': str(INPUT.relative_to(ROOT)),
        'source_sha256': sha(INPUT), 'seed': SEED, 'analysis': 'post-hoc exploratory',
        'all_inspected_partition_fields': list(FIELDS),
        'bootstrap': '50000 paired task resamples; RNG 20260923; percentile 95%; unadjusted',
        'case_selection': 'Positive illustrative episodes 1781 and 1166 selected after outcomes.'})
    plot(rollouts, meta, turns, OUT)
    print(OUT)
    for r in summaries:
        if r['field'] == 'split':
            print(r['value'], r['condition'], r['mean_percent'], r['sample_sd_pp'],
                  r['successes'], r['ssc_minus_nl_final_pp'],
                  (r['paired_bootstrap_95_low_pp'], r['paired_bootstrap_95_high_pp']))


if __name__ == '__main__':
    main()
