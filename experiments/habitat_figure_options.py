"""Offline figure-candidate inventory, including counterexamples and partial seeds.

Exploratory partitions are original metadata or original physical predicates.
No subsets are defined by outcomes. Individual case rankings are descriptive.
"""
from pathlib import Path
from statistics import mean

from experiments.plot_habitat_memory_progress import (
    ROOT, SOURCE, CONDITIONS, HORIZON, padded_values, read, write, sha, csv_write,
)
from experiments.plot_habitat_memory_progress_stats import moments
from experiments.plot_habitat_memory_subgroups import INPUT, metadata

OUT = ROOT / 'reports/habitat_figure_options_20260923'
STYLE = [('dynamic_ssc', 'Dynamic SSC', '#2166ac', '-'),
         ('nl_dynamic_v2', 'Dynamic NL (matched readout)', '#c86b16', '--')]


def predicate_curve(r, predicate):
    padded_values(r)  # Validate full physical record before deriving a view.
    values = []
    for m in r['measurements']:
        goals = [g for g in m['goal_checks'] if g['goal_id'].split(':')[0] == predicate]
        assert goals, 'No silently dropped task without this predicate'
        values.append(mean(g['satisfied'] for g in goals))
    return values + [values[-1]] * (HORIZON - len(values))


def views(rollouts, meta):
    configs = [('all', None, None), ('cooperative', 'category', 'cooperative'),
               ('mixed', 'category', 'mixed'), ('hard', 'split', 'hard'),
               ('standard', 'split', 'standard'),
               ('placement', 'predicate', 'is_on_top'), ('opening', 'predicate', 'is_open')]
    result = {}
    for label, field, value in configs:
        result[label] = {}
        for r in rollouts:
            if field not in (None, 'predicate') and meta[r['task_id']][field] != value:
                continue
            result[label][r['task_id'], r['condition']] = (
                predicate_curve(r, value) if field == 'predicate' else padded_values(r))
    return result


def summarize(data):
    import numpy as np
    summaries, rows = [], []
    for view, curves in data.items():
        tasks = sorted({t for t, c in curves})
        assert set(curves) == {(t, c) for t in tasks for c in CONDITIONS}
        delta = [100 * mean(curves[t, 'dynamic_ssc'][i] - curves[t, 'nl_dynamic_v2'][i]
                            for t in tasks) for i in range(HORIZON)]
        final_diffs = np.array([100 * (curves[t, 'dynamic_ssc'][-1] -
                                      curves[t, 'nl_dynamic_v2'][-1]) for t in tasks])
        rng = np.random.default_rng(20260923)
        ci = np.quantile(rng.choice(final_diffs, (50000, len(tasks))).mean(axis=1), [.025, .975])
        for c in CONDITIONS:
            s = {'view': view, 'condition': c, **moments([curves[t, c][-1] for t in tasks]),
                 'temporal_mean_percent': 100 * mean(mean(curves[t, c]) for t in tasks),
                 'delta_final_pp': delta[-1], 'delta_temporal_mean_pp': mean(delta),
                 'paired_bootstrap95_low_pp': float(ci[0]), 'paired_bootstrap95_high_pp': float(ci[1]),
                 'ssc_ahead_rounds': ','.join(str(i + 1) for i, d in enumerate(delta) if d > 1e-8),
                 'ssc_behind_rounds': ','.join(str(i + 1) for i, d in enumerate(delta) if d < -1e-8)}
            summaries.append(s)
            for i in range(HORIZON):
                rows.append({'view': view, 'condition': c, 'turn': i + 1,
                             **moments([curves[t, c][i] for t in tasks])})
    return summaries, rows


def extract_extra(cfg, index):
    folder = SOURCE / 'rollouts' / f'{index:04d}'
    result = read(folder / 'result.json')
    job = cfg['jobs'][index]
    assert all(result[k] == v for k, v in job.items())
    artifact = Path(result['artifact'])
    manifest = read(artifact / 'manifest.json')
    summary = read(artifact / 'summary.json')
    assert manifest['config_sha256'] == sha(SOURCE / 'frozen_config.json')
    assert manifest['status'] == 'completed' and manifest['model'] == cfg['model']
    assert manifest['memory_view'] == job['memory_view']
    assert manifest['condition'] == job['runtime_condition']
    assert int(manifest['seed']) == job['seed']
    assert summary['historical_benchmark_evaluation']['strict_physical'] == result['strict_physical']
    import json
    raw = [json.loads(l) for l in (artifact / 'physical_checks.jsonl').read_text().splitlines()]
    row = {'job_index': index, 'task_id': job['task_id'], 'seed': job['seed'],
           'condition': job['condition'], 'memory_view': job['memory_view'],
           'terminal_turn': summary['turns'], 'terminated': summary['done'],
           'service_affected': bool(result.get('service_affected', False)),
           'final_success': result['strict_physical']['strict_success'],
           'final_completion': result['strict_physical']['strict_percent_complete'],
           'goal_ids': sorted(g['goal_id'] for g in raw[-1]['checks']),
           'measurements': [{'turn': m['turn'], 'completed_goals': m['completed_goals'],
                            'total_goals': m['total_goals'],
                            'completion': m['strict_percent_complete'],
                            'measurement_valid': m['measurement_valid'],
                            'goal_checks': [{'goal_id': g['goal_id'], 'satisfied': g['satisfied']}
                                            for g in m['checks']]} for m in raw],
           'source_sha256': {name: sha(artifact / name) for name in
                             ('manifest.json', 'summary.json', 'physical_checks.jsonl')}}
    padded_values(row)
    return row


def replication(cfg):
    import json
    extra, coverage = [], []
    selected = read(SOURCE / 'priority72_manifest.json')['selected_indices']
    for index in selected:
        job = cfg['jobs'][index]
        if job['seed'] == 891011:
            continue
        folder = SOURCE / 'rollouts' / f'{index:04d}'
        row = {k: job[k] for k in ('seed', 'task_id', 'condition')}
        row['job_index'] = index
        if (folder / 'result.json').exists():
            extra.append(extract_extra(cfg, index))
            row['status'] = 'completed'
        elif (folder / 'STARTED.json').exists():
            row['status'] = 'started_without_result'
            paths = list((folder / 'benchmark').glob('results/*/closed_loop_artifacts/physical_checks.jsonl'))
            if paths:
                ms = [json.loads(l) for l in paths[0].read_text().splitlines()]
                row['last_observed_turn'] = ms[-1]['turn'] if ms else None
                row['physical_checks_sha256'] = sha(paths[0])
            row['handling'] = 'No final score imputed; not a terminal trajectory or substitute seed.'
        else:
            row['status'] = 'not_started'
        coverage.append(row)
    return extra, coverage


def cases(rollouts, meta):
    rows = []
    for seed in sorted({r['seed'] for r in rollouts}):
        lookup = {(r['task_id'], r['condition']): r for r in rollouts if r['seed'] == seed}
        for t in sorted({t for t, c in lookup}):
            assert all((t, c) in lookup for c in CONDITIONS)
            nl, ssc = [lookup[t, c] for c in CONDITIONS]
            row = {'task_id': t, 'episode_id': meta[t]['episode_id'], 'title': meta[t]['title'],
                   'category': meta[t]['category'], 'split': meta[t]['split'], 'seed': seed}
            for c, r in [('nl', nl), ('ssc', ssc)]:
                row.update({f'{c}_job': r['job_index'], f'{c}_final_percent': 100 * r['final_completion'],
                            f'{c}_strict_success': r['final_success'], f'{c}_terminal_turn': r['terminal_turn'],
                            f'{c}_service_affected': r['service_affected']})
            diff = [100 * (a - b) for a, b in zip(padded_values(ssc), padded_values(nl))]
            row.update(delta_final_pp=diff[-1], delta_temporal_mean_pp=mean(diff),
                       ssc_behind_rounds=','.join(str(i+1) for i,d in enumerate(diff) if d < -1e-8))
            rows.append(row)
    return rows


def plot(rows, panels, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.4), squeeze=False)
    for ax, (view, label) in zip(axs[0], panels):
        for c, legend, color, style in STYLE:
            rs = [r for r in rows if r['view'] == view and r['condition'] == c]
            assert len(rs) == HORIZON
            x = [r['turn'] for r in rs]
            ax.fill_between(x, [r['mean_minus_sd_percent'] for r in rs],
                            [r['mean_plus_sd_percent'] for r in rs], color=color, alpha=.12)
            ax.plot(x, [r['mean_percent'] for r in rs], color=color, linestyle=style,
                    label=legend, linewidth=2.1)
        ax.set(title=label, xlim=(1, 20), ylim=(0, 100), xticks=[1, 5, 10, 15, 20],
               yticks=[0, 25, 50, 75, 100], xlabel='Interaction round')
        ax.grid(axis='y', alpha=.2)
    axs[0][0].set_ylabel('Mean physical goal completion (%)')
    h, l = axs[0][0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=12, fontweight='bold', y=.98)
    fig.legend(h, l, loc='upper center', ncol=1 if len(panels)==1 else 2,
               bbox_to_anchor=(.5, .92), frameon=False, fontsize=9)
    fig.text(.5, .025, 'Exploratory; seed 891011. Mean ± between-task sample SD.\n'
             'Failures retained; terminal states carried forward to round 20.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .78))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output.with_suffix('.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read(SOURCE / 'frozen_config.json')
    rs = read(INPUT)
    meta = metadata(rs, cfg)
    data = views(rs, meta)
    summaries, turns = summarize(data)
    csv_write(OUT / 'figure_options.csv', summaries)
    csv_write(OUT / 'all_views_by_turn.csv', turns)
    extra, coverage = replication(cfg)
    write(OUT / 'additional_seed_measurements.json', extra)
    write(OUT / 'additional_seed_coverage.json', coverage)
    csv_write(OUT / 'all_paired_cases.csv', cases(rs + extra, meta))
    # Partial-seed comparisons always use the same covered tasks in both seeds.
    covered = {r['task_id'] for r in extra}
    partial = [r for r in rs + extra if r['task_id'] in covered]
    write(OUT / 'partial_seed_summary.json', [
        {'seed': seed, 'condition': c, 'n_tasks': len(sub),
         'successes': sum(r['final_success'] for r in sub),
         **moments([r['final_completion'] for r in sub])}
        for seed in sorted({r['seed'] for r in partial}) for c in CONDITIONS
        for sub in [[r for r in partial if r['seed']==seed and r['condition']==c]]])
    plot(turns, [('cooperative', 'Cooperative (all 5 tasks)'), ('mixed', 'Mixed (all 7 tasks)')],
         OUT / 'category_mean_sd', 'Original task categories')
    plot(turns, [('placement', 'Object placement (all 12 tasks)'), ('opening', 'Opening goals (all 12 tasks)')],
         OUT / 'goal_type_mean_sd', 'Physical goal types (task-macro averages)')
    plot(turns, [('cooperative', 'Cooperative (all 5 tasks)')],
         OUT / 'panel_d_cooperative_mean_sd', '(d) Goal completion over interaction')
    long_rows = []
    for view, curves in data.items():
        for (task, c), values in sorted(curves.items()):
            for turn, value in enumerate(values, 1):
                long_rows.append({'view':view, 'task_id':task, 'seed':891011,
                                  'condition':c, 'turn':turn, 'completion':value})
    csv_write(OUT / 'task_level_plot_values.csv', long_rows)
    write(OUT / 'provenance.json', {'input':str(INPUT.relative_to(ROOT)), 'input_sha256':sha(INPUT),
          'config_sha256':sha(SOURCE / 'frozen_config.json'), 'script_sha256':sha(Path(__file__)),
          'scope':'Offline exploratory figure audit. No new experiments.',
          'views':list(data), 'predicate_weighting':'Mean within task, then equal weight across tasks.',
          'additional_seeds':'Incomplete coverage only; never merged into the full 12-task figure.',
          'uncertainty':'Sample SD across tasks, ddof=1. Bootstrap CIs are unadjusted exploratory intervals.'})
    print(OUT)


if __name__ == '__main__':
    main()
