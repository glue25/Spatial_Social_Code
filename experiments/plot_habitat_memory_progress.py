"""Plot real post-action physical completion, never interpolate missing rounds.

Offline only. Exports the complete first-seed paired block already published in
the dynamic-memory comparison. No API calls or changes to experiment workers.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'workdirs/habitat_memory_controls_20260922_v1'
OUTPUT = ROOT / 'reports/habitat_memory_progress_20260923'
CONDITIONS = ('nl_dynamic_v2', 'dynamic_ssc')
SEED = 891011
HORIZON = 20


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def padded_values(rollout):
    checks = rollout['measurements']
    n = rollout['terminal_turn']
    assert 1 <= n <= HORIZON and len(checks) == n
    assert [c['turn'] for c in checks] == list(range(1, n + 1)), 'Missing/duplicate interior round'
    assert n == HORIZON or rollout['terminated'], 'Cannot carry forward an interrupted episode'
    values = []
    for c in checks:
        assert c['measurement_valid'] is True, 'No imputation of invalid physics'
        assert 0 <= c['completed_goals'] <= c['total_goals'] and c['total_goals'] > 0
        assert math.isclose(c['completion'], c['completed_goals'] / c['total_goals'])
        assert c['completed_goals'] == sum(g['satisfied'] for g in c['goal_checks'])
        assert c['total_goals'] == len(c['goal_checks'])
        assert {g['goal_id'] for g in c['goal_checks']} == set(rollout['goal_ids'])
        values.append(c['completion'])
    assert math.isclose(values[-1], rollout['final_completion'])
    return values + [values[-1]] * (HORIZON - n)


def extract(source):
    cfg = read(source / 'frozen_config.json')
    scope = read(source / 'priority72_manifest.json')
    indices = [i for i in scope['selected_indices'] if cfg['jobs'][i]['seed'] == SEED]
    assert len(indices) == 24
    result = []
    for i in indices:
        folder = source / 'rollouts' / f'{i:04d}'
        row = read(folder / 'result.json'); job = cfg['jobs'][i]
        assert all(row[k] == v for k, v in job.items())
        artifact = Path(row['artifact'])
        manifest = read(artifact / 'manifest.json'); summary = read(artifact / 'summary.json')
        assert manifest['config_sha256'] == sha(source / 'frozen_config.json')
        assert manifest['model'] == cfg['model'] and manifest['status'] == 'completed'
        assert manifest['memory_view'] == job['memory_view'] and int(manifest['seed']) == SEED
        assert manifest['condition'] == job['runtime_condition']
        assert summary['historical_benchmark_evaluation']['strict_physical'] == row['strict_physical']
        raw = [json.loads(line) for line in (artifact / 'physical_checks.jsonl').read_text().splitlines() if line.strip()]
        item = {'job_index': i, 'task_id': job['task_id'], 'seed': SEED,
                'condition': job['condition'], 'memory_view': job['memory_view'],
                'terminal_turn': summary['turns'], 'terminated': summary['done'],
                'service_affected': bool(row.get('service_affected', False)),
                'final_success': row['strict_physical']['strict_success'],
                'final_completion': row['strict_physical']['strict_percent_complete'],
                'goal_ids': sorted(c['goal_id'] for c in raw[-1]['checks']),
                'measurements': [{'turn': c['turn'], 'completed_goals': c['completed_goals'],
                     'total_goals': c['total_goals'], 'completion': c['strict_percent_complete'],
                     'measurement_valid': c['measurement_valid'],
                     'goal_checks': [{'goal_id': g['goal_id'], 'satisfied': g['satisfied']} for g in c['checks']]}
                     for c in raw],
                'source_sha256': {'result.json': sha(folder / 'result.json'),
                     'manifest.json': sha(artifact / 'manifest.json'),
                     'summary.json': sha(artifact / 'summary.json'),
                     'physical_checks.jsonl': sha(artifact / 'physical_checks.jsonl')}}
        padded_values(item)
        result.append(item)
    tasks = {r['task_id'] for r in result}
    assert len(tasks) == 12
    assert {(r['task_id'], r['condition']) for r in result} == {(t, c) for t in tasks for c in CONDITIONS}
    for t in tasks:
        a, b = [r for r in result if r['task_id'] == t]
        assert a['goal_ids'] == b['goal_ids']
    return result


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def aggregate(rollouts):
    curves = {c: [padded_values(r) for r in rollouts if r['condition'] == c] for c in CONDITIONS}
    assert all(len(v) == 12 for v in curves.values())
    rows = []
    for t in range(HORIZON):
        means = {c: sum(v[t] for v in curves[c]) / len(curves[c]) for c in CONDITIONS}
        rows.append({'turn': t + 1, 'paired_tasks': 12, **means,
                     'delta_ssc_minus_nl_pp': 100 * (means['dynamic_ssc'] - means['nl_dynamic_v2'])})
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    styles = [('dynamic_ssc', 'Dynamic SSC', '#2166ac', '-', 'o'),
              ('nl_dynamic_v2', 'Dynamic NL (matched state readout)', '#c86b16', '--', 's')]
    for key, label, color, line, marker in styles:
        ax.plot([r['turn'] for r in rows], [100 * r[key] for r in rows],
                label=label, color=color, linestyle=line, linewidth=2.1,
                marker=marker, markersize=3.6)
    ax.set(xlim=(1, 20), ylim=(0, 100), xticks=[1, 5, 10, 15, 20], yticks=list(range(0, 101, 20)),
           xlabel='Interaction round', ylabel='Mean physical goal completion (%)')
    ax.set_title('(d) Goal completion over interaction', loc='left', fontweight='bold', pad=12)
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=9.2)
    ax.annotate('89.6%', (20, 100 * rows[-1]['dynamic_ssc']), xytext=(-6, 9),
                textcoords='offset points', ha='right', color='#2166ac', fontsize=9)
    ax.annotate('84.0%', (20, 100 * rows[-1]['nl_dynamic_v2']), xytext=(-6, -15),
                textcoords='offset points', ha='right', color='#c86b16', fontsize=9)
    fig.text(.12, .035, '12 paired development tasks · seed 891011 · all outcomes retained',
             fontsize=8.5, color='#555555')
    fig.subplots_adjust(left=.12, right=.975, top=.86, bottom=.19)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('panel_d_goal_completion.' + suffix), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--from-export', action='store_true', help='Redraw using the exported measurement file only')
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    rollouts = read(out / 'per_rollout_measurements.json') if args.from_export else extract(args.source)
    write(out / 'per_rollout_measurements.json', rollouts)
    rows = aggregate(rollouts)
    csv_write(out / 'mean_completion_by_turn.csv', rows)
    long_rows = []
    for r in rollouts:
        values = padded_values(r)
        for t, value in enumerate(values, 1):
            long_rows.append({'job_index': r['job_index'], 'task_id': r['task_id'],
                 'seed': r['seed'], 'condition': r['condition'], 'turn': t,
                 'completion': value, 'observed_this_turn': int(t <= r['terminal_turn']),
                 'source_measurement_turn': min(t, r['terminal_turn']),
                 'service_affected': int(r['service_affected'])})
    csv_write(out / 'per_rollout_by_turn.csv', long_rows)
    plot(rows, out)
    avg = {c: sum(r[c] for r in rows) / HORIZON for c in CONDITIONS}
    statistics = {'seed': SEED, 'tasks': 12, 'rollouts': 24, 'rounds': HORIZON,
        'observed_measurements': sum(len(r['measurements']) for r in rollouts),
        'carried_forward_entries': sum(HORIZON - r['terminal_turn'] for r in rollouts),
        'service_affected_rollouts': [r['job_index'] for r in rollouts if r['service_affected']],
        'final_completion': {c: rows[-1][c] for c in CONDITIONS},
        'mean_completion_over_20_rounds': avg,
        'mean_over_rounds_delta_ssc_minus_nl_pp': 100 * (avg['dynamic_ssc'] - avg['nl_dynamic_v2']),
        'ssc_ahead_rounds': [r['turn'] for r in rows if r['delta_ssc_minus_nl_pp'] > 1e-8],
        'ssc_behind_rounds': [r['turn'] for r in rows if r['delta_ssc_minus_nl_pp'] < -1e-8],
        'protocol': 'Post-all-agents-action strict physical score. Each task equal weight, not pooled goals. '
                    'Only genuinely terminated episodes carried forward. No smoothing, interior interpolation, '
                    'cumulative maximum, filtering of failed runs, or extrapolation from final success rates.',
        'scope': 'Same frozen updater/access rules/planner/per-call caps, different readout. '
                 'Realized token use and closed-loop trajectories need not match. NL is a matched '
                 'deterministic readout of revisable state, not an independent learned memory system. '
                 'This is the full first-seed block; later incomplete/paused seeds are not pooled.',
        'interpretation': 'The curves cross. Higher SSC terminal completion does not establish earlier '
                          'completion overall or a monotonically widening advantage. No significance claim.',
        'plot_source_sha256': sha(Path(__file__)),
        'export_sha256': {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'summary.json'}}
    write(out / 'summary.json', statistics)
    print(json.dumps({k: statistics[k] for k in ('observed_measurements', 'carried_forward_entries',
          'final_completion', 'mean_completion_over_20_rounds', 'mean_over_rounds_delta_ssc_minus_nl_pp')}, indent=2))


if __name__ == '__main__':
    main()
