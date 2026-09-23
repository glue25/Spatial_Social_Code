"""Mean +/- between-task sample SD for the already audited 12-task curves.

No inference, sample selection, smoothing, or changes to the original figures.
Variance uses ddof=1; the band is NOT a confidence interval or seed variability.
"""
import argparse
import math
from pathlib import Path
import shutil
from statistics import mean, variance

from experiments.plot_habitat_memory_progress import (
    ROOT, CONDITIONS, HORIZON, SEED, padded_values, read, write, sha, csv_write,
)

OUTPUT = ROOT / 'reports/habitat_memory_progress_stats_20260923'
INPUT = ROOT / 'reports/habitat_memory_progress_20260923/per_rollout_measurements.json'


def moments(values):
    assert len(values) >= 2 and all(math.isfinite(x) for x in values)
    m, v = mean(values), variance(values)
    sd = math.sqrt(v)
    return {'n_tasks': len(values), 'mean_percent': 100 * m,
            'sample_variance_fraction_squared': v,
            'sample_variance_pp_squared': v * 10000,
            'sample_sd_pp': sd * 100,
            'mean_minus_sd_percent': (m - sd) * 100,
            'mean_plus_sd_percent': (m + sd) * 100}


def compute(rollouts):
    assert len(rollouts) == 24 and {r['seed'] for r in rollouts} == {SEED}
    lookup = {(r['task_id'], r['condition']): padded_values(r) for r in rollouts}
    tasks = sorted({r['task_id'] for r in rollouts})
    assert len(tasks) == 12 and len(lookup) == 24
    assert set(lookup) == {(t, c) for t in tasks for c in CONDITIONS}
    rows, differences = [], []
    for t in range(HORIZON):
        for c in CONDITIONS:
            rows.append({'turn': t + 1, 'condition': c,
                         **moments([lookup[k, c][t] for k in tasks])})
        paired = moments([lookup[k, 'dynamic_ssc'][t] - lookup[k, 'nl_dynamic_v2'][t] for k in tasks])
        differences.append({'turn': t + 1, 'n_tasks': len(tasks),
            'mean_difference_pp': paired['mean_percent'],
            'sample_variance_difference_pp_squared': paired['sample_variance_pp_squared'],
            'sample_sd_difference_pp': paired['sample_sd_pp']})
    return rows, differences


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#333333', 'axes.labelcolor': '#333333',
        'xtick.color': '#444444', 'ytick.color': '#444444',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    styles = [('dynamic_ssc', 'Dynamic SSC', '#0072B2', '-', 'o'),
              ('nl_dynamic_v2', 'Dynamic NL (matched readout)', '#D55E00', '--', 's')]
    for condition, _, color, _, _ in styles:
        rr = [r for r in rows if r['condition'] == condition]
        # Bounds are not changed in the data. The physical 0–100% axes clip the
        # rendered bands where mean +/- SD exceeds the possible score range.
        ax.fill_between([r['turn'] for r in rr], [r['mean_minus_sd_percent'] for r in rr],
                        [r['mean_plus_sd_percent'] for r in rr],
                        color=color, alpha=.12, linewidth=0, zorder=1)
    for condition, label, color, line, marker in styles:
        rr = [r for r in rows if r['condition'] == condition]
        ax.plot([r['turn'] for r in rr], [r['mean_percent'] for r in rr],
                color=color, linestyle=line, linewidth=2.25, label=label,
                marker=marker, markersize=4, markeredgewidth=.8,
                markeredgecolor='white', markevery=[0, 4, 9, 14, 19], zorder=3)
    ax.set(xlim=(.8, 20.2), ylim=(0, 100), xticks=[1, 5, 10, 15, 20],
           yticks=list(range(0, 101, 20)), xlabel='Interaction round',
           ylabel='Physical goal completion (%)')
    ax.xaxis.labelpad = 8; ax.yaxis.labelpad = 8
    ax.grid(axis='y', color='#d8dde3', linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5, width=.8)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none',
              framealpha=.90, fontsize=9.2, handlelength=2.5, labelspacing=.65)
    fig.text(.105, .943, '(d)', fontsize=15, fontweight='bold', va='top')
    fig.text(.168, .943, 'Goal completion during interaction', fontsize=13,
             fontweight='semibold', va='top')
    fig.text(.13, .86, 'Mean ± 1 SD across tasks', fontsize=9.4, color='#515b67')
    fig.text(.965, .86, '12 paired tasks · one seed', fontsize=9.4, color='#515b67', ha='right')
    for condition, _, color, _, _ in styles:
        last = [r for r in rows if r['condition'] == condition][-1]
        offset = 10 if condition == 'dynamic_ssc' else -17
        ax.annotate(f"{last['mean_percent']:.1f}%", (20, last['mean_percent']),
                    xytext=(-5, offset), textcoords='offset points', ha='right',
                    color=color, fontsize=9, fontweight='medium', zorder=5,
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    fig.text(.13, .027, 'All outcomes retained; terminal states carried forward. Shading is not a confidence interval.',
             fontsize=7.4, color='#606975')
    fig.subplots_adjust(left=.13, right=.965, top=.82, bottom=.18)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'panel_d_mean_sd.{ext}', dpi=260)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    rollouts = read(args.input); rows, differences = compute(rollouts)
    dest = out / 'per_rollout_measurements.json'
    if args.input.resolve() != dest.resolve(): shutil.copyfile(args.input, dest)
    csv_write(out / 'mean_variance_by_turn.csv', rows)
    csv_write(out / 'paired_difference_stats_by_turn.csv', differences)
    plot(rows, out)
    summary = {'seed': SEED, 'tasks_per_condition': 12, 'rollouts': 24,
        'variance_estimator': 'Sample variance across the12 task fractions at each round, ddof=1.',
        'band': 'Mean +/-1 between-task sample SD. Not SE, confidence interval or variability across seeds.',
        'band_display': 'Axes restricted to0–100%; portions outside are visually clipped, not truncated in CSV.',
        'paired_difference': 'Statistics of within-task SSC minus NL differences; covariance preserved, not independent-arm variance sum.',
        'final_round': {c: next(r for r in rows if r['turn'] == 20 and r['condition'] == c) for c in CONDITIONS},
        'sample_policy': 'Exactly the same complete first-seed12-task cohort as the original curve, including failures and service-affected NL job35; no additional outcomes selected.',
        'scope': 'Matched dynamic natural-language state readout, not general memory systems. Same updater/access rules/per-call caps; actual compute may differ. Later paused/incomplete seeds not pooled.',
        'interpretation': 'Curves still cross; uncertainty display does not turn the result into a monotonic/earlier-completion advantage. One seed cannot estimate between-seed variability.',
        'source_sha256': sha(args.input), 'script_sha256': sha(Path(__file__)),
        'exports_sha256': {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.name != 'summary.json'}}
    write(out / 'summary.json', summary)
    print(summary['final_round'])


if __name__ == '__main__':
    main()
