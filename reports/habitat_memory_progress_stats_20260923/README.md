# Panel (d), revised: means and between-task variance

This is a presentation/statistics revision of the existing first-seed24-rollout
curve, **not new inference, resampling or selection of better-looking outcomes**.
The original figure and measurements remain unchanged.

## Statistical definition

At round t, each of the12 paired tasks contributes its fraction of physically
satisfied goals, x(i,t). Tasks have equal weight, regardless of goal count.

- Mean: sum(x(i,t))/12.
- Sample variance: sum((x(i,t) - mean(t))^2)/11 (ddof=1).
- Sample SD: square root of sample variance.
- Plotted lines: mean in percent; ribbons: mean +/-1 task SD, in percentage points.
- CSV variance is provided both on the0–1 fraction scale and in squared
  percentage points (the latter is10000 times the former).

The SD ribbon describes task heterogeneity. It is **not** a standard error,
confidence interval, or variance across random seeds. Scores across rounds are
not independent replicates. The axes show the physically possible0–100% range;
bands extending outside that range are visually clipped, while CSV bounds are
left unmodified. Large overlap is not a significance test.

## Data and fairness

The exact same12 tasks, seed891011,24 runs,378 observed checkpoints and102
terminal carry-forwards are used. Failed runs and the service-affected NL job35
remain. Current physical goal satisfaction can regress; no monotone envelope,
spline, smoothing or averaging over selected rounds is applied.

Dynamic NL is the matched natural-language **readout of the same revisable
state store**, not an independent general-purpose memory agent. Same updater,
information access and per-call budget; realized trajectories/costs may differ.
Later incomplete/paused seeds are not pooled. This figure remains a one-seed
development diagnostic; their stopping history must not be concealed.

## Endpoint statistics (round20)

| Condition | Mean (%) | Sample SD (pp) | Sample variance (pp squared) |
|---|---:|---:|---:|
| Dynamic SSC | 89.58 | 15.54 | 241.48 |
| Matched dynamic NL readout | 84.03 | 22.04 | 485.59 |

SSC has a higher terminal mean and lower terminal between-task variance in
this cohort. This does not establish greater robustness across seeds, uniformly
earlier completion, or a widening gap: the means still cross, and the average
over rounds remains49.17% SSC versus50.45% NL.

## Files

- `panel_d_mean_sd.png`: revised preview.
- `panel_d_mean_sd.pdf`, `.svg`: vector formats.
- `mean_variance_by_turn.csv`: all40 condition/round means, variances, SDs and bounds.
- `paired_difference_stats_by_turn.csv`: mean/variance/SD of within-task
  SSC-minus-NL differences; preserves pairing/covariance.
- `per_rollout_measurements.json`: unchanged source checkpoints and original hashes.
- `summary.json`: scope, estimator definition, statistics and export hashes.

## Caption

**(d) Physical goal completion during interaction.** Lines denote the mean and
shaded regions +/-1 sample standard deviation across12 paired development
tasks (seed891011). Scores are measured after all agents act in each round;
terminated runs retain their final scores through round20, and all outcomes
are included. Dynamic SSC and the matched dynamic natural-language state
readout share the updater, information-access rules and per-call budgets.
Shading represents between-task variability, not confidence intervals or
between-seed variability; displayed bands are bounded by the0–100% plot range.

Reproduce from the unchanged original measurement export:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/physical_code_mpl \
  MA-EgoQA/.venv/bin/python -m experiments.plot_habitat_memory_progress_stats
```

The script also accepts `--input` and `--output` paths. No API/GPU/simulator is
used. Tests verify ddof, scale conversion, unchanged pairing, zero variance for
constant observations and rejection of a singleton variance estimate.
