# Panel (d): measured goal completion over interaction rounds

This plot uses **all12 paired tasks from the complete first seed891011**,
24 rollouts from the matched Dynamic SSC / dynamic natural-language readout
study. It does not use the separate108-rollout Direct/Static/Dynamic study.
It does not pool the later incomplete/paused seeds or select successful tasks.
The later continuation was paused after observing intermediate results; this
first-seed figure is a descriptive snapshot, not completed multiseed evidence.

## Data provenance and averaging

- Source: each rollout's `physical_checks.jsonl`, from the frozen
  `habitat_memory_controls_20260922_v1` run. The frozen benchmark runner writes
  this score **after all agents have acted in each round**.
- All24 traces have consecutive round numbers1 through their terminal round,
  with no gaps or invalid physical measurements:378 actual checkpoints.
- The strict scorer counts satisfied physical goals after stability/release
  checks. Each plotted value is the equal-weight average of12 per-task fractions,
  not a pooled fraction of all goals or a model's self-reported progress.
- For genuinely terminated rollouts only, the terminal fraction is carried
  forward through round20 (102 additional entries). Failures remain in the
  denominator. Interior missing/invalid rounds would cause an error, not be
  interpolated, zero-filled or silently dropped.
- All24 endpoints agree with their saved final physical evaluation. NL job35,
  which completed despite an exhausted transient API request, remains included
  and flagged. Physical measurement validity is not clean-service equivalence.
- No moving average, spline, cumulative maximum, monotonicity enforcement or
  inference from final success percentages. A previously achieved goal can
  become unsatisfied, so an individual or average curve can decrease.

Both conditions have the same frozen updater, per-agent access boundaries,
planner, update timing and per-call caps. Realized costs/observations/actions
can diverge. The NL condition is a deterministic natural-language readout of the
same revisable state store, **not an independent general-purpose memory agent**.

## What the actual curve says

| Round | Dynamic SSC | Matched dynamic NL readout | SSC minus NL |
|---|---:|---:|---:|
| 5 | 16.67% | 16.67% | 0.00pp |
| 10 | 57.64% | 59.03% | -1.39pp |
| 15 | 63.19% | 78.47% | -15.28pp |
| 20 | 89.58% | 84.03% | +5.56pp |

SSC is ahead at rounds6–9,11,17–20; behind at3–4,10,12–16; tied at1–2,5.
The mean completion over all20 rounds is49.17% for SSC and50.45% for NL
(SSC minus NL: -1.28pp). This descriptive temporal average was computed after
inspection and is not a preregistered primary endpoint.

**The figure does not support a claim of uniformly earlier completion or an
advantage that widens monotonically with interaction.** It shows crossings,
a middle/late SSC regression, then a higher SSC terminal mean in this seed.
No significance or multiseed robustness claim is made; uncertainty bands are
not shown. Round20 values are **goal completion**, not the66.67%/58.33% binary
task-success rates reported elsewhere.

## Suggested caption

**(d) Physical goal completion over interaction rounds.** Curves show the
equal-weight mean across the same12 development tasks (seed891011), measured
after all agents act in each round. Dynamic SSC and the matched dynamic
natural-language state-readout control share the updater, information-access
rules, planner and per-call budgets. Terminated runs retain their final values
through round20; all outcomes, including one service-affected NL run, are
included. The trajectories cross, with Dynamic SSC ending at89.58% versus84.03%;
the result does not establish uniformly faster completion.

## Files

- `panel_d_goal_completion.png`: raster preview.
- `panel_d_goal_completion.pdf`, `.svg`: vector versions for the paper.
- `mean_completion_by_turn.csv`: two means and paired difference at each round.
- `per_rollout_by_turn.csv`:480 entries, with measured vs carried-forward flag.
- `per_rollout_measurements.json`:378 actual reduced checkpoints, individual
  goal satisfaction, endpoint metadata and original artifact hashes.
- `summary.json`: coverage, descriptive metrics and export checksums.

Generation code: `experiments/plot_habitat_memory_progress.py` in the project.
Rebuild from exported measurements, without datasets, simulator or API:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/physical_code_mpl \
  MA-EgoQA/.venv/bin/python experiments/plot_habitat_memory_progress.py \
  --from-export --output reports/habitat_memory_progress_20260923
```

Matplotlib is required only to draw the figure. The associated unittest checks
terminal-only padding, preservation of regression, missing/duplicate rounds,
interrupted episodes, invalid physics and mismatched final scores. Extracted
checkpoints are not an independent replay of the simulator's physics.
