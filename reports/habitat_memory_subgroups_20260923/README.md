# Habitat: original subgroups and illustrative cases

Offline **post-hoc exploratory analysis**, not a new evaluation or a
preregistered subgroup hypothesis. No new API calls or changes to workers.
The complete paired cohort is 12 development tasks, seed 891011, 24 rollouts.
All tasks, including failures and service-affected job 35, remain included.

## Original difficulty split

| Original split | Tasks | NL final completion, mean ± SD | SSC final completion, mean ± SD | NL strict success | SSC strict success |
|---|---:|---:|---:|---:|---:|
| hard | 6 | 79.17% ± 26.74 pp | 90.28% ± 15.29 pp | 3/6 | 4/6 |
| standard | 6 | 88.89% ± 17.21 pp | 88.89% ± 17.21 pp | 4/6 | 4/6 |

Labels come directly from frozen task JSON `split`, not from observed scores.
The hard group includes episodes **1781, 1013, 623, 744, 1166, 772**:
two SSC-only successes, one NL-only success, two shared successes, one shared
failure. No negative hard task was dropped. All six have three agents,
room restrictions, bandwidth limits, and restricted communication. They also
carry `tom_level=3`, but several labels use syntactic fallback rather than a
strict proof: do not describe physical completion as third-order-ToM accuracy.

The hard group's final difference is **+11.11 percentage points**, with an
exploratory paired task-bootstrap 95% interval **[-11.11, +38.89] pp**
(50,000 resamples; RNG seed 20260923; percentile interval, unadjusted for
subgroup exploration). This is suggestive, not statistically established.
The standard group is tied at the endpoint; this does not establish a
statistically significant hard-versus-standard interaction.

`difficulty_mean_sd.{png,pdf,svg}` shows BOTH original splits. Shading is
**between-task sample SD (ddof=1)**, not variation over seeds or a confidence
interval. Raw variance and band bounds are exported to
`subgroup_mean_variance_by_turn.csv`. Plot bounds clip ribbons visually; the
stored values are unmodified. Each checkpoint is actual post-action physical
completion. Early termination carries forward its final score to round 20.
No cumulative maximum or smoothing is applied. Even hard-task curves cross:
the data support higher final completion, **not uniformly faster progress**.

All inspected partitions (`split`, `category`, `tom_level`, `num_agents`) are
exported, not just favorable ones. These overlapping partitions are not
independent confirmations. All task metadata, source hashes, per-task outcomes,
and the selected-case ledger are retained in JSON files in this directory.

## Positive qualitative examples (outcome-selected)

`illustrative_cases.{png,pdf,svg}` shows two single paired trajectories,
explicitly selected after inspecting outcomes. They are NOT a new averaged
evaluation subset, and there is no invented variance for individual runs.

1. **Tea reset, episode 1781, NL job 0 / SSC job 2.**
   SSC reaches 1/3 at round 7, 2/3 at round 9, and 3/3 at round 17.
   NL reaches 1/3 at round 10 and remains there through round 20.
   The strict goals are kettle and teapot on `chair_11`, and the bedroom
   `chest_of_drawers_21` open.
   In SSC, agent_1's round-2 delivered message tells agent_0 the correct
   destination, `chair_11`. Agent_2 nevertheless places the kettle on
   `chair_10` at round 5. Agent_0 picks it up at round 6 and successfully
   places it on `chair_11` at round 7. Its round-7 planner context retains
   `claim.agent_1.chair_11` with `source=delivered_message`,
   `source_agent=agent_1`, `turn=2`, alongside physical action state.
   Agent_2 places the teapot correctly at round 9 and opens the drawer at
   round 17. This is a concrete **correction followed by goal completion**.
   NL also receives correct destination information (agent_1 to agent_0,
   round 3), and that information is still present in its round-7 prose
   context. It is therefore NOT evidence that the NL store forgot the message.
   In NL, kettle placement at round 5 is on `table_13`, and teapot placement
   is on `chair_10`; neither physical placement goal is achieved by round 20.
   Sources: the exact runs' `events.jsonl`, `plans.jsonl`, and
   `physical_checks.jsonl`, resolved via each job's `result.json` artifact.
2. **Guest reset, episode 1166, NL job 20 / SSC job 22.**
   Both succeed: SSC completes all goals at round 8, NL at round 13.
   SSC reaches 1/3 at round 5 and 2/3 at round 6; NL first reaches 1/3 at
   round 9. This illustrates a five-round speedup in this one paired run,
   not a general average speedup.

## Interpretation and recommended use

NL here is a **natural-language rendering of the same revisable state store**.
Both use the same updater, planner, information-access rules, and configured
budgets. Different actions generate different subsequent observations and
messages. Do not claim that only SSC has a dynamic memory or receives evidence;
do not infer causal contribution of evidence links from one trajectory.

Suggested statement: “In an exploratory analysis of the six tasks originally
labeled hard, Dynamic SSC attained higher final physical goal completion than
the matched natural-language readout (90.28% vs 79.17%). The tea-reset example
illustrates a successful correction of an incorrectly placed object, with the
correct destination retained as an attributed communication record.”

Keep the full 12-task mean ± SD result alongside any subgroup/case presentation:
`../habitat_memory_progress_stats_20260923/`. The published figure package is
indexed at `../habitat_figure_options_20260923/README_zh.md`, including a newer
partial-seed audit: the tea example reverses direction in seed 891013. It must
not be presented as a seed-robust case.

Reproduce from repository root:

For a public clone without the private frozen run directory, use
`python -m experiments.verify_habitat_figure_exports --render` (numpy and
matplotlib required). The original extraction commands below require the
local frozen run artifacts.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/physical_code_mpl \
  MA-EgoQA/.venv/bin/python -m experiments.plot_habitat_memory_subgroups
PYTHONDONTWRITEBYTECODE=1 MA-EgoQA/.venv/bin/python -m unittest \
  tests.test_habitat_memory_subgroups
```
