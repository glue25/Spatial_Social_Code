# Habitat dynamic memory controls — partial Flash results

**Status: 24 of 72 planned rollouts completed. This is NOT the completed
multi-seed experiment.** One additional rollout aborted; 47 have not started.

This release contains the entire first-seed block: the same 12 development tasks
under Dynamic SSC and a dynamic natural-language rendering control, using
`gemini-3-flash-preview` and seed `891011`. All first-seed failures are included.
The planned remaining seeds are `891012` and `891013`.

## Paired first-seed results

| Metric | Dynamic NL (`nl_dynamic_v2`) | Dynamic SSC | SSC minus NL |
|---|---:|---:|---:|
| Strict task success | 7/12 (58.33%) | 8/12 (66.67%) | +8.33 percentage points |
| Mean goal completion | 84.03% | 89.58% | +5.56 percentage points |
| Service-affected completed rollouts | 1 | 0 | — |

Paired outcomes: SSC alone succeeds on 3 tasks, NL alone on 2; both succeed on
5 and both fail on 2. Thus the success-rate difference is **one net task**.
These descriptive development-set numbers do not establish statistical
significance, multi-seed robustness, generalization, or superiority to all
natural-language memory systems. No confidence interval is claimed in this
release. Task-clustered paired uncertainty should accompany the full experiment.

## What this control actually tests

Both conditions use the same frozen, revisable per-agent v2 state/updater,
observation types, action feedback, delivered communication, planner, update
timing and action interface. They preserve the same private-information access
boundaries. The NL arm renders those records as deterministic prose with subject,
source, time, layer and uncertainty metadata; **it is not an independently
generated or learned natural-language summarizer**. Both can replace old states.

`frozen_memory_views.py` is an exact copy of the executed renderer. This is a
representation/readout control, not a comparison against every possible dynamic
memory implementation. Closed-loop trajectories and actual costs can differ.
The common budget is 20 environment turns, 8,192 maximum output tokens per call,
and low reasoning effort. The input observation mode is text, grounded in the
real simulator; oracle trajectories are not provided to the planner.

## Cost of the completed first-seed block

| Provider/log metric | Dynamic NL | Dynamic SSC |
|---|---:|---:|
| Logical requests | 654 | 682 |
| Received responses | 653 | 682 |
| Transport attempts, including retries | 661 | 685 |
| Failed transport attempts | 8 | 3 |
| Reported input tokens | 2,628,282 | 2,769,872 |
| Reported output tokens, including reasoning | 124,006 | 113,735 |
| Reported total tokens | 2,752,288 | 2,883,607 |

Usage is summed once per saved response, not once per retry. Failed requests
without a response can have unreported cost. Availability probes and the aborted
second-seed rollout are not included in this first-seed cost table. Equal caps
are not a claim of identical realized compute.

## Failures and coverage — do not hide these

- First-seed NL job **35** completed all 20 turns with valid physical measurement,
  but one API request exhausted three HTTP 500/502 attempts. The original planner
  continued. Its original outcome remains **failure, 2/3 goals complete**, with
  `service_affected=1`. It was not rerun or replaced. The paired SSC rollout also
  failed with 2/3 goals complete, so this pair contributes zero to the observed
  between-condition difference. Service exposure remains a confound to disclose.
- Second-seed NL job **60** aborted at turn 9. The model identity was correct and
  `finish_reason` was `stop`, but the response body was empty. Usage recorded
  7,862 reasoning tokens and no answer text. This does not by itself establish
  whether the cause was budget exhaustion or gateway handling. The incomplete
  trajectory has **no completed physical result** and has not been replaced.
- The scheduler stopped for review. The 24-rollout table is explicitly the
  pre-existing complete first-seed block, **not all 25 attempted rollouts** and
  not a success-filtered estimate of the full 72. `schedule.csv` retains every
  planned task/seed/condition, including the aborted and unstarted entries.

## Files and verification

- `rollouts.csv`: 24 per-rollout physical metrics, service flags, usage and local
  audit-artifact SHA-256 hashes.
- `schedule.csv`: all 72 frozen jobs and their snapshot status.
- `summary.json`: aggregate numbers and export checksums.
- `protocol.json`: sanitized protocol, control definition and source hashes.
- `interrupted_rollout.json`: empty-response failure metadata, without prompts,
  credentials or generated reasoning text.
- `frozen_memory_views.py`: exact executed memory readout implementation.
- `verify.py`: standalone offline coverage, arithmetic and checksum check.

```bash
python reports/habitat_dynamic_memory_partial_20260923/verify.py
```

No datasets, simulator installation, credentials or API calls are needed for
that check. It verifies the published tables, **not a replay of the underlying
physics**. Raw videos and full request/response bodies are not in this compact
release; local artifact hashes are provenance identifiers, not substitutes for
independent trajectory validation. These results are separate from the earlier
108-rollout Direct/Static/Dynamic publication and must not be pooled with it.
