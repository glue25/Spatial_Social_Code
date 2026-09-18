# Gemini 3 Flash: completed Habitat development-suite results

**Scope:** 12 distinct tasks × 3 seeds × 3 conditions = 108 real simulator
rollouts. These are local development-suite results, not official hidden-test
or leaderboard results. Repeated seeds are not 36 independent task clusters.

| Condition | Strict successes | Success rate | Mean task completion |
|---|---:|---:|---:|
| Direct | 8/36 | 22.2% | 57.2% |
| Static SSC | 10/36 | 27.8% | 65.5% |
| Dynamic SSC | 24/36 | 66.7% | 83.1% |

Dynamic SSC improves strict success by 44.4 percentage points and mean
completion by 25.9 points over Direct on this suite. Its per-seed success
counts are 8, 9, 7 (12 tasks each), compared with Direct's 2, 4, 2 and
Static SSC's 4, 3, 3. All three conditions cover the same task/seed pairs;
unsuccessful rollouts remain in the denominator.

## Protocol and cost

- Requested model and logged response identity: `gemini-3-flash-preview`.
  This is the provider-reported identity, not independently verified weights.
- The same model/service/key is used for all three conditions.
- Input is private **text observations** from the simulator, not video input.
  Habitat uses deterministic agent-indexed SSC state updates; it is not the
  typed visual compiler used in the other ongoing benchmarks.
- State variant `v2`; seeds 891011, 891012, 891013; cap of 20 environment
  turns, **not** 20 total API calls; low reasoning; 8192 output-token cap/call.
- Strict physical scoring includes actual simulation checks. This small
  export does not redistribute simulator assets, raw prompts or trajectories.
- Equal task/seed coverage and per-call settings do **not** make this an
  equal-total-compute experiment. Logged costs differ:

| Condition | Logged API calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| Direct | 2324 | 5,853,107 | 513,799 |
| Static SSC | 2358 | 6,428,146 | 567,088 |
| Dynamic SSC | 1921 | 7,389,139 | 467,025 |

Costs come from per-rollout provider usage logs. They do not estimate
unlogged failed network attempts or monetary charges. The 6603 logged calls
agree with the completed suite's saved API-response count.

## Relationship to earlier results

**This is not a Gemini 3 Pro / Flash replication claim.** There are multiple
historical Habitat protocols; they should not be collapsed into a single
"old result." The following two reports used **Qwen3-VL-8B**, not Gemini 3 Pro:

| Historical protocol | Direct mean completion | Method mean completion | Interpretation |
|---|---:|---:|---|
| Task-conditioned four-condition report | 45.83% | Spatial+Social 53.47% | Reported positive completion difference (+7.64 pp) |
| Later three-condition online closed-loop audit | 53.01% | Dynamic SSC 41.67% | Mixed: strict success 4/36 vs 3/36, but lower mean completion |

The [earlier four-condition report](../habitat_spatial_social_status_zh.md)
also reported Social-only at 59.95%, above Spatial+Social. It is now marked
historical and is not re-certified here under the later strict physical
scorer. The [later three-seed audit](../habitat_closed_loop_multiseed/habitat_multiseed_results.md)
is a different online-state protocol. Positive
[individual mechanism cases](../habitat_ssc_executable_pipeline_zh.md)
also exist, but do not estimate overall benchmark performance.

Thus, "all older results were negative" would be inaccurate. Preserve both
the previously reported positive evidence and the later mixed results, each
with its own protocol and scoring qualifications. This export does not
establish how much of their difference is due to state construction, scoring
or other changes. Nor does it attribute the historical-to-Flash difference
to the backbone alone. Positive Pro results on a different task/protocol
do not establish a matched Habitat Pro/Flash comparison.

The supported conclusion is narrower: **Dynamic SSC performs better than
Direct and Static SSC on this completed Flash development suite.** Broader
cross-model robustness requires matched protocols and independent evaluation.

中文说明：这是一项完整、正向的 Flash/Habitat 开发集结果，可以作为补充证据；
不能写成“同一 Habitat 任务上 Gemini 3 Pro 与 Flash 已相互验证”，也不替代旧结果。

## Files and offline verification

- `rollouts.csv`: one row per completed task/seed/condition, outcomes and
  recorded usage only; no credentials, machine paths, prompts or media.
- `summary.json`: aggregate results, per-seed success counts, protocol limits,
  and hashes of this CSV and local source artifacts.
- `verify.py`: dependency-free coverage and arithmetic checks.

```bash
python reports/habitat_flash_108_20260918/verify.py
```

Verification recomputes published aggregates and checks pairing, uniqueness,
usage totals and the CSV hash. It does not independently replay the simulator
or validate the underlying trajectories. Reproduction of full rollouts also
requires the appropriate assets, environment and frozen runner.
