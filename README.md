# Spatial-Social Code

Spatial-Social Code studies whether models reason more reliably from an
explicit, executable representation of a multi-agent world. It separates
grounded physical evidence from social and mental-state inference:

- **Spatial Code** records facts, visibility, events, roles, temporal phases,
  and provenance.
- **Social Code** records answer-independent observations, beliefs, goals,
  intents, and contradictions.
- A restricted AST runtime validates generated programs before executing them
  into canonical JSON traces.

The repository contains two complementary evaluation tracks. In the target
visual formulation, the Spatial and Social compilers read scene evidence only;
the question is used later to construct a read-only view over the compiled
state, and only the shared final answerer sees answer choices. Some historical
visual runs predate this stricter compile-once protocol and are explicitly
marked as diagnostics below.

1. task-conditioned agent-indexed state compilation and closed-loop planning
   on EnactToM/Habitat, with deterministic oracle execution retained only as
   an upper-bound diagnostic;
2. grounded visual Spatial-Social Code evaluation on MMToM and MA-EgoQA.

Direct-answer baselines are stored only for paired evaluation. They never
select, switch, fuse, gate, or overwrite a Spatial-Social answer.

## Current visual results

The method and paired baseline use the same model and the same visual inputs.
They are not call-, token-, latency-, or FLOP-matched unless explicitly noted.

| Benchmark | Spatial-Social Code | Direct baseline | Difference |
|---|---:|---:|---:|
| MMToM option-blind partial run (204 valid pairs) | **142/204 (69.61%)** | 121/204 (59.31%) | **+10.29 pp** |
| EnactToM/Habitat dynamic, 12-task × 3-seed development suite (strict completion) | 41.67% | **53.01%** | **−11.34 pp** |
| CoMind train recording-heldout OOF (bbox IoU@0.5) | **6.47%--23.15%** | 0.11%--0.34% | metric-dependent |

The option-blind MMToM partial run has exact McNemar `p = 0.00145049`, but it
stopped after 205 rows (204 valid pairs) and is not the complete 600-question
benchmark. On Habitat, strict full-task success is 3/36 for Direct, 3/36 for
Static SSC, and 4/36 for Dynamic SSC; Dynamic does not improve the aggregate
mean completion score and uses about 50.5% more tokens. These are development
results, not a frozen confirmation set.

The former MMToM 470/600 and MA-EgoQA 904/1741 runs are retained only in
clearly marked historical audit reports because their compiler protocols do
not satisfy the current question/option-independence requirement.
See the [submission results and baseline policy](reports/submission_results_and_baselines_zh.md)
for the component evidence, Habitat results, official published references,
and the strict comparability boundary.

## CoMind real-world grounding track

**Completed Flash + SAM development replication (2026-09-21):** all 300 paired
examples (100/task) are archived with generated programs, traces, visible API
responses and offline checks. Direct scores 17.21% and SSC+SAM 22.21% on an
explicit 14-field macro-average (+5.00 pp), not an official composite score.
The gain is concentrated in Joint Attention/localization; some SCOIA and
Handover semantic fields regress. All 4 SSC failures remain in the primary
denominator. A separate fresh paired, larger-budget recovery of those 4 cases
is reported without replacing the primary result. SAM is additional detector
compute; no equal-tool-budget Direct+SAM control was run. See the
[full Flash/SAM archive and comparability notes](reports/comind_flash_sam_20260921/README.md).

A deterministic 300-example CoMind train subset (100 examples per task) was
also evaluated with `gpt-5.6-sol`, using matched inputs for Direct and
Spatial-Social inference. All 300 pairs completed. Across 14 reported fields,
Spatial-Social was higher on 4, lower on 7, and tied on 3; it did not uniformly
improve over Direct. The only significant paired difference was a regression
on handover category L1 (0% versus 14%, exact McNemar `p = 0.000122`).

This is an exploratory train-subset comparison, not an official CoMind hidden
test result. See the [full report](reports/comind_gpt56_300_results_zh.md) and
[aggregate JSON](reports/comind_gpt56_300_summary.json). One audited
[SCOIA positive-flip case](reports/comind_success_case_zh.md) pairs the exact
Direct and Spatial-Social outputs, ground truth, executable Code, and all six
original input frames. It is an illustrative field-level success, not a claim
that the 300-example result improves overall.

A subsequent train-only mechanism study processed all 6,877 labeled samples
with explicit visual grounding and recording-heldout policy selection. On the
6,673-sample strict common set, bbox IoU@0.5 improved from near zero to
6.47%--23.15% across the three tasks, without using held-out-recording labels
to select each fold's policy; handover category also improved from 22.82%
(Direct) / 24.16% (raw Spatial-Social) to 31.88%. We report this as the fourth
main environment because it directly tests Spatial grounding in real
multi-view interactions. It remains an OOF analysis of the labeled training
corpus, not an official hidden-test or leaderboard result. See the
[grounding/OOF report](reports/comind_grounding_oof_results_zh.md) and
[machine-readable summary](reports/comind_grounding_oof_summary.json).

The current MMToM, MA-EgoQA, and CoMind ablation matrix and resumable launch
commands are recorded in the
[three-environment ablation status](reports/three_environment_ablation_status_zh.md).

## Target visual pipeline

1. Freeze scene evidence independently of any question or answer choice.
2. Compile query-independent grounded Spatial Code with clip/time provenance.
3. Execute the Spatial program in the restricted runtime.
4. Compile Social Code using only validated Spatial evidence references.
5. Infer answer-independent beliefs, goals, intents, and roles while
   preserving unknowns and contradictions.
6. Construct a question-specific read-only view, then give that view and the
   answer choices to one shared final reasoner.
7. On uncertain MMToM items, use an independent denser visual pass as the
   authoritative method prediction instead of averaging it with sparse
   evidence.

Important constraints include:

- missing evidence is unknown, not negative evidence;
- a world fact is not automatically an agent belief;
- execution or observation does not imply initiation, responsibility, intent,
  or causality;
- Social records can be required to cite valid Spatial evidence IDs;
- `query_evidence` is rejected by the normal runtime; historical replay is
  isolated behind explicitly named `legacy_*` entry points.

The controlled mechanism test in
[`experiments/ssc_mechanism_v1`](experiments/ssc_mechanism_v1/) directly checks
observation projection, information update, joint action feasibility,
compile-once multi-query reuse, and local counterfactual replay. It is a
deterministic conformance test, not a learned-model benchmark.

## EnactToM track

The E0 experiment compiles all 300 public EnactToM tasks into query-independent
oracle state, replays golden trajectories, and evaluates message
deletion/redirection interventions. It tests the representation and reducer,
not learned-agent performance.

The current learned-agent Habitat experiment is separate: a deterministic
compiler gives each Qwen3-VL-8B agent either Spatial Code, Social Code, or both,
then the model plans through the normal private-observation runtime. The
compiler never reads `golden_trajectory`, does not emit action schedules, and
does not reveal another agent's private facts. Direct, Spatial-only,
Social-only, and Spatial+Social use the same tasks and 20-turn budget. See the
[live Habitat status](reports/habitat_spatial_social_status_zh.md). A separate
[hard_20 action-trace case](reports/habitat_hard20_case_zh.md) documents a
75%-progress Standard failure and a 100% Oracle-schedule upper bound. The
historical runs used text observations and saved no RGB frames, so the case is
an action-log audit rather than a visual gallery.

For a non-Oracle learned-agent audit example, the
[episode 772 Physical+Social case](reports/habitat_episode_772_case_zh.md)
exports the complete repeat-3 action/communication trace, exact per-agent Code,
and replay frames. The historical evaluator recorded 100%, but replay shows the
picture frame was still grasped; this is an evaluator-failure case, not evidence
that SSC beat Direct.

The current offline-recomputed evidence, including raw records for Habitat
episodes 1166/1566 and the 204-pair MMToM partial run, is indexed in the
[latest experiment audit bundle](reports/audit_latest_experiments/README_zh.md).

The lightweight public artifact does not require datasets or API credentials:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash scripts/verify_artifact.sh
```

Dataset-dependent tests are automatically skipped when their optional public
data are absent. After preparing EnactToM data, the deterministic E0 track can
be reproduced with:

```bash
python scripts/validate_datasets.py
python scripts/run_enacttom_e0.py
```

See the [method formulation](docs/method_zh.md),
[experiment plan](docs/experiment_plan_zh.md), and
[E0 report](reports/enacttom_e0_oracle_results.md).

## Repository layout

```text
spatial_social_code/
  runtime.py                    restricted Spatial/Social runtimes
  social_code.py                native Social compiler and inference
  enacttom.py                   EnactToM compiler and reducer
  evaluate_mmtom_jdcloud.py     paired MMToM evaluator
  refine_mmtom_uncertain_jdcloud.py
MA-EgoQA/egomas/src/
  run_video_balanced_baseline_jdcloud.py
  run_video_provenance_v3_jdcloud.py
  audit_video_evidence_sparse_jdcloud.py
scripts/                        dataset preparation and EnactToM entrypoints
tests/                          deterministic EnactToM tests
docs/                           method and experiment documentation
```

The executable-hypothesis experiment is included as a failed ablation and is
not part of the reported best method.

## Visual setup

Python 3.10+ is required because the implementation uses
`zip(..., strict=True)`. For all visual, analysis, and local-model dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Smaller editable-install extras are also available: `video`, `analysis`,
`local-model`, `docs`, and `test`. The `test` extra is sufficient for the
lightweight artifact verifier; `requirements.txt` installs the complete
research environment.

Obtain MA-EgoQA/EgoLife and MMToM from their respective authors. Pass explicit
`--benchmark`, `--video-root`, `--questions`, and `--frames` paths to the
evaluators.

Configure a Gemini-compatible endpoint:

```bash
export GEMINI_API_KEY="..."
export GEMINI_API_BASE="https://your-endpoint.example/v1/generateContent"
export GEMINI_MODEL="your-model"
export FFMPEG_BINARY="$(command -v ffmpeg)"
```

For the optional local Qwen backend, set `SPATIAL_SOCIAL_MODEL_PATH` to a
checkpoint or Hugging Face model ID. Set
`SPATIAL_SOCIAL_LOCAL_FILES_ONLY=1` for offline loading.

## Tests

```bash
bash scripts/verify_artifact.sh
```

The MA-EgoQA integration tests additionally require that repository's optional
dependencies and are intentionally not part of the lightweight check.

## Visual evaluation

```bash
python spatial_social_code/evaluate_mmtom_jdcloud.py \
  --questions /path/to/MMToM-QA/Benchmark/questions.json \
  --frames /path/to/required_frames \
  --output results/mmtom.jsonl
```

MA-EgoQA uses a paired two-stage workflow so both conditions see exactly the
same clips:

```bash
cd MA-EgoQA
python egomas/src/run_video_balanced_baseline_jdcloud.py \
  --benchmark /path/to/MA-EgoQA.json \
  --video-root /path/to/egolife_raw \
  --output data/baselines/paired.jsonl

python egomas/src/run_video_provenance_v3_jdcloud.py \
  --benchmark /path/to/MA-EgoQA.json \
  --paired-source data/baselines/paired.jsonl \
  --video-root /path/to/egolife_raw \
  --output data/baselines/spatial_social.jsonl
```

## Reproducibility boundary

- Bulk raw media, downloaded datasets, model weights, and full prediction JSONL
  files are not redistributed. Six explicitly identified CoMind train frames
  are included only to make one paired case auditable.
- Included result summaries contain aggregate metrics only.
- Failed variants are retained for transparency but never replace a better
  official result.
- The repository contains no API keys, endpoint credentials, or local-machine
  paths.

## Complete Flash results and archived SSC code

See [MMToM 600-item Pro/Flash comparison and Habitat 108-rollout full archive](reports/flash_full_results_20260920/README.md) for paired results, complete generated programs and state traces, frozen implementations, offline replay, and comparison limitations.
