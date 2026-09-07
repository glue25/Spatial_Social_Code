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

The repository contains two complementary evaluation tracks. In the primary
visual formulation, Physical and Social compilers may read the task evidence
and option-free question premise, but never answer choices, gold labels,
correctness feedback, or a baseline answer. Only the shared final answerer sees
the answer choices.

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
| MA-EgoQA (video) | **904/1741 (51.92%)** | 848/1741 (48.71%) | **+3.22 pp** |
| MMToM (legacy answer-directed heuristic; withdrawn as a clean SSC claim) | 470/600 (78.33%) | 406/600 (67.67%) | +10.67 pp |
| EnactToM/Habitat frozen-12 (goal completion) | **53.47%** | 45.83% | **+7.64 pp** |
| CoMind train recording-heldout OOF (bbox IoU@0.5) | **6.47%--23.15%** | 0.11%--0.34% | metric-dependent |

McNemar's exact test gives `p = 0.00388` on MA-EgoQA and `p = 3.17e-05`
on the legacy MMToM result. Aggregate machine-readable summaries are included with the code.
See the [submission results and baseline policy](reports/submission_results_and_baselines_zh.md)
for the component evidence, Habitat results, official published references,
and the strict comparability boundary.

## CoMind real-world grounding track

A deterministic 300-example CoMind train subset (100 examples per task) was
also evaluated with `gpt-5.6-sol`, using matched inputs for Direct and
Spatial-Social inference. All 300 pairs completed. Across 14 reported fields,
Spatial-Social was higher on 4, lower on 7, and tied on 3; it did not uniformly
improve over Direct. The only significant paired difference was a regression
on handover category L1 (0% versus 14%, exact McNemar `p = 0.000122`).

This is an exploratory train-subset comparison, not an official CoMind hidden
test result. See the [full report](reports/comind_gpt56_300_results_zh.md) and
[aggregate JSON](reports/comind_gpt56_300_summary.json).

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

## Adopted visual pipeline

1. Retrieve or sample visual evidence without consulting a baseline answer.
2. Compile option-blind grounded Spatial Code with clip/time provenance.
3. Execute the Spatial program in the restricted runtime.
4. Compile Social Code using only validated Spatial evidence references.
5. Infer answer-independent beliefs, goals, intents, and roles while
   preserving unknowns and contradictions.
6. Give the frozen traces and answer choices to one shared final reasoner.
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
[live Habitat status](reports/habitat_spatial_social_status_zh.md).

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

- Raw media, downloaded datasets, model weights, and full prediction JSONL
  files are not redistributed.
- Included result summaries contain aggregate metrics only.
- Failed variants are retained for transparency but never replace a better
  official result.
- The repository contains no API keys, endpoint credentials, or local-machine
  paths.
