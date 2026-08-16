# Spatial-Social Code

Spatial-Social Code studies whether models reason more reliably from an
explicit, executable representation of a multi-agent world. It separates
grounded physical evidence from social and mental-state inference:

- **Spatial Code** records facts, visibility, events, roles, temporal phases,
  and provenance.
- **Social Code** records observations, beliefs, goals, intents, action
  evidence, contradictions, and calibrated hypothesis posteriors.
- A restricted AST runtime validates generated programs before executing them
  into canonical JSON traces.

The repository contains two complementary evaluation tracks:

1. deterministic/oracle-state compilation and counterfactual replay on
   EnactToM with Qwen3-4B;
2. grounded visual Spatial-Social Code evaluation on MMToM and MA-EgoQA.

Direct-answer baselines are stored only for paired evaluation. They never
select, switch, fuse, gate, or overwrite a Spatial-Social answer.

## Current visual results

The method and paired baseline use the same model and the same visual inputs.

| Benchmark | Spatial-Social Code | Direct baseline | Difference |
|---|---:|---:|---:|
| MA-EgoQA (video) | **904/1741 (51.92%)** | 848/1741 (48.71%) | **+3.22 pp** |
| MMToM | **470/600 (78.33%)** | 406/600 (67.67%) | **+10.67 pp** |

McNemar's exact test gives `p = 0.00388` on MA-EgoQA and `p = 3.17e-05`
on MMToM. Aggregate machine-readable summaries are included with the code.

## Adopted visual pipeline

1. Retrieve or sample visual evidence without consulting a baseline answer.
2. Compile choice-blind grounded Spatial Code with clip/time provenance.
3. Execute the Spatial program in the restricted runtime.
4. Compile Social Code using only validated Spatial evidence references.
5. Infer beliefs, goals, intents, roles, and answer-aligned likelihoods while
   preserving unknowns and contradictions.
6. Answer from raw visual evidence plus validated executable traces.
7. On uncertain MMToM items, use an independent denser visual pass as the
   authoritative method prediction instead of averaging it with sparse
   evidence.

Important constraints include:

- missing evidence is unknown, not negative evidence;
- a world fact is not automatically an agent belief;
- execution or observation does not imply initiation, responsibility, intent,
  or causality;
- Social records can be required to cite valid Spatial evidence IDs;
- answer choices are introduced only after candidate-hidden extraction in the
  native Bayesian path.

## EnactToM track

The E0 experiment compiles all 300 public EnactToM tasks into query-independent
oracle state, replays golden trajectories, and evaluates message
deletion/redirection interventions. It tests the representation and reducer,
not learned-agent performance.

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
python3 scripts/validate_datasets.py
python3 scripts/run_enacttom_e0.py
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

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
python spatial_social_code/test_spatial_social.py
cd MA-EgoQA
python -m unittest \
  egomas.src.test_jdcloud_baseline \
  egomas.src.test_spatial_social_jdcloud \
  egomas.src.test_video_spatial_social_jdcloud \
  egomas.src.test_video_grounded_v2_jdcloud \
  egomas.src.test_video_provenance_v3_jdcloud
```

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
