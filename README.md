# Spatial-Social Code

Spatial-Social Code is an executable intermediate-representation method for
video question answering and theory-of-mind reasoning. It separates grounded
physical evidence from social and mental-state inference:

- **Spatial Code** records facts, visibility, events, event roles, temporal
  phases, and provenance.
- **Social Code** records observations, beliefs, goals, intents, action
  evidence, contradictions, and calibrated hypothesis posteriors.
- A restricted AST runtime validates generated programs before executing them
  into canonical JSON traces.

The method is standalone. Direct-video baseline predictions are stored only
for paired evaluation and never select, switch, fuse, gate, or overwrite a
Spatial-Social answer.

## Results

All comparisons use the same model and the same visual inputs for the method
and its paired direct baseline.

| Benchmark | Spatial-Social Code | Direct baseline | Difference |
|---|---:|---:|---:|
| MA-EgoQA (video) | **904/1741 (51.92%)** | 848/1741 (48.71%) | **+3.22 pp** |
| MMToM | **470/600 (78.33%)** | 406/600 (67.67%) | **+10.67 pp** |

For MA-EgoQA, McNemar's exact test gives `p = 0.00388`. For MMToM,
`p = 3.17e-05`. Compact machine-readable summaries are included with the
evaluation code.

## Method

The adopted pipeline is:

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

Important inference constraints include:

- missing evidence is unknown, not negative evidence;
- a world fact is not automatically an agent belief;
- execution or observation does not imply initiation, responsibility, intent,
  or causality;
- social records can be required to cite valid Spatial evidence IDs;
- answer choices are introduced only after candidate-hidden evidence
  extraction in the native Bayesian path.

## Repository layout

```text
spatial_social_code/
  runtime.py                    restricted Spatial/Social runtimes
  social_code.py                native Social Code compiler and inference
  evaluate_mmtom_jdcloud.py     paired MMToM evaluator
  refine_mmtom_uncertain_jdcloud.py
  rescore_mmtom_dense_authoritative.py
  test_spatial_social.py
MA-EgoQA/egomas/src/
  run_video_spatial_social_jdcloud.py
  run_video_balanced_baseline_jdcloud.py
  run_video_provenance_v3_jdcloud.py
  audit_video_evidence_sparse_jdcloud.py
  jdcloud_gemini.py
```

The executable-hypothesis experiment is included as an explicitly failed
ablation. It is not part of the reported best method.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Obtain MA-EgoQA/EgoLife and MMToM from their respective authors and place them
using the paths expected by the evaluators, or pass explicit `--benchmark`,
`--video-root`, `--questions`, and `--frames` arguments.

Configure a Gemini-compatible endpoint with environment variables:

```bash
export GEMINI_API_KEY="..."
export GEMINI_API_BASE="https://your-endpoint.example/v1/generateContent"
export GEMINI_MODEL="your-model"
export FFMPEG_BINARY="$(command -v ffmpeg)"
```

For the optional local Qwen backend, set `SPATIAL_SOCIAL_MODEL_PATH` to a
local checkpoint or Hugging Face model ID. Set
`SPATIAL_SOCIAL_LOCAL_FILES_ONLY=1` for fully offline loading.

Alternatively, set `GEMINI_CREDENTIALS_FILE` to a local shell-style env file.
Credential files are ignored by Git.

## Tests

```bash
python spatial_social_code/test_spatial_social.py
cd MA-EgoQA
python -m unittest \
  egomas.src.test_jdcloud_baseline \
  egomas.src.test_spatial_social_jdcloud \
  egomas.src.test_video_spatial_social_jdcloud \
  egomas.src.test_video_provenance_v3_jdcloud
```

## Evaluation

Paired MMToM evaluation:

```bash
python spatial_social_code/evaluate_mmtom_jdcloud.py \
  --questions /path/to/MMToM-QA/Benchmark/questions.json \
  --frames /path/to/required_frames \
  --output results/mmtom.jsonl
```

MA-EgoQA video evaluation is a two-stage paired workflow. First construct the
answer-symmetric clip set and direct prediction, then run the standalone
provenance-constrained method on exactly those clips:

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

## Reproducibility and scope

- Raw benchmark media and full prediction JSONL files are not redistributed.
- Included result summaries contain aggregate metrics only.
- Failed audit and executable-hypothesis variants are retained for
  transparency but never replace a better official result.
- The repository does not contain API keys, endpoint credentials, or local
  machine paths.
