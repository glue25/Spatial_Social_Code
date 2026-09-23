# MMToM historical component diagnostics

**Archive release, not a new run and not the current option-blind SSC protocol.**
The tables below preserve two existing studies with different backbones and
scoring protocols. Do not combine them or describe them as a clean intervention
on evidence links / agent attribution. All counts were recomputed from the
stored per-question predictions before publication.

## 1. Qwen3-VL-32B-Instruct — frozen70

Seven question types, ten questions each. All70 selected final records are
present with no error rows. Full predictions and stored image metadata match
the primary-run records for every selected question.

| Condition | Correct / total | Accuracy |
|---|---:|---:|
| Direct | 25/70 | 35.71% |
| Spatial-only | 27/70 | 38.57% |
| Social-only | 46/70 | 65.71% |
| Raw Spatial+Social code, without runtime | 42/70 | 60.00% |
| Joint Spatial+Social runtime | **50/70** | **71.43%** |

Joint minus Spatial-only: **+32.86 percentage points**, exact McNemar p=0.00140.
Joint minus Social-only: **+5.71 points**, p=0.34375 (not significant).
For the latter comparison,7 questions favor joint and3 favor Social-only.

### Material protocol limitations

- This historical implementation is **candidate-conditioned**: saved
  `query_evidence` occurs in40/70 Social-only and66/70 joint codes. Social-only
  has an answer posterior in70/70. This is not current option-blind compilation.
  Candidate conditioning by itself does not demonstrate gold-label access.
- Spatial-only uses a VLM answer over frames plus the saved Spatial trace.
  Social-only is generated separately without that Spatial trace and answered
  through a posterior; joint reuses the original runtime prediction. Therefore
  the groups do **not** share one compilation and one identical final answerer.
- Compute budgets/pipelines are not matched. These results do not isolate the
  contribution of explicit cross-layer links.
- Matching saved image metadata is not independent certification of historical
  frame bytes. The70 final records are complete, but any earlier retries that
  were overwritten in the old JSONL cannot be reconstructed from this export.

## 2. Historical Gemini traces —600-question bare-evidence diagnostic

Archive backbone label: Gemini 3-Pro-Preview. This is **offline recomputation**
of frozen records, not600 new API evaluations of the current implementation.

| Bare-evidence condition | Correct / total | Accuracy |
|---|---:|---:|
| Spatial-only | 313/600 | 52.17% |
| Social-only | 298/600 | 49.67% |
| Joint Spatial+Social | **319/600** | **53.17%** |

Joint minus Spatial-only: **+1.00 percentage point**, exact McNemar p=0.51184
(not significant);32 questions favor joint and26 favor Spatial-only.
Joint minus Social-only: +3.50 points, p=0.02753.
These nominal question-level tests are unadjusted and are not a replacement for
episode-cluster uncertainty analysis.

The old recomputation explicitly separated neural evidence from fused cached
programs to avoid importing the full-method posterior as Social-only evidence.
Nevertheless, it remains a historical candidate/posterior-scoring diagnostic,
not an end-to-end test of the current method. The common question/rule scaffold
is removed in this bare-evidence matrix; its scores cannot be compared directly
with the following matrix or with full-system headline scores.

### Required companion result: retained shared scaffold

The same archive also contains the following600-question matrix. It is included
to avoid selectively presenting only positive joint-versus-single comparisons.

| Condition | Correct / total | Accuracy |
|---|---:|---:|
| Shared question/rule scaffold | 418/600 | 69.67% |
| Scaffold + Spatial | **452/600** | **75.33%** |
| Scaffold + Social | 415/600 | 69.17% |
| Scaffold + Spatial + Social | 450/600 | 75.00% |

Here joint is **0.33 points below Spatial-only**. Historical labeled-data OOF
routing is not relabeled as plain SSC in this release.

## Appropriate interpretation

> Joint representations have higher point estimates than either single
> component in the historical Qwen70 and bare-evidence Gemini600 diagnostics.
> Neither comparison establishes a significant advantage over its strongest
> single component. Protocol differences and the nonpositive shared-scaffold
> result prevent attributing these outcomes specifically to evidence links.

These are development diagnostics, not untouched held-out validation. They
must not replace or conceal nonpositive MA/Habitat results, and should not be
relabeled a current Flash/Pro mechanism ablation or proof of universal synergy.

## Reproducible export

- `qwen70_predictions.csv`: all70 labels/predictions and audited protocol flags.
- `gemini600_offline_predictions.csv`: all600 labels/predictions, including both
  the bare-evidence and shared-scaffold matrices.
- `summary.json`: counts, type coverage, paired discordances, p-values/checksums.
- `protocol_and_provenance.json`: source hashes, method differences and scope.
- `verify.py`: standalone offline arithmetic/coverage/checksum verification.

```bash
python reports/mmtom_historical_components_20260923/verify.py
```

No model, API, credentials or datasets are required to verify these tables.
Raw videos, full prompts, model reasoning and model weights are not included.
Verification checks exported records, not a rerun of the historical models.
