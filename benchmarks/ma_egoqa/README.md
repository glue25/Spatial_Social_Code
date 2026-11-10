# MA-EgoQA code structure

| Path | Purpose |
|---|---|
| `egomas/src/run_jdcloud_baseline.py` | Text baseline |
| `egomas/src/run_video_spatial_social_jdcloud.py` | Paired video baseline and SSC evaluation |
| `egomas/src/run_video_grounded_v2_jdcloud.py` | Video grounding and final review |
| `egomas/src/run_video_provenance_v3_jdcloud.py` | Evidence-linked video evaluation |
| `egomas/src/audit_video_evidence_sparse_jdcloud.py` | Evidence review |
| `egomas/src/index_bm25.py` | Retrieval indexing |
| `egomas/src/test_*.py` | Pipeline tests |
| `egomas/utils/` | Prompts, input/output and evaluation utilities |

## Setup and use

Install the [video and analysis dependencies](../../docs/installation.md), obtain
the annotations and EgoLife media, and configure a compatible model service.

```bash
python benchmarks/ma_egoqa/egomas/src/run_video_provenance_v3_jdcloud.py --help
```

Set the benchmark, raw-video root, paired-source and output paths explicitly.
The provenance runner reuses a completed paired retrieval result; it does not
create that input automatically. Shared offline tests for this integration are
included in the root `python run_tests.py` command.

The separate MA-EgoQA component-ablation dependency is not included here; its
release remains a [TODO](../../TODO.md). No new model evaluation is implied by
the source-only tests.
