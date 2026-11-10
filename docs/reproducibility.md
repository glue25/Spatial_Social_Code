# Reproducibility

## What can be checked offline

```bash
python -m examples.minimal
python check_code.py
python run_tests.py
```

- The example executes a synthetic Spatial/Social record pair without a model.
- `check_code.py` verifies the release manifest and parses the listed Python files.
  It reports local edits rather than updating the manifest automatically.
- `run_tests.py` checks the shared representation, benchmark helpers, SAM sidecar,
  information-history diagnostic and Habitat snapshot in isolated processes.
  Dataset/archival-fixture tests may skip explicitly when their inputs are absent.

These checks establish code consistency and tested behavior. They do not repeat
API inference, run Habitat physics or reproduce the paper's numerical results.

## What a live run additionally needs

| Pipeline | Source entry point | Additional inputs |
| --- | --- | --- |
| MMToM | `benchmarks/mmtom/matched_runner.py` | Original question/frame inputs, historical input/config manifests and model service |
| MA-EgoQA | `benchmarks/ma_egoqa/egomas/src/run_video_provenance_v3_jdcloud.py` | Benchmark annotations, EgoLife clips, paired retrieval inputs, FFmpeg and service |
| CoMind | `benchmarks/comind/pipelines/compiler/run.py` | Frozen shard inputs/configs, multi-view media and transcripts, model service; SAM weights/service for the grounding stage |
| Habitat | `benchmarks/habitat/experiments/reevaluation_v3/run_qdd_habitat.py` | Linux/CUDA simulator environment, EnactToM tasks, licensed scenes/assets and model service |
| Information history | `benchmarks/information_history/experiments/information_history_v1/run.py` | Prepared diagnostic inputs, source task vocabulary and service for new predictions |

Consult each benchmark README and its parser before running. A source entry point
is not a promise of a one-command full reproduction. Frozen run configurations,
full media and historical output archives are not included in this source-only
release. The information-history study is a constructed symbolic diagnostic,
not a new video counterfactual or a Habitat physics rollout.

## Protocol and result versions

Do not combine historical and later runs into one comparison merely because
they share a benchmark name. MMToM snapshots can contain question/option-conditioned
reasoning and conditional denser-frame passes; their complete evidence and budgets
need not match Direct. CoMind's historical SSC+SAM pipeline is not itself a
symmetric Direct+SAM control. Later symmetric and component-ablation dependencies
are tracked explicitly in the roadmap.

For any new comparison, record the source commit, input/config hashes, split,
model/service version, seed, tools, frame selection, calls, tokens, parser/scorer
and failure treatment. Preserve invalid outputs and distinguish offline replay
from fresh inference. The cleanup does not certify the semantic correctness of
archived answers or upgrade a development result to a held-out evaluation.

## Updating release checksums

After intentional changes, review the diff, then run:

```bash
python scripts/update_checksums.py
python check_code.py
```

The checksum updater uses Git's tracked and unignored files. Inspect new files
before running it; the manifest is an integrity aid, not a security audit.
The original submitted hashes remain under `docs/provenance/` for comparison.
