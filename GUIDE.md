# Usage and reproducibility

## Install and inspect

Use Python 3.10 or newer. The offline suite is checked on Python 3.10 and 3.12.
Run commands from the source checkout; the installed Python package alone does
not contain all benchmark scripts.

```bash
python -m pip install -e '.[test]'
python manage.py demo
python manage.py check
python manage.py test
```

`demo` executes illustrative, hand-written Spatial/Social records. `check` verifies
SHA-256 hashes and Python syntax. `test` runs shared tests and isolated CoMind tests
without contacting model services. Optional original datasets and archival audit
records are not bundled; their dependent tests explicitly skip.

On Windows use `.venv\Scripts\activate`. The core runtime is portable; live runners
using Unix locking and the Habitat stack require an appropriate Linux environment.

Optional extras: `video` installs PyAV, NumPy, Pillow and tqdm; `analysis` installs
pandas, PyArrow and BM25; `local-model` installs PyTorch/Transformers; `docs` installs
ReportLab. Video decoding may also need FFmpeg. These extras do not install datasets,
SAM weights or a working simulator.

```bash
python -m pip install -e '.[video,analysis]'
cp credentials.env.example credentials.env
# Fill in your own values before loading them:
source credentials.env
```

Endpoint protocols differ: a chat-completions endpoint cannot automatically replace
a generate-content endpoint. Inspect the selected runner's parser and configuration,
and record the actual model/service version used. Never commit credentials.

## Benchmark entry points and required inputs

`python manage.py run NAME --help` forwards help to the original parser. Use `--`
before forwarded arguments when needed. This wrapper sets import paths; it does
not fabricate missing configurations or download datasets.

| Name | Implementation | Required inputs beyond this repository |
| --- | --- | --- |
| `mmtom` | `benchmarks/mmtom/matched_runner.py` | MMToM-QA questions/frames, original reference JSONL and replay-audit manifests, model service |
| `ma-egoqa` | `benchmarks/ma_egoqa/egomas/src/run_video_provenance_v3_jdcloud.py` | MA-EgoQA annotations, EgoLife clips/proxies, retrieval inputs, service and FFmpeg |
| `comind` | `benchmarks/comind/pipelines/compiler/run.py` | Frozen shard inputs/configs and media hashes, multi-view frames/transcripts, service |
| `habitat` | `benchmarks/habitat/run.py` | Pinned upstream code, Linux/CUDA/Habitat environment, licensed scenes, frozen tasks/episodes and model service |
| `information-history` | `benchmarks/information_history/run.py` | Source task vocabulary for preparing diagnostic cases; service for fresh predictions |

MMToM also includes separate native transport, offline replay checks and comparison
scripts. Its `legacy/` folder preserves four historically different reasoning modules;
shared utilities are imported from the canonical packages. Legacy answer-conditioned
`query_evidence` calls are available only through the explicit legacy runtime bridge.
The current shared runtime continues to reject those calls by default.

CoMind has three stages under `benchmarks/comind/pipelines`: `compiler` generates
original structured predictions; `sam` applies the label-blind grounding sidecar;
`recovery` retains the submitted whole-pair recovery policy. The compiler extracts
its original prompt expressions and schemas from the canonical shared module and
explicitly uses the legacy executor. Run `sam_server.py` only with the separately
provisioned upstream service dependencies and weights. Historical launch preparation
still needs the original frozen manifests and diagnostic artifacts; these are not
replaced by synthetic production inputs.

Information history is a constructed symbolic diagnostic. It is not a video
counterfactual benchmark or a Habitat physics rollout. Habitat uses the submitted
Flash v2 closed-loop implementation, with physical outcome checks and separate
participant information stores.

## Habitat dependency setup

The repository stores one patch instead of 218 copied upstream files:

```bash
python manage.py setup-habitat
# Alternatively use an already downloaded, checksum-matching upstream archive:
python manage.py setup-habitat --archive /path/to/EnactTom.tar.gz
```

The command downloads EnactToM revision
`093c4a11b622c8f0b50070e51ae9461b2b1db2f8`, verifies the archive checksum, applies
`benchmarks/habitat/enacttom.patch`, and verifies all 218 submitted file hashes.
It installs source into the ignored `external/EnactTom` directory and preserves
upstream configuration files and license. An existing modified dependency is
rejected rather than overwritten. This step does not install Python/CUDA packages,
scenes, model weights or datasets; follow the pinned upstream installation instructions
in that directory, then provision the task inputs required by `benchmarks/habitat/run.py`.

The patch preserves the 15 submission-modified upstream files and redirects imports
to `benchmarks.habitat`. The actual Habitat/CUDA environment and new model calls were
not executed during this repository cleanup.

## Protocol boundaries

Execution checks call structure and evidence-ID membership; it does not establish
that an inferred belief or interpretation is true. Historical MMToM reasoning may
use question/option-conditioned evidence and conditional denser-frame passes.
CoMind SSC+SAM is not automatically a symmetric Direct+SAM control. Do not merge
these with later revised protocols without documenting information access and budgets.

For every new comparison record source commit, input/config hashes, split, model,
seed, tools, frame selection, API calls, tokens, parser/scorer and failure treatment.
Keep invalid outputs and distinguish offline replay from fresh inference. This
cleanup does not upgrade a development result to a held-out evaluation.

## Source history and maintenance

The full previous release is preserved at commit
[`24fd07f`](https://github.com/glue25/Spatial_Social_Code/tree/24fd07f0f2de96ddf8e9ba3eff32302262157466).
Duplicate snapshots and old analysis/export scripts were removed from the default
tree; obtain the archived release when reproducing their original directory layout.
Do not reuse old frozen source hashes after changing paths: prepare a new source
manifest while keeping the original run and its records intact.

`MANIFEST.json` holds current file hashes, old-to-new source routes, the base release
and the pinned dependency's expected hashes. It does not hash itself. After intentional
edits, inspect newly added files, run tests, then:

```bash
python manage.py manifest
python manage.py check
```

The manifest works in exported ZIPs; updating it requires a Git checkout. Source
hashes establish byte integrity, not scientific validity. Open release issues and
experimental work are tracked together in [TODO.md](TODO.md); consolidated upstream
license text is in [NOTICE](NOTICE).
