<div align="center">

# Spatial–Social Code
### Thinking with Spatial–Social Code for Social Reasoning in Videos

**Xiaoyuan Zhang · Zhaowei Zhang · Wenbo Li<sup>◦</sup> · Alan Yuille · Tianmin Shu · Jieneng Chen**

Johns Hopkins University · <sup>◦</sup>The Chinese University of Hong Kong

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml/badge.svg)](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml)

**[Paper (alphaXiv)](https://www.alphaxiv.org/abs/2611.spatial-social-code-video-reasoning)** · [Quick start](#quick-start) · [Benchmarks](#benchmarks) · [TODO](#todo) · [Citation](#citation)

</div>

**Spatial–Social Code (SSC)** represents physical situations and participant-specific
information as linked, executable records. Spatial Code records objects, events
and perceptual evidence; Social Code records observations, beliefs, goals and
intentions with references to supporting evidence.

This repository contains the shared runtime and five benchmark implementations
from our ICLR submission. Full datasets, weights and frozen run artifacts are
not bundled; source cleanup does not constitute new experimental results.

## Quick start

```bash
git clone https://github.com/glue25/Spatial_Social_Code.git
cd Spatial_Social_Code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m spatial_social_code demo
python -m spatial_social_code test
```

The demo executes a hand-written scene without a GPU or API key. Offline tests
check the runtime, benchmark helpers, structured API transport and SAM sidecar;
tests requiring omitted datasets or historical records explicitly skip. Python
3.10 and 3.12 are checked in CI. Live Unix/Habitat runners need a suitable Linux
environment. On Windows activate the environment with `.venv\Scripts\activate`.

## Benchmarks

Use `python -m spatial_social_code run NAME --help` to inspect each runner.
Arguments after `NAME` are forwarded to the benchmark; use `--` if needed.

| Name | Task | Required inputs |
| --- | --- | --- |
| `mmtom` | Belief and goal inference | MMToM-QA questions/frames, reference JSONL, replay-audit manifests and model service |
| `ma-egoqa` | Multi-agent video QA | MA-EgoQA annotations, EgoLife clips/proxies, retrieval inputs, FFmpeg and service |
| `comind` | Collaborative grounding | Frozen shard configs and media hashes, multi-view frames/transcripts and service; SAM weights/service for grounding |
| `habitat` | Embodied interaction | Linux/CUDA/Habitat environment, licensed scenes, frozen tasks/episodes and service |
| `information-history` | Symbolic information-use diagnostic | Source task vocabulary and prepared cases; service for fresh predictions |

The entry points run from this source checkout; an installed wheel alone does
not include all benchmark inputs and scripts. See the expandable setup and
protocol notes below before starting live inference.

## Code

```text
spatial_social_code/   Shared runtime, reasoning, scoring and command entry point
benchmarks/           Five benchmark pipelines and service configuration example
tests/                Shared and integration tests
third_party/          Upstream helpers and license notices
.github/              CI, citation metadata and release file hashes
```

## TODO

- [x] Consolidate shared code, add authors/citation and verify offline tests in CI.
- [x] Provide a common command entry point and verified, on-demand Habitat source setup.
- [ ] Confirm exact upstream revisions and redistribution terms for MA-EgoQA helpers
  and the cap-x SAM launcher before public release; add their applicable notices.
- [ ] Add the announced arXiv link and a paper-table-to-result index recording
  model, protocol, split, sample counts, seeds, scorer and input hashes.
- [ ] Publish MMToM/CoMind frozen manifests and portable data/config/output arguments,
  plus one end-to-end walkthrough per benchmark. Preserve historical defaults.
- [ ] Package dependencies for the later symmetric CoMind and MA component-ablation
  protocols; keep these distinct from the submitted historical compiler.
- [ ] Validate the pinned Linux/CUDA/Habitat environment on a licensed task and
  support benchmark execution outside the source checkout.
- [ ] Create a named release with source ZIP and checksums after these items are complete.

For research follow-up: freeze evaluation splits and matching rules before runs;
record matched evidence/tools/model versions/calls/token budgets and failures;
report task/seed uncertainty and invalid outputs; complete human annotation or
semantic review before reporting it. These are planned tasks, not completed results.

## Citation

Until the arXiv identifier is announced, cite the repository:

```bibtex
@misc{zhang2026spatialsocialcode,
  title  = {Thinking with Spatial-Social Code for Social Reasoning in Videos},
  author = {Zhang, Xiaoyuan and Zhang, Zhaowei and Li, Wenbo and
            Yuille, Alan and Shu, Tianmin and Chen, Jieneng},
  year   = {2026},
  url    = {https://github.com/glue25/Spatial_Social_Code},
  note   = {Research code; preprint identifier forthcoming}
}
```

Wenbo Li is affiliated with CUHK; the other five authors are affiliated with JHU.
Machine-readable citation metadata is in [.github/CITATION.cff](.github/CITATION.cff).

## License

Original SSC code and documentation use the [MIT License](LICENSE). Upstream
code and assets retain their own terms; see [third-party notices](third_party/NOTICE)
and the unresolved provenance items in the TODO list.

<details>
<summary><strong>Advanced setup and protocol notes</strong></summary>

### Dependencies and services

```bash
python -m pip install -e '.[video,analysis]'
cp benchmarks/credentials.env.example credentials.env
# Fill in your service values before loading them:
source credentials.env
```

Extras: `video` provides PyAV/NumPy/Pillow/tqdm, `analysis` provides
pandas/PyArrow/BM25, `local-model` provides PyTorch/Transformers, and `docs`
provides ReportLab. Video decoding may also require FFmpeg. Extras do not supply
model weights, datasets or a simulator. Endpoint protocols differ: inspect the
runner's configuration and record the actual model/service version used.

### Habitat source setup

The repository stores one patch instead of 218 copied upstream files:

```bash
python -m spatial_social_code setup-habitat
# Alternatively use an already downloaded, checksum-matching upstream archive:
python -m spatial_social_code setup-habitat --archive /path/to/EnactTom.tar.gz
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

### Historical protocol boundaries

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


Execution checks call structure and evidence-ID membership; it does not establish
that an inferred belief or interpretation is true. Historical MMToM reasoning may
use question/option-conditioned evidence and conditional denser-frame passes.
CoMind SSC+SAM is not automatically a symmetric Direct+SAM control. Do not merge
these with later revised protocols without documenting information access and budgets.

For every new comparison record source commit, input/config hashes, split, model,
seed, tools, frame selection, API calls, tokens, parser/scorer and failure treatment.
Keep invalid outputs and distinguish offline replay from fresh inference. This
cleanup does not upgrade a development result to a held-out evaluation.

### Maintenance and source history

For changes, report a minimal failing example and environment, run the tests,
review the diff, then refresh and verify the release hashes:

```bash
python -m spatial_social_code manifest
python -m spatial_social_code check
```

`.github/release-manifest.json` records current file hashes, old-to-new source
routes and the pinned Habitat dependency. It excludes its own hash. Checking works
in exported ZIPs; regenerating the manifest requires Git. Keep credentials,
datasets, weights and generated outputs out of commits.

The complete earlier submission-based release and old analysis scripts remain
in [Git history](https://github.com/glue25/Spatial_Social_Code/tree/24fd07f0f2de96ddf8e9ba3eff32302262157466).
Use that checkout for original directory layouts. After changing source paths,
prepare new source hashes and preserve original run records separately. Offline
checks do not reproduce the paper's model inference or simulator results.

</details>
