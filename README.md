<div align="center">

# Spatial–Social Code
### Thinking with Spatial–Social Code for Social Reasoning in Videos

**Xiaoyuan Zhang · Zhaowei Zhang · Wenbo Li<sup>◦</sup> · Alan Yuille · Tianmin Shu · Jieneng Chen**

Johns Hopkins University · <sup>◦</sup>The Chinese University of Hong Kong

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](docs/installation.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml/badge.svg)](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml)

[Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Documentation](docs/README.md) · [TODO](TODO.md) · [Citation](#citation)

</div>

**Spatial–Social Code (SSC)** represents physical situations and participant-specific
information as linked, executable records. Spatial Code organizes objects,
relations, events and perceptual evidence. Social Code records observations,
beliefs, goals and intentions, with explicit references to supporting evidence.

This repository is based on the **code snapshot prepared for our ICLR submission**.
It contains the shared runtime and five benchmark implementations, with an offline
test suite. The open-source preparation adds documentation, authorship, a runnable
example and release checks. It does not add new experimental results.

## Overview

```mermaid
flowchart LR
    A[Video / task evidence] --> B[Spatial Code]
    B --> C[Social Code]
    B --> D[Validated records]
    C --> D
    D --> E[Question answering / grounding / action]
```

- **Explicit state:** distinguish physical facts from a participant's observations and beliefs.
- **Evidence links:** validate the structure of generated calls and references between records.
- **Multiple tasks:** video question answering, collaborative grounding, embodied interaction and an information-history diagnostic.
- **Offline entry point:** inspect and test the code without API keys, model weights or benchmark media.

Execution validates record structure and evidence-ID membership; it does not prove
that a generated belief or visual interpretation is true. Historical benchmark
pipelines have different information access and budgets; see the
[reproducibility guide](docs/reproducibility.md).

## Quick start

```bash
git clone https://github.com/glue25/Spatial_Social_Code.git
cd Spatial_Social_Code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

python -m examples.minimal
python check_code.py
python run_tests.py
```

On Windows, activate with `.venv\Scripts\activate`. The shared runtime and example
are portable; live experiment launchers that use Unix locking or Habitat require
a suitable Linux environment.

The example uses a small, hand-written scene and produces linked Spatial/Social
JSON records. It makes no model calls. The test runner executes benchmark
snapshots in separate processes so their local imports do not conflict. Tests
requiring omitted datasets or historical result fixtures report explicit skips.

For video dependencies, model-service configuration and simulator prerequisites,
see [installation](docs/installation.md).

## Benchmarks

| Benchmark | Task | Code and protocol |
| --- | --- | --- |
| MMToM-QA | Belief and goal inference | [MMToM guide](benchmarks/mmtom/README.md) |
| MA-EgoQA | Multi-agent egocentric video QA | [MA-EgoQA guide](benchmarks/ma_egoqa/README.md) |
| CoMind | Collaborative visual grounding | [CoMind guide](benchmarks/comind/README.md) |
| Habitat / EnactToM | Embodied interaction and state updates | [Habitat guide](benchmarks/habitat/README.md) |
| Information history | Controlled symbolic information-use diagnostic | [Diagnostic guide](benchmarks/information_history/README.md) |

The source-only snapshot does not include full datasets, model weights, API
credentials or complete result archives. The benchmark launchers require their
original inputs and services; passing offline tests is not a rerun of the paper's
experiments. Availability and remaining work are listed in [TODO.md](TODO.md).

## Repository layout

```text
spatial_social_code/    Shared record language, runtime and reasoning utilities
benchmarks/            Five benchmark-specific implementations and snapshots
experiments/           Shared controls, adapters and analysis utilities
examples/              Minimal example without external data or model services
tests/                 Shared tests (benchmark-local tests live with their code)
docs/                  Installation, reproducibility, architecture and provenance
scripts/               Setup helpers and release-manifest maintenance
third_party/           Retained upstream code and license notices
check_code.py          Offline integrity and Python syntax checks
run_tests.py           Offline shared and benchmark-local test runner
```

Benchmark snapshots intentionally retain separate module trees to preserve the
submitted implementations. See [architecture](docs/architecture.md) before
consolidating duplicate modules. The previous research archive remains accessible
in the [previous research snapshot](https://github.com/glue25/Spatial_Social_Code/tree/d3b8a5c)
and Git history.

## Release roadmap

- [x] Restore the organized ICLR submission code and verify its file hashes.
- [x] Add author information, citation metadata and a minimal example.
- [x] Document installation, benchmark entry points and offline verification.
- [x] Add CI configuration and preserve third-party notices.
- [ ] Add the public arXiv identifier after announcement.
- [ ] Publish a versioned mapping from paper tables to protocols and result artifacts.
- [ ] Provide portable live-run configs and small end-to-end benchmark walkthroughs.
- [ ] Complete the third-party provenance checklist before public release.

See [TODO.md](TODO.md) for concrete completion criteria and known gaps.

## Citation

Until an arXiv identifier is available, cite the repository:

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

Author order and affiliations are recorded in [AUTHORS.md](AUTHORS.md).
GitHub-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Contributing and license

Bug reports, documentation fixes and reproducibility improvements are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the local checks and issue templates.

Original SSC code and documentation are released under the [MIT License](LICENSE).
Retained third-party code, dataset assets and benchmark annotations are **not
relicensed** by that file; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
