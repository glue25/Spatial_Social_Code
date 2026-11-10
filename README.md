<div align="center">

# Spatial–Social Code
### Thinking with Spatial–Social Code for Social Reasoning in Videos

**Xiaoyuan Zhang · Zhaowei Zhang · Wenbo Li<sup>◦</sup> · Alan Yuille · Tianmin Shu · Jieneng Chen**

Johns Hopkins University · <sup>◦</sup>The Chinese University of Hong Kong

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](GUIDE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml/badge.svg)](https://github.com/glue25/Spatial_Social_Code/actions/workflows/tests.yml)

[Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Guide](GUIDE.md) · [TODO](TODO.md) · [Citation](#citation)

</div>

**Spatial–Social Code (SSC)** represents physical situations and participant-specific
information as linked, executable records: Spatial Code records objects, events
and perceptual evidence; Social Code records observations, beliefs, goals and
intentions with references to supporting evidence.

This compact release is based on our **ICLR submission code**. Shared code lives
in one place, five benchmarks have distinct entry points, and Habitat's upstream
source is downloaded only when needed. This cleanup adds no experimental results.

## Quick start

```bash
git clone https://github.com/glue25/Spatial_Social_Code.git
cd Spatial_Social_Code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python manage.py demo
python manage.py check
python manage.py test
```

The demo executes a hand-written scene without data, a GPU or an API key.
Tests cover the shared runtime, benchmark helpers, structured API transport and
SAM sidecar. Tests requiring omitted datasets or historical records skip explicitly.
Live experiment launchers need their datasets, frozen run inputs and services;
offline tests do not reproduce the paper's results.

## Code layout

```text
spatial_social_code/   Shared language, runtime, reasoning and scoring
benchmarks/           mmtom / ma_egoqa / comind / habitat / information_history
tests/                Shared and integration tests
third_party/          Two retained AutoToM helper modules
manage.py             Demo, tests, integrity checks, dependency setup and runners
GUIDE.md              Installation, benchmark commands and protocol notes
TODO.md               Release and research work still to complete
NOTICE                Consolidated third-party license notices
MANIFEST.json         File hashes, source mapping and pinned Habitat dependency
```

| Benchmark | Task | Entry point |
| --- | --- | --- |
| MMToM-QA | Belief and goal inference | `python manage.py run mmtom --help` |
| MA-EgoQA | Multi-agent egocentric video QA | `python manage.py run ma-egoqa --help` |
| CoMind | Collaborative visual grounding | `python manage.py run comind --help` |
| Habitat / EnactToM | Embodied interaction and state updates | `python manage.py run habitat --help` |
| Information history | Controlled symbolic information-use diagnostic | `python manage.py run information-history --help` |

See [GUIDE.md](GUIDE.md) for required inputs and the limits of each protocol.
Materially different historical MMToM prompts remain in `benchmarks/mmtom/legacy`;
shared helpers and duplicate source trees have been merged. The complete earlier
submission-based release, including development scripts, remains in
[Git history](https://github.com/glue25/Spatial_Social_Code/tree/24fd07f0f2de96ddf8e9ba3eff32302262157466).

## Roadmap

- [x] Consolidate duplicate modules and documentation; unify local commands.
- [x] Preserve original authors, citation metadata and upstream notices.
- [x] Pin Habitat's upstream source and verify the SSC patch against the submission.
- [ ] Resolve remaining third-party provenance before public release.
- [ ] Add the announced arXiv link and paper-table-to-result mapping.
- [ ] Publish portable live-run configs and end-to-end reproduction walkthroughs.

Completion criteria and research follow-up are listed in [TODO.md](TODO.md).

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

Author order and affiliations appear above; machine-readable citation metadata is
in [CITATION.cff](CITATION.cff). Wenbo Li is affiliated with CUHK; the other five
authors are affiliated with JHU.

## Contributing and license

Please describe the failing command, environment and a minimal example in an
issue or pull request. Run `python manage.py test`, review your diff, then refresh
and verify file hashes with `python manage.py manifest` and `python manage.py check`.
Keep dataset files, model weights, credentials and generated runs out of commits.

Original SSC code and documentation use the [MIT License](LICENSE). Third-party
code and assets retain their own terms; see [NOTICE](NOTICE) and the unresolved
provenance items in [TODO.md](TODO.md).
