# Installation

## Offline example and tests

Use Python 3.10 or newer. Python 3.12 is the reference review environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m examples.minimal
python check_code.py
python run_tests.py
```

The `test` extra includes the CPU libraries imported by the shared and
benchmark-local tests. No model requests, GPU or original video files are needed.
The complete benchmark source layout is used from the checkout; installing the
Python package alone is not a standalone installation of all five pipelines.

`requirements-review.txt` retains the pinned Python 3.12 environment prepared for
the submission. To use those exact versions, install it in a separate Python 3.12
virtual environment, then install the package with `python -m pip install -e .`.

## Optional dependencies

| Extra | Intended use |
| --- | --- |
| `test` | Offline review tests; NumPy, pandas, Pillow, retrieval helpers and pytest |
| `video` | Frame/video processing using PyAV, NumPy, Pillow and tqdm |
| `analysis` | Dataframes, Parquet and BM25 retrieval |
| `local-model` | Optional Transformers/PyTorch backend; provision compatible GPU dependencies |
| `docs` | PDF/report generation |

```bash
python -m pip install -e '.[video,analysis]'
# Only if you intend to run a local model:
python -m pip install -e '.[local-model]'
```

Video launchers may also need FFmpeg on `PATH`. Habitat and SAM require their
own runtime, assets/weights and service processes; installing an extra does not
provide those. Read the corresponding benchmark guide before launching jobs.

## Model services

Copy the example file and fill in your own service values:

```bash
cp credentials.env.example credentials.env
# Edit credentials.env, then load it into your shell:
source credentials.env
```

`GEMINI_API_KEY`, `GEMINI_API_BASE` and `GEMINI_MODEL` configure the historical
Gemini-compatible client. A chat-completions endpoint is not automatically
interchangeable with a generate-content endpoint. Other launchers use the
additional `SSC_*` variables or experiment config files shown in their source.
Set the model to one your service actually supports and record it with the run.

Keep credential files out of version control. The example contains placeholders;
empty service values must be configured before live inference.

## Datasets and assets

Obtain benchmark annotations, raw media, simulator scenes and model weights
through the respective upstream projects. The submission snapshot does not
redistribute complete datasets or weights. See the
[benchmark guides](README.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
