# CoMind code structure

| Path | Purpose |
|---|---|
| `pipelines/compiler/` | Native-schema generation and source dependencies |
| `pipelines/sam/` | SAM sidecar, coordinate conversion and tests |
| `pipelines/recovery/` | Recovery and scoring logic |
| `pipelines/transport/` | Model transport and source dependencies |
| `cap-x/capx/serving/launch_sam3_server.py` | SAM service launcher |

## Setup and use

```bash
python benchmarks/comind/pipelines/compiler/run.py --help
```

The compiler expects prepared `runs/comind_0` or `runs/comind_1` directories with
`frozen_config.json`, `inputs.json` and media hashes, plus access to their original
inputs and model service. The source-only package does not include these run
directories. SAM additionally needs its model weights and configured service.

Run `python run_tests.py` at the repository root for offline SAM-sidecar tests.
They do not load the SAM model or regenerate dataset predictions. This historical
pipeline is distinct from the later symmetric Direct+SAM/SSC+SAM protocol; see
the [reproduction TODO](../../TODO.md).
