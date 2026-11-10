# MMToM code structure

| Path | Purpose |
|---|---|
| `matched_runner.py` | Model-comparison experiment launcher |
| `legacy_loader.py` | Source snapshot loader |
| `native_transport.py` | Model service transport |
| `service_gate.py` | Service coordination |
| `historical_source/spatial_social_code/` | Spatial/Social compilation, reasoning and refinement |
| `historical_source/MA-EgoQA/` | Shared video and prompt utilities |
| `historical_source/AutoToM/` | Model-interface utilities |

## Setup and use

Install the [test/video dependencies](../../docs/installation.md), then inspect the launcher:

```bash
python benchmarks/mmtom/matched_runner.py --help
```

This is the historical model-substitution runner. It requires the saved reference
predictions, replay audit files, original question/frame inputs and frozen service
configuration used by the run. These are not all in this source-only snapshot;
`--prepare` is not a dataset downloader. Its historical reasoning may depend on
question/options and conditional frame refinement. See the
[reproducibility boundaries](../../docs/reproducibility.md).
