# Habitat / EnactToM code structure

| Path | Purpose |
|---|---|
| `experiments/reevaluation_v3/` | Flash v2 experiment launcher and request guards |
| `experiments/habitat_closed_loop/runtime.py` | Online control and state interface |
| `experiments/habitat_closed_loop/state_store.py` | Private state storage |
| `experiments/habitat_closed_loop/updaters.py` | Observation, action and message updates |
| `experiments/habitat_closed_loop/strict_scorer.py` | Physical-goal scoring |
| `spatial_social_code/` | Reasoning and shared utilities |
| `external/EnactTom/` | EnactToM source and configuration subset |
| `tests/` | Snapshot tests |

The prose-readout adapter is located at `../../experiments/habitat_closed_loop/memory_views.py`.

## Setup and use

Inspect the launcher without starting the simulator:

```bash
python benchmarks/habitat/experiments/reevaluation_v3/run_qdd_habitat.py --help
```

Live execution needs a Linux/CUDA Habitat environment, the original tasks and
scene assets, a configured service, and the protocol's preflight/probe records.
The bundled upstream source subset does not include all simulator dependencies
or assets. See the root [installation guide](../../docs/installation.md).

The root `python run_tests.py` runs shared state/scoring tests and isolated
benchmark guard tests without model calls or physics rollouts. Preserve the
Direct/Static/Dynamic distinctions and original seed/scorer definitions when
setting up a real reproduction.
