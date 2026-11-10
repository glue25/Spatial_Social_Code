# Information-history code structure

| Path | Purpose |
|---|---|
| `experiments/information_history_v1/protocol.py` | Histories, interventions, prompts and scoring targets |
| `experiments/information_history_v1/run.py` | Experiment preparation and inference |
| `experiments/information_history_v1/metrics.py` | Evaluation metrics |
| `experiments/habitat_closed_loop/state_store.py` | State storage dependency |
| `experiments/reevaluation_v3/service_gate.py` | Service coordination |
| `tests/` | Diagnostic tests |

## Setup and use

Run the module from its own source root so relative imports resolve:

```bash
cd benchmarks/information_history
python -m experiments.information_history_v1.run --help
```

The actions are `prepare`, `run`, `probe`, `summarize` and `revise-scoring`;
`--output` is required for an actual action. Preparation needs the source task
vocabulary and protocol inputs. New inference needs a configured model service.

The root `python run_tests.py` includes isolated tests for this snapshot. This
is a constructed symbolic information-history diagnostic, not a new Habitat
physics rollout or visual counterfactual evaluation.
