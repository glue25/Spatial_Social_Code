# Spatial Social Code

Spatial Social Code studies whether a small language model can reason more reliably from an explicit, executable representation of a multi-agent world.

```text
environment annotations / canonical facts
                ↓
deterministic Spatial–Social compiler
                ↓
physical state + agent-indexed information state
                ↓
              Qwen3-4B
                ↓
belief, goal, coordination and counterfactual reasoning
```

The main model is **Qwen3-4B**. Qwen3-VL and other VLMs are external raw-video baselines, not part of the method pipeline. The first stage uses oracle or semi-oracle annotations so that representation quality can be separated from perception errors.

## Current scope

- **MMToM-QA** — primary code-understanding and belief/goal QA benchmark.
- **MA-EgoQA** — held-out evaluation for long-horizon, multi-agent evidence aggregation.
- **EnactToM** — functional ToM transfer through messages and actions.
- **HoloAssist, CoMind, EgoTaskQA** — later perception and task-understanding extensions.

The implemented E0 experiment compiles all 300 public EnactToM tasks into query-independent oracle state, replays golden trajectories, and evaluates message deletion/redirection interventions. It is a representation and reducer test, not learned-agent performance.

## Quick start

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
python3 scripts/validate_datasets.py
python3 scripts/run_enacttom_e0.py
```

Dataset metadata can be prepared with:

```bash
bash scripts/prepare_datasets.sh
```

Large videos, Habitat/HSSD assets, third-party repositories and downloaded datasets stay local and are excluded from Git.

## Repository structure

```text
spatial_social_code/   EnactToM compiler, reducer and interventions
scripts/               dataset preparation, validation and E0 entrypoint
tests/                 deterministic unit and public-task replay tests
experiments/configs/   frozen dataset sources and pilot identifiers
reports/               concise, auditable experiment results
docs/                  method, benchmark landscape and experiment plan
```

## Documents

- [Method formulation](docs/method_zh.md)
- [Benchmark models and SOTA](docs/benchmark_sota_zh.md)
- [Training and experiment plan](docs/experiment_plan_zh.md)
- [Dataset status](docs/dataset_status_zh.md)
- [E0 progress report](docs/progress_2026-07-31.md)

## Result boundary

Early prompt-only Social Code experiments did not beat direct answering and have been removed from the active codebase. The retained E0 result establishes that the explicit state representation is executable and supports local counterfactual interventions. The next scientific test is a controlled comparison using the same Qwen3-4B and token budget:

```text
direct text  vs.  matched free-form memory  vs.  Spatial–Social Code
```
