# Habitat closed-loop SSC

This directory implements the versioned strict scorer, the shared action
serializer, per-agent online state, runner hooks, and auditable D/T/C regression.

Environment and model preflight:

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/check_environment.py \
  --output workdirs/habitat_closed_loop/environment_check/report.json
```

One-task real D/T/C smoke (three rollouts):

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/run_regression.py \
  --mode smoke --output workdirs/habitat_closed_loop/smoke
```

After smoke artifact validation, the 36-rollout development regression is:

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/run_regression.py \
  --mode dev36 --output workdirs/habitat_closed_loop/dev36
```

Both commands require the fixed `qwen3-vl-8b-instruct` model to be served at
`http://127.0.0.1:8891/v1`. The CLI's `--max-llm-calls 20` is a historical
misnomer: in a single-task run it sets 20 environment turns and each agent's
replanning threshold, not 20 aggregate API requests. The artifacts therefore
record planner calls per agent and merge exact provider usage during analysis.

Do not use `--allow-missing-preflight` for a reported run. It exists only for
testing the subprocess failure path.

Validate every real run package before aggregation:

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/validate_run.py \
  /path/to/closed_loop_artifacts
```

After exactly 36 accepted packages exist, generate the editable English CSV,
LaTeX table, and PNG/PDF/SVG plot:

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/analyze_dev36.py \
  workdirs/habitat_closed_loop/dev36 \
  --output workdirs/habitat_closed_loop/dev36/analysis
```

The historical 1566/772 regression boundary is reproduced with:

```bash
.venv-habitat/bin/python experiments/habitat_closed_loop/audit_historical_cases.py \
  --output workdirs/habitat_closed_loop/historical_case_audit.json
```

This audit deliberately reports the old single-snapshot stability field as
`not_measured`; it is not counted as a new rollout.

The already prepared, answer-blind 100-episode MMToM annotation packet is at
`workdirs/priority_ssc_v1/human_audit/`. Its manifest says
`awaiting_two_independent_reviewers`; the adjudication sheet is still blank.
