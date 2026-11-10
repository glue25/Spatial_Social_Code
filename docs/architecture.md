# Code architecture

## Shared runtime

`spatial_social_code/runtime.py` defines Spatial and Social memories, the allowed
record calls and the AST-based execution path. `execute_spatial()` and
`execute_social()` produce JSON-line traces. Supplying `valid_evidence_ids` and
`require_evidence=True` checks Social references against the supplied Spatial IDs.
`social_code.py` provides compiler and readout utilities.

The [minimal example](../examples/minimal.py) demonstrates these interfaces with
hand-written records. Its belief is supplied by the example, not inferred by the
runtime. Structure and reference checks are not a semantic truth checker.

`execute_social_legacy()` exists for historical replay. Keep historical
answer-conditioned machinery distinguishable from current record interfaces.

## Benchmark-specific source snapshots

Each directory under `benchmarks/` groups the source used by that pipeline.
Some snapshots include a local `spatial_social_code/`, `experiments/` or model
transport dependency. These are intentionally separate versions, not accidental
copies to deduplicate without analysis.

`run_tests.py` starts benchmark-local tests in independent subprocesses with
explicit import roots. This avoids one snapshot importing an unrelated module
with the same name from another benchmark. Preserve that separation when adding
new tests or launch commands.

## Shared experiments and dependencies

Root-level `experiments/` contains controls, state interfaces and analysis code
shared by the submitted package. `third_party/` preserves upstream interfaces
and notices; benchmark-local third-party integrations stay beside their runner.

Public APIs are currently source-oriented. A unified CLI and fully portable live
run configuration remain [roadmap items](../TODO.md).
