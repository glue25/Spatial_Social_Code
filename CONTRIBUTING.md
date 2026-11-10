# Contributing

## Development checks

```bash
python -m pip install -e '.[test]'
python -m examples.minimal
python run_tests.py
```

Use a branch and keep changes focused. Explain the problem, changed behavior,
validation and any effect on historical results. For changes to benchmark
snapshots, identify the source version and add a focused regression test when
behavior changes. Do not silently consolidate duplicated snapshots.

Before proposing a release change, review the added files, run
`python scripts/update_checksums.py`, then `python check_code.py`.

## Reporting a problem

Include the command, Python/OS version, commit, benchmark and a small sanitized
traceback. State whether you are running the synthetic example, offline tests
or live inference. Do not attach credentials, private service responses or
restricted benchmark media. Use the bug report or reproduction issue template.

For new experiments, describe the information and budget matching, scorer,
invalid-output policy and split. Keep speculative claims separate from measured
results. See [the roadmap](TODO.md) for useful starting points.

## Licensing

Submit original contributions under the project's MIT terms. Preserve upstream
notices for third-party modifications and identify their source. Dataset/model
access remains governed by the respective provider.
