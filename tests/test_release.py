"""Regression checks for the merged imports and source-only entry points."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def isolated(code):
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT)),
                   capture_output=True, text=True)


def test_historical_runner_uses_historical_runtime_and_shared_helpers():
    isolated('''
from pathlib import Path
from benchmarks.mmtom import matched_runner as runner
import runtime, evaluate_spatial_social, social_code
from spatial_social_code import runtime as current
assert runtime.execute_social is current.execute_social_legacy
assert 'benchmarks/mmtom/legacy' in evaluate_spatial_social.__file__
assert 'benchmarks/mmtom/legacy' in social_code.__file__
assert 'benchmarks/ma_egoqa' in runner.ma.__file__
assert 'query_evidence' in runtime.execute_social('M.query_evidence(0, "A", "answer", "x", 0.9, 0.9, "rule")')
try:
    current.execute_social('M.query_evidence(0, "A", "answer", "x", 0.9, 0.9, "rule")')
except current.UnsafeProgram:
    pass
else:
    raise AssertionError('Current runtime admitted legacy evidence')
assert 'spatial_social_code/runtime.py' in runner.source_hashes()
assert 'benchmarks/mmtom/legacy/runtime.py' in runner.source_hashes()
''')


def test_comind_uses_canonical_prompts_and_explicit_legacy_executor():
    isolated('''
from pathlib import Path
from benchmarks.comind.pipelines.compiler import comind_adapter as adapter
from spatial_social_code import runtime
assert adapter.execute_social is runtime.execute_social_legacy
assert Path(adapter.original.__file__).resolve() == Path('spatial_social_code/run_comind_codex_pilot.py').resolve()
assert set(adapter.PROMPTS) == {'direct', 'spatial', 'social', 'answer'}
''')


@pytest.mark.parametrize('benchmark', ['mmtom', 'ma-egoqa', 'comind', 'habitat', 'information-history'])
def test_benchmark_help_without_data_or_service(benchmark):
    result = subprocess.run([sys.executable, '-m', 'spatial_social_code', 'run', benchmark, '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_frozen_source_lists_resolve():
    from benchmarks.information_history.run import SOURCES
    assert all((ROOT / name).is_file() for name in SOURCES)
