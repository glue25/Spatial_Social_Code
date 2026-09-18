"""Offline arithmetic/coverage check; no datasets, credentials or API calls.

This checks the exported metrics, not the underlying simulator trajectories.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    summary = json.loads((root / 'summary.json').read_text())
    csv_path = root / 'rollouts.csv'
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == summary['sha256']['rollouts.csv']
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary['total_rollouts'] == 108
    expected_conditions = {'direct', 'static_ssc', 'dynamic_ssc'}
    assert set(summary['conditions']) == expected_conditions
    assert {r['condition'] for r in rows} == expected_conditions
    assert {int(r['seed']) for r in rows} == set(summary['seeds'])
    assert len({r['task_id'] for r in rows}) == summary['unique_tasks'] == 12
    keys = {(r['task_id'], int(r['seed']), r['condition']) for r in rows}
    assert len(keys) == len(rows), 'duplicate task/seed/condition'
    for row in rows:
        assert row['strict_success'] in ('0', '1')
        completion = float(row['strict_percent_complete'])
        assert math.isfinite(completion) and 0 <= completion <= 1
        for field in ('logged_api_calls', 'input_tokens', 'output_tokens'):
            assert int(row[field]) >= 0
    paired = None
    for condition in summary['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        pairs = {(r['task_id'], int(r['seed'])) for r in selected}
        assert len(pairs) == 36
        if paired is not None:
            assert pairs == paired
        paired = pairs
        expected = summary['results'][condition]
        successes = sum(int(r['strict_success']) for r in selected)
        assert expected['n'] == len(selected) and expected['successes'] == successes
        assert math.isclose(expected['success_rate'], successes / len(selected), abs_tol=1e-12)
        completion = sum(float(r['strict_percent_complete']) for r in selected) / len(selected)
        assert math.isclose(expected['mean_completion'], completion, abs_tol=1e-12)
        for field in ('logged_api_calls', 'input_tokens', 'output_tokens'):
            assert sum(int(r[field]) for r in selected) == expected[field]
        for seed in summary['seeds']:
            subset = [r for r in selected if int(r['seed']) == seed]
            assert len(subset) == 12
            assert sum(int(r['strict_success']) for r in subset) == summary['successes_by_seed'][condition][str(seed)]
    return summary


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
    print('PASS: 108 unique rollouts, matched task/seed coverage, metrics, usage totals and CSV hash')
