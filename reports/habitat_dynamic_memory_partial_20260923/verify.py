"""Offline table/coverage verification only; no network or simulator required."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def verify(root):
    summary = json.loads((root / 'summary.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in summary['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    with (root / 'rollouts.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    with (root / 'schedule.csv').open(newline='') as f:
        schedule = list(csv.DictReader(f))
    assert len(rows) == summary['completed_rollouts'] == 24
    assert len(schedule) == summary['planned_rollouts'] == 72
    assert len({r['job_index'] for r in schedule}) == 72
    keys = {(r['task_id'], int(r['seed']), r['condition']) for r in schedule}
    tasks = {r['task_id'] for r in schedule}
    assert len(tasks) == protocol['tasks'] == 12
    assert keys == {(t, s, c) for t in tasks for s in protocol['seeds']
                    for c in protocol['conditions']}
    states = Counter(r['status'] for r in schedule)
    assert states == {'completed': 24, 'aborted': 1, 'not_started': 47}
    assert summary['aborted_rollouts'] == states['aborted']
    assert summary['not_started_rollouts'] == states['not_started']
    assert {r['job_index'] for r in rows} == {r['job_index'] for r in schedule if r['status'] == 'completed'}
    assert len({r['job_index'] for r in rows}) == 24
    assert {int(r['seed']) for r in rows} == set(summary['reported_seeds']) == {891011}
    by_index = {r['job_index']: r for r in schedule}
    pairs = defaultdict(dict)
    costs = ('request_count', 'responses_received', 'transport_attempts',
             'failed_transport_attempts', 'prompt_tokens', 'completion_tokens', 'total_tokens')
    for r in rows:
        assert all(r[k] == by_index[r['job_index']][k] for k in ('task_id', 'seed', 'condition'))
        assert r['measurement_valid'] == '1'
        assert r['strict_success'] in ('0', '1') and r['service_affected'] in ('0', '1')
        goals, total = int(r['completed_goals']), int(r['total_goals'])
        assert 0 <= goals <= total and total > 0
        assert math.isclose(float(r['strict_percent_complete']), goals / total)
        assert int(r['strict_success']) == int(goals == total)
        assert all(int(r[k]) >= 0 for k in costs)
        assert int(r['responses_received']) <= int(r['request_count']) <= int(r['transport_attempts'])
        pairs[r['task_id']][r['condition']] = r
    assert len(pairs) == summary['paired_tasks'] == 12
    assert all(set(p) == set(protocol['conditions']) for p in pairs.values())
    for c, expected in summary['results'].items():
        rr = [r for r in rows if r['condition'] == c]
        assert len(rr) == expected['n'] == 12
        successes = sum(int(r['strict_success']) for r in rr)
        assert successes == expected['successes']
        assert math.isclose(successes / len(rr), expected['success_rate'])
        assert math.isclose(sum(float(r['strict_percent_complete']) for r in rr) / len(rr), expected['mean_completion'])
        assert sum(int(r['service_affected']) for r in rr) == expected['service_affected']
        assert all(sum(int(r[k]) for r in rr) == expected[k] for k in costs)
    counts = Counter()
    for p in pairs.values():
        a, b = int(p['dynamic_ssc']['strict_success']), int(p['nl_dynamic_v2']['strict_success'])
        counts['both_success' if a and b else 'ssc_only_success' if a else 'nl_only_success' if b else 'both_fail'] += 1
    assert dict(counts) == summary['paired_success_outcomes']
    for metric, delta in summary['delta_ssc_minus_nl'].items():
        assert math.isclose(delta, summary['results']['dynamic_ssc'][metric] - summary['results']['nl_dynamic_v2'][metric])
    interrupted = json.loads((root / 'interrupted_rollout.json').read_text())
    assert by_index[str(interrupted['job_index'])]['status'] == 'aborted'
    assert interrupted['seed'] == 891012
    assert interrupted['invalid_responses'][0]['choices'][0]['content_characters'] == 0
    return summary


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
    print('PASS: 24 paired first-seed results; 72-job ledger; failures, metrics, costs and checksums')
