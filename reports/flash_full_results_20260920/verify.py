"""Offline integrity and scoring check; Python standard library only, no API."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())
def rows(name):
    with gzip.open(ROOT / name, 'rt') as f:
        yield from (json.loads(line) for line in f)
def main():
    for name, expected in read('SHA256SUMS.json').items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    meta = read('mmtom/final_verified.json')
    scores = meta['records']
    flash = list(rows('mmtom/flash_results.jsonl.gz'))
    pro = list(rows('mmtom/pro_results.jsonl.gz'))
    assert [r['index'] for r in scores] == list(range(600))
    assert [r['index'] for r in flash] == [r['index'] for r in pro] == list(range(600))
    inputs = {r['index']: r for r in read('mmtom/original_frozen_inputs/inputs.json')}
    calls = collections.defaultdict(list)
    accepted = 0
    for call in rows('mmtom/calls.jsonl.gz'):
        calls[call['index']].append(call)
        q = call['files']['request.json']
        assert q['generationConfig']['maxOutputTokens'] == 16384
        if 'response.json' in call['files']:
            response = call['files']['response.json']
            assert response['modelVersion'] == 'gemini-3-flash-preview'
            assert all(c['finishReason'] == 'STOP' for c in response['candidates'])
            accepted += 1
    assert accepted == 2206
    for s, f, p in zip(scores, flash, pro):
        truth = str(inputs[s['index']]['row']['answer']).lower()
        assert truth == s['answer'] == str(p['answer']).lower()
        if s['index'] == 44:
            assert f == {'index': 44, 'error_type': 'GeminiServiceError', 'prediction': None}
            # The Direct answer already existed before SSC transport failure.
            reply = calls[44][0]['files']['response.json']
            text = ''.join(part.get('text', '') for part in reply['candidates'][0]['content']['parts'])
            assert text.strip().lower() == s['direct_prediction']
            assert s['direct_recovered_from_own_reply']
        else:
            assert f['direct_prediction'] == s['direct_prediction']
            assert f['prediction'] == s['ssc_prediction']
            assert f['direct_correct'] == s['direct_correct']
            assert f['correct'] == s['ssc_correct']
        assert s['direct_correct'] == (s['direct_prediction'] == truth)
        assert s['ssc_correct'] == (s['ssc_prediction'] == truth)
        assert p['direct_correct'] == (p['direct_prediction'] == truth)
        assert p['correct'] == (p['prediction'] == truth)
        assert s['pro_direct_correct'] == p['direct_correct']
        assert s['pro_ssc_correct'] == p['correct']
    assert [sum(r[k] for r in scores) for k in ['direct_correct','ssc_correct','pro_direct_correct','pro_ssc_correct']] == [375,439,406,470]
    audit = read('audit/independent_comparison.json')
    for name, data, key in [('flash', scores, 'ssc_correct'), ('pro', pro, 'correct')]:
        both = sum(r['direct_correct'] and r[key] for r in data)
        so = sum(not r['direct_correct'] and r[key] for r in data)
        do = sum(r['direct_correct'] and not r[key] for r in data)
        assert audit['paired'][name] == dict(both_correct=both, ssc_only=so, direct_only=do, both_wrong=600-both-so-do, net_gain=so-do)
        assert so-do == 64
    for field in ['spatial_raw', 'spatial_code', 'social_code']:
        same = sum(f[field] == p[field] for f,p in zip(flash,pro) if f['index'] != 44)
        assert same == audit['complete_code_identity'][field]['identical'] == 0
    for typ, expected in meta['by_type'].items():
        subset = [s for s in scores if s['type'] == typ]
        assert len(subset) == expected['n']
        for k in ['direct_correct','ssc_correct','pro_direct_correct','pro_ssc_correct']:
            assert sum(s[k] for s in subset) == expected[k]
    replay = read('audit/offline_replay.json')
    assert replay['n'] == replay['matched'] == 600 and not replay['mismatches']
    with (ROOT/'mmtom/scores.csv').open() as f:
        csvrows = list(csv.DictReader(f))
    assert len(csvrows) == 600
    for s, row in zip(scores,csvrows):
        assert all(row[k] == ('' if v is None else str(v)) for k,v in s.items())
    habitat = collections.defaultdict(list)
    seen = set()
    for r in rows('habitat/rollouts.jsonl.gz'):
        key = (r['task'],r['condition'],r['seed'])
        assert key not in seen
        seen.add(key)
        physical = r['files']['summary.json']['historical_benchmark_evaluation']['strict_physical']
        assert physical['measurement_valid']
        habitat[r['condition']].append(physical)
    assert len(seen) == 108
    assert {r[2] for r in seen} == {891011,891012,891013}
    assert len({r[0] for r in seen}) == 12
    assert seen == {(task,cond,seed) for task in {r[0] for r in seen} for cond in ['direct','static_ssc','dynamic_ssc'] for seed in [891011,891012,891013]}
    summary = read('habitat/published_summary/summary.json')
    for cond, expected in [('direct',8),('static_ssc',10),('dynamic_ssc',24)]:
        data = habitat[cond]
        assert len(data) == 36 and sum(r['strict_success'] for r in data) == expected
        completion = sum(r['strict_percent_complete'] for r in data)/36
        assert abs(completion-summary['results'][cond]['mean_completion']) < 1e-12
    assert {(r['task'],r['condition'],r['seed']) for r in rows('habitat/scorer_only/checks.jsonl.gz')} == seen
    print('PASS: hashes; MMToM 600 unique paired items, Flash375/439, Pro406/470; 2206 STOP responses; Habitat108 rollouts, Direct8/36, Static10/36, Dynamic24/36.')

if __name__ == '__main__':
    main()
