"""Verify exported historical tables only; no model, dataset or network needed."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def paired(rows, left, right):
    a = sum(r[left] == r['answer'] and r[right] != r['answer'] for r in rows)
    b = sum(r[right] == r['answer'] and r[left] != r['answer'] for r in rows)
    n = a + b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2**n) if n else 1.0
    return {'left': left, 'right': right, 'left_only_correct': a,
            'right_only_correct': b, 'delta_right_minus_left': (b - a) / len(rows),
            'mcnemar_exact_p': p}


def summarize(rows, fields, comparisons):
    result = {'n': len(rows), 'unique_episodes': len({r['episode'] for r in rows}),
              'question_type_counts': dict(sorted(Counter(r['question_type'] for r in rows).items())),
              'conditions': {}, 'paired_comparisons': []}
    for field in fields:
        assert all(r[field] in ('a', 'b') for r in rows)
        correct = sum(r[field] == r['answer'] for r in rows)
        result['conditions'][field] = {'correct': correct, 'total': len(rows), 'accuracy': correct / len(rows)}
    result['paired_comparisons'] = [paired(rows, a, b) for a, b in comparisons]
    return result


QWEN_FIELDS = ('direct', 'spatial_only', 'social_only', 'raw_code', 'joint_runtime')
GEMINI_FIELDS = ('bare_spatial', 'bare_social', 'bare_joint', 'shared_scaffold',
                'shared_spatial', 'shared_social', 'shared_joint')
QWEN_PAIRS = (('spatial_only', 'joint_runtime'), ('social_only', 'joint_runtime'),
              ('raw_code', 'joint_runtime'), ('direct', 'joint_runtime'))
GEMINI_PAIRS = (('bare_spatial', 'bare_joint'), ('bare_social', 'bare_joint'),
                ('shared_spatial', 'shared_joint'), ('shared_social', 'shared_joint'))


def verify(root):
    summary = json.loads((root / 'summary.json').read_text())
    for name, digest in summary['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    q = read_rows(root / 'qwen70_predictions.csv')
    g = read_rows(root / 'gemini600_offline_predictions.csv')
    assert len(q) == len({r['index'] for r in q}) == 70
    assert len(g) == len({r['index'] for r in g}) == 600
    assert {int(r['index']) for r in g} == set(range(600))
    assert set(Counter(r['question_type'] for r in q).values()) == {10}
    assert len({r['question_type'] for r in q}) == 7
    assert all(r['answer'] in ('a', 'b') for r in q + g)
    assert summary['qwen70'] == summarize(q, QWEN_FIELDS, QWEN_PAIRS)
    assert summary['gemini600_offline'] == summarize(g, GEMINI_FIELDS, GEMINI_PAIRS)
    assert [summary['qwen70']['conditions'][c]['correct'] for c in QWEN_FIELDS] == [25, 27, 46, 42, 50]
    assert [summary['gemini600_offline']['conditions'][c]['correct'] for c in GEMINI_FIELDS] == [313, 298, 319, 418, 452, 415, 450]
    assert sum(int(r['social_query_evidence_present']) for r in q) == 40
    assert sum(int(r['joint_query_evidence_present']) for r in q) == 66
    assert all(r['image_metadata_matches_primary'] == '1' for r in q)
    return summary


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
    print('PASS: all70/600 unique predictions, coverage, accuracy, paired tests, protocol flags and checksums')
