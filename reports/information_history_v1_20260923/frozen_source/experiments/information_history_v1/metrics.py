"""Paired base-episode clustered estimates; no success-only filtering."""
import itertools
from .protocol import CONDITIONS, HISTORIES, CONTRASTS


def per_base(rows):
    # Missing/invalid answers fail every correctness/invariance metric. Pending
    # rows are distinguished in coverage, never silently dropped from denominators.
    by = {(r['history'], r['condition']): r for r in rows}
    output = {}
    for condition in CONDITIONS:
        rr = {h: by.get((h, condition), {}) for h in HISTORIES}
        def correct(h, field):
            return bool(rr[h].get('scores', {}).get(field, False))
        def equal(a, b, field):
            return bool(rr[a].get('valid') and rr[b].get('valid') and
                        rr[a]['prediction'][field] == rr[b]['prediction'][field])
        pairs = list(itertools.combinations(HISTORIES, 2))
        n = len(HISTORIES)
        output[condition] = {
            'world_accuracy': sum(correct(h, 'world_location') for h in HISTORIES)/n,
            'target_accuracy': sum(correct(h, 'target_location') for h in HISTORIES)/n,
            'unrelated_accuracy': sum(correct(h, 'unrelated_location') for h in HISTORIES)/n,
            'joint_accuracy': sum(correct(h, 'joint') for h in HISTORIES)/n,
            'supported_target_accuracy': sum(correct(h, 'supported_target') for h in HISTORIES)/n,
            'world_pair_invariance': sum(equal(a,b,'world_location') for a,b in pairs)/len(pairs),
            'world_pair_invariant_correct': sum(equal(a,b,'world_location') and
                correct(a,'world_location') for a,b in pairs)/len(pairs),
            'target_correct_change': sum(correct(a,'target_location') and correct(b,'target_location')
                and not equal(a,b,'target_location') for a,b in CONTRASTS)/len(CONTRASTS),
            'unrelated_pair_invariance': sum(equal(a,b,'unrelated_location') for a,b in pairs)/len(pairs),
            'unrelated_pair_invariant_correct': sum(equal(a,b,'unrelated_location') and
                correct(a,'unrelated_location') for a,b in pairs)/len(pairs),
            'paired_joint_correct': sum(correct(a,'joint') and correct(b,'joint')
                for a,b in CONTRASTS)/len(CONTRASTS),
            'all_four_joint_correct': float(all(correct(h,'joint') for h in HISTORIES)),
        }
    return output


def summarize(bases, rows, bootstrap=10000):
    import numpy as np
    ids = [b['base_id'] for b in bases]
    metrics = [per_base([r for r in rows if r['base_id'] == bid]) for bid in ids]
    rng = np.random.default_rng(20260923)
    selections = rng.integers(0, len(ids), size=(bootstrap, len(ids)))
    result = {'planned_base_episodes': len(ids), 'planned_final_calls': len(ids)*16,
              'completed_final_calls': len(rows), 'valid_final_calls': sum(r['valid'] for r in rows),
              'pending_final_calls': len(ids)*16-len(rows), 'conditions': {}, 'paired_differences': {},
              'bootstrap': {'unit': 'base_episode', 'draws': bootstrap, 'seed': 20260923},
              'per_base': {bid:m for bid,m in zip(ids,metrics)},
              'interpretation': 'Constructed text-state diagnostic; not visual compilation or '
                  'Habitat physics. Link/attribution deletion removes information. Prose is '
                  'lossless serialization, not independently updated natural-language memory.'}
    for c in CONDITIONS:
        result['conditions'][c] = {}
        for metric in metrics[0][c]:
            values = np.array([m[c][metric] for m in metrics])
            lo, hi = np.quantile(values[selections].mean(axis=1), [.025, .975])
            result['conditions'][c][metric] = {'mean':float(values.mean()), 'ci95':[float(lo),float(hi)]}
    # Two co-primary comparisons, Bonferroni 97.5% marginal intervals; no promised significance.
    for other in ('no_links','no_attribution','prose'):
        primary = other != 'prose'; alpha = .025 if primary else .05
        difference = np.array([m['ssc']['paired_joint_correct']-m[other]['paired_joint_correct'] for m in metrics])
        lo,hi = np.quantile(difference[selections].mean(axis=1), [alpha/2,1-alpha/2])
        result['paired_differences']['ssc_minus_'+other] = {
            'metric':'paired_joint_correct', 'mean':float(difference.mean()),
            'ci':[float(lo),float(hi)], 'confidence':1-alpha, 'co_primary':primary}
    return result
