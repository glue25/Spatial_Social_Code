"""Factorized records, oracle replay and four information-history interventions.

Source tasks provide entity vocabularies only. Histories are constructed symbolic
episodes, not the original golden trajectory, real visibility or rendered video.
The existing StateStore holds acquisition records; its historical planner render
is NOT silently replaced or claimed to be the same as this diagnostic adapter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re

from benchmarks.habitat.state_store import Fact, StateStore

CONDITIONS = ('ssc', 'prose', 'no_links', 'no_attribution')
HISTORIES = ('witness', 'unseen', 'delivered', 'undelivered')
CONTRASTS = (('witness', 'unseen'), ('delivered', 'unseen'),
             ('delivered', 'undelivered'), ('witness', 'undelivered'))
SALT = 'ssc-information-history-20260923-v1'


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def entities(task, kind):
    prefix = task['problem_pddl'].split('(:init')[0]
    return ' '.join(re.findall(r'([^()\n]+) - '+kind+r'\s*$', prefix, re.M)).split()


def make_base(task, split, index):
    """No outcome/annotation fields enter construction or selection."""
    seed = int(digest([SALT, task['task_id']])[:16], 16)
    rng = random.Random(seed)
    agents = sorted(entities(task, 'agent')); rng.shuffle(agents)
    objects = sorted(entities(task, 'object')); rng.shuffle(objects)
    places = sorted(entities(task, 'furniture')); rng.shuffle(places)
    assert len(agents) >= 3 and len(objects) >= 2 and len(places) >= 3
    # Role/entity assignments are fixed before any request; names carry no role.
    return {'base_id': f'{split}_{index:03d}', 'split': split, 'seed': seed,
            'task_id': task['task_id'], 'scene_id': str(task['scene_id']),
            'episode_id': str(task['episode_id']), 'target': agents[0],
            'source_agent': agents[1], 'unrelated': agents[2],
            'agents': agents[:3], 'object': objects[0], 'distractor': objects[1],
            'old': places[0], 'new': places[1], 'third': places[2],
            'locations': sorted(places[:3]),
            'intermediate_move': bool(index % 2),
            'provenance': 'New symbolic history; source task vocabulary only. '
                          'No actual Habitat physics, camera visibility or original private secrets.'}


def build(base, history):
    assert history in HISTORIES
    b = base
    # IDs depend on base and event identity, NEVER on condition or history.
    def ident(namespace, label):
        return namespace + digest([b['seed'], namespace, label])[:12]
    events = []
    def physical(label, turn, obj, loc, kind='location', support=None):
        record = {'id': ident('p', label), 'turn': turn, 'kind': kind,
                  'object': obj, 'location': loc}
        if support is not None:
            record['support_event'] = support
        events.append(record)
        return record['id']
    p0 = physical('initial_target', 0, b['object'], b['old'])
    d0 = physical('initial_distractor', 0, b['distractor'], b['third'])
    # A fixed extra physical update in half the bases increases temporal variety.
    if b['intermediate_move']:
        physical('intermediate_target', 1, b['object'], b['third'])
        physical('intermediate_distractor', 1, b['distractor'], b['new'])
    p1 = physical('final_target', 2, b['object'], b['new'])
    d1 = physical('final_distractor', 2, b['distractor'], b['old'])
    # Both report contents always exist, with identical timestamps. Routing is
    # represented ONLY by attributed access records, not leaked in report text.
    pm = physical('target_report', 3, b['object'], b['new'], 'report', p1)
    dm = physical('distractor_report', 3, b['distractor'], b['old'], 'report', d1)
    store = StateStore(b['agents']); access = []
    def acquire(label, holder, source, turn, mode, event, delivered=True):
        sid = ident('s', label)
        entry = {'id': sid, 'agent': holder, 'source_agent': source,
                 'turn': turn, 'mode': mode, 'available': delivered,
                 'physical_id': event}
        access.append(entry)
        if not delivered:
            # The intended receiver never acquires the failed packet. Keep this
            # only in the external audit ledger, not its private StateStore.
            return
        fact = Fact(key=sid, value={'mode': mode, 'available': delivered},
                    source_type='delivered_message' if mode == 'message' else 'observation',
                    source_agent=source, evidence=event, turn=turn,
                    status=('reported' if delivered else 'not_delivered')
                    if mode == 'message' else 'observed')
        # Failed deliveries are retained as audit records, NOT accepted evidence.
        store.update(holder, 'social', [fact])
    for a in b['agents']:
        acquire(f'initial_{a}_target', a, a, 0, 'vision', p0)
        acquire(f'initial_{a}_distractor', a, a, 0, 'vision', d0)
    for event, label in ((p1, 'target'), (d1, 'distractor')):
        acquire('source_final_'+label, b['source_agent'], b['source_agent'], 2, 'vision', event)
    acquire('target_final', b['target'], b['target'], 2, 'vision',
            p1 if history == 'witness' else d1)
    acquire('unrelated_final', b['unrelated'], b['unrelated'], 2, 'vision', d1)
    acquire('target_message', b['target'], b['source_agent'], 3, 'message',
            pm if history == 'delivered' else dm)
    if history == 'undelivered':
        acquire('failed_message', b['target'], b['source_agent'], 3, 'message', pm, False)
    snapshots = {a: store.snapshot(a) for a in sorted(b['agents'])}
    social = []; links = []
    for agent, snapshot in snapshots.items():
        for key, fact in snapshot['social'].items():
            social.append({'id': key, 'agent': agent, 'source_agent': fact['source_agent'],
                           'turn': fact['turn'], **fact['value'], 'status': fact['status']})
            links.append({'social_id': key, 'physical_id': fact['evidence']})
    for attempt in access:
        if not attempt['available']:
            social.append({k: attempt[k] for k in
                           ('id', 'agent', 'source_agent', 'turn', 'mode', 'available')})
            social[-1]['status'] = 'not_delivered'
            links.append({'social_id': attempt['id'], 'physical_id': attempt['physical_id']})
    # No grouping/order channel for agent attribution; IDs are opaque.
    social.sort(key=lambda r: digest([b['seed'], 'social_order', r['id']]))
    events.sort(key=lambda r: digest([b['seed'], 'physical_order', r['id']]))
    links.sort(key=lambda r: digest([b['seed'], 'link_order', r['social_id']]))
    view = {'physical_events': events, 'social_records': social, 'evidence_links': links}
    query = {'object': b['object'], 'target_agent': b['target'],
             'unrelated_agent': b['unrelated'], 'locations': b['locations']}
    truth = oracle(events, access, query)
    expected = b['new'] if history in ('witness', 'delivered') else b['old']
    assert truth['world_location'] == b['new']
    assert truth['target_location'] == expected
    assert truth['unrelated_location'] == b['old']
    return {'base_id': b['base_id'], 'history': history, 'query': query, 'view': view,
            'oracle_access_log': access, 'gold': truth, 'state_store_snapshots': snapshots,
            'world_terminal': {b['object']: b['new'], b['distractor']: b['old']}}


def oracle(events, access, query):
    """Independent event replay. Gold is never copied from a model/compiled belief."""
    by_id = {e['id']: e for e in events}
    world = {}
    for e in sorted(events, key=lambda e: (e['turn'], e['id'])):
        if e['kind'] == 'location':
            world[e['object']] = e['location']
    private = {}; supporting = {}
    for a in sorted(access, key=lambda a: (a['turn'], a['id'])):
        if not a['available']:
            continue
        e = by_id[a['physical_id']]
        # Reports have world-supported, veridical content by construction; this
        # is a declared closed-world assumption, not a theory of human belief.
        if e['kind'] == 'report':
            s = by_id[e['support_event']]
            assert (s['object'], s['location']) == (e['object'], e['location'])
        key = (a['agent'], e['object'])
        private[key] = e['location']; supporting[key] = e['id']
    obj = query['object']; target = (query['target_agent'], obj)
    other = (query['unrelated_agent'], obj)
    return {'world_location': world[obj], 'target_location': private.get(target, 'unknown'),
            'unrelated_location': private.get(other, 'unknown'),
            'target_evidence': supporting.get(target, 'unknown')}


def transform(view, condition):
    assert condition in CONDITIONS
    out = copy.deepcopy(view)
    if condition == 'no_links':
        out['evidence_links'] = []
    if condition == 'no_attribution':
        for r in out['social_records']:
            r['agent'] = None; r['source_agent'] = None
    return out


def prose(view):
    """Lossless deterministic field-wise serialization, with a round-trip check."""
    lines = []
    for section, rows in view.items():
        lines.append('Section '+section+'.')
        for r in rows:
            lines.append('A record has ' + '; '.join(k+' equal to '+canonical(v)
                                                  for k, v in r.items()) + '.')
    return '\n'.join(lines)


def unprose(text):
    view = {}; section = None
    for line in text.splitlines():
        if line.startswith('Section '):
            section = line[8:-1]; view[section] = []
        else:
            r = {}
            for field in line[len('A record has '):-1].split('; '):
                k, v = field.split(' equal to ', 1); r[k] = json.loads(v)
            view[section].append(r)
    return view


INSTRUCTIONS = '''This is a constructed, closed-world text-record diagnostic, not video.
Act as an external evaluator of separately maintained agent information, not as an agent
allowed to use everyone's private knowledge. Infer the final actual object location and
each queried agent's last supported location belief from the supplied records.
Physical location events change the world; report events only describe content. Each
social record is an experience. An evidence link says WHICH physical event it concerns.
Vision available=true is a confirmed observation (not merely possible visibility).
Messages update only the receiving agent when available=true; attempted/undelivered
messages do not. In this constructed world delivered reports are truthful and trusted.
An agent retains its earlier belief unless it receives relevant new evidence; absence
of updates does not reveal an unseen move. Event time, not list order, controls recency.
Record IDs are opaque. Null attribution and absent links really mean that information
is not supplied; do not infer it from ordering or invent a mapping. Answer unknown if
the remaining input does not determine a field. No world-location inference may be
substituted for a private belief. All physical facts are identical across conditions.
Return exactly one JSON object with these four string fields and no commentary:
{"world_location":"location ID or unknown", "target_location":"location ID or unknown",
 "unrelated_location":"location ID or unknown", "target_evidence":"physical event ID or unknown"}.
target_evidence must be the most recent acquired physical event about the target object
supporting the target agent's answer (the report event itself for a delivered report).
'''


def prompt(case, condition):
    view = transform(case['view'], condition)
    body = prose(view) if condition == 'prose' else canonical(view)
    return INSTRUCTIONS+'\nRecords:\n'+body+'\nQuery:\n'+canonical(case['query'])


def score(prediction, case, valid=True):
    fields = ('world_location', 'target_location', 'unrelated_location', 'target_evidence')
    result = {k: bool(valid and prediction.get(k) == case['gold'][k]) for k in fields}
    result['joint'] = all(result[k] for k in fields[:3])
    # Exact evidence AND semantic answer must match the independently replayed log.
    result['supported_target'] = result['target_location'] and result['target_evidence']
    return result


def audit(base, cases):
    assert {c['history'] for c in cases} == set(HISTORIES)
    assert len({digest(c['world_terminal']) for c in cases}) == 1
    assert len({digest(c['view']['physical_events']) for c in cases}) == 1
    for c in cases:
        v = c['view']; assert unprose(prose(v)) == v
        no = transform(v, 'no_links')
        assert no['physical_events'] == v['physical_events']
        assert no['social_records'] == v['social_records'] and not no['evidence_links']
        at = transform(v, 'no_attribution')
        assert at['physical_events'] == v['physical_events']
        assert at['evidence_links'] == v['evidence_links']
        for x, y in zip(v['social_records'], at['social_records']):
            assert {k: z for k, z in x.items() if k not in ('agent', 'source_agent')} == {
                k: z for k, z in y.items() if k not in ('agent', 'source_agent')}
        serialized = canonical(at)
        assert all(a not in serialized for a in base['agents'])
        assert oracle(v['physical_events'], c['oracle_access_log'], c['query']) == c['gold']
    # Three distinct histories become byte-identical after link removal. This
    # proves non-identifiability, not SSC superiority at equal information.
    lost = [transform(c['view'], 'no_links') for c in cases if c['history'] != 'undelivered']
    assert all(x == lost[0] for x in lost)
    return True
