"""Verify published casebook without original datasets, APIs or simulator.

Checks trace consistency, provenance/export hashes, evidence references, scores,
and exact SAM proposal-selection replay. Does not certify generated semantics.
"""
import importlib.util
import math
from pathlib import Path
import re

from experiments.build_cross_environment_casebook import OUT, read, sha
from experiments.build_habitat_update_case import validate as validate_inspection


def near(a,b):
    if a is None or b is None:return a is b
    return len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-6) for x,y in zip(a,b))


def sam_replay(case,c):
    arm=case['result']['conditions'][c]
    base=case['result']['conditions'][c.removesuffix('_sam')]['prediction']
    task=case['input']['task']
    field='action_noun_l1' if task=='scoia' else 'category_l1'
    assert arm['identity']==str(base.get(field) or '').strip().replace('_',' ') or (
        not base.get(field) and arm['identity']=='task relevant object')
    views=arm['sam_proposals'];best={r:(v[0] if v else None) for r,v in views.items()}
    for v in views.values():assert all(v[i]['score']>=v[i+1]['score'] for i in range(len(v)-1))
    def box(role):
        if role is None or best.get(role) is None:return None
        x1,y1,x2,y2=best[role]['box_xyxy']
        return [x1/2+(704 if role=='helper' else 0),y1/2,(x2-x1)/2,(y2-y1)/2]
    if task=='joint_attention':expected={'left_bbox':box('leader'),'right_bbox':box('helper')}
    else:
        ranked=[(v['score'],r) for r,v in best.items() if v]
        chosen=max(ranked)[1] if ranked else None
        assert chosen==arm['selected_view']
        expected={'bbox':box(chosen)}
    for field,b in expected.items():assert near(b,arm['prediction'][field])
    assert {k:v for k,v in base.items() if k not in expected}=={
        k:v for k,v in arm['prediction'].items() if k not in expected}


def verify(root=OUT):
    index=read(root/'index_all.json');assert len(index)==13
    counts={env:sum(x['environment']==env for x in index) for env in ['Habitat','MMToM','MA-EgoQA','CoMind']}
    assert counts=={'Habitat':3,'MMToM':3,'MA-EgoQA':4,'CoMind':3}
    ledger=read(root/'selection_ledger_all.json');assert len(ledger)==1000
    spec=importlib.util.spec_from_file_location('published_casebook_scorer',root/'code/comind_frozen_scorer.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    for item in index:
        folder=root/item['folder']
        if item['id']=='habitat_588':
            d=read(folder/'evidence_chain.json');assert validate_inspection(d)
            for f in d['frames']:assert sha(folder/f['file'])==f['sha256']
            continue
        d=read(folder/'case.json');assert d['id']==item['id']
        for m in d.get('media',[]):
            if 'export' in m:
                assert sha(root/m['export'])==m['sha256']==m['export_sha256']
        if d['environment']=='Habitat':
            e=d['evidence'];msg=e['message']['record'];update=e['update']['record'];inp=e['next_input']['record'];plan=e['next_plan']['record']
            assert msg['turn']+1==plan['turn']
            newkeys=set(update['after']['social'])-set(update['before']['social'])
            assert newkeys and any(k in inp['context'] for k in newkeys)
            assert plan['action']==e['feedback']['record']['action']['serialized']
            assert e['feedback']['record']['outcome']['success']
            assert e['actual_request']['exact_state_present']
        elif d['environment']=='MMToM':
            r=d['result'];gold=d['input']['row']['answer']
            assert r['answer']==gold
            assert r['direct_correct']==(r['direct_prediction']==gold)
            assert r['correct']==(r['prediction']==gold)
            assert re.findall(r'(?im)^\s*([AB])[.\s]*$',r['direct_raw'])[-1].lower()==r['direct_prediction']
            assert len(d['media'])==8 and not r['uncertainty_triggered']
            for layer in ['spatial','social']:assert (folder/f'{layer}_code.py').read_text()==r[f'{layer}_code']+'\n'
        elif d['environment']=='MA-EgoQA':
            assert d['input']['row']['answer']==d['input']['row']['options'][0]
            compiled=d['compiled'];g=compiled['graph'];ids=set(compiled['valid_evidence_ids'])
            for edge in g['evidence_links']:
                assert edge['evidence_id'] in ids
                assert edge['record'] in {r['id'] for r in g['social_records']}
            for eid in d['cited_evidence']:assert eid in ids
            for f in d['preview_frames']:assert sha(folder/f['preview_file'])==f['preview_sha256']
            requests=read(folder/'answer_requests.json')
            seq=[[p['sha256'] for p in req['parts'] if p.get('mimeType')=='video/mp4'] for req in requests.values()]
            assert all(x==seq[0] for x in seq)
            assert sorted(seq[0])==sorted(m['sha256'] for m in d['media'])
            if d['id']=='ma_1597':assert not d['results']['conditions']['ssc_no_references']['valid']
        else:
            for c,arm in d['result']['conditions'].items():
                pred=arm.get('prediction') or {}
                assert scorer.score_prediction(d['input']['task'],pred,d['gold'])==d['scores'][c]
            for c in ['direct_sam','ssc_sam']:sam_replay(d,c)
    for rel,expected in read(root/'SHA256SUMS.json').items():assert sha(root/rel)==expected,rel
    return counts


if __name__=='__main__':
    print('PASS:',verify())
    print('No source dataset, API, simulator, or private run directory required. Semantics are not independently certified.')
