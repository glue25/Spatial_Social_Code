#!/usr/bin/env python3
"""Offline coverage/hash/score/code checks; optional independent gold re-scoring."""
import argparse
import ast
import copy
import math
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE/'code/workdirs/comind_native_schema_20260920'
sys.path.insert(0,str(BASE/'comind_source'))
from spatial_social_code.runtime import execute_spatial,execute_social
sys.path.insert(0,str(HERE/'code/workdirs/comind_flash_sam_20260920'))
import frozen_v2 as V2
from frozen_coordinates import coordinate_correct_original

# Execute only the four pure functions from the exact frozen SAM worker.
# Do not import its live HTTP/file loop or require Pillow for this audit.
sam_source=HERE/'code/workdirs/comind_flash_sam_20260920/run.py'
sam_ast=ast.parse(sam_source.read_text())
pure=ast.Module(body=[n for n in sam_ast.body if isinstance(n,ast.FunctionDef) and n.name in {'identity','proposals','normalized_box','augment'}],type_ignores=[])
sam_functions={'V2':V2,'copy':copy,'math':math}
exec(compile(pure,str(sam_source),'exec'),sam_functions)

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gold',type=Path,help='Original scorer-only gold JSON, obtained separately')
    ap.add_argument('--cohort',choices=['all','primary','recovery'],default='all');args=ap.parse_args()
    sums=json.loads((HERE/'SHA256SUMS.json').read_text())
    for name,h in sums.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    gold=None
    if args.gold:
        info=json.loads((HERE/'audit/label_boundary.json').read_text())
        gold=json.loads(args.gold.read_text())
        canonical=hashlib.sha256(json.dumps(gold,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert hashlib.sha256(args.gold.read_bytes()).hexdigest()==info['gold_source_sha256'] or canonical==info.get('gold_canonical_sha256')
    for label,ids in [('primary',set(range(300))),('recovery',{94,113,236,245})]:
        if args.cohort!='all' and label!=args.cohort:continue
        predictions=rows(HERE/label/'results.jsonl.gz');scores=rows(HERE/label/'scores.jsonl.gz')
        assert len(predictions)==len(scores)==len(ids)
        assert {r['index'] for r in predictions}=={r['index'] for r in scores}==ids
        lookup={r['index']:r for r in scores};totals=defaultdict(Counter);sizes=Counter();states=Counter();replayed=0
        sam_calls={(r['index'],r['view']):r for r in rows(HERE/label/'sam_calls.jsonl.gz')}
        for row in predictions:
            task=row['task'];sizes[task]+=1;states[row['state']]+=1
            ssc=row['base_conditions'].get('ssc',{})
            for code,trace,fn in [('spatial_code','spatial_trace',execute_spatial),('social_code','social_trace',execute_social)]:
                if code in ssc and trace in ssc:assert fn(ssc[code])==ssc[trace];replayed+=1
            if isinstance(row['prediction'],dict):
                bb=set(V2.BBOX_FIELDS[task]);orig=ssc['prediction']
                assert {k:v for k,v in row['prediction'].items() if k not in bb}=={k:v for k,v in orig.items() if k not in bb}
                term,_=sam_functions['identity'](ssc,task);assert term==row['identity']
                for view,proposals in row['sam_proposals'].items():
                    call=sam_calls[(row['index'],view)]
                    assert call['request.json']['prompt']==term
                    if 'response' in call:assert sam_functions['proposals'](call['response'])==proposals
                    else:assert not proposals and view in row['view_errors']
                pred,view=sam_functions['augment'](orig,task,row['sam_proposals'])
                assert pred==row['prediction'] and view==row['selected_view']
            saved=lookup[row['index']]['scores']
            for name in ['direct','ssc','ssc_sam']:
                assert set(saved[name])==set(V2.FIELD_TO_PREDICTION[task])
                assert all(type(v)is bool for v in saved[name].values())
                cond=row['base_conditions'].get(name,{}) if name!='ssc_sam' else {'prediction':row['prediction']}
                valid=isinstance(cond.get('prediction'),dict) and not cond.get('error') and not cond.get('error_type')
                if not valid:assert not any(saved[name].values())
                if gold:
                    if valid:
                        pred=cond['prediction'] if name=='ssc_sam' else coordinate_correct_original(task,cond['prediction'])
                        expected=V2.score_prediction(task,pred,gold[row['id']]['gold'])
                    else:expected=dict.fromkeys(V2.FIELD_TO_PREDICTION[task],False)
                    assert expected==saved[name],(label,row['index'],name)
                totals[(task,name)].update({k:int(v) for k,v in saved[name].items()})
        for call in rows(HERE/label/'calls.jsonl.gz'):
            files=call['files'];cap=files['request.json']['generationConfig']['maxOutputTokens']
            assert cap==(16384 if label=='primary' else 32768)
            if 'response.json' in files:
                r=files['response.json'];assert r['modelVersion']=='gemini-3-flash-preview'
                assert all(c.get('finishReason')=='STOP' for c in r['candidates'])
        macro={name:sum(totals[(task,name)][m]/sizes[task] for task,fs in V2.FIELD_TO_PREDICTION.items() for m in fs)/14 for name in ['direct','ssc_sam']}
        if label=='primary':
            assert sizes=={'joint_attention':100,'scoia':100,'handover':100}
            assert abs(macro['direct']-241/1400)<1e-12 and abs(macro['ssc_sam']-311/1400)<1e-12
            summary=json.loads((HERE/label/'summary.json').read_text())
            for task,conditions in summary['metrics'].items():
                for name,cell in conditions.items():assert dict(totals[(task,name)])==cell['correct']
        print(json.dumps({'cohort':label,'n':len(ids),'states':dict(states),'macro14':macro,'code_traces_replayed':replayed,'gold_rescored':gold is not None}))
    print('PASS: coverage, failures retained, frozen hashes, code execution, equal stage caps, SAM identity/proposal/coordinate replay, non-bbox invariance, aggregate scores')
    if not gold:print('Labels were not redistributed. Default score checks trust saved field booleans; --gold performs independent label re-scoring.')

if __name__=='__main__':main()
