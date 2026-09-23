"""Verify the public snapshot in an isolated process using its frozen sources.

No private dataset/workdir, API credential, model, or simulator is required.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/information_history_v1_20260923'
CORE=r'''
import gzip, hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]);read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in read(root/'SHA256SUMS.json').items():assert sha(root/rel)==h,rel
prov=read(root/'evidence/provenance.json')
for rel,h in prov['source_hashes'].items():assert sha(root/'frozen_source'/rel)==h,rel
from experiments.information_history_v1.protocol import build,audit,prompt,score,digest,CONDITIONS,HISTORIES
from experiments.information_history_v1.run import parse
from experiments.information_history_v1.metrics import summarize
for name,field in [('dataset','data_sha256'),('selection_audit','selection_sha256')]:
    raw=gzip.decompress((root/f'evidence/{name}.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==prov[field]
data=json.loads(gzip.decompress((root/'evidence/dataset.json.gz').read_bytes()))
cfg=read(root/'evidence/frozen_config.public.json')
files={}
for line in gzip.decompress((root/'evidence/all_calls.jsonl.gz').read_bytes()).decode().splitlines():
    rec=json.loads(line);assert hashlib.sha256(rec['text'].encode()).hexdigest()==rec['sha256']
    assert rec['path'] not in files
    files[rec['path']]=json.loads(rec['text'])
by={(c['base_id'],c['history']):c for c in data['cases']}
for b in data['bases']:
    cc=[by[b['base_id'],h] for h in HISTORIES];assert audit(b,cc)
    for c in cc:assert build(b,c['history'])==c
all_rows=[]
for split,n in [('engineering',160),('formal',640)]:
    rows=[]
    for b in data['bases']:
        if b['split']!=split:continue
        for h in HISTORIES:
            c=by[b['base_id'],h]
            for condition in CONDITIONS:
                prefix=f"calls/{split}/{b['base_id']}/{h}/{condition}/"
                req=files[prefix+'request.json'];raw=files.get(prefix+'response.json')
                r=files[prefix+'result.json'];body=req['body']
                assert body['messages']==[{'role':'user','content':prompt(c,condition)}]
                assert body['model']==cfg['model'] and body['temperature']==cfg['temperature']
                assert body['max_tokens']==cfg['max_output_tokens']
                assert body['reasoning_effort']==cfg['reasoning_effort']
                assert digest(body)==req['sha256']==r['request_sha256']
                assert r['config_sha256']==prov['original_config_sha256']
                try:pred=parse(raw or {},c);valid=True
                except Exception:pred={};valid=False
                assert r['valid']==valid and r['prediction']==pred
                assert r['scores']==score(pred,c,valid)
                rows.append(r)
    assert len(rows)==n
    expected=read(root/f'{split}_summary.json')
    actual=summarize([b for b in data['bases'] if b['split']==split],rows,cfg['bootstrap_draws'])
    assert actual==expected,(split,'summary mismatch')
    all_rows+=rows
# Recalculate per-condition costs including every transport attempt, not reuses.
cost={}
for path,attempt in files.items():
    parts=path.split('/')
    if parts[0]!='calls' or not parts[-1].startswith('attempt_'):continue
    key=parts[1]+'/'+parts[4]
    x=cost.setdefault(key,dict(actual_request_attempts=0,input_tokens=0,output_tokens=0,total_tokens=0,
                             usage_missing_attempts=0,elapsed_seconds=0.0))
    x['actual_request_attempts']+=1;x['elapsed_seconds']+=attempt.get('elapsed_seconds',0)
    usage=attempt.get('response',{}).get('usage')
    if not usage:x['usage_missing_attempts']+=1;continue
    for name,rawname in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
        x[name]+=usage.get(rawname,0)
saved=read(root/'costs.json')
for key,x in cost.items():
    for k,v in x.items():
        assert abs(v-saved[key][k])<1e-6,(key,k)
selection=read(root/'case_selection.json');assert len(selection['all40'])==40
assert set(selection['selected_cases'])=={'formal_000','formal_003','formal_007','formal_019'}
for bid in selection['selected_cases']:
    c=read(root/'cases'/bid/'case.json')
    assert c['base']==next(b for b in data['bases'] if b['base_id']==bid)
    for h in HISTORIES:
        assert next(x for x in c['histories'] if x['history']==h)==by[bid,h]
        for condition in CONDITIONS:
            prefix=f'calls/formal/{bid}/{h}/{condition}/'
            assert c['outputs'][h][condition]==files[prefix+'result.json']
            for name in ('request.json','response.json'):
                assert read(root/'cases'/bid/'saved_calls'/h/condition/name)==files[prefix+name]
print(json.dumps({'passed':True,'formal':640,'engineering':160,'selected_full_history_cases':4,
                  'formal_invalid':sum(not r['valid'] for r in all_rows if r['split']=='formal'),
                  'source':'isolated frozen source, public files only'}))
'''


def verify(root=OUT):
    root=Path(root).resolve()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'frozen_source'))
    result=subprocess.run([sys.executable,'-c',CORE,str(root)],cwd=root/'frozen_source',
                          env=env,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return result.stdout.strip()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=OUT)
    print(verify(parser.parse_args().root))
