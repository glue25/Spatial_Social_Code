"""Freeze a NEW 300-pair run after fixed label-blind diagnostics pass."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from run import HERE,ROOT,OLD,read,sha,utc,write

PREVIOUS=HERE.parent/'ma_comind_flash16384_20260920'
NAMES=['comind_0','comind_1']
def prepare():
    assert not (HERE/'runs').exists(),'Refuse overwrite'
    for check in ['FORMAT_PROBE.json','CANARY.json','REPLAY.json']:assert read(HERE/check)['passed'],check
    test=subprocess.run([sys.executable,str(HERE/'test_native_schema.py')],capture_output=True,text=True)
    assert test.returncode==0,test.stdout+test.stderr
    write(HERE/'TESTS.json',{'utc':utc(),'passed':True,'stdout':test.stdout})
    sources=list(HERE.glob('*.py'))+list((HERE/'comind_source').rglob('*.py'))
    sources += [OLD/'legacy_loader.py',OLD/'service_gate.py']+list((OLD/'historical_source').rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    assignment=[];indices=[]
    for name in NAMES:
        assert read(PREVIOUS/'runs'/name/'status.json')['state']=='operator_paused', 'Old worker not paused'
        out=HERE/'runs'/name;out.mkdir(parents=True)
        for fn in ['inputs.json','media_hashes.json']:shutil.copyfile(HERE/'input_snapshot'/name/fn,out/fn)
        cfg=read(HERE/'input_snapshot'/name/'frozen_config.json')
        cfg.update(created_utc=utc(),source_hashes=hashes,input_hash=sha(out/'inputs.json'),media_hash=sha(out/'media_hashes.json'),
            schema_transport='generationConfig.responseMimeType=application/json and original responseJsonSchema, for all Direct/Spatial/Social/SSC calls',
            text_schema_suffix=False,format_revision='native_schema_v1',
            local_validation='Full original JSON schema subset (types, required keys, lengths, uniqueness); no generated-output repair. Original code normalizer and unknown fallback unchanged.',
            scientific_changes=['model/provider vs historical GPT5.6','native structured JSON interface (replaces failed text-schema adapter)','uniform16384cap','bounded10transportattempts','twofixedkeys'],
            method='Original main() prompt expressions, original schemas, Spatial/Social normalizers and executors; schema moved out of text into native API',
            prior_response_reuse=False,fresh_baseline=True,fresh_ssc=True,
            previous_run=str(PREVIOUS/'runs'/name),
            diagnostics='Four synthetic checks and fixed first sample per task full pipeline; no gold read or correctness gating; diagnostic outputs not reused',
            scope='Development format revision after observing failures. New full300 paired run, not hidden-test or untouched confirmation; historical model is GPT5.6, not Pro.')
        write(out/'frozen_config.json',cfg)
        rows=read(out/'inputs.json');part=[r['index'] for r in rows];indices.extend(part)
        assignment.append({'dataset':'comind','shard':name,'key_name':cfg['key_name'],'count':len(rows),'indices':part,'config_sha256':sha(out/'frozen_config.json')})
    assert sorted(indices)==list(range(300)) and len(set(indices))==300
    assert len({a['key_name'] for a in assignment})==2
    write(HERE/'ASSIGNMENT.json',{'utc':utc(),'shards':assignment,'fresh_all_samples':True,'old_failures_preserved_separately':True,'canary_reused':False,'comind_gold_hash':sha(HERE/'SCORER_ONLY_comind_gold.json')})
    print('PREPARED 300 fresh pairs, two fixed150-item shards',flush=True)

def launch():
    assert read(HERE/'CANARY.json')['passed'] and read(HERE/'TESTS.json')['passed']
    launched=[]
    for r in read(HERE/'ASSIGNMENT.json')['shards']:
        name=r['shard'];out=HERE/'runs'/name
        assert read(PREVIOUS/'runs'/name/'status.json')['state']=='operator_paused'
        assert sha(out/'frozen_config.json')==r['config_sha256']
        cfg=read(out/'frozen_config.json')
        for p,h in cfg['source_hashes'].items():assert sha(ROOT/p)==h,p
        session='ssc-comind-native-'+name+'-20260920'
        assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0,'Duplicate session'
        cmd='cd '+shlex.quote(str(ROOT))+' && export PYTHONDONTWRITEBYTECODE=1; exec '+shlex.quote(str(ROOT/'MA-EgoQA/.venv/bin/python'))+' -u '+shlex.quote(str(HERE/'run.py'))+' --shard '+shlex.quote(name)+' >> '+shlex.quote(str(out/'queue.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
        launched.append({'shard':name,'session':session,'count':r['count'],'key_name':r['key_name']})
        write(HERE/'LAUNCHED.json',{'utc':utc(),'workers':launched})
        print('LAUNCHED',session,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:launch()
    else:prepare()
