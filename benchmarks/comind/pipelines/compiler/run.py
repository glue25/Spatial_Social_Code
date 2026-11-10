"""CoMind-only native-schema rerun; never retry a completed invalid sample."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'benchmarks/ma_egoqa')]
from spatial_social_code.service_gate import service_gate
sys.path.insert(0,str(HERE))
from budget_transport import Client, Paused, ModelIdentityError, PermanentServiceError, write

FATAL = (Paused, ModelIdentityError, PermanentServiceError)
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(name):
    out=HERE/'runs'/name
    cfg=read(out/'frozen_config.json')
    assert cfg['dataset']=='comind', 'This revision must never launch or modify MA'
    rows=read(out/'inputs.json')
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((out/n).exists() for n in ['COMPLETE','OPERATOR_PAUSED.json','FAILED.json','SERVICE_BLOCKED.json']):
            return
        assert sha(out/'inputs.json')==cfg['input_hash']
        assert sha(out/'media_hashes.json')==cfg['media_hash']
        hashes=read(out/'media_hashes.json')
        if cfg['dataset']=='maegoqa':
            import matched_runner as legacy
        else:
            import comind_adapter
        for path,value in cfg['source_hashes'].items():assert sha(ROOT/path)==value, path
        client=Client(out,cfg,service_gate)
        consecutive_transport=0
        for position,item in enumerate(rows):
            if (out/'OPERATOR_PAUSED.json').exists():raise Paused('operator pause')
            target=out/'results'/f"{item['index']:04d}.json"
            if target.exists():
                result=read(target)
            else:
                for path,value in cfg['source_hashes'].items():assert sha(ROOT/path)==value,path
                for path in item['media_files']:assert sha(path)==hashes[path], 'frozen media changed'
                write(out/'status.json',dict(utc=utc(),state='sample_running',completed=position,total=len(rows),index=item['index'],pid=os.getpid()))
                client.begin(item['index'])
                try:
                    if cfg['dataset']=='maegoqa':
                        result=legacy.process_ma(item,client)
                    else:
                        result=comind_adapter.infer(item,client,write,FATAL)
                except FATAL:
                    raise
                except Exception as exc:
                    result={'index':item['index'],'error_type':type(exc).__name__,'prediction':None}
                    # Preserve an already completed independent Direct result.
                write(target,result)
            errors=[result.get('error_type')]+[v.get('error_type') for v in result.get('conditions',{}).values()]
            transport_bad=any(e=='GeminiServiceError' for e in errors)
            consecutive_transport=consecutive_transport+1 if transport_bad else 0
            write(out/'status.json',dict(utc=utc(),state='sample_completed',completed=position+1,total=len(rows),index=item['index'],pid=os.getpid()))
            print(json.dumps({'utc':utc(),'shard':name,'completed':position+1,'total':len(rows),'index':item['index'],'errors':[e for e in errors if e]}),flush=True)
            if consecutive_transport>=3:
                write(out/'SERVICE_BLOCKED.json',dict(utc=utc(),reason='Three consecutive samples exhausted transport; stop flooding unavailable service. Failures remain scored, no replacement.'))
                write(out/'status.json',dict(utc=utc(),state='service_blocked',completed=position+1,total=len(rows)))
                return
        write(out/'COMPLETE',{'utc':utc(),'processed':len(rows),'failures_retained':True})
        write(out/'status.json',dict(utc=utc(),state='complete',completed=len(rows),total=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',required=True,choices=['comind_0','comind_1']);a=p.parse_args()
    try:run(a.shard)
    except Paused:
        write(HERE/'runs'/a.shard/'status.json',dict(utc=utc(),state='operator_paused'))
    except Exception as exc:
        write(HERE/'runs'/a.shard/'FAILED.json',dict(utc=utc(),error_type=type(exc).__name__))
        write(HERE/'runs'/a.shard/'status.json',dict(utc=utc(),state='failed',error_type=type(exc).__name__))
        raise
