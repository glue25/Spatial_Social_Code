"""One fresh paired recovery for every terminal SSC truncation; never splice."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=HERE.parent/'comind_native_schema_20260920'
SAM=HERE.parent/'comind_flash_sam_20260920'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('original_runtime',BASE/'run.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
from comind_adapter import infer
sys.path.insert(0,str(SAM))
spec=importlib.util.spec_from_file_location('sam_runtime',SAM/'run.py')
sam=importlib.util.module_from_spec(spec);spec.loader.exec_module(sam)
sam.HERE=HERE/'sam'
read,write,sha,utc=runtime.read,runtime.write,runtime.sha,runtime.utc

def run(name):
    out=HERE/'runs'/name
    cfg=read(out/'frozen_config.json');plan=read(HERE/'PROTOCOL.json')
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(out/'inputs.json')==cfg['input_hash']
        for path,h in plan['source_hashes'].items():assert sha(ROOT/path)==h,path
        client=runtime.Client(out,cfg,runtime.service_gate)
        for item in read(out/'inputs.json'):
            idx=item['index'];result_path=out/'results'/f'{idx:04d}.json'
            for path,h in plan['source_hashes'].items():assert sha(ROOT/path)==h,path
            write(out/'status.json',{'utc':utc(),'state':'sample_running','index':idx,'pid':os.getpid()})
            if not result_path.exists():
                for path in item['media_files']:assert sha(path)==plan['media_hashes'][path]
                client.begin(idx)
                result=infer(item,client,write,runtime.FATAL)
                write(result_path,result)
            target=HERE/'sam/results'/f'{idx:04d}.json'
            if not target.exists():
                write(out/'status.json',{'utc':utc(),'state':'sam_running','index':idx,'pid':os.getpid()})
                sam_item=dict(item,upstream_result=str(result_path.relative_to(ROOT)),
                    frame_hashes={str(p):plan['media_hashes'][str(p)] for p in sam.frames(item).values()})
                result=sam.process(sam_item,plan['sam_config']);write(target,result)
            print({'index':idx,'state':read(target)['state'],'fresh_paired':True},flush=True)
        write(out/'COMPLETE',{'utc':utc(),'processed':len(read(out/'inputs.json'))})
        write(out/'status.json',{'utc':utc(),'state':'complete','processed':len(read(out/'inputs.json'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',required=True,choices=['comind_0','comind_1']);a=p.parse_args()
    try:run(a.shard)
    except Exception as e:
        write(HERE/'runs'/a.shard/'FAILED.json',{'utc':utc(),'type':type(e).__name__,'message':str(e)[:500]});raise
