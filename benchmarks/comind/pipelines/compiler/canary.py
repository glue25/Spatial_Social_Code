"""Fixed first sample per task; schema/execution checks only, no gold or scoring.
These diagnostic responses are never copied into the new evaluation run.
"""
import concurrent.futures
import json
from run import HERE,read,sha,Client,FATAL,write,utc,service_gate
from comind_adapter import infer

TASKS=['joint_attention','scoia','handover']

def run_key(name,items):
    cfg=read(HERE/'input_snapshot'/name/'frozen_config.json')
    out=HERE/'canary'/name
    hashes={}
    for shard in ['comind_0','comind_1']:hashes.update(read(HERE/'input_snapshot'/shard/'media_hashes.json'))
    client=Client(out,cfg,service_gate)
    records=[]
    for item in items:
        for p in item['media_files']:assert sha(p)==hashes[p]
        client.begin(item['index'])
        result=infer(item,client,write,FATAL)
        write(out/'results'/f"{item['index']:04d}.json",result)
        failures={k:v.get('error_type') for k,v in result['conditions'].items() if v.get('error_type')}
        record={'task':item['task'],'index':item['index'],'shard':name,'format_valid':not failures,
                'calls':client.counter,'failures':failures,
                'spatial_lines':len(result['conditions'].get('ssc',{}).get('spatial_code','').splitlines()),
                'social_lines':len(result['conditions'].get('ssc',{}).get('social_code','').splitlines())}
        records.append(record)
        print(json.dumps(record),flush=True)
    return records

if __name__=='__main__':
    assert read(HERE/'FORMAT_PROBE.json')['passed']
    assert not (HERE/'CANARY.json').exists(),'Preserve diagnostic outcomes'
    rows=sorted([r for name in ['comind_0','comind_1'] for r in read(HERE/'input_snapshot'/name/'inputs.json')],key=lambda r:r['index'])
    selected=[next(r for r in rows if r['task']==t) for t in TASKS]
    write(HERE/'CANARY_SELECTION.json',{'selection':'First original manifest index per task; chosen before requests; no labels loaded','indices':[r['index'] for r in selected],'tasks':TASKS})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_key,name,selected[i::2]) for i,name in enumerate(['comind_0','comind_1'])]
        records=[r for f in futures for r in f.result()]
    report={'utc':utc(),'passed':len(records)==3 and all(r['format_valid'] and r['calls']==4 for r in records),
            'records':records,'labels_read':False,'scored':False,'reuse_in_evaluation':False,
            'limitation':'Format/execution gate, not semantic correctness or model-performance evidence'}
    write(HERE/'CANARY.json',report)
    assert report['passed'],'Multimodal format gate failed; evaluation must not launch'
