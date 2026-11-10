"""Two fixed synthetic format checks per key. No benchmark labels or scoring."""
import concurrent.futures
import json
from pathlib import Path
from run import HERE,Client,read,write,service_gate,utc
from comind_adapter import original,validate_schema

TESTS=[
    ('spatial',"Return a spatial_code array containing exactly this literal string: S.unknown(0, 'relation', 'scene', 'target').",original.code_schema('spatial_code')),
    ('answer','Return a synthetic joint_attention prediction: left_bbox [1,2,3,4], right_bbox [5,6,7,8], cue_type ["gaze"], category_l1 "test_object".',original.prediction_schema('joint_attention')),
]

def check(name):
    cfg=read(HERE/'input_snapshot'/name/'frozen_config.json')
    out=HERE/'format_probe'/name
    client=Client(out,cfg,service_gate)
    records=[]
    for i,(stage,prompt,schema) in enumerate(TESTS):
        client.begin(i)
        row={'shard':name,'stage':stage,'format_valid':False}
        try:
            raw=client.generate_parts([{'text':prompt}],16384,response_schema=schema)
            payload=json.loads(raw)
            validate_schema(payload,schema)
            row.update(format_valid=True,payload=payload)
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc)[:500])
        records.append(row);write(out/'status.json',records)
        print(json.dumps(row),flush=True)
        if not row['format_valid']:break
    return records

if __name__=='__main__':
    assert not (HERE/'FORMAT_PROBE.json').exists(),'Preserve prior diagnostic outcomes'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=[r for subset in pool.map(check,['comind_0','comind_1']) for r in subset]
    report={'utc':utc(),'passed':len(records)==4 and all(r['format_valid'] for r in records),'records':records,
            'labels_read':False,'evaluation_reuse':False,'schema_transport':'generationConfig.responseMimeType + responseJsonSchema'}
    write(HERE/'FORMAT_PROBE.json',report)
    assert report['passed'], 'Native-schema preflight failed; do not launch evaluation'
