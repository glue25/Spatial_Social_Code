"""Offline scoring; only this program reads labels, never the worker."""
from collections import Counter
from run import HERE,ROOT,read,write,sha,utc,valid_ssc
import frozen_v2 as V2
from frozen_coordinates import coordinate_correct_original

def main():
    cfg=read(HERE/'PROTOCOL.json');gold_path=ROOT/cfg['scorer_only_gold_path']
    assert sha(gold_path)==cfg['scorer_only_gold_sha256']
    gold=read(gold_path);items=read(HERE/'inputs.json')
    metrics={task:{name:{'n':0,'invalid_output':0,'missing_bbox':0,'correct':dict.fromkeys(fields,0)}
                   for name in ['direct','ssc','ssc_sam']} for task,fields in V2.FIELD_TO_PREDICTION.items()}
    states=Counter();upstream=0;ids=[]
    paired={t:{m:{'sam_only':0,'ssc_only':0} for m in fs} for t,fs in V2.FIELD_TO_PREDICTION.items()}
    for item in items:
        upstream+=(ROOT/item['upstream_result']).exists()
        p=HERE/'results'/f"{item['index']:04d}.json"
        if not p.exists():continue
        row=read(p);task=row['task'];states[row['state']]+=1;ids.append(row['index'])
        assert sha(ROOT/item['upstream_result'])==row['upstream_sha256']
        scores={}
        for name in ['direct','ssc','ssc_sam']:
            cell=metrics[task][name];cell['n']+=1
            source=row['base_conditions'].get(name,{}) if name!='ssc_sam' else {'prediction':row['prediction']}
            if not valid_ssc(source):
                cell['invalid_output']+=1
                cell['missing_bbox']+=1
                scores[name]=dict.fromkeys(V2.FIELD_TO_PREDICTION[task],False);continue
            pred=source['prediction'] if name=='ssc_sam' else coordinate_correct_original(task,source['prediction'])
            cell['missing_bbox']+=any(V2.finite_box(pred.get(f)) is None for f in V2.BBOX_FIELDS[task])
            scores[name]=V2.score_prediction(task,pred,gold[row['id']]['gold'])
            for k,v in scores[name].items():cell['correct'][k]+=int(v)
        for m in paired[task]:
            paired[task][m]['sam_only']+=int(scores['ssc_sam'][m] and not scores['ssc'][m])
            paired[task][m]['ssc_only']+=int(scores['ssc'][m] and not scores['ssc_sam'][m])
    for conditions in metrics.values():
        for cell in conditions.values():cell['rates']={m:v/cell['n'] if cell['n'] else None for m,v in cell['correct'].items()}
    report={'utc':utc(),'processed':len(ids),'selected':len(items),'upstream_completed':upstream,
            'upstream_completed_pending_sam':upstream-len(ids),'states':dict(states),'metrics':metrics,'paired':paired,
            'scope':'Same completed sidecar IDs, all failures retained, not successful-only. SSC non-bbox fields unchanged even on SAM failure. No depth. Additional detector compute; no Direct+SAM control yet.',
            'worker':read(HERE/'status.json') if (HERE/'status.json').exists() else None}
    write(HERE/'comparison_latest.json',report)
    import json
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
