"""Independent recovery scoring; no report is used for inference or retries."""
import json
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'sam'))
import frozen_v2 as V2
from frozen_coordinates import coordinate_correct_original

def main():
    plan=json.loads((HERE/'PROTOCOL.json').read_text())
    gold=json.loads((ROOT/plan['sam_config']['scorer_only_gold_path']).read_text())
    records=[];states=Counter()
    for sel in plan['selected']:
        p=HERE/'sam/results'/f"{sel['index']:04d}.json"
        if not p.exists():continue
        r=json.loads(p.read_text());scores={};states[r['state']]+=1
        for name in ['direct','ssc_sam']:
            condition=r['base_conditions'].get('direct',{}) if name=='direct' else {'prediction':r['prediction']}
            pred=condition.get('prediction')
            if not isinstance(pred,dict) or condition.get('error_type') or condition.get('error'):
                scores[name]=dict.fromkeys(V2.FIELD_TO_PREDICTION[r['task']],False)
            else:
                if name=='direct':pred=coordinate_correct_original(r['task'],pred)
                scores[name]=V2.score_prediction(r['task'],pred,gold[r['id']]['gold'])
        records.append({'index':r['index'],'id':r['id'],'task':r['task'],'state':r['state'],'scores':scores,
            'direct_valid':isinstance(r['base_conditions'].get('direct',{}).get('prediction'),dict),
            'ssc_valid':isinstance(r['base_conditions'].get('ssc',{}).get('prediction'),dict),
            'sam_view_errors':r.get('view_errors',{})})
    report={'utc':datetime.now(timezone.utc).isoformat(),'selected':4,'processed':len(records),'states':dict(states),'records':records,
        'all_four_processed':len(records)==4,'uniform_output_cap':32768,'primary_300_modified':False,
        'scope':'Single fresh paired recovery of all4 failure-selected IDs; separate from primary300, no outcome-based resampling or replacement.'}
    (HERE/'RECOVERY_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
