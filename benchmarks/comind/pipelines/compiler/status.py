"""Offline scoring only; never used to select retries, prompts or continuation."""
import importlib.util
import json
from run import HERE,ROOT,read,sha,write,utc

def main():
    assignment=read(HERE/'ASSIGNMENT.json')
    assert sha(HERE/'SCORER_ONLY_comind_gold.json')==assignment['comind_gold_hash']
    gold=read(HERE/'SCORER_ONLY_comind_gold.json')
    p=HERE/'comind_source/experiments/comind_optimization_v2/compare_balanced_pilot.py'
    spec=importlib.util.spec_from_file_location('comparison',p);cmp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cmp)
    old={}
    for line in (ROOT/'pipelines/spatial_social_optimization_v1/comind_codex_300_predictions.jsonl').read_text().splitlines():
        r=json.loads(line);old[(r['task'],str(r['recording_id']),str(r['sample_id']),r['condition'])]=r
    metrics={task:{name:{'n':0,'invalid':0,'correct':dict.fromkeys(fields,0)} for name in ['direct','ssc','historical_gpt56_direct','historical_gpt56_ssc']} for task,fields in cmp.V2.FIELD_TO_PREDICTION.items()}
    workers=[];seen=set()
    for s in assignment['shards']:
        out=HERE/'runs'/s['shard'];inputs={r['index']:r for r in read(out/'inputs.json')}
        results=[read(p) for p in (out/'results').glob('*.json')]
        for result in results:
            idx=result['index'];assert idx not in seen;seen.add(idx);item=inputs[idx];task=item['task']
            for name,condition in [('direct','direct'),('ssc','ssc'),('historical_gpt56_direct','direct'),('historical_gpt56_ssc','spatial_social')]:
                r=result.get('conditions',{}).get(condition,{}) if not name.startswith('historical') else old[(task,str(item['recording_id']),str(item['sample_id']),condition)]
                cell=metrics[task][name];cell['n']+=1
                if not isinstance(r.get('prediction'),dict) or r.get('error_type') or r.get('error'):
                    cell['invalid']+=1;continue
                scores=cmp.V2.score_prediction(task,cmp.coordinate_correct_original(task,r['prediction']),gold[item['id']]['gold'])
                for k,v in scores.items():cell['correct'][k]+=int(v)
        workers.append({'shard':s['shard'],'processed':len(results),'total':len(inputs),'state':read(out/'status.json') if (out/'status.json').exists() else {'state':'not_started'},'transport':read(out/'transport_status.json') if (out/'transport_status.json').exists() else None})
    for variants in metrics.values():
        for cell in variants.values():cell['rates']={k:v/cell['n'] if cell['n'] else None for k,v in cell['correct'].items()}
    report={'utc':utc(),'processed':len(seen),'selected':300,'workers':workers,'metrics':metrics,'scope':'New native-schema run only; invalids retained; missing separate; no old/diagnostic predictions merged; historical comparison on same completed IDs, coordinate-corrected uniformly.'}
    write(HERE/'comparison_latest.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
