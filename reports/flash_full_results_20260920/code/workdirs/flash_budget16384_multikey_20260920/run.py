"""Fixed disjoint MMToM shards; same method/caps, enhanced transport retries."""
import argparse,fcntl,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'flash_model_only_legacy_20260918'
sys.path.insert(0,str(OLD));import matched_runner as legacy
sys.path.insert(0,str(HERE))
from budget_transport import Client,OutputTruncated,ModelIdentityError,Paused,write
from egomas.src.jdcloud_gemini import _extract_text
PREVIOUS=['flash_budget16384_pilot_20260919','flash_budget16384_extension20_20260919','flash_budget16384_remaining570_20260919']
ALLOWED=['api_key','maegoqa_key','comind_key','habitat_key']
def read(p):return json.loads(p.read_text())
def own_hashes():return {n:legacy.sha(HERE/n) for n in ['run.py','budget_transport.py']}
def prepare():
 assert not (HERE/'runs').exists(),'Refuse overwrite'
 probes=read(HERE/'KEY_PROBES.json');keys=[r for r in probes['keys'] if r['usable']]
 assert keys and all(k['key_name'] in ALLOWED for k in keys)
 processed=[]
 for name in PREVIOUS:
  processed.extend(read(p)['index'] for p in (HERE.parent/name/'runs/mmtom/results').glob('*.json'))
 assert sorted(processed)==list(range(45)),'Review changed prior results before scheduling'
 original=OLD/'runs/mmtom';base=read(original/'frozen_config.json');legacy.verify(original,base)
 rows=read(original/'inputs.json')[45:600];assert len(rows)==555
 all_media=read(original/'media_hashes.json');needed={p for item in rows for p in item['media_files']}
 for p in needed:assert legacy.sha(p)==all_media[p]
 assignment=[]
 for number,key in enumerate(keys):
  name=key['key_name'];out=HERE/'runs'/name;items=rows[number::len(keys)]
  media={p:all_media[p] for item in items for p in item['media_files']}
  write(out/'inputs.json',items);write(out/'media_hashes.json',media)
  cfg=dict(base)
  cfg.update(created_utc=legacy.utc(),shard=name,key_file=key['key_file'],shared_gate=key['shared_gate'],samples=len(items),input_hash=legacy.sha(out/'inputs.json'),media_hash=legacy.sha(out/'media_hashes.json'),pilot_hashes=own_hashes(),uniform_output_cap=16384,transport_max_attempts=10,output_caps='16384 uniformly for all Direct and SSC calls including original conditional retries/refinement',retry_policy='10 attempts for transport/transient HTTP/empty text; 600s per-request timeout; exponential 5s backoff capped at120s; no retry of semantic errors, truncation or identity mismatch',scientific_changes=['model','uniform 16384 output ceiling','reject truncated outputs','increase transport attempts from5 to10','fixed key-based disjoint shards'],selection='Frozen indices45–599 round-robin over pretested keys, independent of labels or outcomes; no reassignment or reruns',scope='Budget-adjusted historical-protocol development replication; credential/transport-policy change after first45; not provider-controlled or total-compute matched')
  write(out/'frozen_config.json',cfg)
  assignment.append({'shard':name,'count':len(items),'indices':[x['index'] for x in items],'config_sha256':legacy.sha(out/'frozen_config.json')})
 write(HERE/'ASSIGNMENT.json',{'utc':legacy.utc(),'previous_processed':45,'remaining':555,'keys':len(keys),'shards':assignment,'key_probe_sha256':legacy.sha(HERE/'KEY_PROBES.json'),'frozen_inference_hashes':own_hashes()})
 print(json.dumps({'prepared':555,'shards':[(x['shard'],x['count']) for x in assignment]}),flush=True)
def summarize(out):
 items={x['index']:x for x in read(out/'inputs.json')};results=[read(p) for p in sorted((out/'results').glob('*.json'))]
 cells={m:{'n':len(results),'correct':0,'invalid':0} for m in ['direct','ssc']}
 for r in results:
  row=items[r['index']]['row'];truth=str(row['answer']).lower();direct=r.get('direct_prediction')
  if not direct:
   for p in sorted((out/'calls'/f"{r['index']:04d}").glob('*/request.json')):
    q=read(p);response=p.parent/'response.json'
    if response.exists() and q['parts'][-1].get('text') in [t.format(question=row['question']) for t in [legacy.mm.DIRECT_PROMPT,legacy.mm.DIRECT_REPAIR_PROMPT]]:direct=legacy.mm.parse_ab(_extract_text(read(response)))
  for m,pred in [('direct',direct),('ssc',r.get('prediction'))]:
   valid=pred in ['a','b'];cells[m]['invalid']+=not valid;cells[m]['correct']+=valid and pred==truth
 for c in cells.values():c['accuracy']=c['correct']/c['n'] if c['n'] else None
 write(out/'comparison_latest.json',{'utc':legacy.utc(),'completed':len(results),'total':len(items),'errors_retained':True,'comparison':cells})
def verify(out,cfg):
 legacy.verify(out,cfg);assert own_hashes()==cfg['pilot_hashes']
 manifest=read(HERE/'ASSIGNMENT.json');entry=next(x for x in manifest['shards'] if x['shard']==cfg['shard'])
 assert legacy.sha(out/'frozen_config.json')==entry['config_sha256']
def run(name):
 out=HERE/'runs'/name;cfg=read(out/'frozen_config.json')
 with (out/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if any((out/n).exists() for n in ['FAILED.json','COMPLETE.json','OPERATOR_PAUSED.json']):raise RuntimeError('Terminal marker exists')
  verify(out,cfg);items=read(out/'inputs.json');media=read(out/'media_hashes.json');client=Client(out,cfg,legacy.GATE.service_gate)
  for number,item in enumerate(items):
   if (out/'OPERATOR_PAUSED.json').exists():raise Paused('operator pause')
   verify(out,cfg)
   for p in item['media_files']:assert legacy.sha(p)==media[p]
   target=out/'results'/f"{item['index']:04d}.json"
   if target.exists():continue
   write(out/'status.json',dict(utc=legacy.utc(),state='sample_running',shard=name,completed=number,total=len(items),index=item['index'],pid=os.getpid()))
   print(f"START shard={name} index={item['index']} completed={number}/{len(items)}",flush=True);client.begin(item['index'])
   try:result=legacy.process_mm(item,client)
   except (ModelIdentityError,Paused):raise
   except Exception as e:
    write(target,{'index':item['index'],'error_type':type(e).__name__,'prediction':None});summarize(out);raise
   write(target,result);summarize(out)
   write(out/'progress.json',{'utc':legacy.utc(),'completed':number+1,'total':len(items)})
   print(f"DONE shard={name} index={item['index']} completed={number+1}/{len(items)}",flush=True)
  write(out/'COMPLETE.json',{'utc':legacy.utc(),'completed':len(items)})
  write(out/'status.json',{'utc':legacy.utc(),'state':'complete','completed':len(items),'total':len(items)})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--shard',choices=ALLOWED);args=parser.parse_args()
 if args.prepare:prepare()
 else:
  if not args.shard:parser.error('--shard is required')
  out=HERE/'runs'/args.shard
  try:run(args.shard)
  except Exception as e:
   write(out/'FAILED.json',{'utc':legacy.utc(),'error_type':type(e).__name__})
   write(out/'status.json',{'utc':legacy.utc(),'state':'stopped_error','error_type':type(e).__name__,'pid':os.getpid()})
   raise
