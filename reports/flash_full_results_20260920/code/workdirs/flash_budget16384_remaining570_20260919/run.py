"""Remaining 570 frozen MMToM questions; unchanged budget-adjusted paired inference."""
import fcntl,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'flash_model_only_legacy_20260918'
sys.path.insert(0,str(OLD))
import matched_runner as legacy
sys.path.insert(0,str(HERE))
from budget_transport import Client,OutputTruncated,ModelIdentityError,Paused,write
from egomas.src.jdcloud_gemini import _extract_text
OUT=HERE/'runs/mmtom'
def read(p):return json.loads(p.read_text())
def own_hashes():return {n:legacy.sha(HERE/n) for n in ('run.py','budget_transport.py')}
def prepare():
 assert not OUT.exists(),'Refuse overwrite'
 cfg=read(OLD/'runs/mmtom/frozen_config.json');legacy.verify(OLD/'runs/mmtom',cfg)
 items=read(OLD/'runs/mmtom/inputs.json')[30:600]
 all_media=read(OLD/'runs/mmtom/media_hashes.json')
 media={p:all_media[p] for item in items for p in item['media_files']}
 for p,h in media.items():assert legacy.sha(p)==h
 write(OUT/'inputs.json',items);write(OUT/'media_hashes.json',media)
 cfg.update(created_utc=legacy.utc(),samples=len(items),input_hash=legacy.sha(OUT/'inputs.json'),media_hash=legacy.sha(OUT/'media_hashes.json'),uniform_output_cap=16384,output_caps='16384 uniformly for all Direct and SSC requests including original conditional retries',scientific_changes=['model','uniform output ceiling 16384','reject MAX_TOKENS before downstream use'],scope='Remaining 570 questions in the 600-question budget-adjusted historical-protocol development replication; not original-cap or provider-controlled comparison',selection='Frozen MMToM entries 30 through 599 in order; no correctness-based selection or reruns; first thirty preserved in two prior cohorts',pilot_hashes=own_hashes())
 write(OUT/'frozen_config.json',cfg)
 print('Prepared 570 fixed continuation samples',flush=True)
def summarize():
 items={x['index']:x for x in read(OUT/'inputs.json')};results=[read(p) for p in sorted((OUT/'results').glob('*.json'))]
 cells={m:{'n':len(results),'correct':0,'invalid':0} for m in ('direct','ssc')}
 for r in results:
  row=items[r['index']]['row'];truth=str(row['answer']).lower();direct=r.get('direct_prediction')
  if not direct:
   for p in sorted((OUT/'calls'/f"{r['index']:04d}").glob('*/request.json')):
    q=read(p);prompt=q['parts'][-1].get('text');response=p.parent/'response.json'
    if response.exists() and prompt in [x.format(question=row['question']) for x in (legacy.mm.DIRECT_PROMPT,legacy.mm.DIRECT_REPAIR_PROMPT)]:direct=legacy.mm.parse_ab(_extract_text(read(response)))
  for m,pred in [('direct',direct),('ssc',r.get('prediction'))]:
   valid=pred in ('a','b');cells[m]['invalid']+=not valid;cells[m]['correct']+=valid and pred==truth
 for c in cells.values():c['accuracy']=c['correct']/c['n'] if c['n'] else None
 write(OUT/'comparison_latest.json',{'utc':legacy.utc(),'completed':len(results),'total':len(items),'errors_retained':True,'scope':'budget-adjusted historical-protocol development replication; errors retained; no premature improvement claim','comparison':cells})
def run():
 cfg=read(OUT/'frozen_config.json')
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if any((OUT/n).exists() for n in ['FAILED.json','COMPLETE.json','OPERATOR_PAUSED.json']):raise RuntimeError('Terminal marker exists')
  assert own_hashes()==cfg['pilot_hashes']
  legacy.verify(OUT,cfg);items=read(OUT/'inputs.json');media=read(OUT/'media_hashes.json')
  client=Client(OUT,cfg,legacy.GATE.service_gate)
  for number,item in enumerate(items):
   if (OUT/'OPERATOR_PAUSED.json').exists():raise Paused('operator pause')
   legacy.verify(OUT,cfg);assert own_hashes()==cfg['pilot_hashes']
   for p in item['media_files']:assert legacy.sha(p)==media[p]
   target=OUT/'results'/f"{item['index']:04d}.json"
   if target.exists():continue
   write(OUT/'status.json',dict(utc=legacy.utc(),state='sample_running',completed=number,total=len(items),index=item['index'],pid=os.getpid()))
   print(f"START index={item['index']} completed={number}/{len(items)}",flush=True)
   client.begin(item['index'])
   try:result=legacy.process_mm(item,client)
   except (ModelIdentityError,Paused):raise
   except Exception as e:
    result={'index':item['index'],'error_type':type(e).__name__,'prediction':None}
    write(target,result);summarize();raise
   write(target,result);summarize()
   write(OUT/'progress.json',{'utc':legacy.utc(),'completed':number+1,'total':len(items)})
   print(f"DONE index={item['index']} completed={number+1}/{len(items)}",flush=True)
  write(OUT/'COMPLETE.json',{'utc':legacy.utc(),'completed':len(items)})
  write(OUT/'status.json',{'utc':legacy.utc(),'state':'complete','completed':len(items),'total':len(items)})
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:
  try:run()
  except Exception as e:
   write(OUT/'FAILED.json',{'utc':legacy.utc(),'error_type':type(e).__name__})
   write(OUT/'status.json',{'utc':legacy.utc(),'state':'stopped_error','error_type':type(e).__name__,'pid':os.getpid()})
   raise
