"""At most two minimal calls per provisioned key; no dataset labels read."""
import concurrent.futures,json,time,sys
from datetime import datetime,timezone
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'flash_model_only_legacy_20260918'
sys.path.insert(0,str(OLD));from service_gate import service_gate
CFG=json.loads((OLD/'runs/mmtom/frozen_config.json').read_text())
NAMES=['api_key','maegoqa_key','comind_key','habitat_key']
def gate_for(name):return CFG['shared_gate'] if name=='habitat_key' else str(HERE/'locks'/f'{name}.lock')
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def probe(name):
 path=Path('/tmp/physical_code_qdd_credentials')/name
 row={'key_name':name,'key_file':str(path),'shared_gate':gate_for(name),'usable':False,'attempts':[]}
 if not path.exists():row['error']='missing';return row
 key=path.read_text().strip()
 body={'contents':[{'role':'user','parts':[{'text':'Reply with exactly OK.'}]}],'generationConfig':{'temperature':0.0,'maxOutputTokens':16384}}
 for i in range(1,3):
  a={'attempt':i,'utc':datetime.now(timezone.utc).isoformat()};start=time.monotonic()
  try:
   with service_gate(gate_for(name),15):
    resp=requests.post(CFG['endpoint'],json=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},timeout=(20,90),allow_redirects=False)
   a['http_status']=resp.status_code
   try:raw=resp.json()
   except ValueError:raw={'non_json':resp.text[:1000]}
   raw=json.loads(json.dumps(raw).replace(key,'[REDACTED]'));write(HERE/'key_probes'/name/f'attempt_{i}.json',raw)
   a['modelVersion']=raw.get('modelVersion');a['finishReasons']=[c.get('finishReason') for c in raw.get('candidates',[])]
   txt='\n'.join(p.get('text','') for c in raw.get('candidates',[]) for p in c.get('content',{}).get('parts',[]) if not p.get('thought'))
   a['visible_chars']=len(txt);row['usable']=resp.status_code==200 and a['modelVersion']==CFG['model'] and bool(txt) and a['finishReasons']==['STOP']
   if resp.status_code!=200:a['error']=str(raw)[:1000]
  except requests.RequestException as e:a.update(error_type=type(e).__name__,error=str(e).replace(key,'[REDACTED]'))
  a['elapsed_seconds']=round(time.monotonic()-start,2);row['attempts'].append(a)
  write(HERE/'key_probes'/name/'status.json',row);print(json.dumps({'key_name':name,**a},ensure_ascii=False),flush=True)
  if row['usable']:break
  if a.get('http_status')==200 or a.get('http_status') in [400,401,403,404]:break
  if i<2:time.sleep(20)
 return row
if __name__=='__main__':
 assert not (HERE/'KEY_PROBES.json').exists(),'Preserve existing key probe report'
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,NAMES))
 report={'utc':datetime.now(timezone.utc).isoformat(),'requested_model':CFG['model'],'keys':rows,'usable_keys':[r['key_name'] for r in rows if r['usable']]}
 write(HERE/'KEY_PROBES.json',report);print('USABLE',report['usable_keys'],flush=True)
