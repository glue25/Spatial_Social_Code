import sys,json,collections,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path('/pfs/zhangxiaoyuan/project/physical_code');OLD=ROOT/'workdirs/flash_model_only_legacy_20260918'
sys.path.insert(0,str(OLD));import matched_runner as legacy
from egomas.src.jdcloud_gemini import _extract_text,GeminiServiceError
F=ROOT/'workdirs/flash_budget16384_multikey_20260920/FINAL_VERIFIED_20260920.json'
report=json.loads(F.read_text());items={r['index']:r for r in json.loads((OLD/'runs/mmtom/inputs.json').read_text())};hashes=json.loads((OLD/'runs/mmtom/media_hashes.json').read_text())
class Replay:
 def __init__(self,root,index):self.root=root;self.index=index;self.counter=0;self.config=SimpleNamespace(model='gemini-3-flash-preview')
 def generate_parts(self,parts,max_output_tokens=4096):
  self.counter+=1;p=self.root/'calls'/f'{self.index:04d}'/f'{self.counter:02d}';q=json.loads((p/'request.json').read_text());assert q['generationConfig']=={'temperature':0.0,'maxOutputTokens':16384}
  assert q['legacy_requested_cap']==max_output_tokens
  assert len(parts)==len(q['parts'])
  for generated,saved in zip(parts,q['parts']):
   if '_path' in generated:assert hashes[generated['_path']]==saved['sha256']
   else:assert generated==saved
  if not (p/'response.json').exists():raise GeminiServiceError('Preserved missing transport response')
  raw=json.loads((p/'response.json').read_text());assert raw['modelVersion']==self.config.model;assert all(c['finishReason']=='STOP' for c in raw['candidates']);return _extract_text(raw)
 def generate(self,prompt,max_output_tokens=4096):return self.generate_parts([{'text':prompt}],max_output_tokens)
results=[]
with patch.object(legacy.mm,'image_parts',lambda paths:[{'_path':str(p.resolve())} for p in paths]),patch.object(legacy.dense,'image_parts',lambda paths:[{'_path':str(p.resolve())} for p in paths]),patch('requests.post',side_effect=AssertionError('No network allowed in replay')):
 for m in report['records']:
  p=ROOT/m['source'];old=json.loads(p.read_text());idx=old['index'];c=Replay(p.parents[1],idx)
  try:
   fresh=legacy.process_mm(items[idx],c)
   differing=[k for k,v in old.items() if k not in ['elapsed_seconds','refinement_seconds'] and fresh.get(k)!=v]
   results.append({'index':idx,'matched':not differing,'different_fields':differing,'calls':c.counter})
  except GeminiServiceError:
   assert idx==44 and old.get('error_type')=='GeminiServiceError';results.append({'index':idx,'matched':True,'preserved_transport_failure':True,'calls':c.counter})
  if len(results)%100==0:print('replayed',len(results),'mismatches',sum(not r['matched'] for r in results),flush=True)
out={'scope':'No API; archived pipeline replay from exact saved candidate text; checks original per-stage requested caps, effective cap16384, full text prompts and frozen frame hashes; output fields checked except wall-clock timings','n':len(results),'matched':sum(r['matched'] for r in results),'mismatches':[r for r in results if not r['matched']],'records':results}
Path('/tmp/flash_replay_audit_20260921.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='records'})
