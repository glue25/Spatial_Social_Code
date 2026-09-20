"""Wait for the unchanged first ten, then execute the next twenty once."""
import json,os,sys,time,fcntl
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
FIRST=HERE.parent/'flash_budget16384_pilot_20260919/runs/mmtom'
OUT=HERE/'runs/mmtom'
def status(state,**kw):
 d={'utc':datetime.now(timezone.utc).isoformat(),'state':state,'pid':os.getpid(),'first_batch':str(FIRST),'continuation_indices':list(range(10,30)),**kw}
 p=HERE/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
with (HERE/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 missing=0
 while True:
  blockers=[str(p) for r in [FIRST,OUT] for n in ['FAILED.json','OPERATOR_PAUSED.json','GATE_FAILED.json'] if (p:=r/n).exists()]
  if blockers:status('stopped_terminal_marker',markers=blockers);sys.exit(1)
  if (OUT/'COMPLETE.json').exists():status('continuation_already_complete');sys.exit(0)
  if (FIRST/'COMPLETE.json').exists():
   count=len(list((FIRST/'results').glob('*.json')))
   if count!=10:status('stopped_bad_first_batch_count',count=count);sys.exit(1)
   status('launching_continuation')
   os.execv(sys.executable,[sys.executable,'-u',str(HERE/'run.py')])
  p=FIRST/'status.json';s=json.loads(p.read_text()) if p.exists() else {};pid=s.get('pid');alive=False
  if pid:
   try:os.kill(pid,0);alive=True
   except ProcessLookupError:pass
  missing=0 if alive else missing+1
  if missing>=3:status('stopped_first_worker_missing');sys.exit(1)
  status('waiting_for_first10',first_completed=len(list((FIRST/'results').glob('*.json'))))
  time.sleep(20)
