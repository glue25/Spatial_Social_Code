import json
from pathlib import Path
from datetime import datetime,timezone
home=Path(__file__).resolve().parent
roots=[home.parent/'flash_budget16384_pilot_20260919/runs/mmtom',home/'runs/mmtom']
report={'utc':datetime.now(timezone.utc).isoformat(),'total':30,'completed':0,'cohorts':[],'comparison':{m:{'n':0,'correct':0,'invalid':0} for m in ('direct','ssc')},'scope':'30-question budget-adjusted pilot; completed-sample invalids retained; not final benchmark'}
for root in roots:
 count=len(list((root/'results').glob('*.json')));status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'state':'not_started'}
 report['cohorts'].append({'path':str(root),'completed':count,'status':status,'terminal_markers':[n for n in ['FAILED.json','COMPLETE.json','OPERATOR_PAUSED.json'] if (root/n).exists()]});report['completed']+=count
 p=root/'comparison_latest.json'
 if p.exists():
  d=json.loads(p.read_text())
  for m in report['comparison']:
   for k in ['n','correct','invalid']:report['comparison'][m][k]+=d['comparison'][m][k]
for c in report['comparison'].values():c['accuracy']=c['correct']/c['n'] if c['n'] else None
print(json.dumps(report,indent=2))
