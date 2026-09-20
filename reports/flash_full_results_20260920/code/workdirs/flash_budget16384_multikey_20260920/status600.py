"""Read-only aggregate; prior failed sample is retained, never replaced."""
import json
from pathlib import Path
from datetime import datetime,timezone
home=Path(__file__).resolve().parent
names=['flash_budget16384_pilot_20260919','flash_budget16384_extension20_20260919','flash_budget16384_remaining570_20260919']
roots=[home.parent/n/'runs/mmtom' for n in names]
manifest=json.loads((home/'ASSIGNMENT.json').read_text());roots.extend(home/'runs'/s['shard'] for s in manifest['shards'])
report={'utc':datetime.now(timezone.utc).isoformat(),'total':600,'processed':0,'failed_sample_records':0,'cohorts':[],'comparison':{m:{'n':0,'correct':0,'invalid':0} for m in ['direct','ssc']},'scope':'Budget-adjusted historical-protocol development run; first45 used5attempts, remaining555 use10attempts and four fixed keys; not original-cap or total-compute matched'}
seen=set()
for root in roots:
 results=[json.loads(p.read_text()) for p in (root/'results').glob('*.json')];ids={r['index'] for r in results}
 assert not seen.intersection(ids),'Overlapping scored sample IDs';seen.update(ids)
 count=len(results);status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'state':'not_started'}
 row={'path':str(root),'processed':count,'status':status,'terminal_markers':[n for n in ['FAILED.json','COMPLETE.json','OPERATOR_PAUSED.json'] if (root/n).exists()]}
 if (root/'transport_status.json').exists():row['transport_status']=json.loads((root/'transport_status.json').read_text())
 report['cohorts'].append(row);report['processed']+=count;report['failed_sample_records']+=sum('error_type' in r for r in results)
 p=root/'comparison_latest.json'
 if p.exists():
  d=json.loads(p.read_text())
  for m in report['comparison']:
   for k in ['n','correct','invalid']:report['comparison'][m][k]+=d['comparison'][m][k]
for c in report['comparison'].values():c['accuracy']=c['correct']/c['n'] if c['n'] else None
report['results_not_yet_in_score_snapshot']=report['processed']-report['comparison']['direct']['n']
print(json.dumps(report,indent=2))
