"""Start each frozen shard once in its own detached tmux session."""
import json,subprocess,shlex
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
manifest=json.loads((root/'ASSIGNMENT.json').read_text())
launched=[]
for shard in manifest['shards']:
 name=shard['shard'];assert name in ['api_key','maegoqa_key','comind_key','habitat_key']
 session=f'ssc-mm4-{name}-20260920'
 probe=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
 if probe.returncode==0:raise RuntimeError('Refuse duplicate session: '+session)
 command='bash '+shlex.quote(str(root/'launch_one.sh'))+' '+shlex.quote(name)
 subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
 launched.append({'shard':name,'session':session,'count':shard['count']})
 (root/'LAUNCHED.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'sessions':launched},indent=2)+'\n')
 print('LAUNCHED',session,'samples',shard['count'],flush=True)
