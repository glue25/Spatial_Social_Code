"""Wait for preserved inference, then compile and render the independent demo."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();source=Path(__file__).resolve().parent
    status={'utc':datetime.now(timezone.utc).isoformat(),'state':'waiting_for_inference','model_calls_in_this_finisher':0}
    def save(state):
        status.update(state=state,utc=datetime.now(timezone.utc).isoformat());(out/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(cmd,name):
        with (out/name).open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=3600)
    try:
        save('waiting_for_inference');deadline=time.time()+5400
        while not (out/'INFERENCE_COMPLETE').exists():
            if time.time()>deadline or 'Traceback (most recent call last)' in (out/'inference.log').read_text():raise RuntimeError('inference incomplete or failed; no placeholder substitution')
            time.sleep(10)
        save('compiling_observation_states_and_render_tracks')
        run([sys.executable,str(source/'build_temporal.py'),'--output',str(out)],'compile.log')
        save('rendering_blender_animation')
        run(['xvfb-run','-a','blender','-b','--python-exit-code','1','--python',str(source/'blender_temporal.py'),'--',str(out)],'render.log')
        save('making_synchronized_preview')
        run([sys.executable,str(source/'make_preview.py'),'--output',str(out)],'preview.log')
        save('render_complete_pending_visual_audit')
    except Exception as error:
        status['error_type']=type(error).__name__;status['error']=str(error);save('failed_preserved');raise


if __name__=='__main__':main()
