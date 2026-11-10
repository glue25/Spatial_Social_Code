"""Frozen remote-Flash version of the existing 12-task,3-seed D/T/C suite.

Uses public run.sh, real Habitat physics and unchanged private-state reducers.
No Qwen fallback and no oracle trajectories in planner input.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('direct', 'static_ssc', 'dynamic_ssc')
SEEDS = (891011,891012,891013)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def event(out, state, **data):
    row={'utc':datetime.now(timezone.utc).isoformat(),'state':state,**data}
    with (out/'status.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def prepare(out, project, probe):
    evidence=json.loads(probe.read_text())
    assert evidence['http_status']==200 and evidence['response']['model']=='gemini-3-flash-preview'
    out.mkdir(parents=True,exist_ok=False);source=out/'source'
    files=[]
    for folder in ['experiments/habitat_closed_loop','spatial_social_code',
                   'external/EnactTom/enacttom','external/EnactTom/habitat_llm']:
        files += [p for p in (project/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.py','.sh','.yaml','.yml','.json','.txt')]
    files += [project/'experiments/reevaluation_v3'/n for n in (
        '__init__.py','run_qdd_habitat.py','habitat_qdd_guard.py')]
    files += [project/'tests/test_qdd_habitat.py']
    files += list((project/'reports/cases').glob('habitat_ep*_state.json'))
    for p in files:
        dest=source/p.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Only the new copy is adapted; original/live EnactToM files are unchanged.
    path=source/'external/EnactTom/habitat_llm/llm/openai_chat.py'
    text=path.read_text()
    marker='            response = self.client.chat.completions.create(**completion_kwargs)'
    assert text.count(marker)==1
    text=text.replace(marker,
        '            from experiments.reevaluation_v3.habitat_qdd_guard import request_options, validate_and_log\n'
        '            completion_kwargs = request_options(completion_kwargs)\n'+marker+'\n'
        '            validate_and_log(response)')
    path.write_text(text)
    (source/'external/EnactTom/data').symlink_to(project/'external/EnactTom/data',target_is_directory=True)
    tasks=sorted((project/'workdirs/habitat_enacttom/frozen12/tasks').glob('*.json'))
    assert len(tasks)==12
    # First task validates every condition; task order independent of performance.
    tasks.sort(key=lambda p:(not p.stem.endswith('_1566'),p.name))
    for p in tasks:
        dest=source/'tasks'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    episode=project/'workdirs/habitat_enacttom/frozen12/episodes.json.gz'
    shutil.copy2(episode,out/'episodes.json.gz')
    shutil.copy2(probe,out/'availability_probe.json')
    jobs=[{'task':str(source/'tasks'/p.name),'condition':c,'seed':seed}
          for seed in SEEDS for p in tasks for c in CONDITIONS]
    cfg={'protocol':'qdd_flash_habitat12x3seeds_DTC_v1','root':str(project),'source':str(source),
        'scope':'existing previously inspected12-task development suite; not full official EnactToM corpus',
        'model':'gemini-3-flash-preview','base_url':os.environ.get("SSC_OPENAI_BASE_URL", ""),
        'reported_model':'gemini-3-flash-preview','conditions':list(CONDITIONS),'seeds':list(SEEDS),
        'jobs':jobs,'rollouts':len(jobs),'max_environment_turns':20,'observation_mode':'text',
        'state_variant':'v2','reasoning_effort':'low','max_tokens_per_call':8192,
        'timeout_seconds_per_rollout':3600,'baseline_reuse':False,'oracle':False,
        'budget_note':'20 is environment turns, not total API calls; costs differ by condition',
        'source_hashes':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},
        'episode_hash':sha(out/'episodes.json.gz'),'probe_hash':sha(out/'availability_probe.json')}
    write(out/'frozen_config.json',cfg);event(out,'prepared',rollouts=len(jobs))


def verify(out,cfg):
    source=Path(cfg['source'])
    for rel,h in cfg['source_hashes'].items():assert sha(source/rel)==h,rel
    assert sha(out/'episodes.json.gz')==cfg['episode_hash']
    assert sha(out/'availability_probe.json')==cfg['probe_hash']


def command(argv,cwd,env,log,timeout):
    with log.open('w') as f:
        p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise RuntimeError('owned rollout timed out: '+str(log))
    if rc:raise RuntimeError(f'command failed rc={rc}: {log}')


def run(out,cfg):
    project=Path(cfg['root']);source=Path(cfg['source']);python=str(project/'.venv-habitat/bin/python')
    env=dict(os.environ)
    assert env.get('GEMINI_API_KEY'),'missing role-specific credential'
    env.update(PYTHONPATH=f'{source}:{source}/external/EnactTom',GEMINI_BASE_URL=cfg['base_url'],
        ENACTTOM_PYTHON=python,ENACTTOM_EPISODES_PATH=str(out/'episodes.json.gz'),
        QDD_HABITAT_OUTPUT=str(out),SSC_MODEL=cfg['model'],SSC_MODEL_CHECKPOINT='remote_api_no_local_checkpoint',
        SSC_CONFIG_SHA256=sha(out/'frozen_config.json'),SSC_MAX_TURNS='20',SSC_STABLE_SAMPLES='3',
        SSC_PHYSICS_STEPS_PER_SAMPLE='10',SSC_PHYSICS_DT=str(1/60),SSC_OPEN_THRESHOLD='0.10',
        SSC_STATE_VARIANT='v2',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
    command(['nvidia-smi','-L'],project,env,out/'gpu.log',30)
    command([python,'-m','pytest','-q','tests/test_qdd_habitat.py',
             'experiments/habitat_closed_loop/test_closed_loop.py'],source,env,out/'tests.log',300)
    command([python,'-c','from pathlib import Path; import experiments.habitat_closed_loop.runtime as r; '
             f'assert Path(r.__file__).is_relative_to(Path({str(source)!r})); print(r.__file__)'],
             source/'external/EnactTom',env,out/'import_check.log',30)
    results=[]
    for i,job in enumerate(cfg['jobs']):
        verify(out,cfg)
        if (out/'OPERATOR_PAUSED.json').exists():event(out,'operator_paused');return
        folder=out/f"seed{job['seed']}"/'results'/job['condition']/Path(job['task']).stem
        if (folder/'validated_result.json').exists():
            results.append(json.loads((folder/'validated_result.json').read_text()));continue
        folder.mkdir(parents=True,exist_ok=False)
        e=dict(env,ENACTTOM_BENCHMARK_SEED=str(job['seed']),SSC_SEED=str(job['seed']),
               SSC_CLOSED_LOOP_CONDITION=job['condition'])
        argv=[str(source/'external/EnactTom/enacttom/run.sh'),'benchmark','--task',job['task'],
              '--output-dir',str(folder),'--model',cfg['model'],'--llm','openai_chat',
              '--num-times','1','--max-llm-calls','20','--benchmark-run-mode','standard',
              '--observation-mode','text','--no-calibration','--video']
        event(out,'rollout_started',number=i+1,total=len(cfg['jobs']),**job)
        write(folder/'launch.json',{'argv':argv})
        command(argv,source/'external/EnactTom',e,folder/'raw_console.log',cfg['timeout_seconds_per_rollout'])
        summaries=list(folder.glob('**/closed_loop_artifacts/summary.json'));assert len(summaries)==1
        artifact=summaries[0].parent
        command([python,str(source/'experiments/habitat_closed_loop/validate_run.py'),str(artifact)],
                source,env,folder/'acceptance.json',120)
        manifest=json.loads((artifact/'manifest.json').read_text())
        assert manifest['condition']==job['condition'] and str(manifest['seed'])==str(job['seed'])
        assert manifest['config_sha256']==env['SSC_CONFIG_SHA256']
        strict=json.loads(summaries[0].read_text())['historical_benchmark_evaluation']['strict_physical']
        assert strict['measurement_valid'] is True
        result={**job,'strict_success':strict['strict_success'],
                'strict_percent_complete':strict['strict_percent_complete'],'artifact':str(artifact)}
        write(folder/'validated_result.json',result);results.append(result)
        write(out/'partial_results.json',results)
        write(out/'progress.json',{'completed':len(results),'total':len(cfg['jobs'])})
        event(out,'rollout_validated',completed=len(results),total=len(cfg['jobs']))
    assert len(results)==108
    analysis=source/'experiments/habitat_closed_loop/analyze_multiseed.py'
    command([python,str(analysis),*[str(out/f'seed{s}') for s in SEEDS],
             '--output',str(out/'analysis')],source,env,out/'analysis.log',300)
    (out/'COMPLETE').write_text('108 real rollouts validated; development suite only\n')
    event(out,'complete',rollouts=108)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--probe',type=Path);a=p.parse_args();out=a.output.resolve()
    if a.prepare:prepare(out,a.root.resolve(),a.probe.resolve());return
    cfg=json.loads((out/'frozen_config.json').read_text());verify(out,cfg)
    with (out/'queue.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((out/n).exists() for n in ('FAILED.json','COMPLETE','OPERATOR_PAUSED.json')):return
        try:run(out,cfg)
        except Exception as exc:
            write(out/'FAILED.json',{'type':type(exc).__name__,'reason':str(exc)})
            event(out,'failed',reason=str(exc));raise


if __name__=='__main__':main()
