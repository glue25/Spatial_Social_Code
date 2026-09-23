"""Prepare once; isolated engineering gate; recoverable, bounded API evaluation.

The background process must run from OUTPUT/source, whose files are SHA-frozen.
Only transport failures may retry. HTTP-200 model outputs are cached before
validation; invalid/wrong outputs never trigger a new generation.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

from .protocol import (CONDITIONS, HISTORIES, SALT, canonical, digest, entities,
                       make_base, build, audit, prompt, score)
from .metrics import summarize
from experiments.reevaluation_v3.service_gate import service_gate

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'gemini-3-flash-preview'
ENDPOINT = 'https://api2.aigcbest.top/v1/chat/completions'
SOURCES = ['experiments/information_history_v1/'+n for n in
           ('__init__.py','protocol.py','metrics.py','run.py','README.md')]+[
    'experiments/habitat_closed_loop/state_store.py',
    'experiments/reevaluation_v3/service_gate.py',
    'tests/test_information_history_v1.py']
FIELDS = ('world_location','target_location','unrelated_location','target_evidence')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(p)


def prepare(out):
    assert not out.exists(), 'No overwrite of a frozen experiment'
    corpus_paths = [ROOT/'external_datasets/enacttom'/f'{s}.jsonl' for s in ('standard','hard')]
    excluded = {('107734479_176000442','742')}
    exclusion_sources = []
    # All already materialized task definitions in these historical pipelines,
    # not just successful runs. No prediction/score file is consulted.
    roots = ['workdirs/habitat_enacttom', 'workdirs/habitat_closed_loop',
             'workdirs/habitat_memory_controls_20260922_v1/source',
             'workdirs/qdd_four_env_20260917/habitat108_v2/source']
    files = set()
    for root in roots:
        files.update((ROOT/root).glob('**/tasks/*.json'))
        files.update((ROOT/root).glob('mechanism_tasks/*/*.json'))
    for p in sorted(files):
        d = read(p)
        if 'scene_id' in d and 'episode_id' in d:
            key = (str(d['scene_id']),str(d['episode_id'])); excluded.add(key)
            exclusion_sources.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(p),
                                      'episode_key':key})
    candidates = {}; ledger = []
    for p in corpus_paths:
        for n,line in enumerate(p.read_text().splitlines(),1):
            if not line.strip(): continue
            row = json.loads(line); key=(str(row['scene_id']),str(row['episode_id']))
            reason = ('already_materialized_or_manually_inspected' if key in excluded else
                      'insufficient_entities' if len(entities(row,'agent'))<3 or
                      len(entities(row,'object'))<2 or len(entities(row,'furniture'))<3 else 'eligible')
            ledger.append({'task_id':row['task_id'],'episode_key':key,'eligibility':reason,
                           'rank':digest([SALT,row['task_id']]), 'path':str(p.relative_to(ROOT)), 'line':n})
            if reason=='eligible':
                item=(digest([SALT,row['task_id']]), row, p, n)
                if key not in candidates or item[0]<candidates[key][0]: candidates[key]=item
    ranked=sorted(candidates.values(),key=lambda x:x[0]); assert len(ranked)>=50,len(ranked)
    chosen=ranked[:50]; bases=[]; cases=[]
    out.mkdir(parents=True)
    for i,(_,task,p,n) in enumerate(chosen):
        split='engineering' if i<10 else 'formal'
        base=make_base(task,split,i if i<10 else i-10)
        base['source']={'path':str(p.relative_to(ROOT)), 'line':n, 'file_sha256':sha(p),
                        'task_sha256':digest(task)}
        bases.append(base)
        built=[build(base,h) for h in HISTORIES]; assert audit(base,built)
        cases.extend(built)
        write(out/'source_tasks'/f"{base['base_id']}.json",task)
    assert len({(b['scene_id'],b['episode_id']) for b in bases})==50
    source=out/'source'
    for rel in SOURCES:
        dst=source/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,dst)
    credentials=[]; unique=set()
    for name in ('habitat_key','api_key','comind_key','maegoqa_key'):
        p=Path('/tmp/physical_code_qdd_credentials')/name
        if not p.exists() or not p.read_text().strip(): continue
        value=p.read_text().strip(); fingerprint=hashlib.sha256(value.encode()).hexdigest()
        if fingerprint in unique: continue
        unique.add(fingerprint)
        # Use original account lock paths so concurrent legacy processes cooperate.
        gate=ROOT/'workdirs/flash_budget16384_multikey_20260920/locks'/f'{name}.lock'
        credentials.append({'alias':name,'file':str(p),'gate':str(gate)})
    assert credentials, 'No existing provisioned credentials'
    data={'bases':bases,'cases':cases}
    write(out/'dataset.json',data)
    write(out/'selection_audit.json',{'corpus':[{ 'path':str(p.relative_to(ROOT)), 'sha256':sha(p)} for p in corpus_paths],
        'excluded_episode_keys':sorted(excluded),'exclusion_sources':exclusion_sources,'candidate_ledger':ledger,
        'selected_task_ids':[b['task_id'] for b in bases],
        'independence_scope':'Disjoint from all materialized task definitions found under the listed local '
            'historical roots and inspected episode742. No guarantee of globally unseen task vocabulary '
            'or backbone pretraining. These 50 symbolic histories are newly constructed.',
        'selection_rule':'Unique (scene,episode), fixed SHA rank; first10 engineering, next40 formal. '
                         'No model outputs or outcome labels used.'})
    cfg={'schema':1,'created_utc':utc(),'model':MODEL,'endpoint':ENDPOINT,
         'source_root':str(source), 'source_hashes':{rel:sha(source/rel) for rel in SOURCES},
         'data_sha256':sha(out/'dataset.json'), 'selection_sha256':sha(out/'selection_audit.json'),
         'conditions':list(CONDITIONS),'histories':list(HISTORIES),
         'engineering_bases':10,'formal_bases':40,'engineering_calls':160,'formal_calls':640,
         'credentials':credentials,'max_output_tokens':2048,'temperature':0,'reasoning_effort':'low',
         'minimum_key_interval':15,'request_timeout_seconds':180,'transport_attempt_limit':12,
         'retry_backoff_seconds':[15,30,60,120,240,300],
         'max_workers':len(credentials),'bootstrap_draws':10000,
         'engineering_gate':{'valid_output_fraction_at_least':.95,'all_structural_tests':True,
                             'accuracy_threshold':None},
         'retry_policy':'Retry only network/non-200 transient service errors. No retries for cached '
             'HTTP200 outputs, even if incorrect, truncated, or invalid. Transport failures retained.',
         'model_identity':'Require exact provider-reported model string. Not cryptographic backend attestation.',
         'primary_comparisons':['ssc - no_links: paired_joint_correct','ssc - no_attribution: paired_joint_correct'],
         'source_commit_identity':{'kind':'frozen_source_sha256_snapshot','git_commit':None,
             'note':'Not attributed to an unrelated current/historical git commit. Exact run-time sources frozen.'},
         'scope':'Constructed symbolic fixed-record diagnostic using EnactToM entity vocabularies and '
             'StateStore. NOT simulator physics, original task execution, native camera visibility, '
             'visual compilation, original Habitat planner serialization or independent NL summarizer.',
         'information_loss':'no_links preserves all physical and social records but removes their '
             'join. no_attribution nulls holder/sender without swapping identities. Unknown can be '
             'the rational ablated answer; a score gap does not prove equal-information reasoning superiority.',
         'publication_dir':str(ROOT/'reports/information_history_v1_20260923')}
    write(out/'frozen_config.json',cfg)
    write(out/'status.json',{'utc':utc(),'state':'prepared','formal_completed':0,'formal_planned':640})
    print(json.dumps({'prepared':str(out),'eligible_unique_episodes':len(ranked),
                      'engineering':10,'formal':40,'credential_aliases':[x['alias'] for x in credentials]}))


def verify(out):
    cfg=read(out/'frozen_config.json'); source=Path(cfg['source_root'])
    assert ROOT.resolve()==source.resolve(), 'Run only from frozen OUTPUT/source'
    for rel,value in cfg['source_hashes'].items(): assert sha(source/rel)==value,rel
    assert sha(out/'dataset.json')==cfg['data_sha256']
    assert sha(out/'selection_audit.json')==cfg['selection_sha256']
    data=read(out/'dataset.json')
    for b in data['bases']: assert audit(b,[c for c in data['cases'] if c['base_id']==b['base_id']])
    return cfg,data


def parse(raw, case=None):
    assert raw.get('model')==MODEL, 'model_identity_mismatch'
    choices=raw.get('choices',[]); assert len(choices)==1, 'choice_count'
    assert choices[0].get('finish_reason')=='stop', 'not_complete'
    text=choices[0].get('message',{}).get('content'); assert isinstance(text,str), 'no_text'
    prediction=json.loads(text)
    assert isinstance(prediction,dict) and set(prediction)==set(FIELDS), 'field_set'
    assert all(isinstance(prediction[k],str) for k in FIELDS), 'field_types'
    # Wrong location/citation values are semantic errors of THAT field. They must
    # not zero otherwise correct state answers or be counted as transport/format
    # failures. The oracle scorer compares all four fields independently.
    return prediction


def revise_scoring(old,out):
    """Engineering-only, no-call correction preserving every generated answer."""
    assert not out.exists(),'No overwrite'
    oldcfg=read(old/'frozen_config.json')
    assert (old/'PAUSE').exists()
    assert not list((old/'calls/formal').glob('**/request.json')), 'Formal inference already started'
    with (old/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # must have reached quiescence
        assert sha(old/'dataset.json')==oldcfg['data_sha256']
        assert sha(old/'selection_audit.json')==oldcfg['selection_sha256']
        assert sha(ROOT/'experiments/information_history_v1/protocol.py')==oldcfg['source_hashes'][
            'experiments/information_history_v1/protocol.py'], 'No generator/prompt/transform changes permitted'
        out.mkdir(parents=True)
        for name in ('dataset.json','selection_audit.json','preflight.json'):
            shutil.copy2(old/name,out/name)
        for name in ('source_tasks','probes','calls'):
            if (old/name).exists():shutil.copytree(old/name,out/name,
                ignore=shutil.ignore_patterns('result.json','*.tmp'))
        source=out/'source'
        for rel in SOURCES:
            dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        cfg={**oldcfg,'created_utc':utc(),'source_root':str(source),
             'source_hashes':{rel:sha(source/rel) for rel in SOURCES},
             'scoring_revision':2,
             'engineering_cache_parent':{'output':str(old),'config_sha256':sha(old/'frozen_config.json'),
                 'reason':'Semantic citation/location errors now affect their own metric only, not '
                          'syntactic validity or all state scores. JSON/type/truncation failures unchanged.',
                 'no_formal_requests_in_parent':True,'prompts_and_generation_unchanged':True,
                 'cached_http200_resampled':False}}
        write(out/'frozen_config.json',cfg)
        write(out/'status.json',{'utc':utc(),'state':'prepared_scoring_revision',
                                'formal_completed':0,'formal_planned':640})
        write(old/'SUPERSEDED_BY.json',{'output':str(out),'utc':utc(),
            'reason':'Engineering-only semantic-vs-format scoring separation; retain all original artifacts.'})
        print(json.dumps({'revised_output':str(out),'cached_responses':len(list((out/'calls').glob('**/response.json')))}))


class Transport:
    def __init__(self,out,cfg,aliases=None):
        self.out=out;self.cfg=cfg;self.disabled=set();self.mutex=threading.Lock();self.abort=threading.Event()
        self.credentials=[c for c in cfg['credentials'] if aliases is None or c['alias'] in aliases]
        self.keys={c['alias']:Path(c['file']).read_text().strip() for c in self.credentials}

    def body(self,text):
        return {'model':MODEL,'messages':[{'role':'user','content':text}],
                'temperature':self.cfg['temperature'],'max_tokens':self.cfg['max_output_tokens'],
                'reasoning_effort':self.cfg['reasoning_effort'],'response_format':{'type':'json_object'}}

    def request(self,folder,text,preferred=0,probe=False):
        import requests
        folder.mkdir(parents=True,exist_ok=True); body=self.body(text)
        request={'endpoint':ENDPOINT,'body':body,'sha256':digest(body)}
        if (folder/'request.json').exists(): assert read(folder/'request.json')==request
        else: write(folder/'request.json',request)
        if (folder/'response.json').exists(): return read(folder/'response.json'),None
        if (folder/'exhausted.json').exists(): return None,read(folder/'exhausted.json')['reason']
        used=len(list(folder.glob('attempt_*.json')))
        limit=3 if probe else self.cfg['transport_attempt_limit']
        last='service_unavailable'
        for attempt in range(used,limit):
            if self.abort.is_set(): raise RuntimeError('run_aborted')
            if (self.out/'PAUSE').exists(): raise RuntimeError('operator_pause')
            with self.mutex:
                healthy=[c for c in self.credentials if c['alias'] not in self.disabled]
            if not healthy:
                if probe:
                    write(folder/'exhausted.json',{'utc':utc(),'reason':last});return None,last
                raise RuntimeError('all_credentials_service_blocked')
            cred=healthy[(preferred+attempt)%len(healthy)]; alias=cred['alias']; key=self.keys[alias]
            path=folder/f'attempt_{attempt:03d}.json'
            entry={'utc':utc(),'attempt':attempt+1,'credential_alias':alias,'state':'request_started'}
            with service_gate(cred['gate'],self.cfg['minimum_key_interval']):
                if (self.out/'PAUSE').exists(): raise RuntimeError('operator_pause')
                write(path,entry); start=time.monotonic()
                try:
                    response=requests.post(ENDPOINT,json=body,headers={'Authorization':'Bearer '+key},
                        timeout=(20,self.cfg['request_timeout_seconds']),allow_redirects=False)
                    try: raw=response.json()
                    except ValueError: raw={'non_json_response':response.text[:500]}
                    serialized=json.dumps(raw)
                    for secret in self.keys.values(): serialized=serialized.replace(secret,'[REDACTED]')
                    raw=json.loads(serialized)
                    entry.update(state='response_received',http_status=response.status_code,
                                 elapsed_seconds=time.monotonic()-start,response=raw)
                    write(path,entry)
                    if response.status_code==200:
                        write(folder/'response.json',raw)  # BEFORE semantic/format validation
                        return raw,None
                    last='HTTP_'+str(response.status_code)
                    if response.status_code in (401,402,403):
                        with self.mutex: self.disabled.add(alias)
                        write(self.out/'credential_health'/f'{alias}.json',{'utc':utc(),'disabled':True,'reason':last})
                    elif response.status_code not in (408,409,429,500,502,503,504):
                        write(folder/'exhausted.json',{'reason':last,'utc':utc()});return None,last
                except requests.RequestException as exc:
                    last=type(exc).__name__
                    entry.update(state='transport_error',error_type=last,elapsed_seconds=time.monotonic()-start)
                    write(path,entry)
            if attempt+1<limit:
                delay=self.cfg['retry_backoff_seconds'][min(attempt,len(self.cfg['retry_backoff_seconds'])-1)]
                write(folder/'retry_status.json',{'utc':utc(),'next_attempt':attempt+2,'delay_seconds':delay,'reason':last})
                time.sleep(delay)
        write(folder/'exhausted.json',{'utc':utc(),'reason':last})
        return None,last


def probes(out,cfg):
    target=out/'preflight.json'
    if target.exists(): return read(target)['usable_aliases']
    def one(c):
        client=Transport(out,cfg,[c['alias']])
        raw,error=client.request(out/'probes'/c['alias'],
            'Return exactly this JSON object: '+canonical({k:'unknown' for k in FIELDS}),probe=True)
        try: parse(raw or {}); valid=True
        except Exception: valid=False
        return {'alias':c['alias'],'valid':valid,'error':error,
                'provider_reported_model':(raw or {}).get('model')}
    with cf.ThreadPoolExecutor(max_workers=len(cfg['credentials'])) as pool:
        results=list(pool.map(one,cfg['credentials']))
    aliases=[r['alias'] for r in results if r['valid']]
    write(target,{'utc':utc(),'results':results,'usable_aliases':aliases,
                 'selection':'Technical availability before engineering, not answer accuracy.'})
    assert aliases,'No usable same-model endpoint; inspect preflight.json'
    return aliases


def results(out,split):
    return [read(p) for p in sorted((out/'calls'/split).glob('*/*/*/result.json'))]


def run_phase(out,cfg,data,split,aliases):
    bases=[b for b in data['bases'] if b['split']==split]
    client=Transport(out,cfg,aliases)
    def one_base(item):
        i,b=item;case_list=[c for c in data['cases'] if c['base_id']==b['base_id']]
        jobs=[(c,k) for c in case_list for k in CONDITIONS]
        random.Random(b['seed']+1923).shuffle(jobs)
        for c,condition in jobs:
            if client.abort.is_set(): raise RuntimeError('run_aborted')
            folder=out/'calls'/split/b['base_id']/c['history']/condition
            if (folder/'result.json').exists():
                old=read(folder/'result.json')
                assert old['config_sha256']==sha(out/'frozen_config.json')
                assert old['request_sha256']==digest(client.body(prompt(c,condition)))
                continue
            write(out/'workers'/f"{b['base_id']}.json",{'utc':utc(),'state':'request',
                  'base_id':b['base_id'],'history':c['history'],'condition':condition})
            raw,error=client.request(folder,prompt(c,condition),preferred=i)
            identity_ok=bool(raw and raw.get('model')==MODEL)
            try: pred=parse(raw or {},c); valid=True
            except Exception as exc:
                pred={};valid=False
                if error is None: error=type(exc).__name__+':'+str(exc)[:120]
            row={'base_id':b['base_id'],'split':split,'history':c['history'],'condition':condition,
                 'utc':utc(),'valid':valid,'prediction':pred,'scores':score(pred,c,valid),
                 'error':error,'provider_identity_ok':identity_ok,'config_sha256':sha(out/'frozen_config.json'),
                 'request_sha256':digest(client.body(prompt(c,condition))),
                 'artifact_path':str(folder),'source_hashes':cfg['source_hashes']}
            write(folder/'result.json',row)
            print(json.dumps({k:row[k] for k in ('base_id','history','condition','valid','error')}),flush=True)
            if raw is not None and not identity_ok:
                client.abort.set();raise RuntimeError('provider_model_identity_mismatch')
        write(out/'workers'/f"{b['base_id']}.json",{'utc':utc(),'state':'complete','base_id':b['base_id']})
    with cf.ThreadPoolExecutor(max_workers=min(cfg['max_workers'],len(aliases))) as pool:
        pending={pool.submit(one_base,item) for item in enumerate(bases)}
        while pending:
            done,pending=cf.wait(pending,timeout=15,return_when=cf.FIRST_COMPLETED)
            for fut in done: fut.result()
            rr=results(out,split)
            write(out/'status.json',{'utc':utc(),'pid':os.getpid(),'state':split+'_running',
                  'completed':len(rr),'valid':sum(r['valid'] for r in rr),'planned':len(bases)*16,
                  'formal_completed':len(results(out,'formal')),'formal_planned':640})
    rr=results(out,split); assert len(rr)==len(bases)*16
    summary=summarize(bases,rr,cfg['bootstrap_draws']);write(out/f'{split}_summary.json',summary)
    return rr,summary


def costs(out):
    result={}
    for p in out.glob('calls/*/*/*/*/attempt_*.json'):
        _,split,b,h,condition,name=p.relative_to(out).parts
        key=split+'/'+condition
        x=result.setdefault(key,{'actual_request_attempts':0,'input_tokens':0,'output_tokens':0,
                                 'total_tokens':0,'usage_missing_attempts':0,'elapsed_seconds':0.0})
        d=read(p);x['actual_request_attempts']+=1;x['elapsed_seconds']+=d.get('elapsed_seconds',0)
        usage=d.get('response',{}).get('usage')
        if not usage: x['usage_missing_attempts']+=1;continue
        for field,name in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
            value=usage.get(name)
            if isinstance(value,(int,float)):x[field]+=value
    return result


def publish_summary(out,cfg):
    folder=Path(cfg['publication_dir']);folder.mkdir(parents=True,exist_ok=True)
    for name in ('formal_summary.json','engineering_summary.json','engineering_gate.json'):
        if (out/name).exists():shutil.copy2(out/name,folder/name)
    write(folder/'costs.json',costs(out))
    write(folder/'run_manifest.json',{'output':str(out),'config_sha256':sha(out/'frozen_config.json'),
        'data_sha256':cfg['data_sha256'],'selection_sha256':cfg['selection_sha256'],
        'source_hashes':cfg['source_hashes'],'source_commit_identity':cfg['source_commit_identity'],
        'model':MODEL,'scope':cfg['scope'],'conditions':cfg['conditions'],
        'task_ids':[{'base_id':b['base_id'],'task_id':b['task_id'],'scene_id':b['scene_id'],
                     'episode_id':b['episode_id'],'seed':b['seed']} for b in read(out/'dataset.json')['bases']],
        'result_block_to_artifact':'calls/{split}/{base_id}/{history}/{condition}/result.json',
        'information_loss':cfg['information_loss']})


def run(out,probe_only=False):
    cfg,data=verify(out)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        testlog=out/'frozen_tests.log'
        with testlog.open('w') as handle:
            subprocess.run([os.sys.executable,'-m','unittest','discover','-s','tests',
                            '-p','test_information_history_v1.py'],cwd=ROOT,stdout=handle,stderr=handle,check=True)
        aliases=probes(out,cfg)
        if probe_only:
            print(json.dumps({'preflight_usable_aliases':aliases}));return
        start=time.monotonic()
        rr,summary=run_phase(out,cfg,data,'engineering',aliases)
        passed=(len(rr)==160 and sum(r['valid'] for r in rr)/160>=.95 and
                all(r['provider_identity_ok'] for r in rr if r['valid']))
        gate={'utc':utc(),'passed':passed,'completed':len(rr),'valid':sum(r['valid'] for r in rr),
              'test_source_sha256':cfg['source_hashes']['tests/test_information_history_v1.py'],
              'accuracy_used_for_gate':False,'criterion':cfg['engineering_gate']}
        write(out/'engineering_gate.json',gate);publish_summary(out,cfg)
        if not passed:
            write(out/'status.json',{'utc':utc(),'state':'engineering_gate_failed_requires_review',
                                    'formal_completed':0,'formal_planned':640});return
        run_phase(out,cfg,data,'formal',aliases)
        publish_summary(out,cfg)
        write(out/'COMPLETE.json',{'utc':utc(),'formal_final_calls':640,'engineering_final_calls':160,
                                  'wall_seconds_this_invocation':time.monotonic()-start})
        write(out/'status.json',{'utc':utc(),'state':'complete','formal_completed':640,'formal_planned':640})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','probe','summarize','revise-scoring'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--from-output',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(out);return
    if args.action=='revise-scoring':
        assert args.from_output is not None
        revise_scoring(args.from_output.resolve(),out);return
    try:
        if args.action=='summarize':
            cfg,data=verify(out)
            for split in ('engineering','formal'):
                write(out/f'{split}_summary.json',summarize([b for b in data['bases'] if b['split']==split],
                      results(out,split),cfg['bootstrap_draws']))
            publish_summary(out,cfg)
        else:run(out,args.action=='probe')
    except Exception as exc:
        write(out/'status.json',{'utc':utc(),'state':'blocked_review_required',
              'error_type':type(exc).__name__,'error':str(exc)[:300]})
        raise


if __name__=='__main__':main()
