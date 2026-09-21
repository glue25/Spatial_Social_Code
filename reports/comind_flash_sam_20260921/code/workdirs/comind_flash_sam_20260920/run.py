"""Label-blind SAM sidecar: follow immutable Flash SSC results, never rewrite them."""
import base64
import copy
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import urllib.request
import urllib.error
from PIL import Image
import frozen_v2 as V2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)

def identity(ssc, task):
    field = 'action_noun_l1' if task == 'scoia' else 'category_l1'
    candidates = V2.trace_candidates(ssc, task).get(field, [])
    if candidates:
        value, confidence, source = candidates[0]
    else:
        value = ssc['prediction'].get(field)
        confidence, source = None, 'ssc_final_fallback'
    term = str(value or '').strip().replace('_', ' ')
    if not term: term, source = 'task relevant object', 'unknown_fallback'
    return term, {'field': field, 'source': source, 'confidence': confidence, 'candidates': candidates}

def frames(item):
    result = {}
    for im in item['images']:
        if im['kind'] == 'prediction':
            assert im['role'] not in result
            result[im['role']] = Path(im['path'])
    required = ['helper'] if item['task'] == 'scoia' else ['leader', 'helper']
    assert all(role in result for role in required)
    return {role: result[role] for role in required}

def proposals(raw):
    assert isinstance(raw.get('results'), list), 'SAM response lacks results'
    valid = []
    for p in raw['results']:
        try:
            b = [float(v) for v in p['box']]; score = float(p['score'])
            if len(b) != 4 or not all(math.isfinite(v) for v in b + [score]): continue
            x1,y1,x2,y2 = b
            if not (0 <= x1 < x2 <= 1408 and 0 <= y1 < y2 <= 1408): continue
            valid.append({'box_xyxy': b, 'score': score, 'label': p.get('label')})
        except (KeyError, ValueError, TypeError): continue
    return sorted(valid, key=lambda x: -x['score'])

def normalized_box(box, role, task):
    if box is None: return None
    x1,y1,x2,y2 = box
    if task == 'scoia': return [x1,y1,x2-x1,y2-y1]
    return [x1/2+(704 if role=='helper' else 0),y1/2,(x2-x1)/2,(y2-y1)/2]

def augment(base, task, views):
    """Only bbox fields change. Failed/no-detection views receive null boxes."""
    pred = copy.deepcopy(base)
    best = {r: v[0] if v else None for r,v in views.items()}
    def box(role):
        p = best.get(role)
        return normalized_box(p['box_xyxy'], role, task) if p else None
    selected = None
    if task == 'joint_attention': pred.update(left_bbox=box('leader'), right_bbox=box('helper'))
    elif task == 'scoia': pred['bbox'] = box('helper')
    else:
        # Fixed label-free tie break: larger role name, matching historical max tuple.
        ranked = [(p['score'], role) for role,p in best.items() if p]
        selected = max(ranked)[1] if ranked else None
        pred['bbox'] = box(selected) if selected else None
    return pred, selected

def valid_ssc(ssc):
    return isinstance(ssc.get('prediction'), dict) and not ssc.get('error_type') and not ssc.get('error')

def segment(path, term, folder, cfg, expected_hash):
    folder.mkdir(parents=True, exist_ok=True)
    assert sha(path) == expected_hash, 'Frozen prediction frame changed'
    sig = {'image_sha256': expected_hash, 'prompt': term, 'model': cfg['sam_model']}
    request_path = folder/'request.json'
    if request_path.exists(): assert read(request_path) == sig
    else: write(request_path, sig)
    raw_path = folder/'response.json.gz'
    if raw_path.exists():
        return proposals(json.loads(gzip.decompress(raw_path.read_bytes())))
    state_path = folder/'attempts.json'
    state = read(state_path) if state_path.exists() else {'attempts': 0, 'errors': []}
    with Image.open(path) as im:
        assert im.size == (1408,1408), 'Unexpected frame geometry'
        buf=io.BytesIO();im.convert('RGB').save(buf,format='PNG')
    payload=json.dumps({'image_base64':base64.b64encode(buf.getvalue()).decode(), 'text_prompt':term}).encode()
    if state.get('terminal'): raise RuntimeError('Saved terminal SAM failure')
    for attempt in range(state['attempts']+1, cfg['max_attempts_per_view']+1):
        state.update(attempts=attempt, utc=utc());write(state_path,state)
        try:
            req=urllib.request.Request(cfg['sam_url']+'/segment',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=cfg['request_timeout_seconds']) as response:
                raw=json.loads(response.read())
            # Save raw masks/boxes before parsing; restarts cannot resample semantic errors.
            tmp=raw_path.with_suffix('.tmp');tmp.write_bytes(gzip.compress(json.dumps(raw).encode()));tmp.replace(raw_path)
            return proposals(raw)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            state['errors'].append({'attempt':attempt,'type':type(exc).__name__,'utc':utc()})
            permanent=isinstance(exc,urllib.error.HTTPError) and 400<=exc.code<500 and exc.code not in (408,429)
            state['terminal']=permanent or attempt==cfg['max_attempts_per_view'];write(state_path,state)
            if state['terminal']: raise RuntimeError('SAM transport exhausted or permanent failure') from exc
            time.sleep(cfg['retry_delay_seconds'])
        finally:
            time.sleep(cfg['min_interval_seconds'])
    raise RuntimeError('Persisted SAM transport attempts exhausted')

def verify(cfg):
    for p,h in cfg['source_hashes'].items(): assert sha(ROOT/p)==h, p
    assert sha(HERE/'inputs.json')==cfg['inputs_sha256']

def process(item, cfg):
    idx=item['index'];folder=HERE/'artifacts'/f'{idx:04d}'
    upstream=ROOT/item['upstream_result'];raw=read(upstream)
    assert raw['index']==item['index']
    assert all(raw[k]==item[k] for k in ['id','task','recording_id','sample_id'] if k in raw)
    if 'model' in raw: assert raw['model']=='gemini-3-flash-preview'
    snapshot=folder/'upstream.json'
    if snapshot.exists(): assert read(snapshot)==raw, 'Upstream result changed'
    else: write(snapshot,raw)
    ssc=raw.get('conditions',{}).get('ssc',{})
    out={k:item[k] for k in ['index','id','task','recording_id','sample_id']}
    out.update(utc=utc(),upstream_sha256=sha(upstream),base_conditions=raw.get('conditions',{}),
               prediction=None,coordinate_space='official_task_xywh',depth_used=False)
    if not valid_ssc(ssc):
        out.update(state='upstream_ssc_invalid',error_type='UpstreamSSCInvalid');return out
    assert raw.get('model')=='gemini-3-flash-preview'
    term,source=identity(ssc,item['task']);out.update(identity=term,identity_provenance=source)
    views={};errors={}
    for role,path in frames(item).items():
        try: views[role]=segment(path,term,folder/role,cfg,item['frame_hashes'][str(path)])
        except Exception as exc:
            views[role]=[];errors[role]={'type':type(exc).__name__,'message':str(exc)[:300]}
    pred,selected=augment(ssc['prediction'],item['task'],views)
    out.update(prediction=pred,sam_proposals=views,selected_view=selected,view_errors=errors,
               state='sam_failed' if errors else ('no_detection' if any(not v for v in views.values()) else 'complete'))
    # Preserve all non-bbox fields exactly; extra compute changes localization only.
    fields=set(V2.BBOX_FIELDS[item['task']])
    assert {k:v for k,v in pred.items() if k not in fields}=={k:v for k,v in ssc['prediction'].items() if k not in fields}
    return out

def main():
    cfg=read(HERE/'PROTOCOL.json');verify(cfg);items=read(HERE/'inputs.json')
    with (HERE/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            verify(cfg)
            done={int(p.stem) for p in (HERE/'results').glob('*.json')}
            if len(done)==len(items):
                write(HERE/'COMPLETE',{'utc':utc(),'processed':len(done)})
                write(HERE/'status.json',{'utc':utc(),'state':'complete','processed':len(done),'total':len(items),'pid':os.getpid()});return
            available=[i for i in items if i['index'] not in done and (ROOT/i['upstream_result']).exists()]
            if (HERE/'OPERATOR_PAUSED.json').exists(): return
            try:
                with urllib.request.urlopen(cfg['sam_url']+'/health',timeout=5) as r: health=json.loads(r.read())
                assert health.get('ready') and health.get('model')==cfg['sam_model']
            except Exception:
                write(HERE/'status.json',{'utc':utc(),'state':'waiting_for_sam_service','processed':len(done),'available':len(available),'total':len(items),'pid':os.getpid()})
                time.sleep(30);continue
            write(HERE/'status.json',{'utc':utc(),'state':'processing' if available else 'waiting_for_upstream','processed':len(done),'available':len(available),'total':len(items),'pid':os.getpid()})
            for item in available:
                if (HERE/'OPERATOR_PAUSED.json').exists(): return
                verify(cfg)
                write(HERE/'status.json',{'utc':utc(),'state':'sample_running','index':item['index'],'processed':len(done),'total':len(items),'pid':os.getpid()})
                result=process(item,cfg);write(HERE/'results'/f"{item['index']:04d}.json",result);done.add(item['index'])
                print(json.dumps({'utc':utc(),'index':item['index'],'processed':len(done),'state':result['state']}),flush=True)
            time.sleep(cfg['poll_seconds'])

if __name__=='__main__':
    try: main()
    except Exception as exc:
        write(HERE/'FAILED.json',{'utc':utc(),'type':type(exc).__name__,'message':str(exc)[:500]});raise
