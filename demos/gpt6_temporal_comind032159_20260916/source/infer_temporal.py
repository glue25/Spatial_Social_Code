"""Fresh GPT-6 visual time-series annotation with immutable evidence and no tools."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import requests

REGISTRY = {
    'work_counter': 'Fixed kitchen worktop; apparent image motion can be egocentric camera motion.',
    'red_colander_counter': 'Red perforated colander/filter basket initially on countertop near sink. NEVER the solid red drawer bowl.',
    'drawer_lower': 'Lower pull-out drawer under worktop near the other person. Out of view before visible is NOT closed.',
    'red_bowl_drawer': 'Solid red bowl in drawer, distinct from countertop perforated colander. Register unknown when not actually visible.',
    'nested_metal_bowl': 'Metallic bowl nested in solid red drawer bowl; not an ordinary white dish on countertop.',
    'vegetable_scrap': 'Small green vegetable/leaf remnant in camera wearer hand; NOT whole cabbage. Identity continuity uncertain if hidden.',
    'cabbage_head': 'Large whole green cabbage, distinct from small remnant.',
    'wearer_left_hand': 'Camera wearer anatomical left hand, generally small green remnant; do not assume holding in every frame.',
    'wearer_right_hand': 'Camera wearer anatomical right hand. Do not swap with other person hands.',
    'other_person': 'Visible dark-shirt person. Anonymous visual track; not linked to any transcript speaker or mental state.',
    'other_person_bracelet_hand': 'Other person hand with visible silver wrist accessory. If identity not visually resolvable use unknown, do not switch IDs.',
    'other_person_other_hand': 'Other person hand without that accessory; use unknown where identity unresolved.',
}


def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def enum(values):
    return {'type':'string','enum':values}


SOURCE = enum(['observed','inferred','unknown'])
TEXT = {'type':'string'}
CONF = {'type':'number','minimum':0,'maximum':1}
IDS = enum(list(REGISTRY))
REFS = {'type':'array','items':{'type':'integer','minimum':0,'maximum':50},'minItems':1}
ENTITY = obj({'id':IDS,'visibility':enum(['visible','partially_visible','occluded','out_of_view','unknown']),
              'bbox':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':4,'maxItems':4}]},
              'state':TEXT,'epistemic_status':SOURCE,'confidence':CONF,'basis':TEXT})
RELATION = obj({'predicate':enum(['on_top_of','inside','near','contact','unknown']), 'subject':IDS,'object':IDS,'epistemic_status':SOURCE,'confidence':CONF,'evidence_indices':REFS,'basis':TEXT})
INTERACTION = obj({'actor':IDS,'predicate':enum(['holding','touching','reaching_toward','lifting','moving_with_hand','placing','releasing','unknown']), 'target':IDS,'epistemic_status':SOURCE,'confidence':CONF,'evidence_indices':REFS,'basis':TEXT})
FRAME = obj({'observation_index':{'type':'integer','minimum':0,'maximum':50},'entities':{'type':'array','items':ENTITY,'minItems':len(REGISTRY),'maxItems':len(REGISTRY)},
             'relations':{'type':'array','items':RELATION},'interactions':{'type':'array','items':INTERACTION},
             'camera_motion':obj({'status':enum(['present','not_detected','unknown']),'basis':TEXT}),
             'uncertainties':{'type':'array','items':TEXT}})
SCHEMA = obj({'frames':{'type':'array','items':FRAME}})

INSTRUCTIONS = """Annotate ACTUAL images in chronological order for an OFFLINE retrospective Spatial-Social temporal demo, not a blind benchmark. Return only JSON matching schema. No tools.
Use the fixed ID registry and output every entity exactly once for each TARGET observation. Reference images/overlap are context only; do not output extra frames.
Image bbox is [x_min,y_min,x_max,y_max] in normalized original square image coordinates (0 left/top, 1 right/bottom), NOT pixels/metres. Tight approximate visible bounds; bbox=null for hidden/out-of-view/uncertain identities. Never invent hidden coordinates. Confidence refers to visual judgement, not calibrated probability.
Distinguish the RED PERFORATED COUNTER COLANDER from SOLID RED DRAWER BOWL and nested METAL bowl, and whole cabbage from small green remnant. They are distinct persistent IDs. Register later-seen entities as unknown/out-of-view earlier, NOT absent/destroyed. Avoid assigning out-of-view if cannot tell occlusion vs framing; unknown is preferred.
Observed = directly supported by that actual frame or explicit adjacent-frame evidence. Inferred = interpretation/continuity assumption. Unknown = cannot determine. Entity state must be unknown unless supported. Drawer state only open/closed/partially_open/unknown; do NOT carry forward closed/open through invisibility. For other entities use brief visible description, not actions/goals.
Track hand-remnant interaction, colander grasp/lift/move/put-down, other person's reach toward drawer and drawer state, BUT DO NOT assume these happen or preassign times. Lifting/moving/placing requires actual adjacent images supporting object movement relative to counter AND hand, not screen displacement alone. Use holding/contact if motion direction unclear. Use evidence_indices listing only supplied images. No exact sub-frame action times.
Camera changes are large. Treat background-wide image shifts as camera effects. Do not convert image bbox displacements to world movement. No 3-D coordinates, no metres, no inferred physical velocity. Relations are recorded only when visually supported, otherwise omit or mark unknown.
Other person hands: distinguish wrist accessory visually; unknown if identity ambiguous. Occlusion/out-of-frame does not remove entity from world. No social/mental/speaker assertions: transcript has no word timestamps or speaker alignment. Do not invent a conversation.
Keep basis short (one precise sentence), include uncertainties and camera judgement. Be conservative rather than filling gaps.
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(payload, targets, allowed):
    frames = payload['frames']
    if [f['observation_index'] for f in frames] != targets:
        raise ValueError('target indices missing, duplicated or reordered')
    for f in frames:
        if len(f['entities']) != len(REGISTRY) or {e['id'] for e in f['entities']} != set(REGISTRY):
            raise ValueError('persistent entity IDs differ')
        for e in f['entities']:
            b=e['bbox']
            if b is not None and not (len(b)==4 and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):
                raise ValueError('invalid image bbox')
            if e['visibility'] not in ['visible','partially_visible'] and b is not None:
                raise ValueError('hidden entity must not invent coordinates')
        for r in f['relations']+f['interactions']:
            if not set(r['evidence_indices']) <= set(allowed):
                raise ValueError('evidence not supplied to model')
    return payload


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); manifest=json.loads((out/'observations_manifest.json').read_text()); frames=manifest['frames']
    calls=out/'model_calls'; calls.mkdir(exist_ok=True)
    (out/'entity_registry.json').write_text(json.dumps(REGISTRY,indent=2)+'\n')
    (out/'annotation_schema.json').write_text(json.dumps(SCHEMA,indent=2)+'\n')
    key=os.environ['PROXY_OPENAI_API_KEY']
    for batch,start in enumerate(range(0,51,9)):
        targets=list(range(start,min(start+9,51)))
        allowed=sorted(set([0,50]+list(range(max(0,start-2),min(start+10,51)))))
        folder=calls/f'batch_{batch:02d}'; folder.mkdir(exist_ok=True)
        prompt=INSTRUCTIONS+'\nID registry:\n'+json.dumps(REGISTRY)+'\nTARGET indices: '+str(targets)+'\nActual evidence index/time/frame mapping: '+json.dumps([{k:frames[i][k] for k in ['observation_index','timestamp_seconds','source_frame_index_zero_based']} for i in allowed])
        (folder/'prompt.txt').write_text(prompt)
        signature=hashlib.sha256((prompt+json.dumps(SCHEMA,sort_keys=True)+''.join(frames[i]['model_image_sha256'] for i in allowed)).encode()).hexdigest()
        meta={'model_requested':'gt-6-as-a','target_indices':targets,'supplied_indices':allowed,'signature':signature,'prompt_sha256':sha(folder/'prompt.txt'),'model_images':[{k:frames[i][k] for k in ['observation_index','model_image_path','model_image_sha256']} for i in allowed],'retrospective_context':True,'tools':[]}
        if (folder/'request_metadata.json').exists():
            assert json.loads((folder/'request_metadata.json').read_text())['signature']==signature
        else: (folder/'request_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        rawpath=folder/'api_response.json'
        if not rawpath.exists():
            content=[{'type':'input_text','text':prompt}]
            for i in allowed:
                path=out/frames[i]['model_image_path']; assert sha(path)==frames[i]['model_image_sha256']
                content += [{'type':'input_text','text':f"ACTUAL observation {i}, source frame {frames[i]['source_frame_index_zero_based']}, time {frames[i]['timestamp_seconds']:.6f}s"}, {'type':'input_image','detail':'original','image_url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()}]
            body={'model':'gt-6-as-a','input':[{'role':'user','content':content}],'tools':[],'store':False,'reasoning':{'effort':'medium'},'max_output_tokens':16384,'text':{'format':{'type':'json_schema','name':'temporal_spatial_code','strict':True,'schema':SCHEMA}}}
            for retry in range(3):
                print(json.dumps({'state':'request_started','batch':batch,'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
                response=requests.post('https://modelservice.jdcloud.com/v1/responses',json=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},timeout=600)
                raw=response.json(); safe=json.loads(json.dumps(raw).replace(key,'[REDACTED]'))
                attempt=folder/f'attempt_{len(list(folder.glob("attempt_*.json")))+1:03d}.json'
                attempt.write_text(json.dumps({'http_status':response.status_code,'response':safe},indent=2)+'\n')
                if response.status_code==200:
                    rawpath.write_text(json.dumps(safe,indent=2)+'\n'); break
                if response.status_code not in [429,500,502,503,504] or retry==2:
                    raise RuntimeError(f'API HTTP{response.status_code}; saved sanitized response')
                time.sleep(120*(retry+1))
        raw=json.loads(rawpath.read_text())
        assert raw.get('model')=='gpt-6-astra' and raw.get('status')=='completed', 'model identity/completion gate'
        assert all(x.get('type') in ['message','reasoning'] for x in raw['output']), 'tool output prohibited'
        text=''.join(c['text'] for x in raw['output'] if x.get('type')=='message' for c in x.get('content',[]) if c.get('type')=='output_text')
        payload=json.loads(text)
        (folder/'raw_payload.json').write_text(json.dumps(payload,indent=2)+'\n')
        validate(payload,targets,allowed)
        (folder/'validated.json').write_text(json.dumps({'valid':True,'frames':targets,'server_model':raw['model'],'usage':raw.get('usage')},indent=2)+'\n')
        print(json.dumps({'state':'batch_complete','batch':batch,'frames':len(targets),'usage':raw.get('usage')}),flush=True)
    (out/'INFERENCE_COMPLETE').write_text('6 GPT-6 temporal batches validated; compile and render still required\n')


if __name__=='__main__':
    main()
