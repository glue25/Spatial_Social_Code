"""Compile evidence states and separate rendering-only interpolation tracks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read(path): return json.loads(path.read_text())
def write(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def interpolate_tracks(states, registry, render_fps=30):
    """Image-plane proxies only: never infer object world motion from screen motion."""
    samples={e:[] for e in registry}
    for s in states:
        for e in s['entities']:
            b=e['bbox']
            samples[e['id']].append({'relative_seconds':s['relative_seconds'],'evidence_id':s['evidence_id'],
                'bbox':b,'image_center':None if b is None else [(b[0]+b[2])/2,(b[1]+b[3])/2],
                'visibility':e['visibility'],'epistemic_status':e['epistemic_status'],'state':e['state'],
                'world_position':None,'world_velocity':None})
    frames=[]
    for k in range(301):
        t=k/render_fps; lo=min(k//6,50); hi=min(lo+1,50); alpha=(k%6)/6
        values={}
        for entity in registry:
            a,b=samples[entity][lo],samples[entity][hi]
            source=[a['evidence_id']]
            bbox=a['bbox']; mode='sample_proxy' if k%6==0 else 'held_sample'
            if alpha and a['bbox'] is not None and b['bbox'] is not None:
                bbox=[(1-alpha)*x+alpha*y for x,y in zip(a['bbox'],b['bbox'])]
                mode='render_only_linear_interpolation'; source.append(b['evidence_id'])
            if bbox is None:
                previous=next((x for x in reversed(samples[entity][:lo+1]) if x['bbox'] is not None),None)
                bbox=previous['bbox'] if previous else None
                mode='last_seen_ghost_NOT_current_position' if previous else 'never_localized_hidden_proxy'
                if previous: source=[previous['evidence_id'],a['evidence_id']]
            values[entity]={'proxy_location':None if bbox is None else [10*((bbox[0]+bbox[2])/2-.5),10*(.5-(bbox[1]+bbox[3])/2),0],
                            'proxy_size':None if bbox is None else [10*(bbox[2]-bbox[0]),10*(bbox[3]-bbox[1])],
                            'mode':mode,'source_evidence_ids':source,'current_visibility':a['visibility'],'state':a['state'],
                            'physical_existence':'not_deleted','ghost':a['bbox'] is None}
        frames.append({'render_frame':k+1,'relative_seconds':t,'on_observation_grid':k%6==0,'entities':values})
    return {'coordinate_system':'normalized original image plane; x=(u-.5)*10,y=(.5-v)*10,z=renderer_layer',
            'units':'arbitrary proxy units, NOT metres','motion_semantics':'apparent image motion includes camera motion; NOT physical object displacement',
            'metric_3d_recovered':False,'camera_pose_3d':None,'sample_tracks':samples}, {
            'schema':'render_interpolation_v1','fps':30,'new_observations_created':0,'interpolation':'linear only between adjacent localized same-ID observations; no interpolation over missing boxes',
            'visibility_policy':'unlocalized after first sight -> ghost at last seen position, explicitly not current position; never seen -> hidden mesh, persistent ID',
            'frames':frames}


def emit_python(out,data):
    meta={k:v for k,v in data.items() if k not in ['states','social_context']}
    runtime='''# Readable, executable temporal SSC. No API calls and no rendering interpolation.
import json
from pathlib import Path

class TemporalCode:
    def __init__(self, metadata):
        self.data = dict(metadata, social_context=[], states=[])
    def social_context(self, **value): self.data['social_context'].append(value)
    def at(self, **value):
        self.data['states'].append(dict(value, entities=[], relations=[], interactions=[], state_changes=[], social_updates=[]))
    def entity_state(self, **value): self.data['states'][-1]['entities'].append(value)
    def relation(self, **value): self.data['states'][-1]['relations'].append(value)
    def interaction(self, **value): self.data['states'][-1]['interactions'].append(value)
    def state_change(self, **value): self.data['states'][-1]['state_changes'].append(value)
    def write(self, path): path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2)+'\\n')

'''
    lines=[runtime,'T = TemporalCode('+repr(meta)+')']
    for context in data['social_context']: lines.append('T.social_context(**'+repr(context)+')')
    for s in data['states']:
        lines += ['',f"# {s['timestamp_seconds']:.6f}s | source frame {s['source_frame_index_zero_based']} | {s['evidence_id']} (real observation)"]
        lines.append('T.at(**'+repr({k:v for k,v in s.items() if k not in ['entities','relations','interactions','state_changes','social_updates']})+')')
        for field,method in [('entities','entity_state'),('relations','relation'),('interactions','interaction'),('state_changes','state_change')]:
            for row in s[field]: lines.append('T.'+method+'(**'+repr(row)+')')
    lines += ['','if __name__ == "__main__":',"    T.write(Path(__file__).with_name('temporal_code.replayed.json'))"]
    (out/'temporal_code.py').write_text('\n'.join(lines)+'\n')


def camera_diagnostics(out,states,frames):
    """Background image homographies, not a calibrated moving-camera reconstruction."""
    orb=cv2.ORB_create(nfeatures=1800); matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    previous=None; rows=[]
    for state,frame in zip(states,frames):
        im=cv2.imread(str(out/frame['model_image_path'])); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        for e in state['entities']:
            if e['id']=='work_counter' or e['bbox'] is None: continue
            x0,y0,x1,y1=(np.array(e['bbox'])*704).astype(int)
            cv2.rectangle(mask,(max(0,x0-12),max(0,y0-12)),(min(703,x1+12),min(703,y1+12)),0,-1)
        kp,desc=orb.detectAndCompute(gray,mask)
        if previous is not None:
            pk,pd=previous; row={'from_evidence':frames[state['observation_index']-1]['evidence_id'],'to_evidence':frame['evidence_id'],'status':'unknown','homography':None,'metric_camera_pose':None,'applied_to_world_tracks':False}
            if pd is not None and desc is not None and len(desc)>=2:
                pairs=matcher.knnMatch(pd,desc,k=2); good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance]
                row['candidate_matches']=len(good)
                if len(good)>=8:
                    H,inliers=cv2.findHomography(np.float32([pk[m.queryIdx].pt for m in good]),np.float32([kp[m.trainIdx].pt for m in good]),cv2.RANSAC,3)
                    if H is not None:
                        count=int(inliers.sum()); row['inliers']=count
                        if count>=25 and count/len(good)>=.5:
                            row.update(status='inferred_background_image_transform',homography=H.tolist())
            rows.append(row)
        previous=(kp,desc)
    write(out/'camera_motion_diagnostics.json',{'scope':'2D background RANSAC diagnostic only; parallax/occlusion/rolling-shutter can invalidate it; never interpreted as metric camera or object movement','pairs':rows})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); out=a.output.resolve()
    assert (out/'INFERENCE_COMPLETE').exists()
    manifest=read(out/'observations_manifest.json'); registry=read(out/'entity_registry.json'); states=[]; corrections=[]
    for folder in sorted((out/'model_calls').glob('batch_*')):
        assert read(folder/'validated.json')['valid']
        for raw in read(folder/'raw_payload.json')['frames']:
            s=copy.deepcopy(raw); frame=manifest['frames'][s['observation_index']]
            s.update({k:frame[k] for k in ['evidence_id','timestamp_seconds','relative_seconds','source_frame_index_zero_based']})
            s['social_updates']=[]; s['state_changes']=[]
            for e in s['entities']:
                e['evidence_id']=frame['evidence_id']; e['world_position']=None; e['existence']='persistent_identity; no destruction inference'
                if e['visibility'] not in ['visible','partially_visible'] and e['state']!='unknown':
                    corrections.append({'index':s['observation_index'],'id':e['id'],'field':'state','before':e['state'],'after':'unknown','reason':'unseen state must not be silently carried forward'})
                    e['state']='unknown'
            for r in s['relations']+s['interactions']:
                if r['epistemic_status']=='observed' and (s['observation_index'] not in r['evidence_indices'] or max(r['evidence_indices'])>s['observation_index']):
                    corrections.append({'index':s['observation_index'],'field':'epistemic_status','before':'observed','after':'inferred','reason':'uses non-current/future reference evidence','record':copy.deepcopy(r)})
                    r['epistemic_status']='inferred'
            states.append(s)
    assert [s['observation_index'] for s in states]==list(range(51))
    # Only definite visible drawer state changes; no changes at visibility loss.
    previous=None
    for s in states:
        drawer=next(e for e in s['entities'] if e['id']=='drawer_lower')
        if drawer['state'] not in ['open','closed','partially_open'] or drawer['bbox'] is None: continue
        if previous and previous[1]!=drawer['state']:
            s['state_changes'].append({'entity':'drawer_lower','from':previous[1],'to':drawer['state'],'epistemic_status':'inferred','transition_time_interval_seconds':[previous[0]['timestamp_seconds'],s['timestamp_seconds']],'evidence_ids':[previous[0]['evidence_id'],s['evidence_id']],'exact_transition_time':None})
        previous=(s,drawer['state'])
    data={'schema_version':'temporal_spatial_social_code_v1','sample_id':'CoMind032159','mode':'offline retrospective model annotation, not online causal tracker',
          'source_video_sha256':manifest['source_video_sha256'],'sampling_fps':5,'observation_count':51,'coordinates':'normalized image bbox; no metric 3D','entity_registry':registry,
          'annotation_caveat':'GPT-6 visual estimates, not independent ground truth; stable IDs constrained by registry, continuity through occlusion is uncertain',
          'social_context':[{'evidence_id':'TRANSCRIPT_UNTIMED','text':'do you need? I need another big bowl to put the garbage.','epistemic_status':'observed','source':'existing dataset transcript','speaker':None,'timestamp_seconds':None,'scope_seconds':manifest['actual_interval_seconds'],'alignment':'clip-level context only; no word/speaker time alignment','update_policy':'context is not injected as a timestamped event; no new timed social evidence, therefore per-frame social_updates remain empty'}],
          'states':states}
    write(out/'temporal_code.json',data); emit_python(out,data); write(out/'audit_corrections.json',corrections)
    tracks,render=interpolate_tracks(states,registry); write(out/'blender_animation_tracks.json',tracks); write(out/'render_interpolation.json',render)
    evidence={f['evidence_id']:{k:f[k] for k in ['path','sha256','source_frame_index_zero_based','source_frame_number_one_based','pts','time_base','timestamp_seconds','relative_seconds','model_image_path','model_image_sha256']} for f in manifest['frames']}
    evidence['TRANSCRIPT_UNTIMED']=data['social_context'][0]; write(out/'evidence_index.json',evidence)
    events=[]
    for s in states:
        for r in s['interactions']:
            if r['predicate'] in ['unknown','touching','holding','reaching_toward','lifting','moving_with_hand','placing','releasing']:
                indices=r['evidence_indices']; events.append(dict(r,reported_at_seconds=s['timestamp_seconds'],evidence_span_seconds=[manifest['frames'][min(indices)]['timestamp_seconds'],manifest['frames'][max(indices)]['timestamp_seconds']],exact_action_onset_seconds=None))
    write(out/'interaction_evidence.json',{'interpretation':'sample-supported interactions, not exact onset/offset annotations; gaps and camera motion are not filled','records':events})
    camera_diagnostics(out,states,manifest['frames'])
    annotated=out/'annotated_observations'; annotated.mkdir(exist_ok=True)
    colors={'red_colander_counter':'#ff4a58','red_bowl_drawer':'#fca336','nested_metal_bowl':'#c2cbdb','vegetable_scrap':'#9cf361','cabbage_head':'#5ce587'}
    for s,f in zip(states,manifest['frames']):
        with Image.open(out/f['model_image_path']) as im:
            draw=ImageDraw.Draw(im)
            for e in s['entities']:
                if e['bbox'] is None or e['id']=='work_counter': continue
                b=[int(x*704) for x in e['bbox']]; c=colors.get(e['id'],'#62cfea')
                draw.rectangle(b,outline=c,width=2); draw.text((b[0],max(0,b[1]-12)),e['id'],fill=c)
            im.save(annotated/(f['evidence_id']+'.jpg'),quality=94)
    import runpy
    replay=runpy.run_path(str(out/'temporal_code.py'))['T'].data
    assert replay==data,'readable code and JSON differ'
    assert not any(s['social_updates'] for s in states)
    write(out/'compile_validation.json',{'states':51,'persistent_ids':len(registry),'render_frames':301,'readable_python_equals_json':True,'raw_observation_interpolation':False,'metric_coordinates_claimed':False,'timed_social_updates':0,'conservative_corrections':len(corrections)})
    print(json.dumps(read(out/'compile_validation.json')))


if __name__=='__main__':main()
