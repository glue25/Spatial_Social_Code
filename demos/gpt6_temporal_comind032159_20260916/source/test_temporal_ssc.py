import copy
import json
import runpy

from experiments.astra_temporal_v1.build_temporal import interpolate_tracks, emit_python
from experiments.astra_temporal_v1.infer_temporal import REGISTRY, validate


def sequence():
    return [{'relative_seconds':i/5,'evidence_id':f'F{i}',
             'entities':[{'id':'object','bbox':[.2+i*.005,.2,.3+i*.005,.3], 'visibility':'visible','epistemic_status':'observed','state':'unknown'}]}
            for i in range(51)]


def test_interpolation_is_separate_and_has_no_metric_motion():
    states=sequence(); saved=copy.deepcopy(states)
    tracks,render=interpolate_tracks(states,{'object':'test'})
    assert states==saved
    assert len(render['frames'])==301
    assert render['new_observations_created']==0
    assert sum(r['on_observation_grid'] for r in render['frames'])==51
    assert all(s['world_position'] is None and s['world_velocity'] is None for s in tracks['sample_tracks']['object'])
    assert render['frames'][3]['entities']['object']['mode']=='render_only_linear_interpolation'
    assert render['frames'][3]['entities']['object']['source_evidence_ids']==['F0','F1']


def test_occlusion_never_deleted_or_interpolated_through():
    states=sequence(); states[1]['entities'][0].update(bbox=None,visibility='occluded',epistemic_status='unknown')
    tracks,render=interpolate_tracks(states,{'object':'test'})
    value=render['frames'][9]['entities']['object']
    assert value['physical_existence']=='not_deleted'
    assert value['mode']=='last_seen_ghost_NOT_current_position'
    assert value['ghost']
    assert tracks['sample_tracks']['object'][1]['image_center'] is None
    assert render['frames'][3]['entities']['object']['mode']=='held_sample'


def test_never_seen_entity_has_no_proxy_position():
    states=sequence()
    for s in states:s['entities'][0].update(bbox=None,visibility='unknown')
    _,render=interpolate_tracks(states,{'object':'test'})
    assert all(r['entities']['object']['proxy_location'] is None for r in render['frames'])


def test_readable_python_exactly_replays_epistemic_states(tmp_path):
    data={'schema_version':'test','social_context':[{'text':'untimed','timestamp_seconds':None,'speaker':None}], 'states':[
        {'observation_index':0,'timestamp_seconds':1059.066666,'source_frame_index_zero_based':31772,'evidence_id':'F031772','camera_motion':{'status':'unknown'},'uncertainties':['occlusion'],
         'entities':[{'id':'obj','bbox':None,'epistemic_status':'unknown'}],'relations':[],'interactions':[],'state_changes':[],'social_updates':[]}]}
    emit_python(tmp_path,data)
    assert runpy.run_path(str(tmp_path/'temporal_code.py'))['T'].data==data


def test_identity_and_hidden_coordinate_gates():
    import pytest
    entities=[{'id':k,'bbox':None,'visibility':'unknown'} for k in REGISTRY]
    payload={'frames':[{'observation_index':0,'entities':entities,'relations':[],'interactions':[]}]}
    validate(payload,[0],[0,50])
    entities[0]['bbox']=[0,0,.1,.1]
    with pytest.raises(ValueError,match='hidden entity'):validate(payload,[0],[0,50])


def test_red_objects_and_cabbage_are_distinct_registry_ids():
    assert {'red_colander_counter','red_bowl_drawer','nested_metal_bowl','vegetable_scrap','cabbage_head'} <= set(REGISTRY)
