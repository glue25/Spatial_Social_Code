"""Read the saved .blend and compare baked tracks to the exported proxy timeline."""
import json
from pathlib import Path
import sys
import bpy

out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
timeline=json.loads((out/'render_interpolation.json').read_text())
scene=bpy.context.scene
assert scene.render.fps==30 and scene.frame_start==1 and scene.frame_end==301
ids=set(json.loads((out/'entity_registry.json').read_text()))
assert all(bpy.data.objects.get(e) is not None for e in ids)
position_checks=0;visibility_checks=0
for frame in timeline['frames']:
    scene.frame_set(frame['render_frame'])
    for entity,value in frame['entities'].items():
        root=bpy.data.objects[entity]
        assert root['persistent_identity']
        assert root['ghost_not_current_position']==int(value['ghost'])
        if value['proxy_location'] is not None:
            assert all(abs(root.location[i]-value['proxy_location'][i])<1e-4 for i in [0,1])
            assert all(abs(root.scale[i]-value['proxy_size'][i])<1e-4 for i in [0,1])
            position_checks+=1
        else:
            assert all(child.hide_render for child in root.children)
        visibility_checks+=1
        if entity=='drawer_lower':
            assert root['drawer_open_state_proxy']=={'closed':0,'partially_open':.5,'open':1}.get(value['state'],-1)
            assert bpy.data.objects['drawer_closed_front_proxy'].hide_render==(value['proxy_location'] is None or value['state']!='closed')
assert json.loads(bpy.data.texts['temporal_code.json'].as_string())==json.loads((out/'temporal_code.json').read_text())
result={'all_passed':True,'render_frames':301,'persistent_entity_ids':sorted(ids),'baked_position_scale_checks':position_checks,'visibility_persistence_checks':visibility_checks,'drawer_state_proxy_checked':True,'embedded_observation_json_exact':True,'claim':'engineering track consistency only; not 3D truth or model annotation accuracy'}
(out/'blender/independent_track_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
