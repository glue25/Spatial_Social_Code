"""Preserve geometry/frames; fix integer rounding of a fractional custom property."""
import json
from pathlib import Path
import sys
import bpy

out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
ob=bpy.data.objects['drawer_lower']; name='drawer_open_state_proxy'; path='["'+name+'"]'
action=ob.animation_data.action
for curve in list(action.fcurves):
    if curve.data_path==path:action.fcurves.remove(curve)
del ob[name]; ob[name]=-1.0
for frame in json.loads((out/'render_interpolation.json').read_text())['frames']:
    state=frame['entities']['drawer_lower']['state']
    ob[name]=float({'closed':0,'partially_open':.5,'open':1}.get(state,-1))
    ob.keyframe_insert(path,frame=frame['render_frame'])
for curve in action.fcurves:
    if curve.data_path==path:
        curve.update_autoflags(ob)
        for point in curve.keyframe_points:point.interpolation='CONSTANT'
bpy.context.scene.frame_set(1)
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/'blender/temporal_scene.blend'))
