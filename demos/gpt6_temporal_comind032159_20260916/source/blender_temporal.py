"""Bake editable 2.5D image-plane proxy tracks, NOT metric 3D reconstruction."""
import json
import math
from pathlib import Path
import sys
import bpy

out=Path(sys.argv[sys.argv.index('--')+1]).resolve()
tracks=json.loads((out/'render_interpolation.json').read_text())
registry=json.loads((out/'entity_registry.json').read_text())
render=out/'blender'; render.mkdir(exist_ok=True)
png=render/'frames'; png.mkdir(exist_ok=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.render.engine='BLENDER_EEVEE'; scene.eevee.taa_render_samples=12
scene.render.resolution_x=640; scene.render.resolution_y=640; scene.render.resolution_percentage=100
scene.render.fps=30; scene.frame_start=1; scene.frame_end=2 if '--smoke' in sys.argv else 301
scene.render.image_settings.file_format='PNG'
scene.world.color=(.035,.045,.07)
scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'; scene.view_settings.exposure=0; scene.view_settings.gamma=1

def material(name,color,emission=False):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*color,1); bs.inputs['Roughness'].default_value=.38
    if emission: bs.inputs['Emission'].default_value=(*color,1); bs.inputs['Emission Strength'].default_value=.7
    return m

def cube(name,loc,scale,mat,parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc); ob=bpy.context.object; ob.name=name; ob.scale=scale
    ob.data.materials.append(mat); ob.parent=parent
    return ob

def sphere(name,loc,scale,mat,parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=10,radius=1,location=loc); ob=bpy.context.object; ob.name=name; ob.scale=scale
    ob.data.materials.append(mat); ob.parent=parent
    return ob

def torus(name,r,z,mat,parent):
    bpy.ops.mesh.primitive_torus_add(major_segments=40,minor_segments=8,location=(0,0,z),major_radius=r,minor_radius=.018)
    ob=bpy.context.object; ob.name=name; ob.data.materials.append(mat); ob.parent=parent; return ob

def text(name,body,loc,size,mat):
    cu=bpy.data.curves.new(name,'FONT'); cu.body=body; cu.size=size; cu.align_x='CENTER'
    ob=bpy.data.objects.new(name,cu); scene.collection.objects.link(ob); ob.location=loc; cu.materials.append(mat); return ob

bg=material('background',(.018,.025,.045)); grid=material('grid',(.05,.075,.105)); labelmat=material('labels',(.75,.83,.94),True)
cube('image_plane_background',(0,0,-1),(11.8,11.8,.05),bg)
for i in range(-5,6):
    cube('grid_v',(i,0,-.9),(.012,10,.01),grid); cube('grid_h',(0,i,-.9),(10,.012,.01),grid)

colors={'work_counter':(.08,.15,.20),'red_colander_counter':(.93,.045,.095),'red_bowl_drawer':(.80,.12,.05),'nested_metal_bowl':(.57,.65,.74),'vegetable_scrap':(.18,.85,.12),'cabbage_head':(.20,.60,.13),'drawer_lower':(.40,.25,.11),'wearer_left_hand':(.90,.57,.25),'wearer_right_hand':(.92,.63,.33),'other_person':(.08,.19,.34),'other_person_bracelet_hand':(.32,.67,.94),'other_person_other_hand':(.20,.53,.79)}
short={'work_counter':'worktop','red_colander_counter':'RED COLANDER','red_bowl_drawer':'RED DRAWER BOWL','nested_metal_bowl':'METAL BOWL','vegetable_scrap':'small remnant','cabbage_head':'whole cabbage','drawer_lower':'drawer','wearer_left_hand':'wearer L','wearer_right_hand':'wearer R','other_person':'other person','other_person_bracelet_hand':'other wristband hand','other_person_other_hand':'other hand'}
roots={}; parts={}; mats={}; labels={}
layers={'work_counter':-.5,'other_person':-.15,'drawer_lower':.02,'red_colander_counter':.12,'red_bowl_drawer':.15,'nested_metal_bowl':.19,'cabbage_head':.2,'vegetable_scrap':.4}
for entity in registry:
    root=bpy.data.objects.new(entity,None); scene.collection.objects.link(root); root['entity_id']=entity; root['coordinate_semantics']='image-plane proxy; not world metric motion'; root['persistent_identity']=True
    roots[entity]=root; m=material(entity,colors[entity]); mats[entity]=m
    before=set(bpy.data.objects)
    if entity=='red_colander_counter':
        for r in [.16,.28,.40,.46]: torus(entity+'_perforation_ring',r,.07,m,root)
        for j in range(24):
            a=2*math.pi*j/24; ob=cube(entity+'_spoke',(.3*math.cos(a),.3*math.sin(a),.055),(.32,.012,.016),m,root); ob.rotation_euler[2]=a
        cube(entity+'_handle',(0,-.47,.06),(.18,.10,.04),m,root)
    elif entity in ['red_bowl_drawer','nested_metal_bowl']:
        sphere(entity+'_dish',(0,0,-.015),(.45,.45,.055),m,root); torus(entity+'_rim',.46,.05,m,root)
    elif entity=='drawer_lower':
        for x,y,sx,sy in [(-.46,0,.045,.95),(.46,0,.045,.95),(0,.46,.95,.045),(0,-.46,.95,.045)]: cube(entity+'_wall',(x,y,.05),(sx,sy,.06),m,root)
        cube(entity+'_handle',(0,-.50,.10),(.35,.055,.06),labelmat,root)
    elif entity=='work_counter':
        cube(entity+'_proxy',(0,0,0),(.96,.96,.03),m,root)
    elif entity=='other_person':
        sphere(entity+'_body',(0,-.07,0),(.27,.34,.04),m,root); sphere(entity+'_head',(0,.31,0),(.15,.15,.05),m,root)
    elif entity in ['cabbage_head','vegetable_scrap']:
        sphere(entity+'_leaf',(0,0,0),(.43,.46,.06),m,root)
        for x in [-.2,0,.2]: cube(entity+'_vein',(x,0,.06),(.015,.60,.012),labelmat,root)
    else:
        sphere(entity+'_palm',(0,-.10,0),(.30,.30,.025),m,root)
        for j in range(4): sphere(entity+'_finger',(-.22+j*.15,.19,0),(.06,.28,.024),m,root)
    parts[entity]=list(set(bpy.data.objects)-before)
    labels[entity]=text(entity+'_label',short[entity],(0,0,.8),.12,labelmat)

for frame in tracks['frames']:
    k=frame['render_frame']
    for entity,v in frame['entities'].items():
        ob=roots[entity]; location=v['proxy_location']; label=labels[entity]
        if location is not None:
            ob.location=(location[0],location[1],layers.get(entity,.30)); ob.scale=(*v['proxy_size'],1)
            label.location=(location[0],location[1]+v['proxy_size'][1]/2+.045,.8)
        ob.keyframe_insert('location',frame=k); ob.keyframe_insert('scale',frame=k)
        label.keyframe_insert('location',frame=k)
        for child in parts[entity]+[label]:
            child.hide_render=location is None; child.hide_viewport=location is None
            child.keyframe_insert('hide_render',frame=k); child.keyframe_insert('hide_viewport',frame=k)
        color=(.16,.18,.21) if v['ghost'] else colors[entity]
        bs=mats[entity].node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*color,1); bs.inputs['Base Color'].keyframe_insert('default_value',frame=k)
        ob['last_state']=v['state']; ob['last_visibility']=v['current_visibility']

for ob in bpy.data.objects:
    if ob.animation_data and ob.animation_data.action:
        for curve in ob.animation_data.action.fcurves:
            for kp in curve.keyframe_points: kp.interpolation='CONSTANT' if curve.data_path in ['hide_render','hide_viewport'] else 'LINEAR'
text('title','IMAGE-PLANE PROXIES',(0,5.32,.8),.23,labelmat)
text('boundary','screen motion != physical motion | grey = last-seen ghost',(0,-5.40,.8),.14,labelmat)
bpy.ops.object.camera_add(location=(0,0,16)); camera=bpy.context.object; camera.name='Fixed_display_camera_NOT_recovered_source_camera'; camera.data.type='ORTHO'; camera.data.ortho_scale=11.6; scene.camera=camera
bpy.ops.object.light_add(type='AREA',location=(0,1,9)); light=bpy.context.object; light.data.energy=1100; light.data.size=10
scene['claim']='2.5D image-plane proxies; no metric reconstruction; interpolation not observation; display camera not original camera'
for name in ['temporal_code.py','temporal_code.json','blender_animation_tracks.json','render_interpolation.json','entity_registry.json']:
    tx=bpy.data.texts.new(name); tx.write((out/name).read_text())
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(render/'temporal_scene.blend'))
scene.render.filepath=str(png/'frame_')
bpy.ops.render.render(animation=True)
checks={'frames_rendered':len(list(png.glob('frame_*.png'))),'entities':len(roots),'all_ids_present':set(roots)==set(registry),'metric_3d_claim':False,'animation_fps':30,'render_only_interpolation':True,'scene_editable':True}
assert checks['frames_rendered']==scene.frame_end and checks['all_ids_present']
(render/'render_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
