#!/usr/bin/env python3
"""Blender-side generic generator for normalized task-local worlds."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_geometry import containment_scale, containment_center


def cli_args() -> tuple[Path, Path, int]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 3:
        raise SystemExit("usage: blender -b --python blender_generate.py -- SCENE_JSON OUTPUT_DIR SEED")
    return Path(values[0]).resolve(), Path(values[1]).resolve(), int(values[2])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def material(name: str, color: tuple[float, float, float, float], metallic: float = 0.0, roughness: float = 0.45):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return mat


def tag(obj, owner: str, role: str, entity_type: str = "") -> None:
    obj["ssc_owner"] = owner
    obj["ssc_role"] = role
    if role == "entity":
        obj["ssc_id"] = owner
        obj["ssc_entity_type"] = entity_type


def box(name: str, location, dimensions, mat, owner: str = "", role: str = "part"):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    if owner:
        tag(obj, owner, role)
    return obj


def sphere(name: str, location, radius: float, mat, owner: str = "", role: str = "part"):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    if mat:
        obj.data.materials.append(mat)
    if owner:
        tag(obj, owner, role)
    return obj


def cylinder_between(name: str, start: Vector, end: Vector, radius: float, mat, owner: str):
    delta = end - start
    length = max(delta.length, 0.001)
    midpoint = (start + end) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(delta.normalized())
    obj.data.materials.append(mat)
    tag(obj, owner, "limb")
    return obj


def make_text(name: str, text: str, location, size: float, mat):
    bpy.ops.object.text_add(location=location, rotation=(math.radians(72), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.006
    obj.data.materials.append(mat)
    return obj


def world_aabb(obj) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        [min(point[i] for point in corners) for i in range(3)],
        [max(point[i] for point in corners) for i in range(3)],
    )


def main() -> None:
    scene_path, output_dir, seed = cli_args()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    statements = payload["statements"]
    random.seed(seed)
    clear_scene()

    mats = {
        "floor": material("DefaultFloor", (0.20, 0.22, 0.24, 1.0), roughness=0.8),
        "counter": material("DefaultCounter", (0.48, 0.52, 0.56, 1.0), metallic=0.65, roughness=0.3),
        "wood": material("DefaultDrawer", (0.42, 0.22, 0.08, 1.0), roughness=0.6),
        "red": material("DefaultBowlA", (0.65, 0.04, 0.03, 1.0), roughness=0.35),
        "metal": material("DefaultBowlB", (0.55, 0.60, 0.65, 1.0), metallic=0.9, roughness=0.2),
        "helper": material("DefaultHelper", (0.12, 0.35, 0.85, 1.0), roughness=0.5),
        "partner": material("DefaultPartner", (0.88, 0.30, 0.10, 1.0), roughness=0.5),
        "skin": material("DefaultSkin", (0.72, 0.48, 0.34, 1.0), roughness=0.6),
        "green": material("DefaultVegetable", (0.18, 0.55, 0.08, 1.0), roughness=0.8),
        "text": material("DefaultText", (0.95, 0.95, 0.95, 1.0), roughness=0.5),
    }
    box("DEFAULT_floor", (0, 0, -0.08), (8, 7, 0.16), mats["floor"])

    entity_rows = [row for row in statements if row["kind"] == "entity"]
    entity_types = {row["values"]["entity_id"]: row["values"]["entity_type"] for row in entity_rows}
    states = [row for row in statements if row["kind"] == "state"]
    relations = [row for row in statements if row["kind"] == "relation"]
    poses = [row for row in statements if row["kind"] == "pose"]
    open_states = {row["values"]["entity_id"]: bool(row["values"]["value"]) for row in states if row["values"]["state"] == "open"}
    objects: dict[str, bpy.types.Object] = {}
    groups: dict[str, list[bpy.types.Object]] = {}
    defaults: list[dict] = []
    trace: list[dict] = []

    def register(entity_id: str, obj, parts: list, entity_type: str) -> None:
        objects[entity_id] = obj
        groups[entity_id] = [obj] + parts
        tag(obj, entity_id, "entity", entity_type)
        for part in parts:
            if "ssc_owner" not in part:
                tag(part, entity_id, "part")

    counters = sorted(key for key, value in entity_types.items() if value in {"counter", "surface"})
    for index, entity_id in enumerate(counters):
        obj = box(entity_id, (0, 1.35 + 1.5 * index, 1.0), (4.2, 1.2, 0.18), mats["counter"], entity_id, "entity")
        register(entity_id, obj, [], entity_types[entity_id])
        defaults.append({"entity": entity_id, "field": "metric_size_and_position", "source": "default", "value": {"dimensions": [4.2, 1.2, 0.18], "location": list(obj.location)}})

    drawers = sorted(key for key, value in entity_types.items() if value == "drawer")
    for index, entity_id in enumerate(drawers):
        is_open = open_states.get(entity_id, False)
        # The cabinet origin is at y=0.42. A closed drawer is aligned to that
        # origin; an open drawer is translated outward along the slide axis.
        y = -0.72 if is_open else 0.42
        x = 0.0 + index * 2.4
        cabinet = box(f"{entity_id}__cabinet", (x, 0.42, 0.65), (2.05, 1.10, 0.72), mats["wood"], entity_id, "cabinet")
        bottom = box(entity_id, (x, y, 0.66), (1.85, 1.10, 0.10), mats["wood"], entity_id, "entity")
        left = box(f"{entity_id}__side_l", (x - 0.88, y, 0.84), (0.09, 1.10, 0.42), mats["wood"], entity_id)
        right = box(f"{entity_id}__side_r", (x + 0.88, y, 0.84), (0.09, 1.10, 0.42), mats["wood"], entity_id)
        back = box(f"{entity_id}__back", (x, y + 0.50, 0.84), (1.85, 0.09, 0.42), mats["wood"], entity_id)
        handle = box(f"{entity_id}__handle", (x, y - 0.59, 0.83), (0.60, 0.08, 0.08), mats["metal"], entity_id, "handle")
        interior = box(f"{entity_id}__interior", (x, y - 0.02, 0.84), (1.62, 0.88, 0.34), None, entity_id, "interior")
        interior.hide_render = True
        interior.display_type = "WIRE"
        register(entity_id, bottom, [left, right, back, handle, interior, cabinet], "drawer")
        defaults.append({"entity": entity_id, "field": "drawer_template", "source": "default", "value": {"open_offset_y": -1.14, "closed_offset_y": 0.0, "interior_dimensions": [1.62, 0.88, 0.34]}})

    bowl_ids = sorted(key for key, value in entity_types.items() if value == "bowl")
    for index, entity_id in enumerate(bowl_ids):
        # Template choice is stable under relation interventions. It must not
        # depend on whether the current code happens to place another object
        # inside this bowl, otherwise a support edit would also change shape.
        template_slot = int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest(), 16) % 2
        radius = 0.25 if template_slot == 0 else 0.42
        bowl_material = mats["metal"] if template_slot == 0 else mats["red"]
        major = radius * 0.78
        minor = radius * 0.18
        loc = Vector((-0.65 + index * 0.7, 2.2, 1.35))
        bpy.ops.mesh.primitive_torus_add(major_segments=32, minor_segments=12, location=loc, major_radius=major, minor_radius=minor)
        rim = bpy.context.object
        rim.name = entity_id
        rim.data.materials.append(bowl_material)
        tag(rim, entity_id, "entity", "bowl")
        base = box(f"{entity_id}__base", (loc.x, loc.y, loc.z - radius * 0.20), (radius * 1.35, radius * 1.35, radius * 0.12), bowl_material, entity_id)
        interior = box(f"{entity_id}__interior", (loc.x, loc.y, loc.z), (radius * 1.45, radius * 1.45, radius * 0.55), None, entity_id, "interior")
        interior.hide_render = True
        interior.display_type = "WIRE"
        register(entity_id, rim, [base, interior], "bowl")
        defaults.append({"entity": entity_id, "field": "bowl_template", "source": "default", "value": {"radius": radius, "stable_template_slot": template_slot, "selection_rule": "sha256(entity_id) modulo template count"}})

    agent_ids = sorted(key for key, value in entity_types.items() if value == "agent")
    for index, entity_id in enumerate(agent_ids):
        x = -2.0 if index % 2 == 0 else 2.0
        y = -1.25 - 0.25 * (index // 2)
        mat = mats["helper"] if index % 2 == 0 else mats["partner"]
        torso = box(entity_id, (x, y, 1.05), (0.55, 0.35, 1.05), mat, entity_id, "entity")
        head = sphere(f"{entity_id}__head", (x, y, 1.82), 0.25, mats["skin"], entity_id)
        hand_l = sphere(f"{entity_id}__hand_l", (x - 0.45, y - 0.25, 1.12), 0.11, mats["skin"], entity_id, "hand")
        hand_r = sphere(f"{entity_id}__hand_r", (x + 0.45, y - 0.25, 1.12), 0.11, mats["skin"], entity_id, "hand")
        register(entity_id, torso, [head, hand_l, hand_r], "agent")
        defaults.append({"entity": entity_id, "field": "agent_proxy", "source": "default", "value": "box torso, sphere head/hands, cylinder arms; no identity/appearance reconstruction"})

    scraps = sorted(key for key, value in entity_types.items() if value == "vegetable_scrap")
    for index, entity_id in enumerate(scraps):
        obj = sphere(entity_id, (-1.0 + index * 0.3, 1.8, 1.25), 0.12, mats["green"], entity_id, "entity")
        obj.scale = (1.0, 0.55, 0.45)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        register(entity_id, obj, [], "vegetable_scrap")
        defaults.append({"entity": entity_id, "field": "shape_material", "source": "default", "value": "small green ellipsoid"})

    missing_entities = sorted(set(entity_types) - set(objects))
    if missing_entities:
        raise RuntimeError(f"no geometry template for entities {missing_entities}")

    def group_center(entity_id: str) -> Vector:
        return objects[entity_id].location.copy()

    def move_group(entity_id: str, location: Vector) -> None:
        delta = location - group_center(entity_id)
        for obj in groups[entity_id]:
            if obj.name.endswith("__cabinet"):
                continue
            obj.location += delta

    located_by = {}
    for row in relations:
        if row["values"]["relation"] in {"inside", "on_top_of"}:
            subject = row["values"]["subject"]
            if subject in located_by:
                raise RuntimeError(f"conflicting placement constraints for {subject}: {located_by[subject]} and {row['values']}")
            located_by[subject] = row["values"]
    held_targets = {row["values"]["target"] for row in poses if row["values"]["pose"] == "holding"}
    conflict = sorted(held_targets & set(located_by))
    if conflict:
        raise RuntimeError(f"objects cannot be both placed and held: {conflict}")

    for row in relations:
        values = row["values"]
        if values["relation"] != "near":
            continue
        subject, target = values["subject"], values["object"]
        target_location = group_center(target)
        subject_location = group_center(subject)
        current_xy_distance = math.hypot(
            subject_location.x - target_location.x,
            subject_location.y - target_location.y,
        )
        if current_xy_distance <= 2.25:
            operation = "near_satisfied_by_default"
        else:
            direction = -1.0 if subject_location.x <= target_location.x else 1.0
            move_group(subject, target_location + Vector((1.35 * direction, -0.75, 0.25)))
            operation = "near_layout"
        trace.append({"statement_id": row["statement_id"], "operation": operation, "subject": subject, "object": target})

    inside_rows = [row for row in relations if row["values"]["relation"] == "inside"]
    for _ in range(max(1, len(inside_rows) + 1)):
        for row in inside_rows:
            subject, target = row["values"]["subject"], row["values"]["object"]
            interior = bpy.data.objects.get(f"{target}__interior")
            if interior is None:
                raise RuntimeError(f"inside target {target} has no geometric interior")
            # A relation's layout must not depend on its position in the full
            # list: removing/moving another bowl used to shift this container.
            key = json.dumps([subject, target], ensure_ascii=False)
            slot = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 3
            bpy.context.view_layer.update()
            target_min, target_max = world_aabb(interior)
            visible = [obj for obj in groups[subject] if obj.type == 'MESH'
                       and obj.get('ssc_role') not in {'interior', 'cabinet'}]
            def bounds():
                boxes = [world_aabb(obj) for obj in visible]
                return ([min(b[0][i] for b in boxes) for i in range(3)],
                        [max(b[1][i] for b in boxes) for i in range(3)])
            subject_min, subject_max = bounds()
            factor = containment_scale(subject_min, subject_max, target_min, target_max)
            if factor < 1 - 1e-9:
                origin = group_center(subject)
                for obj in groups[subject]:
                    if obj.get('ssc_role') == 'cabinet':
                        continue
                    obj.location = origin + (obj.location - origin) * factor
                    obj.scale *= factor
                defaults.append({'entity': subject, 'field': 'containment_uniform_scale',
                                 'source': 'default', 'value': factor, 'container': target,
                                 'basis': 'proxy AABB fit, NOT image-derived metric geometry'})
                bpy.context.view_layer.update()
                subject_min, subject_max = bounds()
            desired_center = Vector(containment_center(subject_min, subject_max, target_min, target_max,
                                                       offset_x=(slot - 1) * 0.08))
            measured_center = (Vector(subject_min) + Vector(subject_max)) / 2
            move_group(subject, group_center(subject) + desired_center - measured_center)
    for row in inside_rows:
        trace.append({"statement_id": row["statement_id"], "operation": "inside_constraint",
                      "offset_source": "default", "offset_rule": "sha256(JSON([subject,target])) modulo 3; 0.08 metre slot spacing",
                      **row["values"]})

    for row in relations:
        values = row["values"]
        if values["relation"] != "on_top_of":
            continue
        subject, target = values["subject"], values["object"]
        subject_min, subject_max = world_aabb(objects[subject])
        target_min, target_max = world_aabb(objects[target])
        half_height = (subject_max[2] - subject_min[2]) / 2.0
        move_group(subject, Vector((objects[target].location.x, objects[target].location.y, target_max[2] + half_height + 0.01)))
        trace.append({"statement_id": row["statement_id"], "operation": "support_constraint", **values})

    for row in poses:
        values = row["values"]
        agent, target = values["agent"], values["target"]
        if values["pose"] == "reaching_toward":
            hand = bpy.data.objects[f"{agent}__hand_r"]
            handle = bpy.data.objects.get(f"{target}__handle") or objects[target]
            hand.location = handle.location + Vector((0.0, -0.10, 0.06))
        elif values["pose"] == "holding":
            hand = bpy.data.objects[f"{agent}__hand_l"]
            move_group(target, hand.location + Vector((0.0, 0.0, 0.02)))
        trace.append({"statement_id": row["statement_id"], "operation": "pose_constraint", **values})

    for agent in agent_ids:
        torso = objects[agent]
        for side in ("l", "r"):
            hand = bpy.data.objects[f"{agent}__hand_{side}"]
            shoulder = torso.location + Vector(((-0.30 if side == "l" else 0.30), 0.0, 0.35))
            cylinder_between(f"{agent}__arm_{side}", shoulder, hand.location, 0.085, mats["skin"], agent)

    label_z = 2.65
    for label_index, entity_id in enumerate(sorted(objects)):
        # Small deterministic lateral offsets keep labels for nested entities
        # from occupying exactly the same pixels.
        loc = objects[entity_id].location + Vector((((label_index % 3) - 1) * 0.16, 0, 0.55 + 0.08 * (label_index % 2)))
        make_text(f"LABEL_{entity_id}", entity_id, loc, 0.12, mats["text"])
    utterances = [row for row in statements if row["kind"] == "utterance"]
    if utterances:
        text = utterances[0]["values"]["text"].replace("? ", "?\n", 1)
        make_text("SSC_UTTERANCE", f'unknown speaker: "{text}"', (-0.65, 2.45, label_z), 0.14, mats["text"])
    for row in statements:
        if row["kind"] in {"entity", "relation", "state", "pose"}:
            if not any(item.get("statement_id") == row["statement_id"] for item in trace):
                trace.append({"statement_id": row["statement_id"], "operation": f"{row['kind']}_materialized", "values": row["values"]})
        elif row["kind"] in {"utterance", "social", "unknown"}:
            trace.append({"statement_id": row["statement_id"], "operation": f"{row['kind']}_stored_or_rendered", "values": row["values"]})

    scene = bpy.context.scene
    scene["ssc_scene_path"] = str(scene_path)
    scene["ssc_scene_sha256"] = sha256(scene_path)
    scene["ssc_statement_count"] = len(statements)
    scene["ssc_case_id"] = payload.get("case_id", "base")
    # EEVEE produces an editable, fast preview. On display-less servers the
    # command must be wrapped in ``xvfb-run -a`` (documented in README).
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = 900
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.035, 0.045)

    bpy.ops.object.light_add(type="AREA", location=(0, -1.5, 6.0))
    key = bpy.context.object
    key.name = "DEFAULT_key_light"
    key.data.energy = 1100
    key.data.size = 5.0
    bpy.ops.object.light_add(type="AREA", location=(4, 3, 3.0))
    fill = bpy.context.object
    fill.name = "DEFAULT_fill_light"
    fill.data.energy = 650
    fill.data.size = 3.0
    bpy.ops.object.camera_add(location=(6.4, -7.7, 5.7))
    camera = bpy.context.object
    camera.name = "DEFAULT_camera"
    target = Vector((0, 0.15, 1.05))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 48
    scene.camera = camera
    scene.render.filepath = str(output_dir / "render.png")

    snapshot = {}
    for entity_id, obj in sorted(objects.items()):
        lower, upper = world_aabb(obj)
        snapshot[entity_id] = {
            "entity_type": entity_types[entity_id],
            "location": [round(float(value), 6) for value in obj.location],
            "aabb_min": [round(float(value), 6) for value in lower],
            "aabb_max": [round(float(value), 6) for value in upper],
        }
    generation = {
        "schema_version": "ssc_blender_generation_v1",
        "accepted": len(trace) == len(statements),
        "case_id": payload.get("case_id", "base"),
        "seed": seed,
        "input_scene": str(scene_path),
        "input_scene_sha256": sha256(scene_path),
        "supported_statements": len(statements),
        "executed_statement_ids": sorted(item["statement_id"] for item in trace),
        "unhandled_statement_ids": sorted({row["statement_id"] for row in statements} - {item["statement_id"] for item in trace}),
        "operations": trace,
        "defaults": defaults + [
            {"field": "camera", "source": "default", "value": list(camera.location)},
            {"field": "lighting", "source": "default", "value": "two area lights"},
            {"field": "floor", "source": "default", "value": "8m x 7m neutral plane"},
        ],
        "claim_boundary": "constraint layout and proxy geometry only; not metric reconstruction or physics validation",
    }
    normalized_text = bpy.data.texts.new("SSC_NORMALIZED_SCENE.json")
    normalized_text.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    trace_text = bpy.data.texts.new("SSC_GENERATION_TRACE.json")
    trace_text.write(json.dumps(generation, indent=2, ensure_ascii=False) + "\n")
    source_code_path = scene_path.parent / "world_code.py"
    if source_code_path.is_file():
        code_text = bpy.data.texts.new("SSC_WORLD_CODE.py")
        code_text.write(source_code_path.read_text(encoding="utf-8"))
    (output_dir / "generation_trace.json").write_text(json.dumps(generation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "geometry_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "scene.blend"))
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"accepted": generation["accepted"], "output": str(output_dir), "entities": len(objects), "statements": len(statements)}, indent=2))


if __name__ == "__main__":
    main()
