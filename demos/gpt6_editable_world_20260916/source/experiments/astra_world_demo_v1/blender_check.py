#!/usr/bin/env python3
"""Independent geometry checks for an already generated SSC Blender scene.

The checker reads expected constraints from normalized scene JSON, but derives
all measured values from Blender object transforms and bounds. It never treats
the input relation string itself as evidence that a constraint was satisfied.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


TOLERANCES = {
    "inside_m": 0.025,
    "support_gap_m": 0.035,
    "holding_distance_m": 0.18,
    "reaching_distance_m": 0.25,
    "near_xy_m": 2.25,
    "open_displacement_m": 0.50,
    "closed_displacement_m": 0.15,
}


def cli_args() -> tuple[Path, Path]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 2:
        raise SystemExit("usage: blender -b scene.blend --python blender_check.py -- SCENE_JSON OUTPUT_JSON")
    return Path(values[0]).resolve(), Path(values[1]).resolve()


def vec(values) -> list[float]:
    return [round(float(value), 6) for value in values]


def aabb(objects) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise ValueError("cannot compute an AABB for an empty object collection")
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def owned(entity_id: str, include_interior: bool = False):
    result = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.get("ssc_owner") != entity_id:
            continue
        if not include_interior and obj.get("ssc_role") in {"interior", "cabinet"}:
            continue
        result.append(obj)
    return result


def entity(entity_id: str):
    matches = [obj for obj in bpy.context.scene.objects if obj.get("ssc_id") == entity_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one primary object for {entity_id!r}, found {len(matches)}")
    return matches[0]


def interior(entity_id: str):
    obj = bpy.data.objects.get(f"{entity_id}__interior")
    if obj is None or obj.type != "MESH":
        raise ValueError(f"entity {entity_id!r} has no geometric interior volume")
    return obj


def distance_xy(a: Vector, b: Vector) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def check_inside(subject: str, target: str) -> tuple[bool, dict]:
    subject_min, subject_max = aabb(owned(subject))
    target_min, target_max = aabb([interior(target)])
    tol = TOLERANCES["inside_m"]
    margins = {
        "x_low": subject_min.x - target_min.x,
        "x_high": target_max.x - subject_max.x,
        "y_low": subject_min.y - target_min.y,
        "y_high": target_max.y - subject_max.y,
        "z_low": subject_min.z - target_min.z,
        "z_high": target_max.z - subject_max.z,
    }
    passed = all(value >= -tol for value in margins.values())
    return passed, {
        "method": "visible subject AABB contained by target interior-volume AABB",
        "subject_aabb": [vec(subject_min), vec(subject_max)],
        "target_interior_aabb": [vec(target_min), vec(target_max)],
        "signed_margins_m": {key: round(value, 6) for key, value in margins.items()},
        "tolerance_m": tol,
    }


def check_on_top(subject: str, target: str) -> tuple[bool, dict]:
    subject_min, subject_max = aabb(owned(subject))
    target_min, target_max = aabb(owned(target))
    gap = subject_min.z - target_max.z
    overlap_x = min(subject_max.x, target_max.x) - max(subject_min.x, target_min.x)
    overlap_y = min(subject_max.y, target_max.y) - max(subject_min.y, target_min.y)
    tol = TOLERANCES["support_gap_m"]
    passed = abs(gap) <= tol and overlap_x > 0 and overlap_y > 0
    return passed, {
        "method": "vertical support gap and horizontal AABB overlap",
        "vertical_gap_m": round(gap, 6),
        "horizontal_overlap_m": [round(overlap_x, 6), round(overlap_y, 6)],
        "gap_tolerance_m": tol,
    }


def check_near(subject: str, target: str) -> tuple[bool, dict]:
    measured = distance_xy(entity(subject).matrix_world.translation, entity(target).matrix_world.translation)
    threshold = TOLERANCES["near_xy_m"]
    return measured <= threshold, {
        "method": "XY distance between primary geometry origins",
        "distance_m": round(measured, 6),
        "maximum_m": threshold,
    }


def check_open(entity_id: str, expected: bool) -> tuple[bool, dict]:
    drawer = entity(entity_id)
    cabinet = bpy.data.objects.get(f"{entity_id}__cabinet")
    if cabinet is None:
        raise ValueError(f"drawer {entity_id!r} has no cabinet reference geometry")
    displacement = abs(float(drawer.matrix_world.translation.y - cabinet.matrix_world.translation.y))
    if expected:
        passed = displacement >= TOLERANCES["open_displacement_m"]
        criterion = f">={TOLERANCES['open_displacement_m']}"
    else:
        passed = displacement <= TOLERANCES["closed_displacement_m"]
        criterion = f"<={TOLERANCES['closed_displacement_m']}"
    return passed, {
        "method": "absolute drawer-to-cabinet displacement along slide axis",
        "displacement_m": round(displacement, 6),
        "criterion_m": criterion,
    }


def check_pose(pose: str, agent: str, target: str) -> tuple[bool, dict]:
    hand_suffix = "hand_l" if pose == "holding" else "hand_r"
    hand = bpy.data.objects.get(f"{agent}__{hand_suffix}")
    if hand is None:
        raise ValueError(f"agent {agent!r} lacks {hand_suffix!r} geometry")
    if pose == "reaching_toward":
        target_obj = bpy.data.objects.get(f"{target}__handle") or entity(target)
        threshold = TOLERANCES["reaching_distance_m"]
    else:
        target_obj = entity(target)
        threshold = TOLERANCES["holding_distance_m"]
    measured = float((hand.matrix_world.translation - target_obj.matrix_world.translation).length)
    return measured <= threshold, {
        "method": "3D distance from designated hand proxy to target geometry origin",
        "hand_object": hand.name,
        "target_object": target_obj.name,
        "distance_m": round(measured, 6),
        "maximum_m": threshold,
    }


def main() -> None:
    scene_path, output_path = cli_args()
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    rows = payload["statements"]
    checks = []
    declared_entities = {
        row["values"]["entity_id"]
        for row in rows
        if row["kind"] == "entity"
    }
    generated_entities = {
        str(obj.get("ssc_id"))
        for obj in bpy.context.scene.objects
        if obj.get("ssc_role") == "entity" and obj.get("ssc_id")
    }
    for entity_id in sorted(declared_entities):
        passed = entity_id in generated_entities
        checks.append({
            "check_id": f"entity:{entity_id}",
            "kind": "entity_presence",
            "passed": passed,
            "measured": {"primary_object_count": 1 if passed else 0},
        })

    for row in rows:
        kind, values = row["kind"], row["values"]
        try:
            if kind == "relation":
                relation = values["relation"]
                if relation == "inside":
                    passed, measured = check_inside(values["subject"], values["object"])
                elif relation == "on_top_of":
                    passed, measured = check_on_top(values["subject"], values["object"])
                elif relation == "near":
                    passed, measured = check_near(values["subject"], values["object"])
                else:
                    raise ValueError(f"unsupported relation checker {relation!r}")
            elif kind == "state" and values["state"] == "open":
                passed, measured = check_open(values["entity_id"], bool(values["value"]))
            elif kind == "pose":
                passed, measured = check_pose(values["pose"], values["agent"], values["target"])
            else:
                continue
            checks.append({
                "check_id": row["statement_id"],
                "kind": kind,
                "expected": values,
                "passed": bool(passed),
                "measured": measured,
                "source": row["source"],
                "evidence_id": row["evidence_id"],
            })
        except Exception as exc:
            checks.append({
                "check_id": row["statement_id"],
                "kind": kind,
                "expected": values,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    failed = [row["check_id"] for row in checks if not row["passed"]]
    result = {
        "schema_version": "ssc_geometry_checks_v1",
        "case_id": payload.get("case_id", "base"),
        "scene_blend": bpy.data.filepath,
        "scope": "proxy-geometry consistency only; not real-world metric reconstruction or physics proof",
        "tolerances": TOLERANCES,
        "declared_entity_count": len(declared_entities),
        "generated_entity_count": len(generated_entities),
        "checked_constraint_count": len(checks) - len(declared_entities),
        "all_passed": not failed,
        "failed_check_ids": failed,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "checks": len(checks), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
