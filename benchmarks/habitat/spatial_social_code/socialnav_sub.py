"""Query-independent Spatial-Social Code adapter for SocialNav-SUB.

The official dataset stores trajectories in its own pickle-heavy layout.  This
module deliberately consumes a small canonical JSON record so extraction from
that layout and scientific reasoning remain separate and auditable.  No
question or answer choices enter :func:`compile_scene`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime import execute_social, execute_spatial


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompiledSocialNavScene:
    scene_id: str
    source_sha256: str
    spatial_code: str
    spatial_trace: str
    social_code: str
    social_trace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "query_independent": True,
            "source_sha256": self.source_sha256,
            "spatial_code": self.spatial_code,
            "spatial_trace": self.spatial_trace,
            "social_code": self.social_code,
            "social_trace": self.social_trace,
        }


def _safe_id(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip())
    return normalized[:48] or "unknown"


def _literal(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _sample_pose(sample: dict[str, Any], key: str) -> tuple[float, float]:
    pose = sample[key]
    return _finite(pose["x"], f"{key}.x"), _finite(pose["y"], f"{key}.y")


def _relative_label(robot: dict[str, Any], person: dict[str, Any]) -> str:
    rx, ry = _finite(robot["x"], "robot.x"), _finite(robot["y"], "robot.y")
    px, py = _finite(person["x"], "person.x"), _finite(person["y"], "person.y")
    yaw = _finite(robot.get("yaw", 0.0), "robot.yaw")
    dx, dy = px - rx, py - ry
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if abs(forward) >= abs(left):
        return "ahead" if forward >= 0 else "behind"
    return "left" if left >= 0 else "right"


def _distance(robot: dict[str, Any], person: dict[str, Any]) -> float:
    rx, ry = _finite(robot["x"], "robot.x"), _finite(robot["y"], "robot.y")
    px, py = _finite(person["x"], "person.x"), _finite(person["y"], "person.y")
    return math.hypot(px - rx, py - ry)


def _blocks_goal_path(
    robot: dict[str, Any], person: dict[str, Any], goal: dict[str, Any], threshold: float
) -> bool:
    rx, ry = _sample_pose({"robot": robot}, "robot")
    px, py = _sample_pose({"person": person}, "person")
    gx, gy = _finite(goal["x"], "goal.x"), _finite(goal["y"], "goal.y")
    vx, vy = gx - rx, gy - ry
    norm_sq = vx * vx + vy * vy
    if norm_sq <= 1e-9:
        return False
    projection = ((px - rx) * vx + (py - ry) * vy) / norm_sq
    if projection < 0.0 or projection > 1.0:
        return False
    nearest_x, nearest_y = rx + projection * vx, ry + projection * vy
    return math.hypot(px - nearest_x, py - nearest_y) <= threshold


def _canonical_bytes(scene: dict[str, Any]) -> bytes:
    return json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonicalize_official_scene(raw: dict[str, Any], scene_id: str) -> dict[str, Any]:
    """Convert one official ``localized_obj_coords`` record to our JSON schema."""
    robot_poses = raw.get("robot_global_poses")
    trajectories = raw.get("track_id_to_object_trajectory")
    labels = raw.get("tracking_id_to_annotated_label_id", {})
    goal = raw.get("goal_position")
    if not isinstance(robot_poses, list) or len(robot_poses) < 2:
        raise ValueError("official scene has fewer than two robot poses")
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError("official scene has no person trajectories")
    if not isinstance(goal, dict):
        raise ValueError("official scene has no goal position")

    x0 = _finite(robot_poses[0]["center_x"], "robot_global_poses[0].center_x")
    y0 = _finite(robot_poses[0]["center_y"], "robot_global_poses[0].center_y")
    yaw0 = _finite(robot_poses[0].get("yaw", 0.0), "robot_global_poses[0].yaw")

    def robot_local(pose: dict[str, Any]) -> dict[str, float]:
        dx = _finite(pose["center_x"], "robot.center_x") - x0
        dy = _finite(pose["center_y"], "robot.center_y") - y0
        return {
            "x": math.cos(yaw0) * dx + math.sin(yaw0) * dy,
            "y": -math.sin(yaw0) * dx + math.cos(yaw0) * dy,
            "yaw": _finite(pose.get("yaw", yaw0), "robot.yaw") - yaw0,
        }

    people_at_t: dict[int, dict[str, dict[str, float]]] = {
        index: {} for index in range(len(robot_poses))
    }
    for track_id, trajectory in trajectories.items():
        label = labels.get(str(track_id))
        # ``value`` in the official trajectory records is the semantic class
        # (1 == person), not the annotated person identity.  Falling back to
        # it aliases every unused track to person_1 and can make a person
        # appear to disappear and then reappear as a different individual.
        if label is None:
            continue
        person_id = f"person_{label}"
        for record in trajectory or []:
            timestep = int(record["timestep"])
            if timestep not in people_at_t:
                continue
            people_at_t[timestep][person_id] = {
                "x": _finite(record["center_x"], f"{person_id}.center_x"),
                "y": _finite(record["center_y"], f"{person_id}.center_y"),
            }

    samples = [
        {"t": index, "robot": robot_local(pose), "people": people_at_t[index]}
        for index, pose in enumerate(robot_poses)
    ]
    return {
        "scene_id": scene_id,
        "coordinate_frame": "initial_robot_local",
        "goal": {
            "x": _finite(goal["center_x"], "goal.center_x"),
            "y": _finite(goal["center_y"], "goal.center_y"),
        },
        "samples": samples,
        "source_imagepath": str(raw.get("imagepath", "")),
    }


def compile_scene(scene: dict[str, Any], obstruction_threshold_m: float = 0.75) -> CompiledSocialNavScene:
    """Compile one scene without accepting question or answer-option inputs."""
    scene_id = str(scene.get("scene_id", "")).strip()
    samples = scene.get("samples")
    goal = scene.get("goal")
    if not scene_id or not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("scene_id and at least two trajectory samples are required")
    if not isinstance(goal, dict):
        raise ValueError("goal with x/y coordinates is required")

    ordered = sorted(samples, key=lambda row: int(row["t"]))
    people = sorted(
        {person_id for sample in ordered for person_id in sample.get("people", {})}
    )
    if not people:
        raise ValueError("at least one person visible at both endpoints is required")

    spatial_lines: list[str] = []
    social_lines: list[str] = []
    evidence_ids: set[str] = set()
    for person_id in people:
        pid = str(person_id)
        safe_pid = _safe_id(pid)
        visible = [sample for sample in ordered if person_id in sample.get("people", {})]
        first, last = visible[0], visible[-1]
        t0, t1 = int(first["t"]), int(last["t"])
        start_person = first["people"][person_id]
        end_person = last["people"][person_id]
        start_relation = _relative_label(first["robot"], start_person)
        end_relation = _relative_label(last["robot"], end_person)
        start_distance = _distance(first["robot"], start_person)
        end_distance = _distance(last["robot"], end_person)
        delta = end_distance - start_distance
        trend = "same_distance" if abs(delta) < 0.30 else ("farther" if delta > 0 else "closer")
        obstructs = any(
            person_id in sample.get("people", {})
            and _blocks_goal_path(
                sample["robot"], sample["people"][person_id], goal, obstruction_threshold_m
            )
            for sample in ordered
        )

        endpoint_records = (
            (t0, "relative_position", start_relation, "begin"),
            (t1, "relative_position", end_relation, "end"),
            (t1, "distance_change", trend, "distance"),
            (t1, "obstructs_goal_path", str(obstructs).lower(), "obstruction"),
        )
        person_evidence: list[str] = []
        for timestamp, relation, value, suffix in endpoint_records:
            evidence_id = f"sn_{safe_pid}_{suffix}"
            evidence_ids.add(evidence_id)
            person_evidence.append(evidence_id)
            spatial_lines.append(
                f"S.fact({timestamp}, {_literal(relation)}, {_literal(pid)}, {_literal(value)}, "
                f"source=\"trajectory_annotation\", confidence=1.0, evidence_id={_literal(evidence_id)}, "
                f"clip_id={_literal(scene_id)}, observer=\"robot\")"
            )

        evidence_csv = ",".join(person_evidence)
        social_lines.append(
            f"M.observe({t1}, \"robot\", {_literal(f'{pid}: {start_relation} to {end_relation}; {trend}; obstructs={str(obstructs).lower()}')}, "
            f"source=\"spatial_trace\", confidence=1.0, evidence_ids={_literal(evidence_csv)})"
        )
        if obstructs:
            social_lines.append(
                f"M.intent({t1}, \"robot\", {_literal(f'maintain clearance from {pid}')}, confidence=0.7, "
                f"evidence=\"person intersects the shortest robot-goal corridor\", evidence_ids={_literal(person_evidence[-1])})"
            )
        else:
            social_lines.append(f"M.unknown({_literal(pid)}, \"navigation_intent\")")

    spatial_code = "\n".join(spatial_lines)
    social_code = "\n".join(social_lines)
    spatial_trace = execute_spatial(spatial_code)
    social_trace = execute_social(
        social_code, valid_evidence_ids=evidence_ids, require_evidence=False
    )
    return CompiledSocialNavScene(
        scene_id=scene_id,
        source_sha256=hashlib.sha256(_canonical_bytes(scene)).hexdigest(),
        spatial_code=spatial_code,
        spatial_trace=spatial_trace,
        social_code=social_code,
        social_trace=social_trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Canonical scene JSON or JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8").strip()
    scenes = [json.loads(line) for line in text.splitlines()] if args.input.suffix == ".jsonl" else [json.loads(text)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for scene in scenes:
            output.write(json.dumps(compile_scene(scene).to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
