#!/usr/bin/env python3
"""Compile an SSC environment code into a standalone Habitat episode and task.

The compiler consumes only the code JSON.  It selects already installed assets,
applies explicit code-level poses, and emits a dataset that Habitat can load.
It does not synthesize meshes.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE = ROOT / "reports/cases/habitat_ep1166_ssc_generated_environment.json"
DEFAULT_OUTPUT = ROOT / "reports/code_to_environment/habitat_ep1166_generated_v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_pose(matrix: list[list[float]], dx: float, dz: float, yaw_deg: float) -> list[list[float]]:
    """Apply an explicit planar SSC placement intervention."""
    value = copy.deepcopy(matrix)
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    old = [row[:3] for row in value[:3]]
    yaw = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
    rotated = [
        [sum(yaw[i][k] * old[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    for i in range(3):
        for j in range(3):
            value[i][j] = round(rotated[i][j], 8)
    value[0][3] = round(float(value[0][3]) + dx, 8)
    value[2][3] = round(float(value[2][3]) + dz, 8)
    return value


def instantiate(
    code: dict[str, Any], *, episode_id: str, asset_index: int,
    dx: float, dz: float, yaw_deg: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if code.get("schema_version") != "ssc_asset_linked_environment_v2":
        raise ValueError("SSC code must use schema ssc_asset_linked_environment_v2")
    if not code.get("accepted"):
        raise ValueError("refusing to instantiate an unvalidated SSC code")
    assets = copy.deepcopy(code["spatial_code"]["rigid_asset_instances"])
    if not 0 <= asset_index < len(assets):
        raise ValueError(f"asset_index {asset_index} outside 0..{len(assets)-1}")

    source_transform = copy.deepcopy(assets[asset_index]["transform_4x4"])
    generated_transform = transform_pose(source_transform, dx, dz, yaw_deg)
    assets[asset_index]["transform_4x4"] = generated_transform
    assets[asset_index]["generated_pose_intervention"] = {
        "translation_x_m": dx, "translation_z_m": dz, "yaw_deg": yaw_deg,
    }

    generated_code = copy.deepcopy(code)
    generated_code["schema_version"] = "ssc_generated_environment_instance_v1"
    generated_code["status"] = "compiled_new_episode_instance_using_existing_hssd_assets"
    generated_code["source_code_sha256"] = canonical_hash(code)
    generated_code["generated_episode_id"] = episode_id
    generated_code["spatial_code"]["rigid_asset_instances"] = assets
    generated_code["generation_intervention"] = {
        "type": "explicit_asset_pose",
        "asset_index": asset_index,
        "instance_handle": assets[asset_index]["instance_handle"],
        "source_transform_4x4": source_transform,
        "generated_transform_4x4": generated_transform,
        "translation_x_m": dx,
        "translation_z_m": dz,
        "yaw_deg": yaw_deg,
        "semantic_relations_intended_unchanged": True,
    }

    episode = copy.deepcopy(code["habitat_episode_blueprint"])
    episode["episode_id"] = str(episode_id)
    episode["rigid_objs"] = [
        [asset["config_name"], asset["transform_4x4"]] for asset in assets
    ]
    episode["info"].update({
        "generated_from_ssc": True,
        "ssc_source_code_sha256": generated_code["source_code_sha256"],
        "generation_intervention": generated_code["generation_intervention"],
    })

    task = copy.deepcopy(code["task_blueprint"])
    task["task_id"] = f"sscenv-{episode_id}-code-generated"
    task["title"] = f"SSC-generated environment instance {episode_id}"
    task["episode_id"] = str(episode_id)
    task["runtime_semantics_version"] = "ssc_code_generated_environment_v1"
    task["ssc_generated_environment"] = {
        "source_code_sha256": generated_code["source_code_sha256"],
        "generated_code_sha256": canonical_hash(generated_code),
        "episode_id": str(episode_id),
        "pose_intervention": generated_code["generation_intervention"],
        "new_mesh_synthesis": False,
    }
    for key in (
        "golden_trajectory", "golden_trajectory_metadata", "literal_tom_probes",
        "calibration", "oracle_schedule", "coordination_cards",
        "coordination_replay_responses", "replay_audit_specs",
    ):
        task.pop(key, None)

    manifest = {
        "schema_version": "ssc_to_habitat_compile_manifest_v1",
        "accepted": True,
        "claim": "SSC code compiled to a distinct executable Habitat episode using existing HSSD meshes.",
        "generated_episode_id": str(episode_id),
        "source_scene_id": episode["scene_id"],
        "asset_count": len(assets),
        "pose_changed": source_transform != generated_transform,
        "changed_asset_index": asset_index,
        "changed_instance_handle": assets[asset_index]["instance_handle"],
        "source_transform_sha256": canonical_hash(source_transform),
        "generated_transform_sha256": canonical_hash(generated_transform),
        "source_code_sha256": generated_code["source_code_sha256"],
        "generated_code_sha256": canonical_hash(generated_code),
        "answer_bearing_fields_present": any(
            key in task for key in ("golden_trajectory", "literal_tom_probes", "oracle_schedule")
        ),
        "new_mesh_synthesis": False,
    }
    return generated_code, episode, task, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-id", default="911166")
    parser.add_argument("--asset-index", type=int, default=0)
    parser.add_argument("--dx", type=float, default=-0.08)
    parser.add_argument("--dz", type=float, default=0.06)
    parser.add_argument("--yaw-deg", type=float, default=35.0)
    args = parser.parse_args()

    code_path = args.code.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = json.loads(code_path.read_text(encoding="utf-8"))
    generated_code, episode, task, manifest = instantiate(
        code, episode_id=args.episode_id, asset_index=args.asset_index,
        dx=args.dx, dz=args.dz, yaw_deg=args.yaw_deg,
    )
    code_out = output / "generated_ssc_code.json"
    episodes_out = output / "episodes.json.gz"
    task_out = output / "task.json"
    code_out.write_text(json.dumps(generated_code, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with gzip.open(episodes_out, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": [episode]}, handle, ensure_ascii=False)
    task_out.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest.update({
        "source_code": str(code_path.relative_to(ROOT)),
        "source_code_file_sha256": file_hash(code_path),
        "outputs": {
            "generated_code": str(code_out.relative_to(ROOT)),
            "episodes": str(episodes_out.relative_to(ROOT)),
            "task": str(task_out.relative_to(ROOT)),
        },
        "output_sha256": {
            "generated_code": file_hash(code_out),
            "episodes": file_hash(episodes_out),
            "task": file_hash(task_out),
        },
    })
    manifest_out = output / "compile_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
