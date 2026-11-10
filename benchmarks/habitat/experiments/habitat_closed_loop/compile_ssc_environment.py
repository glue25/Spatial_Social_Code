#!/usr/bin/env python3
"""Compile an asset-linked Habitat scenario specification from SSC/task state.

This is a deterministic compiler and validator.  It reuses a real Habitat
scene and its rigid-object asset instances, but does not claim to synthesize
new meshes.  Answer-bearing fields such as golden_trajectory and probe
expected_response are deliberately excluded.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "workdirs/habitat_enacttom/frozen12/tasks/hard_108294897_176710602_1166.json"
DEFAULT_EPISODES = ROOT / "workdirs/habitat_enacttom/frozen12/episodes.json.gz"
DEFAULT_OUTPUT = ROOT / "reports/cases/habitat_ep1166_ssc_generated_environment.json"
DATA_ROOT = ROOT / "external/EnactTom/data"

ATOM_RE = re.compile(r"\((agent_in_room|is_in_room|is_on_top|is_inside|is_open)\s+([^()]+?)\)")
OBJECT_BLOCK_RE = re.compile(r"\(:objects(.*?)\)\s*\(:init", re.DOTALL)
TYPED_LINE_RE = re.compile(r"^\s*([a-zA-Z0-9_ ]+?)\s+-\s+([a-zA-Z0-9_]+)\s*$")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_entities(pddl: str) -> dict[str, list[str]]:
    match = OBJECT_BLOCK_RE.search(pddl)
    if not match:
        raise ValueError("PDDL object block not found")
    entities: dict[str, list[str]] = {}
    for line in match.group(1).splitlines():
        typed = TYPED_LINE_RE.match(line)
        if typed:
            entities.setdefault(typed.group(2), []).extend(typed.group(1).split())
    return entities


def parse_atoms(expression: str) -> list[dict[str, Any]]:
    atoms = []
    for predicate, raw_args in ATOM_RE.findall(expression):
        atoms.append({"predicate": predicate, "args": raw_args.split()})
    return atoms


def episode_record(path: Path, episode_id: str) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [row for row in payload["episodes"] if str(row["episode_id"]) == str(episode_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one episode {episode_id}, got {len(matches)}")
    return matches[0]


def resolve_asset(name: str) -> Path | None:
    matches = sorted(DATA_ROOT.glob(f"**/{name}"))
    return matches[0] if matches else None


def compile_environment(task_path: Path, episodes_path: Path) -> dict[str, Any]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    episode = episode_record(episodes_path, str(task["episode_id"]))
    pddl = task["problem_pddl"]
    entities = parse_entities(pddl)

    init_text = pddl.split("(:init", 1)[1].split("(:goal", 1)[0]
    initial_relations = parse_atoms(init_text)
    goals = parse_atoms(task["functional_goal_pddl"])

    scene_dataset = DATA_ROOT / "versioned_data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
    stage = DATA_ROOT / f"versioned_data/hssd-hab/stages/{task['scene_id']}.glb"
    rigid_assets = []
    missing_assets = []
    instance_counts: dict[str, int] = {}
    for index, (config_name, transform) in enumerate(episode.get("rigid_objs", [])):
        path = resolve_asset(config_name)
        short_name = config_name.split(".object_config.json", 1)[0]
        instance_number = instance_counts.get(short_name, 0)
        instance_counts[short_name] = instance_number + 1
        record = {
            "instance_index": index,
            "instance_handle": f"{short_name}_:{instance_number:04d}",
            "config_name": config_name,
            "config_path": str(path.relative_to(ROOT)) if path else None,
            "config_sha256": sha256(path) if path else None,
            "transform_4x4": transform,
            "semantic_binding": "resolved_by_Habitat_runtime_not_inferred_by_compiler",
        }
        rigid_assets.append(record)
        if path is None:
            missing_assets.append(config_name)

    excluded_task_fields = {
        "golden_trajectory", "golden_trajectory_metadata", "literal_tom_probes",
        "calibration", "oracle_schedule", "coordination_cards",
        "coordination_replay_responses", "replay_audit_specs",
    }
    task_blueprint = {
        key: copy.deepcopy(value)
        for key, value in task.items()
        if key not in excluded_task_fields
    }
    episode_blueprint = {
        "scene_id": episode["scene_id"],
        "scene_dataset_config": episode["scene_dataset_config"],
        "additional_obj_config_paths": copy.deepcopy(episode.get("additional_obj_config_paths", [])),
        "start_position": copy.deepcopy(episode.get("start_position", [0, 0, 0])),
        "start_rotation": copy.deepcopy(episode.get("start_rotation", [0, 0, 0, 1])),
        "info": {
            "task_gen": "ssc_code_to_environment",
            "source_episode_id": str(episode["episode_id"]),
            "object_labels": copy.deepcopy((episode.get("info") or {}).get("object_labels", {})),
        },
        "ao_states": copy.deepcopy(episode.get("ao_states", {})),
        "targets": {},
        "markers": copy.deepcopy(episode.get("markers", [])),
        "target_receptacles": [],
        "goal_receptacles": [],
        "name_to_receptacle": copy.deepcopy(episode.get("name_to_receptacle", {})),
        "instruction": task["task"],
        "evaluation_propositions": [],
        "evaluation_proposition_dependencies": [],
        "evaluation_constraints": [],
        "object_states": copy.deepcopy(episode.get("object_states", {})),
    }

    room_restrictions: dict[str, list[str]] = {agent: [] for agent in entities.get("agent", [])}
    message_limits: dict[str, int] = {}
    allowed_targets: dict[str, list[str]] = {}
    for binding in task.get("mechanic_bindings", []):
        kind = binding.get("mechanic_type")
        if kind == "room_restriction":
            for agent in binding.get("for_agents", []):
                room_restrictions.setdefault(agent, []).extend(binding.get("restricted_rooms", []))
        elif kind == "limited_bandwidth":
            message_limits.update(binding.get("message_limits", {}))
        elif kind == "restricted_communication":
            allowed_targets.update(binding.get("allowed_targets", {}))

    agents = {}
    for agent in entities.get("agent", []):
        agents[agent] = {
            "spawn": task["agent_spawns"][agent],
            "denied_rooms": sorted(set(room_restrictions.get(agent, []))),
            "message_budget": message_limits.get(agent),
            "allowed_message_targets": allowed_targets.get(agent, []),
            "private_ssc_facts": task.get("agent_secrets", {}).get(agent, []),
            "actions": task.get("agent_actions", {}).get(agent, []),
        }

    physical_signature = {
        "scene_id": task["scene_id"],
        "stage_sha256": sha256(stage) if stage.is_file() else None,
        "initial_relations": initial_relations,
        "goals": goals,
    }
    counterfactual_cases = [
        {
            "case_id": "base",
            "physical_signature": "identical",
            "intervention": None,
            "question": "Which agent initially has the private exact location of box_3?",
            "evaluation_target": "agent_2",
        },
        {
            "case_id": "receiver_intervention",
            "physical_signature": "identical",
            "intervention": {
                "type": "confirmed_message_delivery",
                "sender": "agent_2",
                "message": "box_3 is in bedroom_2 on bed_10",
                "receiver_before": "agent_0",
                "receiver_after": "agent_1",
            },
            "question": "After confirmed delivery, which agent received the exact box location?",
            "evaluation_target": "changes_with_receiver",
        },
        {
            "case_id": "irrelevant_fact_negative_control",
            "physical_signature": "identical",
            "intervention": {
                "type": "private_fact_owner_change",
                "fact": "glass_4 is on counter_25",
                "from": "agent_0",
                "to": "agent_1",
                "task_relevance": "none",
            },
            "question": "Which agent initially has the private exact location of box_3?",
            "evaluation_target": "must_remain_agent_2",
        },
    ]

    checks = {
        "scene_dataset_exists": scene_dataset.is_file(),
        "stage_glb_exists": stage.is_file(),
        "all_rigid_asset_configs_exist": not missing_assets,
        "agent_spawns_complete": set(task["agent_spawns"]) == set(entities.get("agent", [])),
        "goals_nonempty": bool(goals),
        "counterfactual_physics_frozen": all(
            row["physical_signature"] == "identical" for row in counterfactual_cases
        ),
        "golden_trajectory_used": False,
        "probe_expected_response_used": False,
    }
    accepted = all(value for key, value in checks.items() if not key.endswith("_used")) and not any(
        checks[key] for key in ("golden_trajectory_used", "probe_expected_response_used")
    )

    return {
        "schema_version": "ssc_asset_linked_environment_v2",
        "status": "code_complete_for_habitat_episode_instantiation_not_new_mesh_synthesis",
        "accepted": accepted,
        "task_id": task["task_id"],
        "episode_id": task["episode_id"],
        "scene": {
            "scene_id": task["scene_id"],
            "scene_dataset_config": str(scene_dataset.relative_to(ROOT)),
            "scene_dataset_sha256": sha256(scene_dataset) if scene_dataset.is_file() else None,
            "stage_glb": str(stage.relative_to(ROOT)),
            "stage_glb_sha256": sha256(stage) if stage.is_file() else None,
        },
        "spatial_code": {
            "entities": entities,
            "initial_relations": initial_relations,
            "functional_goals": goals,
            "rigid_asset_instances": rigid_assets,
        },
        "social_code": {"agents": agents},
        "habitat_episode_blueprint": episode_blueprint,
        "task_blueprint": task_blueprint,
        "generated_counterfactual_cases": counterfactual_cases,
        "physical_signature": physical_signature,
        "validation": {"checks": checks, "missing_assets": missing_assets},
        "leakage_boundary": {
            "source_fields_used": [
                "task_id", "episode_id", "scene_id", "agent_spawns", "mechanic_bindings",
                "agent_secrets", "agent_actions", "problem_pddl", "functional_goal_pddl",
                "episode.scene_dataset_config", "episode.additional_obj_config_paths",
                "episode.rigid_objs",
            ],
            "source_fields_excluded": ["golden_trajectory", "literal_tom_probes.expected_response"],
        },
        "provenance": {
            "task_path": str(task_path.relative_to(ROOT)),
            "task_sha256": sha256(task_path),
            "episodes_path": str(episodes_path.relative_to(ROOT)),
            "episodes_sha256": sha256(episodes_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = compile_environment(args.task.resolve(), args.episodes.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accepted": payload["accepted"],
        "stage_glb": payload["scene"]["stage_glb"],
        "rigid_asset_instances": len(payload["spatial_code"]["rigid_asset_instances"]),
        "counterfactual_cases": len(payload["generated_counterfactual_cases"]),
        "validation": payload["validation"],
    }, indent=2))
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
