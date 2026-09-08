#!/usr/bin/env python3
"""Export one audited Habitat run without copying raw benchmark media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPATIAL_PREFIX = "[Task-Conditioned Spatial Code]\n"
SOCIAL_PREFIX = "[Task-Conditioned Social Code]\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def extract_codes(task: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for agent, entries in task.get("agent_secrets", {}).items():
        agent_codes: dict[str, str] = {}
        for entry in entries:
            if entry.startswith(SPATIAL_PREFIX):
                agent_codes["spatial_code"] = entry[len(SPATIAL_PREFIX) :]
            elif entry.startswith(SOCIAL_PREFIX):
                agent_codes["social_code"] = entry[len(SOCIAL_PREFIX) :]
        if agent_codes:
            result[agent] = agent_codes
    return result


def result_record(summary: dict[str, Any]) -> dict[str, Any]:
    result = summary["results"][0]
    evaluation = result["evaluation"]
    return {
        "model": summary["model"],
        "llm_provider": summary["llm_provider"],
        "observation_mode": summary["benchmark_observation_mode"],
        "run_mode": result["run_mode"],
        "success": result["success"],
        "functional_success": evaluation["functional_success"],
        "completion_rate": evaluation["percent_complete"],
        "completed_subtasks": evaluation["completed_subtasks"],
        "total_subtasks": evaluation["total_subtasks"],
        "steps": result["steps"],
        "turns": result["turns"],
        "done": result["done"],
        "api_cost_summary": result.get("api_cost_summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--planner-log", type=Path, required=True)
    parser.add_argument("--direct-summary", type=Path, required=True)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--code-output", type=Path, required=True)
    args = parser.parse_args()

    task = load_json(args.task)
    summary = load_json(args.summary)
    planner = load_json(args.planner_log)
    direct_summary = load_json(args.direct_summary)
    manifest = load_json(args.manifest)
    status_records = [
        json.loads(line)
        for line in args.status_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_id = task["task_id"]
    source_ids = {
        task_id,
        summary["results"][0]["task_id"],
        planner["task_id"],
        direct_summary["results"][0]["task_id"],
    }
    if len(source_ids) != 1:
        raise ValueError(f"Source task IDs do not match: {sorted(source_ids)}")
    if task.get("spatial_social_condition") != args.condition:
        raise ValueError("Task condition does not match requested condition")
    matching_status = [
        record
        for record in status_records
        if record.get("repeat") == args.repeat
        and record.get("seed") == args.seed
        and record.get("condition") == args.condition
        and record.get("task") == args.task.stem
    ]
    if len(matching_status) != 1:
        raise ValueError(
            f"Expected one matching status record, found {len(matching_status)}"
        )
    if manifest.get("condition") != args.condition or not any(
        item.get("task_id") == task_id for item in manifest.get("tasks", [])
    ):
        raise ValueError("Compiler manifest does not match task and condition")

    actions = [entry for entry in planner["action_history"] if "agent" in entry]
    communications = [
        {
            "sim_step": entry["sim_step"],
            "turn": entry["turn"],
            "agent": entry["agent"],
            "action": entry["action"],
            "result": entry["result"],
            "delivered": entry["result"].startswith("Message delivered"),
        }
        for entry in actions
        if entry["action"].startswith("Communicate[")
    ]
    frame_paths = [
        frame
        for entry in actions
        for frame in entry.get("selected_frame_paths", [])
    ]
    media_extensions = {".png", ".jpg", ".jpeg", ".mp4", ".webm", ".gif"}
    historical_media = sorted(
        repo_path(path)
        for path in args.summary.parents[1].rglob("*")
        if path.is_file() and path.suffix.lower() in media_extensions
    )

    method_result = result_record(summary)
    case = {
        "schema_version": "habitat_case_export_v1",
        "task_id": task_id,
        "title": task["title"],
        "split": task["split"],
        "scene_id": task["scene_id"],
        "episode_id": task["episode_id"],
        "task_instruction": task["task"],
        "repeat": args.repeat,
        "actual_seed": args.seed,
        "run_status_record": matching_status[0],
        "method_condition": args.condition,
        **method_result,
        "mechanics": {
            "active": task.get("active_mechanics", []),
            "bindings": task.get("mechanic_bindings", []),
        },
        "action_history": planner["action_history"],
        "communication_records": communications,
        "communication_metrics": planner.get("communication_metrics"),
        "paired_direct_same_seed": result_record(direct_summary),
        "visual_artifacts": {
            "historical_media": historical_media,
            "selected_frame_paths": frame_paths,
            "historical_frames_saved": bool(historical_media or frame_paths),
            "action_records_checked": len(actions),
            "replay": {
                "performed": False,
                "status": "blocked_before_simulator_start",
                "reason": "NVIDIA driver unavailable on the export host; no replay frames are claimed or fabricated.",
            },
        },
        "spatial_social_code_file": args.code_output.name,
        "source_provenance": {
            "task": {"path": repo_path(args.task), "sha256": sha256(args.task)},
            "summary": {
                "path": repo_path(args.summary),
                "sha256": sha256(args.summary),
            },
            "planner_log": {
                "path": repo_path(args.planner_log),
                "sha256": sha256(args.planner_log),
            },
            "direct_summary": {
                "path": repo_path(args.direct_summary),
                "sha256": sha256(args.direct_summary),
            },
            "status_log": {
                "path": repo_path(args.status_log),
                "sha256": sha256(args.status_log),
            },
            "compiler_manifest": {
                "path": repo_path(args.manifest),
                "sha256": sha256(args.manifest),
            },
        },
    }
    code = {
        "schema_version": "habitat_task_conditioned_code_v1",
        "task_id": task_id,
        "episode_id": task["episode_id"],
        "repeat": args.repeat,
        "actual_seed": args.seed,
        "method_condition": args.condition,
        "compiler": manifest["compiler"],
        "uses_golden_trajectory": manifest["uses_golden_trajectory"],
        "runtime_execution": manifest["runtime_execution"],
        "privacy": manifest["privacy"],
        "per_agent": extract_codes(task),
        "source_task_sha256": sha256(args.task),
    }

    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    args.code_output.parent.mkdir(parents=True, exist_ok=True)
    args.case_output.write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.code_output.write_text(
        json.dumps(code, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
