#!/usr/bin/env python3
"""Engineering acceptance gate for one real closed-loop rollout package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "manifest.json", "events.jsonl", "state_before_after.jsonl", "plans.jsonl",
    "physical_checks.jsonl", "predictions.jsonl", "usage.jsonl",
    "labels_SCORER_ONLY.jsonl", "summary.json",
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--allow-empty-frames", action="store_true")
    args = parser.parse_args()
    root = args.artifact_dir
    errors = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        print(json.dumps({"accepted": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    manifest = json.loads((root / "manifest.json").read_text())
    plans = read_jsonl(root / "plans.jsonl")
    states = read_jsonl(root / "state_before_after.jsonl")
    checks = read_jsonl(root / "physical_checks.jsonl")
    predictions_text = (root / "predictions.jsonl").read_text()
    condition = manifest.get("condition")
    planner_inputs = [row for row in plans if row.get("phase") == "planner_input"]
    planner_outputs = [row for row in plans if row.get("phase") == "planner_output"]
    if not planner_inputs or not planner_outputs:
        errors.append("missing planner input/output trace")
    if condition == "direct" and any(row.get("state_included") for row in planner_inputs):
        errors.append("Direct received SSC state")
    if condition == "direct" and any(row.get("runtime_state_block_count", 0) for row in planner_outputs):
        errors.append("Direct actual prompt contains SSC runtime state")
    if condition in {"static_ssc", "dynamic_ssc"} and not all(row.get("state_included") for row in planner_inputs):
        errors.append("SSC condition has planner input without state")
    if condition in {"static_ssc", "dynamic_ssc"} and not all(
        row.get("runtime_state_block_count") == 1 for row in planner_outputs
    ):
        errors.append("SSC actual prompt must contain exactly one runtime state block")
    if condition == "dynamic_ssc":
        if not any(row.get("turn", 0) > 0 and row.get("reason") != "initial_private_compile" for row in states):
            errors.append("Dynamic SSC has no online state updates")
        if not all(row.get("state_marker_found_in_actual_prompt") for row in planner_outputs):
            errors.append("latest state marker missing from actual planner prompt")
    if condition == "static_ssc" and any(row.get("turn", 0) > 0 for row in states):
        errors.append("Static SSC changed after initial compile")
    if not checks:
        errors.append("no strict physical checks")
    if "strict_success" in predictions_text or "strict_goal_checks" in predictions_text:
        errors.append("scorer truth leaked into predictions")

    events = read_jsonl(root / "events.jsonl")
    deliveries = {
        (row.get("turn"), row.get("agent_id")): row
        for row in events
        if row.get("event_type") == "message_delivered"
    }
    for row in events:
        if row.get("event_type") != "action_feedback":
            continue
        action = row.get("action") or {}
        outcome = row.get("outcome") or {}
        if action.get("name") != "Communicate" or not outcome.get("success"):
            continue
        arguments = action.get("arguments") or []
        explicit = [
            value for value in arguments[1:]
            if isinstance(value, str) and value.startswith("agent_")
        ]
        if not explicit:
            continue
        delivered = deliveries.get((row.get("turn"), row.get("agent_id")))
        if delivered is None:
            errors.append(
                f"successful targeted message lacks delivery event: "
                f"{row.get('agent_id')} turn {row.get('turn')}"
            )
            continue
        if sorted(delivered.get("receivers") or []) != sorted(explicit):
            errors.append(
                f"targeted message receiver mismatch: {row.get('agent_id')} "
                f"turn {row.get('turn')}"
            )
    frames = list((root / "frames").rglob("*.png")) if (root / "frames").is_dir() else []
    if not frames and not args.allow_empty_frames:
        errors.append("no real environment frame exported")
    result = {
        "accepted": not errors,
        "condition": condition,
        "planner_inputs": len(planner_inputs),
        "planner_outputs": len(planner_outputs),
        "state_updates": len(states),
        "strict_checkpoints": len(checks),
        "frame_count": len(frames),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
