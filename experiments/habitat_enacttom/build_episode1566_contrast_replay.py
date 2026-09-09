#!/usr/bin/env python3
"""Build deterministic replay tasks for the episode-1566 SSC/Direct contrast."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


TASK_ID = (
    "emtom-108294897_176710602-1566-"
    "mixed-safety-prep-with-a-private-checklist-ite-2016757d"
)
AUDIT_SPECS = [
    {"target": "spray_bottle_3", "support": "couch_13"},
    {"target": "box_5", "support": "counter_25"},
    {"target": "fridge_34", "support": None},
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", type=Path, required=True)
    parser.add_argument("--planner-log", type=Path, required=True)
    parser.add_argument("--condition", choices=("direct", "physical_social"), required=True)
    parser.add_argument("--seed", type=int, default=891003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_task = json.loads(args.source_task.read_text(encoding="utf-8"))
    planner_log = json.loads(args.planner_log.read_text(encoding="utf-8"))
    if source_task.get("task_id") != TASK_ID:
        raise ValueError("Source task is not episode 1566")
    if planner_log.get("task_id") != TASK_ID:
        raise ValueError("Planner log is not episode 1566")

    cards: dict[str, list[str]] = defaultdict(list)
    responses: dict[str, list[str]] = defaultdict(list)
    turns: dict[str, list[int]] = defaultdict(list)
    for row in planner_log["action_history"]:
        agent_id = row.get("agent_id")
        action = row.get("action")
        if not agent_id or not action:
            continue
        cards[str(agent_id)].append(str(action))
        responses[str(agent_id)].append(str(row.get("result", "")))
        turns[str(agent_id)].append(int(row["turn"]))

    expected_agents = {"agent_0", "agent_1"}
    if set(cards) != expected_agents:
        raise ValueError(f"Unexpected agents: {sorted(cards)}")
    for agent_id, sequence in turns.items():
        if sequence != list(range(1, max(sequence) + 1)):
            raise ValueError(f"Non-contiguous turns for {agent_id}: {sequence}")

    final_turn = max(max(sequence) for sequence in turns.values())
    expected_turn = 20 if args.condition == "direct" else 18
    if final_turn != expected_turn:
        raise ValueError(f"Unexpected final turn: {final_turn} != {expected_turn}")

    replay_task = dict(source_task)
    replay_task.pop("golden_trajectory", None)
    replay_task.pop("golden_trajectory_metadata", None)
    replay_task["coordination_cards"] = dict(cards)
    replay_task["coordination_target_grounding"] = True
    replay_task["coordination_runtime_execute"] = True
    replay_task["coordination_advance_on_attempt"] = True
    replay_task["coordination_replay_responses"] = dict(responses)
    replay_task["replay_audit_specs"] = [
        {"turn": final_turn, **spec} for spec in AUDIT_SPECS
    ]
    replay_task["replay_visualization_only"] = True
    replay_task["replay_provenance"] = {
        "source": "saved learned-agent action_history",
        "episode_id": "1566",
        "repeat": 3,
        "condition": args.condition,
        "seed": args.seed,
        "source_planner_log": args.planner_log.as_posix(),
        "source_planner_log_sha256": digest(args.planner_log),
        "source_task": args.source_task.as_posix(),
        "source_task_sha256": digest(args.source_task),
        "uses_model_generation": False,
        "uses_golden_trajectory": False,
        "preserves_recorded_failed_actions_as_noops": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay_task, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
