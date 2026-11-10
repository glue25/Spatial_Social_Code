#!/usr/bin/env python3
"""Reproduce the 1566/772 audit boundary without inventing stable samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.habitat_closed_loop.strict_scorer import PhysicalGoal, parse_functional_goals, score_exported_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = args.root / "reports/cases"
    goals = parse_functional_goals("(and (is_on_top spray_bottle_3 couch_13) (is_on_top box_5 counter_25) (is_open fridge_34))")

    def load_1566(prefix):
        rows = {}
        for goal in goals:
            path = cases / f"habitat_ep1566_{prefix}_final_{goal.entity}_state.json"
            rows[goal.entity] = json.loads(path.read_text())
        return score_exported_audit(goals, rows)

    goal_772 = PhysicalGoal("is_on_top", "picture_frame_0", "counter_25")
    row_772 = json.loads((cases / "habitat_ep772_turn14_state.json").read_text())
    result = {
        "status": "historical_single_snapshot_audit_not_new_rollouts",
        "episode_1566_direct": load_1566("direct"),
        "episode_1566_ssc": load_1566("ssc"),
        "episode_772_turn14": score_exported_audit([goal_772], {goal_772.entity: row_772}),
        "interpretation": {
            "episode_1566": "instantaneous physical audit preserves Direct 2/3 versus SSC 3/3; strict stability remains not measured for both",
            "episode_772": "picture frame is still grasped, so strict placement is false even though an above/support query listed the counter",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
