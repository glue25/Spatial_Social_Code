#!/usr/bin/env python3
"""Score the frozen SocialNav router on a completed held-out run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    latest = {
        (row["condition"], row["scene_id"], row["question_key"]): row for row in rows
    }
    selected = []
    for (condition, scene, question), direct in latest.items():
        if condition != "direct":
            continue
        chosen_condition = (
            "visual_predicted_ps"
            if direct["reasoning_group"] == "Social reasoning"
            else "direct"
        )
        selected.append(latest[(chosen_condition, scene, question)])
    agreements = [
        float(row["human_distribution"].get(row.get("answer"), 0.0)) for row in selected
    ]
    modal = [
        row.get("answer") == max(row["human_distribution"], key=row["human_distribution"].get)
        for row in selected
    ]
    result = {
        "router": "visual_predicted_ps for Social reasoning; direct otherwise",
        "frozen_from": "pilot_v2 official metadata slice [0:20]",
        "heldout_slice": "official metadata slice [20:60]",
        "n": len(selected),
        "mean_human_probability_agreement": sum(agreements) / len(agreements),
        "human_mode_accuracy": sum(modal) / len(modal),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
