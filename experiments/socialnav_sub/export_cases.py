#!/usr/bin/env python3
"""Export readable SocialNav pilot cases with images, code, and answers."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "workdirs/socialnav_sub/pilot_v1"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    predictions = rows(PILOT / "jdcloud_pilot/predictions.jsonl")
    # Retries append new records; the most recent row is authoritative.
    latest = {
        (row["scene_id"], row["question_key"], row["condition"]): row
        for row in predictions
    }
    visual = {row["scene_id"]: row for row in rows(PILOT / "jdcloud_pilot/visual_compiled.jsonl")}
    oracle = {row["scene_id"]: row for row in rows(PILOT / "compiled_oracle.jsonl")}
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# SocialNav-SUB pilot cases", ""]
    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        lines.extend(
            [
                f"## {scene_id} ({scene['n_people']} people)",
                "",
                f"- First frame: [sample_with_bev_0.png](../official/{scene_id}/sample_with_bev_0.png)",
                f"- Last frame: [sample_with_bev_9.png](../official/{scene_id}/sample_with_bev_9.png)",
                "",
                "### Frozen visual code",
                "",
                "```json",
                json.dumps(visual.get(scene_id, {}).get("representation"), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Oracle executable code",
                "",
                "```text",
                oracle[scene_id]["spatial_code"],
                oracle[scene_id]["social_code"],
                "```",
                "",
                "### Answers",
                "",
                "| Question | Human mode | Direct | Oracle code | Visual P+S |",
                "|---|---|---|---|---|",
            ]
        )
        question_keys = sorted(
            {key for sid, key, _ in latest if sid == scene_id}
        )
        for question_key in question_keys:
            row = next(latest[(scene_id, question_key, c)] for c in ("direct", "oracle_code", "visual_predicted_ps") if (scene_id, question_key, c) in latest)
            human_mode = max(row["human_distribution"], key=row["human_distribution"].get)
            answers = [
                latest.get((scene_id, question_key, condition), {}).get("answer")
                for condition in ("direct", "oracle_code", "visual_predicted_ps")
            ]
            lines.append(
                f"| {row['question']} | {human_mode} | {answers[0]} | {answers[1]} | {answers[2]} |"
            )
        lines.append("")
    (PILOT / "jdcloud_pilot/cases.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
