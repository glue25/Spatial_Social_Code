#!/usr/bin/env python3
"""Freeze the small EnactToM/HSSD canary used for baseline comparisons."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


TASK_IDS = {
    "standard": {
    # Standard: balanced category, ToM depth 1/2, 2/3 agents, and remote control.
    "emtom-104862639_172226823-618-cooperative-walkthrough-briefing-under-access-and-co-7854fc84",
    "emtom-104862639_172226823-619-cooperative-gallery-reset-with-a-relay-only-reveal-d649a9e6",
    "emtom-107734479_176000442-1526-mixed-courier-prep-with-a-hidden-cabinet-choic-74e271e4",
    "emtom-104348082_171512994-588-cooperative-inspection-staging-with-a-hidden-correct-8487f942",
    "emtom-104348082_171512994-983-mixed-staging-check-jug-placement-cabinet-stat-eccb2531",
    "emtom-108294897_176710602-1566-mixed-safety-prep-with-a-private-checklist-ite-2016757d",
    },
    # Hard: ToM depth 3, both categories, same four scenes to minimize assets.
    "hard": {
    "emtom-104862639_172226823-1013-mixed-quiet-museum-reset-with-a-relayed-drawer-5f2c5fa7",
    "emtom-104862639_172226823-623-mixed-laundry-to-kitchen-reset-with-relayed-fr-38041058",
    "emtom-107734479_176000442-744-mixed-relay-the-correct-family-room-table-for--d901a915",
    "emtom-104348082_171512994-1781-cooperative-tea-reset-with-a-relayed-bedroom-table-c-5e9e0b5c",
    "emtom-108294897_176710602-772-cooperative-exhibit-reset-with-split-object-target-k-41a5f8ab",
    "emtom-108294897_176710602-1166-mixed-guest-reset-with-a-hidden-bedroom-box-an-96b292f9",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", type=Path, default=Path("external_datasets/enacttom"))
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("external/EnactTom/data/datasets/enacttom_episodes/v0_0/train_2k.json.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("workdirs/habitat_enacttom/frozen12"))
    args = parser.parse_args()

    selected = []
    for split in ("standard", "hard"):
        for row in read_jsonl(args.public_dir / f"{split}.jsonl"):
            if row["task_id"] in TASK_IDS[split]:
                frozen_row = {"split": split, **row}
                # The released task files use the historical `emtom` alias,
                # while the current runtime pins the identical domain as
                # `enacttom`. Normalize only the frozen runtime copy.
                if frozen_row.get("pddl_domain") == "emtom":
                    frozen_row["pddl_domain"] = "enacttom"
                    frozen_row["problem_pddl"] = frozen_row["problem_pddl"].replace(
                        "(:domain emtom)", "(:domain enacttom)", 1
                    )
                selected.append(frozen_row)
    found_ids = {split: {row["task_id"] for row in selected if row["split"] == split} for split in TASK_IDS}
    for split in TASK_IDS:
        if found_ids[split] != TASK_IDS[split]:
            raise RuntimeError(
                f"{split} task selection mismatch: missing={sorted(TASK_IDS[split] - found_ids[split])} "
                f"extra={sorted(found_ids[split] - TASK_IDS[split])}"
            )

    with gzip.open(args.episodes, "rt") as handle:
        episode_data = json.load(handle)
    episode_keys = {(str(row["scene_id"]), str(row["episode_id"])) for row in selected}
    episodes = [
        episode
        for episode in episode_data["episodes"]
        if (str(episode["scene_id"]), str(episode["episode_id"])) in episode_keys
    ]
    found_episode_keys = {(str(row["scene_id"]), str(row["episode_id"])) for row in episodes}
    if found_episode_keys != episode_keys:
        raise RuntimeError(f"episode mismatch: missing={sorted(episode_keys - found_episode_keys)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in selected))
    task_dir = args.output_dir / "tasks"
    task_dir.mkdir(exist_ok=True)
    for row in selected:
        task_path = task_dir / f"{row['split']}_{row['scene_id']}_{row['episode_id']}.json"
        task_path.write_text(json.dumps(row, indent=2) + "\n")
    episodes_path = args.output_dir / "episodes.json.gz"
    with gzip.open(episodes_path, "wt") as handle:
        json.dump({"config": episode_data.get("config"), "episodes": episodes}, handle)

    rigid_configs = sorted({rigid[0] for episode in episodes for rigid in episode.get("rigid_objs", [])})
    scene_ids = sorted({str(row["scene_id"]) for row in selected})
    manifest = {
        "purpose": "Small paper-subset canary for equal-budget EnactToM baseline comparisons",
        "selection": "6 standard + 6 hard tasks across four shared scenes; balances standard category and covers ToM depths 1/2/3, 2/3 agents, remote control, room restriction, limited bandwidth, and restricted communication",
        "runtime_compatibility": "Frozen copies normalize the historical pddl domain alias emtom to enacttom; goals, initial states, actions, and annotations are unchanged",
        "task_count": len(selected),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "tom_level_counts": dict(Counter(str(row["tom_level"]) for row in selected)),
        "agent_counts": dict(Counter(str(row.get("num_agents", len(row.get("agent_spawns", {})))) for row in selected)),
        "scene_ids": scene_ids,
        "episode_keys": sorted([list(key) for key in episode_keys]),
        "rigid_object_config_names": rigid_configs,
        "files": {
            "tasks.jsonl": {"sha256": sha256(tasks_path)},
            "episodes.json.gz": {"sha256": sha256(episodes_path)},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
