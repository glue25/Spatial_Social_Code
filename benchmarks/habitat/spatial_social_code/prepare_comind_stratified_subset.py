"""Create a deterministic recording-stratified subset from frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


TASKS = ("joint_attention", "scoia", "handover")


def stable_key(seed: str, *parts: str) -> str:
    return hashlib.sha256("|".join((seed, *parts)).encode("utf-8")).hexdigest()


def select(rows: list[dict], per_task: int, seed: str) -> list[dict]:
    selected = []
    for task in TASKS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("task") == task:
                groups[str(row["recording_id"])].append(row)
        queues = {}
        for recording_id, values in groups.items():
            values.sort(key=lambda row: stable_key(seed, task, recording_id, str(row["sample_id"])))
            queues[recording_id] = deque(values)
        recording_ids = sorted(queues, key=lambda recording_id: stable_key(seed, task, recording_id))
        task_rows = []
        while len(task_rows) < per_task:
            progressed = False
            for recording_id in recording_ids:
                if queues[recording_id] and len(task_rows) < per_task:
                    task_rows.append(queues[recording_id].popleft())
                    progressed = True
            if not progressed:
                break
        if len(task_rows) != per_task:
            raise ValueError(f"requested {per_task} {task} rows but only selected {len(task_rows)}")
        selected.extend(task_rows)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("workdirs/comind_full_recording"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--seed", default="spatial-social-optimization-v1")
    args = parser.parse_args()
    rows = []
    manifests = sorted(args.manifest_root.glob("*/frames/manifest.jsonl"))
    for manifest in manifests:
        rows.extend(json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    chosen = select(rows, args.per_task, args.seed)
    missing_images = [image["path"] for row in chosen for image in row["images"] if not Path(image["path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} selected frozen frame files are missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen), encoding="utf-8")
    metadata = {
        "protocol": "deterministic SHA-256 order and round-robin recording stratification; no labels read",
        "seed": args.seed,
        "source_manifests": len(manifests),
        "total": len(chosen),
        "by_task": dict(Counter(row["task"] for row in chosen)),
        "recordings_by_task": {task: len({row["recording_id"] for row in chosen if row["task"] == task}) for task in TASKS},
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "missing_images": 0,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
