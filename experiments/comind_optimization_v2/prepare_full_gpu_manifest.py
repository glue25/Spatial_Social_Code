#!/usr/bin/env python3
"""Merge the per-recording CoMind frame manifests into one label-free manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fold_for(recording_id: str, folds: int) -> int:
    digest = hashlib.sha256(recording_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("workdirs/comind_full_recording"))
    parser.add_argument("--output", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_full_manifest.jsonl"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = {}
    sources = sorted(args.root.glob("*/frames/manifest.jsonl"))
    for source in sources:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), str(row["recording_id"]), str(row["sample_id"]))
            if key in rows:
                raise ValueError(f"duplicate manifest key: {key}")
            row["recording_fold"] = fold_for(key[1], args.folds)
            row["output_contract"] = {
                "coordinate_space": "original_frame_pixels_1408x1408",
                "bbox_format": "xyxy",
                "identity": "concrete visible object noun; never an abstract superclass",
            }
            rows[key] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(rows[key], ensure_ascii=False) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    by_task = {task: sum(key[0] == task for key in rows) for task in sorted({key[0] for key in rows})}
    print(json.dumps({"sources": len(sources), "samples": len(rows), "by_task": by_task, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
