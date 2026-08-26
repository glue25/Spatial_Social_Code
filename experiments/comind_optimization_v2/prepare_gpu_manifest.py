#!/usr/bin/env python3
"""Freeze a label-free CoMind train manifest for a future GPU grounding run."""

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
    parser.add_argument("--source", type=Path, default=Path("workdirs/spatial_social_optimization_v1/comind_codex_300_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_manifest.jsonl"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    frozen = []
    for row in rows:
        frozen.append({
            "task": row["task"], "recording_id": row["recording_id"], "sample_id": row["sample_id"],
            "recording_fold": fold_for(str(row["recording_id"]), args.folds),
            "images": row["images"], "transcript_10s": row.get("transcript_10s", ""),
            "output_contract": {
                "coordinate_space": "original_frame_pixels_1408x1408",
                "bbox_format": "xyxy",
                "identity": "concrete visible object noun; never replace with a superclass such as food, dishware, container, tool, utensil, or kitchenware",
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(frozen), "labels_in_manifest": False}, indent=2))


if __name__ == "__main__":
    main()
