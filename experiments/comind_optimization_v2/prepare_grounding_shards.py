#!/usr/bin/env python3
"""Split unfinished grounding rows into deterministic, cost-balanced shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row: dict) -> tuple[str, str, str]:
    return str(row["task"]), str(row["recording_id"]), str(row["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    existing = [json.loads(line) for line in args.existing.read_text().splitlines() if line.strip()]
    done = {key(row) for row in existing if not row.get("error") and isinstance(row.get("prediction"), dict)}
    remaining = [row for row in manifest if key(row) not in done]

    # Image count is a useful deterministic proxy for VLM cost.  Greedy
    # assignment keeps two-view JA/Handover and one-view SCOIA balanced.
    buckets: list[list[dict]] = [[] for _ in range(args.shards)]
    costs = [0 for _ in range(args.shards)]
    for row in sorted(remaining, key=lambda item: (-len(item.get("images", [])), key(item))):
        index = min(range(args.shards), key=lambda value: (costs[value], value))
        buckets[index].append(row)
        costs[index] += len(row.get("images", []))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, rows in enumerate(buckets):
        path = args.output_prefix.with_name(f"{args.output_prefix.name}_{index}.jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        outputs.append({"shard": index, "path": str(path), "rows": len(rows), "image_cost": costs[index]})
    print(json.dumps({"manifest": len(manifest), "already_complete": len(done), "remaining": len(remaining), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
