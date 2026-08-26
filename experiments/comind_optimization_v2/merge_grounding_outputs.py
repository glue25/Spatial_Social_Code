#!/usr/bin/env python3
"""Merge resumable grounding outputs, preferring one successful row per key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row: dict) -> tuple[str, str, str]:
    return str(row["task"]), str(row["recording_id"]), str(row["sample_id"])


def success(row: dict) -> bool:
    return not row.get("error") and isinstance(row.get("prediction"), dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    order = [key(row) for row in manifest]
    chosen: dict[tuple[str, str, str], dict] = {}
    attempts = 0
    for path in args.inputs:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            attempts += 1
            item_key = key(row)
            previous = chosen.get(item_key)
            if previous is None or (success(row) and not success(previous)):
                chosen[item_key] = row

    args.output.write_text(
        "".join(json.dumps(chosen[item_key], ensure_ascii=False) + "\n" for item_key in order if item_key in chosen),
        encoding="utf-8",
    )
    successful = sum(item_key in chosen and success(chosen[item_key]) for item_key in order)
    print(json.dumps({"manifest": len(order), "attempt_rows": attempts, "keys_written": sum(k in chosen for k in order), "successful": successful, "remaining": len(order) - successful, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
