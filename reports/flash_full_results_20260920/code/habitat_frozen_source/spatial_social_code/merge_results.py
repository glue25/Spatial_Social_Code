"""Merge disjoint evaluator shards and regenerate a global summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_mmtom import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    records = {}
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                index = int(record["index"])
                if index in records:
                    raise ValueError(f"duplicate question index {index}")
                records[index] = record
    with args.output.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    report = summarize(records.values())
    args.summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
