"""Merge disjoint standalone Spatial-Social Code evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_spatial_social import SCHEMA_VERSION, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    records: dict[int, dict] = {}
    sources: set[str] = set()
    external_settings: set[bool] = set()
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"incompatible record schema in {path}")
            sources.add(str(record.get("social_source")))
            external_settings.add(bool(record.get("external_bayesian_evidence", False)))
            index = int(record["index"])
            if index in records:
                raise ValueError(f"duplicate question index {index}")
            records[index] = record
    if len(sources) > 1:
        raise ValueError(f"cannot merge different Social Code sources: {sorted(sources)}")
    if len(external_settings) > 1:
        raise ValueError("cannot merge native-only and external-evidence records")

    with args.output.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    report = summarize(records.values())
    report["social_source"] = next(iter(sources), None)
    report["external_bayesian_evidence"] = next(iter(external_settings), False)
    args.summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
