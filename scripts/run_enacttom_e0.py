#!/usr/bin/env python3
"""Run the deterministic EnactToM E0 oracle-state experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_social_code.enacttom import render_markdown, run_e0, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay EnactToM oracle state and run first-order interventions."
    )
    parser.add_argument(
        "--standard",
        type=Path,
        default=ROOT / "external_datasets/enacttom/standard.jsonl",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        default=ROOT / "external_datasets/enacttom/hard.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/enacttom_e0_oracle_results.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "reports/enacttom_e0_oracle_results.md",
    )
    args = parser.parse_args()

    paths = [("standard", args.standard), ("hard", args.hard)]
    result = run_e0(paths)
    write_json(args.out, result)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "out_md": str(args.out_md),
                "scope": result["scope"],
                "aggregate": result["aggregate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
