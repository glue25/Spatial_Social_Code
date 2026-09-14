#!/usr/bin/env python3
"""Aggregate paired Habitat D/T/C runs across completed independent seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("direct", "static_ssc", "dynamic_ssc")


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def extract(run_root: Path) -> list[dict]:
    summaries = list(run_root.glob("results/*/*/**/closed_loop_artifacts/summary.json"))
    if len(summaries) != 36:
        raise ValueError(f"{run_root}: expected 36 summaries, found {len(summaries)}")
    rows = []
    for path in summaries:
        artifact = path.parent
        summary = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        evaluation = summary.get("historical_benchmark_evaluation", {})
        strict = evaluation.get("strict_physical", {})
        usage_files = list(path.parents[1].glob("**/api_usage.jsonl"))
        usage = [row for file in usage_files for row in jsonl(file)]
        rows.append({
            "task_id": manifest["task_id"],
            "scene_id": manifest["scene_id"],
            "condition": manifest["condition"],
            "seed": int(manifest["seed"]),
            "strict_success": int(bool(strict.get("strict_success"))),
            "strict_percent_complete": float(strict.get("strict_percent_complete", 0.0)),
            "benchmark_success": int(bool(evaluation.get("historical_benchmark_success"))),
            "api_calls": sum(int(item.get("api_calls", 0) or 0) for item in usage),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in usage),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in usage),
            "artifact_dir": str(artifact),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for root in args.run_roots for row in extract(root)]
    seeds = sorted({row["seed"] for row in rows})
    expected = len(seeds) * 12 * len(CONDITIONS)
    unique = {(row["task_id"], row["condition"], row["seed"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise SystemExit(f"invalid coverage: rows={len(rows)} unique={len(unique)} expected={expected}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "habitat_multiseed_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["seed"], row["condition"], row["task_id"])))

    grouped: dict[str, list[dict]] = defaultdict(list)
    by_pair = {}
    for row in rows:
        grouped[row["condition"]].append(row)
        by_pair[(row["task_id"], row["seed"], row["condition"])] = row

    aggregate = []
    for condition in CONDITIONS:
        values = grouped[condition]
        successes = sum(row["strict_success"] for row in values)
        low, high = wilson(successes, len(values))
        aggregate.append({
            "condition": condition,
            "n_task_seed_runs": len(values),
            "strict_successes": successes,
            "strict_success_rate": successes / len(values),
            "strict_ci95_wilson": [low, high],
            "mean_strict_percent_complete": sum(row["strict_percent_complete"] for row in values) / len(values),
            "benchmark_success_rate": sum(row["benchmark_success"] for row in values) / len(values),
            "mean_api_calls": sum(row["api_calls"] for row in values) / len(values),
            "mean_total_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in values) / len(values),
        })

    per_seed = []
    for seed in seeds:
        for condition in CONDITIONS:
            values = [row for row in grouped[condition] if row["seed"] == seed]
            per_seed.append({
                "seed": seed,
                "condition": condition,
                "strict_successes": sum(row["strict_success"] for row in values),
                "n_tasks": len(values),
                "mean_strict_percent_complete": sum(row["strict_percent_complete"] for row in values) / len(values),
            })

    comparisons = []
    task_seed_keys = sorted({(row["task_id"], row["seed"]) for row in rows})
    for candidate in ("static_ssc", "dynamic_ssc"):
        candidate_only = 0
        direct_only = 0
        both = 0
        neither = 0
        for task_id, seed in task_seed_keys:
            direct = by_pair[(task_id, seed, "direct")]["strict_success"]
            other = by_pair[(task_id, seed, candidate)]["strict_success"]
            if direct and other:
                both += 1
            elif other:
                candidate_only += 1
            elif direct:
                direct_only += 1
            else:
                neither += 1
        comparisons.append({
            "comparison": f"{candidate}_vs_direct",
            "both_success": both,
            "candidate_only_success": candidate_only,
            "direct_only_success": direct_only,
            "neither_success": neither,
            "exact_mcnemar_p": exact_mcnemar(candidate_only, direct_only),
        })

    payload = {
        "status": "three_seed_extension_not_final_frozen_protocol",
        "seeds": seeds,
        "unique_tasks": 12,
        "task_seed_clusters": len(task_seed_keys),
        "total_rollouts": len(rows),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "paired_comparisons": comparisons,
        "statistical_note": "Task-seed rows are paired, but repeated seeds of the same task are not claimed as independent task clusters.",
    }
    (args.output / "habitat_multiseed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Habitat closed-loop three-seed extension", "",
        f"Seeds: {', '.join(map(str, seeds))}; 12 tasks; {len(rows)} real rollouts.", "",
        "This is a multi-seed extension of the engineering configuration, not a frozen final protocol.", "",
        "| Condition | Strict success | Mean completion | Benchmark success | API calls | Tokens |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        md.append(
            f"| {row['condition']} | {row['strict_successes']}/{row['n_task_seed_runs']} "
            f"({100*row['strict_success_rate']:.1f}%) | {100*row['mean_strict_percent_complete']:.1f}% | "
            f"{100*row['benchmark_success_rate']:.1f}% | {row['mean_api_calls']:.1f} | "
            f"{row['mean_total_tokens']:.0f} |"
        )
    md.extend(["", "## Paired comparisons", ""])
    for row in comparisons:
        md.append(
            f"- {row['comparison']}: candidate-only={row['candidate_only_success']}, "
            f"direct-only={row['direct_only_success']}, exact McNemar p={row['exact_mcnemar_p']:.6g}."
        )
    md.extend(["", payload["statistical_note"], ""])
    (args.output / "habitat_multiseed_results.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
