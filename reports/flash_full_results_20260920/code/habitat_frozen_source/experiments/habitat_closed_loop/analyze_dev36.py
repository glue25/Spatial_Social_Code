#!/usr/bin/env python3
"""Generate English tables/figures only after all 36 real rollouts exist."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("direct", "static_ssc", "dynamic_ssc")


def jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return 0.0, 0.0
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0, center - half), min(1, center + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = list(args.run_root.glob("results/*/*/**/closed_loop_artifacts/summary.json"))
    if len(summaries) != 36:
        raise SystemExit(f"refusing partial analysis: expected 36 real summaries, found {len(summaries)}")
    rows = []
    for path in summaries:
        summary = json.loads(path.read_text())
        artifact = path.parent
        manifest = json.loads((artifact / "manifest.json").read_text())
        evaluation = summary.get("historical_benchmark_evaluation", {})
        strict = evaluation.get("strict_physical", {})
        if strict.get("measurement_valid") is False:
            raise SystemExit(f"{path}: invalid strict physical measurement; refusing aggregation")
        usage_files = list(path.parents[1].glob("**/api_usage.jsonl"))
        usage = [row for file in usage_files for row in jsonl(file)]
        rows.append({
            "task_id": manifest.get("task_id"), "scene_id": manifest.get("scene_id"),
            "condition": manifest.get("condition"), "seed": manifest.get("seed"),
            "strict_success": int(bool(strict.get("strict_success"))),
            "strict_percent_complete": strict.get("strict_percent_complete", 0.0),
            "historical_benchmark_success": int(bool(evaluation.get("historical_benchmark_success"))),
            "api_calls": sum(int(item.get("api_calls", 0) or 0) for item in usage),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in usage),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in usage),
        })
    keys = {(row["task_id"], row["condition"]) for row in rows}
    if len(keys) != 36 or set(row["condition"] for row in rows) != set(CONDITIONS):
        raise SystemExit("invalid task/condition coverage; refusing analysis")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "h1_dev36_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: (x["condition"], x["task_id"])))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    aggregate = []
    for condition in CONDITIONS:
        values = grouped[condition]
        wins = sum(row["strict_success"] for row in values)
        low, high = wilson(wins, len(values))
        aggregate.append({
            "condition": condition, "n_tasks": len(values), "strict_successes": wins,
            "strict_success_rate": wins / len(values), "ci95_low": low, "ci95_high": high,
            "benchmark_success_rate": sum(row["historical_benchmark_success"] for row in values) / len(values),
            "mean_api_calls": sum(row["api_calls"] for row in values) / len(values),
            "mean_total_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in values) / len(values),
        })
    (args.output / "h1_dev36_summary.json").write_text(json.dumps({"status": "engineering_regression_one_seed", "independent_task_clusters": 12, "rows": aggregate}, indent=2) + "\n")
    with (args.output / "h1_dev36_table.tex").open("w") as handle:
        handle.write("\\begin{tabular}{lrrrr}\n\\toprule\nCondition & Strict success & Benchmark success & API calls & Tokens \\\\\n\\midrule\n")
        for row in aggregate:
            handle.write(f"{row['condition'].replace('_', ' ').title()} & {100*row['strict_success_rate']:.1f}\\% & {100*row['benchmark_success_rate']:.1f}\\% & {row['mean_api_calls']:.1f} & {row['mean_total_tokens']:.0f} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
    try:
        import matplotlib.pyplot as plt
        labels = [row["condition"].replace("_", " ").title() for row in aggregate]
        rates = [100 * row["strict_success_rate"] for row in aggregate]
        # Clamp tiny floating-point residues (for example a Wilson lower bound
        # of 2.8e-17 at 0/12) because matplotlib rejects negative yerr values.
        low = [max(0.0, 100 * (row["strict_success_rate"] - row["ci95_low"])) for row in aggregate]
        high = [max(0.0, 100 * (row["ci95_high"] - row["strict_success_rate"])) for row in aggregate]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.bar(labels, rates, yerr=[low, high], capsize=4, color=["#7f8c8d", "#4c78a8", "#f58518"])
        ax.set_ylabel("Strict shared physical success (%)")
        ax.set_title("Habitat Development Engineering Regression (12 tasks, 1 seed)")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(args.output / f"h1_dev36_strict_success.{suffix}", dpi=200)
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
