"""Held-out statistics for one Spatial-Social Code source configuration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONDITIONS = ("direct", "spatial", "social", "spatial_social")


def wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def paired(rows: list[dict], candidate: str, baseline: str) -> dict:
    wins = losses = both_correct = both_wrong = 0
    for row in rows:
        a = row["predictions"][candidate]["prediction"] == row["answer"]
        b = row["predictions"][baseline]["prediction"] == row["answer"]
        if a and not b:
            wins += 1
        elif b and not a:
            losses += 1
        elif a:
            both_correct += 1
        else:
            both_wrong += 1
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "exact_mcnemar_p_two_sided": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    rows.sort(key=lambda row: (str(row["question_type"]), int(row["index"])))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_type"]), []).append(row)
    test_rows = []
    validation_rows = []
    for group in grouped.values():
        count = round(len(group) * args.validation_fraction)
        validation_rows.extend(group[:count])
        test_rows.extend(group[count:])

    scores = {}
    for condition in CONDITIONS:
        correct = sum(
            row["predictions"][condition]["prediction"] == row["answer"]
            for row in test_rows
        )
        scores[condition] = {
            "correct": correct,
            "total": len(test_rows),
            "accuracy": correct / len(test_rows) if test_rows else None,
            "wilson_95_ci": wilson(correct, len(test_rows)),
        }
    report = {
        "method": "Spatial-Social Code",
        "social_design": (
            rows[0].get("social_design", "native_state_fusion_v1") if rows else None
        ),
        "social_source": rows[0].get("social_source") if rows else None,
        "external_bayesian_evidence": (
            bool(rows[0].get("external_bayesian_evidence", False)) if rows else None
        ),
        "validation_examples_reserved": len(validation_rows),
        "held_out_examples": len(test_rows),
        "held_out": scores,
        "paired_tests": {
            f"spatial_social_vs_{baseline}": paired(
                test_rows, "spatial_social", baseline
            )
            for baseline in ("direct", "spatial", "social")
        },
    }
    output = args.output or args.input.with_name(args.input.stem + "_analysis.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
