"""Learn a per-question-family fusion gate on validation and score held-out data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def accuracy(rows: list[dict], condition: str) -> float:
    if not rows:
        return 0.0
    return sum(
        row["predictions"][condition]["prediction"] == row["answer"] for row in rows
    ) / len(rows)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def paired_comparison(rows: list[dict], candidate: str, baseline: str) -> dict:
    wins = losses = ties_correct = ties_wrong = 0
    for row in rows:
        candidate_correct = (
            row["predictions"][candidate]["prediction"] == row["answer"]
        )
        baseline_correct = (
            row["predictions"][baseline]["prediction"] == row["answer"]
        )
        if candidate_correct and not baseline_correct:
            wins += 1
        elif baseline_correct and not candidate_correct:
            losses += 1
        elif candidate_correct:
            ties_correct += 1
        else:
            ties_wrong += 1
    discordant = wins + losses
    if discordant:
        tail = sum(
            math.comb(discordant, index) for index in range(min(wins, losses) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "both_correct": ties_correct,
        "both_wrong": ties_wrong,
        "exact_mcnemar_p_two_sided": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--validation-per-type", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["index"]))

    gates: dict[str, str] = {}
    validation_scores = {}
    test_rows = []
    for question_type, group in sorted(grouped.items()):
        validation_count = (
            args.validation_per_type
            if args.validation_per_type is not None
            else max(1, round(len(group) * args.validation_fraction))
        )
        validation = group[:validation_count]
        test = group[validation_count:]
        scores = {
            "autotom": accuracy(validation, "autotom"),
            "autotom_rerank": accuracy(validation, "autotom_rerank"),
            "spatial_social": accuracy(validation, "spatial_social"),
        }
        # Conservative tie break: retain AutoToM unless joint code proves useful.
        candidates = ("autotom", "autotom_rerank", "spatial_social")
        # Stable, conservative tie break follows candidate order.
        gate = max(candidates, key=lambda condition: scores[condition])
        gates[question_type] = gate
        validation_scores[question_type] = scores
        test_rows.extend(test)

    adaptive_correct = sum(
        row["predictions"][gates[str(row["question_type"])]]["prediction"]
        == row["answer"]
        for row in test_rows
    )
    oracle_correct = sum(
        any(
            row["predictions"][condition]["prediction"] == row["answer"]
            for condition in ("autotom", "spatial_social")
        )
        for row in test_rows
    )
    test_scores = {
        condition: {
            "correct": sum(
                row["predictions"][condition]["prediction"] == row["answer"]
                for row in test_rows
            ),
            "total": len(test_rows),
        }
        for condition in (
            "direct",
            "spatial",
            "autotom",
            "autotom_rerank",
            "spatial_social",
        )
    }
    for values in test_scores.values():
        values["accuracy"] = values["correct"] / values["total"]
        values["wilson_95_ci"] = wilson_interval(values["correct"], values["total"])
    test_scores["validation_gated"] = {
        "correct": adaptive_correct,
        "total": len(test_rows),
        "accuracy": adaptive_correct / len(test_rows),
        "wilson_95_ci": wilson_interval(adaptive_correct, len(test_rows)),
    }
    test_scores["oracle_union_ceiling"] = {
        "correct": oracle_correct,
        "total": len(test_rows),
        "accuracy": oracle_correct / len(test_rows),
    }
    test_scores["oracle_union_ceiling"]["wilson_95_ci"] = wilson_interval(
        oracle_correct, len(test_rows)
    )

    report = {
        "validation_fraction": args.validation_fraction,
        "validation_examples": len(rows) - len(test_rows),
        "test_examples": len(test_rows),
        "validation_scores": validation_scores,
        "learned_gate": gates,
        "held_out_test": test_scores,
        "paired_tests": {
            "spatial_social_vs_autotom": paired_comparison(
                test_rows, "spatial_social", "autotom"
            ),
            "spatial_social_vs_autotom_rerank": paired_comparison(
                test_rows, "spatial_social", "autotom_rerank"
            ),
        },
    }
    output = args.output or args.input.with_name(args.input.stem + "_gated.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
