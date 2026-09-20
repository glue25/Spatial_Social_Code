"""Paired full-set comparison of native Spatial-Social Code and AutoToM."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total**2)
    ) / denominator
    return [center - radius, center + radius]


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_metric(prefix: str, index: int) -> dict | None:
    path = (
        ROOT
        / "AutoToM"
        / "results"
        / "metrics"
        / f"{prefix}_MMToM-QA_{index}_back1_reduce1_seed42_metrics.json"
    )
    return json.loads(path.read_text()) if path.exists() else None


def score(rows: list[dict], key: str) -> dict:
    correct = sum(bool(row[key]) for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "wilson_95_ci": wilson(correct, len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_jsonl", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json",
    )
    parser.add_argument(
        "--aligned-prefix", default="automated_qwen3vl_mm_required_aligned"
    )
    parser.add_argument("--fallback-prefix", default="automated_qwen3vl_mm_required")
    parser.add_argument("--condition", default="spatial_social")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples-output", type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    native_rows = {
        int(row["index"]): row
        for line in args.native_jsonl.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    if set(native_rows) != set(range(len(questions))):
        missing = sorted(set(range(len(questions))) - set(native_rows))
        raise ValueError(
            f"native coverage is {len(native_rows)}/{len(questions)}; "
            f"first missing indices: {missing[:20]}"
        )

    rows = []
    examples = []
    aligned_used = 0
    for index, question in enumerate(questions):
        native = native_rows[index]
        if "predictions" in native:
            native_prediction = native["predictions"][args.condition]
            prediction = native_prediction["prediction"].lower()
        else:
            prediction = str(native["prediction"]).lower()
            native_prediction = {
                "prediction": prediction,
                "prob_a": native.get("prob_a"),
                "prob_b": native.get("prob_b"),
            }
        answer = question["answer"].lower()
        metric = load_metric(args.aligned_prefix, index)
        metric_prefix = args.aligned_prefix
        if metric is None:
            metric = load_metric(args.fallback_prefix, index)
            metric_prefix = args.fallback_prefix
        else:
            aligned_used += 1
        if metric is None:
            raise FileNotFoundError(f"missing AutoToM metric for index {index}")
        native_correct = prediction == answer
        autotom_correct = bool(metric["Correctness"])
        row = {
            "index": index,
            "question_type": str(question["question_type"]),
            "native_correct": native_correct,
            "autotom_correct": autotom_correct,
            "autotom_prefix": metric_prefix,
        }
        rows.append(row)
        if native_correct and not autotom_correct:
            examples.append(
                {
                    **row,
                    "episode": question["episode"],
                    "question": question["question"],
                    "answer": answer,
                    "native_prediction": prediction,
                    "native_probabilities": native_prediction,
                    "spatial_code": native.get("spatial_code", native.get("codes", {}).get("spatial", "")),
                    "spatial_trace": native.get("spatial_trace", native.get("traces", {}).get("spatial", "")),
                    "social_code": native.get("social_code", native.get("codes", {}).get("social", "")),
                    "social_trace": native.get("social_trace", native.get("traces", {}).get("social", "")),
                    "social_reasoning": native.get("social_reasoning", {}),
                    "autotom_probabilities": metric.get("probs"),
                }
            )

    by_type = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["question_type"]].append(row)
    for question_type, group in sorted(grouped.items()):
        by_type[question_type] = {
            "native": score(group, "native_correct"),
            "autotom": score(group, "autotom_correct"),
        }

    wins = sum(row["native_correct"] and not row["autotom_correct"] for row in rows)
    losses = sum(row["autotom_correct"] and not row["native_correct"] for row in rows)
    report = {
        "native_condition": args.condition,
        "examples": len(rows),
        "aligned_autotom_metrics_used": aligned_used,
        "overall": {
            "native": score(rows, "native_correct"),
            "autotom": score(rows, "autotom_correct"),
        },
        "by_question_type": by_type,
        "paired": {
            "native_only_correct": wins,
            "autotom_only_correct": losses,
            "both_correct": sum(
                row["native_correct"] and row["autotom_correct"] for row in rows
            ),
            "both_wrong": sum(
                not row["native_correct"] and not row["autotom_correct"] for row in rows
            ),
            "exact_mcnemar_p_two_sided": exact_mcnemar(wins, losses),
        },
        "native_only_correct_examples": len(examples),
    }
    output = args.output or args.native_jsonl.with_name(
        args.native_jsonl.stem + "_vs_autotom.json"
    )
    examples_output = args.examples_output or args.native_jsonl.with_name(
        args.native_jsonl.stem + "_native_wins.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    examples_output.write_text(json.dumps(examples, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
