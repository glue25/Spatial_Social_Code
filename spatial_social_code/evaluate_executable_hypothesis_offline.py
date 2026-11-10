"""Reproducible offline ablation over saved Spatial evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT / "spatial_social_code", ROOT / "third_party/autotom" / "model"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from executable_hypothesis import compile_payload, deterministic_payload  # noqa: E402
from mmtom_visual import load_questions  # noqa: E402
from social_code import answer_options, question_without_options  # noqa: E402


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["index"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def mcnemar_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def metrics(values: list[bool]) -> dict[str, Any]:
    return {
        "correct": sum(values),
        "total": len(values),
        "accuracy": sum(values) / len(values) if values else None,
    }


def write_records(path: Path, records: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method = metrics([row["method_correct"] for row in rows])
    primary = metrics([row["primary_correct"] for row in rows])
    baseline = metrics([row["baseline_correct"] for row in rows])
    primary_only = sum(row["primary_correct"] and not row["method_correct"] for row in rows)
    method_only = sum(row["method_correct"] and not row["primary_correct"] for row in rows)
    both = sum(row["primary_correct"] and row["method_correct"] for row in rows)
    return {
        "executable_hypothesis": method,
        "saved_primary": primary,
        "paired_direct_baseline": baseline,
        "delta_vs_saved_primary": method["accuracy"] - primary["accuracy"],
        "delta_vs_paired_baseline": method["accuracy"] - baseline["accuracy"],
        "contingency_vs_saved_primary": {
            "both_correct": both,
            "saved_primary_only_correct": primary_only,
            "executable_only_correct": method_only,
            "both_wrong": len(rows) - both - primary_only - method_only,
            "net_executable_wins": method_only - primary_only,
            "mcnemar_exact_p": mcnemar_exact(primary_only, method_only),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("ma", "mmtom"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ma-questions", type=Path, default=ROOT / "benchmarks/ma_egoqa" / "data" / "MA-EgoQA.json")
    parser.add_argument("--mmtom-questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json")
    args = parser.parse_args()

    source = load_jsonl(args.source)
    cohorts: dict[int, list[str]] = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected = [int(item["index"]) for item in manifest["indices"]]
        cohorts = {int(item["index"]): list(item.get("cohorts", [])) for item in manifest["indices"]}
    else:
        selected = sorted(source)
    if args.benchmark == "ma":
        questions = json.loads(args.ma_questions.read_text(encoding="utf-8"))
    else:
        questions = load_questions(args.mmtom_questions)

    predictions: dict[int, dict[str, Any]] = load_jsonl(args.output) if args.output.exists() else {}
    comparison_rows: list[dict[str, Any]] = []
    newly_completed = 0
    for index in selected:
        row = questions[index]
        saved = source[index]
        if args.benchmark == "ma":
            option_pairs = [
                (chr(ord("A") + offset), str(text))
                for offset, text in enumerate(row["options"])
            ]
            premise = str(row["question"])
            gold = row["options"].index(row["answer"])
            primary_prediction = int(saved["prediction_index"])
            baseline_prediction = int(saved["baseline_prediction_index"])
            prediction_to_index = {label: offset for offset, (label, _) in enumerate(option_pairs)}
            group = str(row["category"])
            prefix = "S"
        else:
            option_pairs = answer_options(str(row["question"]))
            premise = question_without_options(str(row["question"]))
            gold_label = str(row["answer"]).lower()
            gold = 0 if gold_label == "a" else 1
            primary_prediction = 0 if str(saved["prediction"]).lower() == "a" else 1
            baseline_prediction = 0 if str(saved["direct_prediction"]).lower() == "a" else 1
            prediction_to_index = {"A": 0, "B": 1}
            group = str(row["question_type"])
            prefix = "M"
        if index in predictions and predictions[index].get("benchmark") == args.benchmark:
            prediction_index = int(predictions[index]["prediction_index"])
        else:
            payload, indexed_trace = deterministic_payload(
                premise, option_pairs, str(saved.get("spatial_trace", "")), evidence_prefix=prefix
            )
            code, indexed_trace, decision = compile_payload(
                payload,
                option_pairs,
                indexed_trace,
                premise,
                evidence_prefix=prefix,
            )
            prediction_index = prediction_to_index[decision.choice]
            predictions[index] = {
                "schema_version": 1,
                "method": "Deterministic Executable Hypothesis Layer offline ablation",
                "benchmark": args.benchmark,
                "index": index,
                "spatial_trace": indexed_trace,
                "hypothesis_payload": payload,
                "hypothesis_code": code,
                "hypothesis_trace": decision.trace,
                "runtime_scores": decision.scores,
                "runtime_posteriors": decision.posteriors,
                "runtime_diagnostics": decision.diagnostics,
                "prediction_index": prediction_index,
            }
            newly_completed += 1
            if newly_completed % 25 == 0:
                write_records(args.output, predictions)
        comparison_rows.append({
            "index": index,
            "method_correct": prediction_index == gold,
            "primary_correct": primary_prediction == gold,
            "baseline_correct": baseline_prediction == gold,
            "group": group,
            "cohorts": cohorts.get(index, []),
        })

    write_records(args.output, predictions)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        by_group[row["group"]].append(row)
        for cohort in row["cohorts"]:
            by_cohort[cohort].append(row)
    summary = {
        "schema_version": 1,
        "method": "Deterministic Executable Hypothesis Layer offline ablation",
        "benchmark": args.benchmark,
        "source_spatial_evidence": str(args.source),
        "inference_inputs": "query/options and frozen Spatial trace only; no gold, primary prediction, baseline prediction, category, or question-type routing",
        "comparison_timing": "primary/baseline fields read only after each runtime prediction was final",
        "completed": len(comparison_rows),
        "overall": compare(comparison_rows),
        "by_group_posthoc": {name: compare(group) for name, group in sorted(by_group.items())},
        "by_cohort_posthoc": {name: compare(group) for name, group in sorted(by_cohort.items())},
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary_summary.replace(summary_path)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
