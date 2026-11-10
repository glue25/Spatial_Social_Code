"""Offline component ablation over the frozen MMToM visual traces.

This script never calls a model and never reads a baseline prediction while
constructing an ablation prediction.  It recomputes the native Social Code
posterior from progressively richer deterministic components, then compares
those predictions with the already saved hybrid and direct conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOTOM_MODEL = ROOT / "third_party/autotom" / "model"
if str(AUTOTOM_MODEL) not in sys.path:
    sys.path.insert(0, str(AUTOTOM_MODEL))

from mmtom_visual import load_questions  # noqa: E402
from social_code import (  # noqa: E402
    answer_posterior_probs,
    compile_native_bayesian,
    compile_query_candidates,
    compile_rules,
    compile_spatial_bridge,
)


def load_jsonl(path: Path) -> dict[int, dict]:
    """Load indexed result rows without importing any model backend."""
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "index" not in row:
                raise ValueError(f"{path}:{line_number} has no index")
            rows[int(row["index"])] = row
    return rows


def predict(parts: list, question: str) -> tuple[str, dict[str, float]]:
    posterior = compile_native_bayesian(question, parts)
    probabilities = answer_posterior_probs(posterior.trace)
    if probabilities is None:
        raise RuntimeError("ablation produced no A/B posterior")
    return ("a" if probabilities["A"] >= probabilities["B"] else "b"), probabilities


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def metrics(rows: list[dict], condition: str) -> dict:
    correct = sum(row["predictions"][condition] == row["answer"] for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def comparison(rows: list[dict], left: str, right: str) -> dict:
    left_only = sum(
        row["predictions"][left] == row["answer"]
        and row["predictions"][right] != row["answer"]
        for row in rows
    )
    right_only = sum(
        row["predictions"][left] != row["answer"]
        and row["predictions"][right] == row["answer"]
        for row in rows
    )
    return {
        "left": left,
        "right": right,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_right_wins": right_only - left_only,
        "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
    }


def summarize(rows: list[dict]) -> dict:
    conditions = list(rows[0]["predictions"])
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    return {
        "schema_version": 1,
        "benchmark": "MMToM",
        "protocol": (
            "offline recomputation on frozen sparse Spatial traces; query/rules/bridge "
            "conditions exclude neural Social extraction"
        ),
        "completed": len(rows),
        "conditions": {
            condition: {
                **metrics(rows, condition),
                "by_question_type": {
                    name: metrics(group, condition) for name, group in sorted(by_type.items())
                },
            }
            for condition in conditions
        },
        "paired_comparisons": [
            comparison(rows, "query_only", "query_rules"),
            comparison(rows, "query_rules", "query_rules_spatial"),
            comparison(rows, "query_rules_spatial", "full_hybrid_sparse"),
            comparison(rows, "full_hybrid_sparse", "full_hybrid_avg_dense"),
            comparison(rows, "full_hybrid_avg_dense", "full_hybrid_dense_authoritative"),
            comparison(rows, "direct", "full_hybrid_dense_authoritative"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json"
    )
    parser.add_argument(
        "--sparse", type=Path, default=ROOT / "spatial_social_code" / "mmtom_robust_gemini3pro.jsonl"
    )
    parser.add_argument(
        "--avg-dense",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_uncertainty_role_rerun_gemini3pro.jsonl",
    )
    parser.add_argument(
        "--dense-authoritative",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_dense_authoritative_gemini3pro.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_component_ablation_offline.jsonl",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    sparse = load_jsonl(args.sparse)
    avg_dense = load_jsonl(args.avg_dense)
    dense = load_jsonl(args.dense_authoritative)
    indices = sorted(set(sparse) & set(avg_dense) & set(dense))
    rows: list[dict] = []
    for index in indices:
        saved = sparse[index]
        question = str(questions[index]["question"])
        query = compile_query_candidates(question)
        query_prediction, query_probs = predict([query], question)
        rules = compile_rules(question)
        rules_prediction, rules_probs = predict([query, rules], question)
        bridge = compile_spatial_bridge(question, str(saved.get("spatial_trace", "")))
        spatial_prediction, spatial_probs = predict([query, rules, bridge], question)
        rows.append(
            {
                "index": index,
                "question_type": str(saved["question_type"]),
                "answer": str(saved["answer"]).lower(),
                "predictions": {
                    "direct": str(saved["direct_prediction"]).lower(),
                    "query_only": query_prediction,
                    "query_rules": rules_prediction,
                    "query_rules_spatial": spatial_prediction,
                    "full_hybrid_sparse": str(saved["prediction"]).lower(),
                    "full_hybrid_avg_dense": str(avg_dense[index]["prediction"]).lower(),
                    "full_hybrid_dense_authoritative": str(dense[index]["prediction"]).lower(),
                },
                "probabilities": {
                    "query_only": query_probs,
                    "query_rules": rules_probs,
                    "query_rules_spatial": spatial_probs,
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
