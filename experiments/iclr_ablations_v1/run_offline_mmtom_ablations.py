#!/usr/bin/env python3
"""Recompute label-independent MMToM component ablations from frozen traces.

This program deliberately reads ground truth only after every prediction has
been produced.  The cached neural Social Code was generated without answer
labels; all remaining compilation and execution is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOCIAL_CODE_DIR = ROOT / "spatial_social_code"
AUTOTOM_MODEL = ROOT / "AutoToM" / "model"
for import_path in (SOCIAL_CODE_DIR, AUTOTOM_MODEL):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from mmtom_visual import load_questions  # noqa: E402
from runtime import execute_social  # noqa: E402
from social_code import (  # noqa: E402
    SocialCodeResult,
    answer_posterior_probs,
    compile_native_bayesian,
    compile_query_candidates,
    compile_rules,
    compile_spatial_bridge,
    fuse_social_results,
)


CONDITIONS = (
    "spatial_only_executed",
    "social_only_executed",
    "spatial_social_executed_recomputed",
    "spatial_social_executed_saved",
)


def load_jsonl(path: Path) -> dict[int, dict]:
    return {
        int(row["index"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_from(parts: list[SocialCodeResult], question: str) -> tuple[str, dict]:
    posterior = compile_native_bayesian(question, parts)
    fused = fuse_social_results("offline_ablation", [*parts, posterior])
    probabilities = answer_posterior_probs(fused.trace)
    if probabilities is None:
        raise RuntimeError("No A/B posterior produced")
    prediction = "a" if probabilities["A"] >= probabilities["B"] else "b"
    return prediction, probabilities


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def paired_bootstrap_ci(
    rows: list[dict], left: str, right: str, *, seed: int = 20260825, draws: int = 20000
) -> list[float]:
    rng = random.Random(seed)
    values = [
        int(row["predictions"][right] == row["answer"])
        - int(row["predictions"][left] == row["answer"])
        for row in rows
    ]
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]]


def metrics(rows: list[dict], condition: str) -> dict:
    correct = sum(row["predictions"][condition] == row["answer"] for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson_95_ci": wilson(correct, len(rows)),
    }


def compare(rows: list[dict], left: str, right: str) -> dict:
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
        "accuracy_delta_right_minus_left": (
            metrics(rows, right)["accuracy"] - metrics(rows, left)["accuracy"]
        ),
        "paired_bootstrap_95_ci": paired_bootstrap_ci(rows, left, right),
        "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json"
    )
    parser.add_argument(
        "--cached-traces",
        type=Path,
        default=SOCIAL_CODE_DIR / "mmtom_candidate_aligned_full.jsonl",
    )
    parser.add_argument(
        "--frozen-primary",
        type=Path,
        default=SOCIAL_CODE_DIR / "mmtom_dense_authoritative_gemini3pro.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1"
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    cached = load_jsonl(args.cached_traces)
    primary = load_jsonl(args.frozen_primary)
    if sorted(cached) != list(range(600)) or sorted(primary) != list(range(600)):
        raise ValueError("Expected complete, aligned 600-example MMToM inputs")

    # Prediction phase: no access to answer, correctness, or question_type.
    prediction_rows: list[dict] = []
    reconstruction_mismatches: list[int] = []
    maximum_reconstruction_probability_error = 0.0
    for index in sorted(cached):
        saved = cached[index]
        question = str(questions[index]["question"])
        neural_code = str(saved["codes"]["social_raw_evidence"])
        neural = SocialCodeResult(
            "neural", neural_code, execute_social(neural_code), ("neural",), raw_code=neural_code
        )
        rules = compile_rules(question)
        query = compile_query_candidates(question)
        bridge = compile_spatial_bridge(question, str(saved["traces"]["spatial"]))

        spatial_prediction, spatial_probs = prediction_from([rules, query, bridge], question)
        social_prediction, social_probs = prediction_from([neural, rules, query], question)
        full_prediction, full_probs = prediction_from([neural, rules, query, bridge], question)
        saved_prediction = str(saved["predictions"]["spatial_social"]["prediction"]).lower()
        saved_probs = saved["predictions"]["spatial_social"]
        probability_error = max(
            abs(full_probs["A"] - float(saved_probs["prob_a"])),
            abs(full_probs["B"] - float(saved_probs["prob_b"])),
        )
        maximum_reconstruction_probability_error = max(
            maximum_reconstruction_probability_error, probability_error
        )
        if full_prediction != saved_prediction:
            reconstruction_mismatches.append(index)
        prediction_rows.append(
            {
                "index": index,
                "predictions": {
                    "spatial_only_executed": spatial_prediction,
                    "social_only_executed": social_prediction,
                    "spatial_social_executed_recomputed": full_prediction,
                    "spatial_social_executed_saved": saved_prediction,
                },
                "probabilities": {
                    "spatial_only_executed": spatial_probs,
                    "social_only_executed": social_probs,
                    "spatial_social_executed_recomputed": full_probs,
                },
            }
        )

    # Scoring phase begins only here.
    for row in prediction_rows:
        index = row["index"]
        row["answer"] = str(questions[index]["answer"]).lower()
        row["question_type"] = str(questions[index]["question_type"])

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        by_type[row["question_type"]].append(row)

    condition_metrics = {}
    for condition in CONDITIONS:
        condition_metrics[condition] = {
            **metrics(prediction_rows, condition),
            "by_question_type": {
                question_type: metrics(group, condition)
                for question_type, group in sorted(by_type.items())
            },
        }

    primary_rows = []
    for index, saved in sorted(primary.items()):
        primary_rows.append(
            {
                "index": index,
                "question_type": str(saved["question_type"]),
                "answer": str(saved["answer"]).lower(),
                "predictions": {
                    "direct": str(saved["direct_prediction"]).lower(),
                    "frozen_spatial_social": str(saved["prediction"]).lower(),
                },
            }
        )

    primary_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in primary_rows:
        primary_by_type[row["question_type"]].append(row)
    primary_metrics = {
        condition: {
            **metrics(primary_rows, condition),
            "by_question_type": {
                name: metrics(group, condition) for name, group in sorted(primary_by_type.items())
            },
        }
        for condition in ("direct", "frozen_spatial_social")
    }

    uncertainty_count = sum(bool(row.get("uncertainty_triggered")) for row in primary.values())
    direct_retries = sum(bool(row.get("direct_retry_used")) for row in primary.values())
    summary = {
        "schema_version": 1,
        "status": "complete_offline; equal-budget direct pending",
        "scientific_boundary": {
            "prediction_function_used_gold": False,
            "question_type_used_for_routing": False,
            "mmtom_previously_inspected": True,
            "interpretation": "diagnostic ablation, not clean held-out validation",
        },
        "inputs": {
            "cached_trace_path": str(args.cached_traces.relative_to(ROOT)),
            "cached_trace_sha256": sha256(args.cached_traces),
            "frozen_primary_path": str(args.frozen_primary.relative_to(ROOT)),
            "frozen_primary_sha256": sha256(args.frozen_primary),
        },
        "strict_cached_trace_ablation": {
            "backbone": "Qwen3-VL-8B-Instruct (cache-producing local backend default)",
            "examples": len(prediction_rows),
            "reconstruction_mismatches_vs_saved_full": reconstruction_mismatches,
            "maximum_reconstruction_probability_error": maximum_reconstruction_probability_error,
            "conditions": condition_metrics,
            "paired_comparisons": [
                compare(prediction_rows, "spatial_only_executed", "spatial_social_executed_recomputed"),
                compare(prediction_rows, "social_only_executed", "spatial_social_executed_recomputed"),
                compare(prediction_rows, "spatial_only_executed", "social_only_executed"),
            ],
        },
        "frozen_primary_same_model_comparison": {
            "backbone": "Gemini 3-Pro-Preview",
            "examples": len(primary_rows),
            "conditions": primary_metrics,
            "paired_comparisons": [
                compare(primary_rows, "direct", "frozen_spatial_social")
            ],
        },
        "identifiability": {
            "direct": "available on frozen primary",
            "spatial_only": "strictly recomputed from frozen spatial trace with no neural Social evidence",
            "social_only": "strictly recomputed from frozen candidate-hidden neural Social evidence with no Spatial bridge",
            "spatial_social_executed": "strictly recomputed and exactly checked against cached saved prediction",
            "spatial_social_unexecuted": (
                "not identifiable from cached outputs: no final-answer prediction over raw Spatial+Social code "
                "without executed traces was saved; must be generated by a new model call"
            ),
        },
        "compute_accounting": {
            "unit": "model inference calls; deterministic compiler/runtime calls excluded",
            "direct_primary": {
                "base_calls": 600,
                "format_repair_calls": direct_retries,
                "total_calls": 600 + direct_retries,
                "calls_per_example": (600 + direct_retries) / 600,
            },
            "frozen_spatial_social": {
                "base_spatial_and_social_calls": 1200,
                "uncertainty_examples": uncertainty_count,
                "refinement_calls": 2 * uncertainty_count,
                "total_calls": 1200 + 2 * uncertainty_count,
                "calls_per_example": (1200 + 2 * uncertainty_count) / 600,
            },
            "token_usage": "unavailable in saved Gemini responses; call-count parity is not token parity",
            "latency": {
                "primary_pipeline_total_seconds": sum(float(row.get("elapsed_seconds", 0)) for row in primary.values()),
                "refinement_total_seconds": sum(float(row.get("refinement_seconds", 0)) for row in primary.values()),
                "warning": "elapsed_seconds includes paired Direct plus base Spatial-Social calls and cannot be separated post hoc",
            },
        },
        "pending_registered_conditions": [
            "spatial_social_unexecuted final-answer ablation",
            "call-budget-matched Direct self-consistency",
            "token-budget-matched Direct baseline",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "mmtom_offline_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"conditions": condition_metrics, "reconstruction_mismatches": len(reconstruction_mismatches)}, indent=2))


if __name__ == "__main__":
    main()
