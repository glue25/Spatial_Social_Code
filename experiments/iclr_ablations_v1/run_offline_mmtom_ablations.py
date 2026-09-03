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


STRICT_CONDITIONS = (
    "spatial_only_executed",
    "social_only_executed",
    "spatial_social_executed_recomputed",
)

SHARED_BASE_CONDITIONS = (
    "shared_scaffold",
    "shared_plus_physical",
    "shared_plus_social",
    "shared_plus_physical_social",
)

ROUTED_CONDITION = "oof_physical_social_router"


def recover_neural_evidence(raw_code: str, rules_code: str) -> str:
    """Recover only the cached neural extractor output from a fused raw program.

    Historical checkpoints stored ``SocialCodeResult.raw_code`` under the
    misleading key ``social_raw_evidence``.  That value concatenates neural,
    rules, query, spatial-bridge, and posterior programs in that order.  Using
    it directly leaks the saved full-method posterior into every ablation.
    The deterministic rules program is therefore the stable boundary after
    the neural prefix.
    """
    marker = "\n" + rules_code
    boundary = raw_code.find(marker)
    if boundary < 0:
        raise ValueError("cannot locate deterministic-rules boundary in cached Social Code")
    neural_code = raw_code[:boundary].strip()
    forbidden = ("M.candidate(", "M.query_evidence(", "M.posterior(")
    if not neural_code or any(token in neural_code for token in forbidden):
        raise ValueError("recovered neural evidence is empty or answer-contaminated")
    execute_social(neural_code)
    return neural_code


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


def episode_cluster_bootstrap_ci(
    rows: list[dict], left: str, right: str, *, seed: int = 20260825, draws: int = 20000
) -> list[float]:
    rng = random.Random(seed)
    clustered: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        clustered[str(row["episode"])].append(
            int(row["predictions"][right] == row["answer"])
            - int(row["predictions"][left] == row["answer"])
        )
    episodes = sorted(clustered)
    estimates = []
    for _ in range(draws):
        sampled = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        values = [value for episode in sampled for value in clustered[episode]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]]


def metrics(rows: list[dict], condition: str) -> dict:
    correct = sum(row["predictions"][condition] == row["answer"] for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson_95_ci": wilson(correct, len(rows)),
    }


def episode_fold(episode: str, folds: int = 5) -> int:
    """Assign every episode, and therefore all its questions, to one fold."""
    digest = hashlib.sha256(str(episode).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def add_oof_source_router(
    rows: list[dict],
    *,
    physical: str = "shared_plus_physical",
    social: str = "shared_plus_social",
    output: str = ROUTED_CONDITION,
    folds: int = 5,
) -> list[dict]:
    """Route by question type using labels from other episode folds only.

    This is a diagnostic cross-validated policy, not a hidden-test result.  A
    deterministic Physical tie-break keeps the fitting rule fully specified.
    """
    policies: list[dict] = []
    for fold in range(folds):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        policy: dict[str, str] = {}
        evidence: dict[str, dict] = {}
        for question_type in sorted({row["question_type"] for row in rows}):
            group = [row for row in train if row["question_type"] == question_type]
            physical_correct = sum(
                row["predictions"][physical] == row["answer"] for row in group
            )
            social_correct = sum(
                row["predictions"][social] == row["answer"] for row in group
            )
            source = "social" if social_correct > physical_correct else "physical"
            policy[question_type] = source
            evidence[question_type] = {
                "training_examples": len(group),
                "physical_correct": physical_correct,
                "social_correct": social_correct,
                "selected": source,
            }
        for row in test:
            source = policy[row["question_type"]]
            source_key = social if source == "social" else physical
            row["predictions"][output] = row["predictions"][source_key]
            row["probabilities"][output] = row["probabilities"][source_key]
            row["routing"] = {
                "fold": fold,
                "source": source,
                "source_condition": source_key,
            }
        policies.append(
            {
                "fold": fold,
                "held_out_examples": len(test),
                "held_out_episodes": len({row["episode"] for row in test}),
                "policy": policy,
                "training_evidence": evidence,
            }
        )
    if any(output not in row["predictions"] for row in rows):
        raise RuntimeError("OOF router did not assign every example")
    return policies


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
        "episode_cluster_bootstrap_95_ci": episode_cluster_bootstrap_ci(
            rows, left, right
        ),
        "episode_clusters": len({str(row["episode"]) for row in rows}),
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
    for index in sorted(cached):
        saved = cached[index]
        question = str(questions[index]["question"])
        rules = compile_rules(question)
        neural_code = recover_neural_evidence(
            str(saved["codes"]["social_raw_evidence"]), rules.code
        )
        neural = SocialCodeResult(
            "neural", neural_code, execute_social(neural_code), ("neural",), raw_code=neural_code
        )
        # The strict matrix removes the common question-reasoning scaffold to
        # inspect the raw evidence sources.  It is intentionally not compared
        # directly with the historical shared-scaffold component ablation.
        strict_query = compile_query_candidates(question, include_heuristic_evidence=False)

        # In the comparable matrix every condition receives the same
        # question/rule reasoning floor; only Physical and Social evidence is
        # added or removed.  This restores the protocol used by the ~75%
        # component results while retaining the cache-contamination fixes.
        shared_query = compile_query_candidates(question)
        bridge = compile_spatial_bridge(question, str(saved["traces"]["spatial"]))

        spatial_prediction, spatial_probs = prediction_from([strict_query, bridge], question)
        social_prediction, social_probs = prediction_from([neural, rules, strict_query], question)
        full_prediction, full_probs = prediction_from(
            [neural, rules, strict_query, bridge], question
        )
        scaffold_prediction, scaffold_probs = prediction_from([shared_query, rules], question)
        shared_physical_prediction, shared_physical_probs = prediction_from(
            [shared_query, rules, bridge], question
        )
        shared_social_prediction, shared_social_probs = prediction_from(
            [shared_query, rules, neural], question
        )
        shared_full_prediction, shared_full_probs = prediction_from(
            [shared_query, rules, neural, bridge], question
        )
        prediction_rows.append(
            {
                "index": index,
                "episode": str(questions[index]["episode"]),
                "predictions": {
                    "spatial_only_executed": spatial_prediction,
                    "social_only_executed": social_prediction,
                    "spatial_social_executed_recomputed": full_prediction,
                    "shared_scaffold": scaffold_prediction,
                    "shared_plus_physical": shared_physical_prediction,
                    "shared_plus_social": shared_social_prediction,
                    "shared_plus_physical_social": shared_full_prediction,
                },
                "probabilities": {
                    "spatial_only_executed": spatial_probs,
                    "social_only_executed": social_probs,
                    "spatial_social_executed_recomputed": full_probs,
                    "shared_scaffold": scaffold_probs,
                    "shared_plus_physical": shared_physical_probs,
                    "shared_plus_social": shared_social_probs,
                    "shared_plus_physical_social": shared_full_probs,
                },
            }
        )

    # Scoring phase begins only here.
    for row in prediction_rows:
        index = row["index"]
        row["answer"] = str(questions[index]["answer"]).lower()
        row["question_type"] = str(questions[index]["question_type"])
        row["fold"] = episode_fold(row["episode"])

    router_policies = add_oof_source_router(prediction_rows)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        by_type[row["question_type"]].append(row)

    condition_metrics = {}
    for condition in (*STRICT_CONDITIONS, *SHARED_BASE_CONDITIONS, ROUTED_CONDITION):
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
                "episode": str(saved["episode"]),
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
        "schema_version": 2,
        "status": "complete_offline; equal-budget direct pending",
        "scientific_boundary": {
            "prediction_function_used_gold": False,
            "question_type_used_for_routing": True,
            "mmtom_previously_inspected": True,
            "interpretation": "diagnostic ablation, not clean held-out validation",
            "cached_social_recovery": (
                "neural prefix recovered before deterministic-rules boundary; saved candidate, "
                "query-evidence, spatial-bridge, and posterior records excluded"
            ),
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
            "reconstruction_mismatches_vs_saved_full": "not comparable: corrected strict ablation removes shared answer heuristics and cached posteriors",
            "maximum_reconstruction_probability_error": None,
            "conditions": condition_metrics,
            "paired_comparisons": [
                compare(prediction_rows, "spatial_only_executed", "spatial_social_executed_recomputed"),
                compare(prediction_rows, "social_only_executed", "spatial_social_executed_recomputed"),
                compare(prediction_rows, "spatial_only_executed", "social_only_executed"),
            ],
        },
        "shared_scaffold_ablation": {
            "backbone": "Qwen3-VL-8B-Instruct (cache-producing local backend default)",
            "examples": len(prediction_rows),
            "protocol": (
                "Every condition receives the identical question-derived candidate/rule "
                "scaffold; clean Physical and neural Social evidence are then added or removed."
            ),
            "conditions": {
                condition: condition_metrics[condition]
                for condition in (*SHARED_BASE_CONDITIONS, ROUTED_CONDITION)
            },
            "oof_router": {
                "protocol": (
                    "five-fold episode-held-out diagnostic; source selected per question type "
                    "using only the other four folds; no held-out episode label used"
                ),
                "fold_assignment": "int(sha256(str(episode))[:8], 16) % 5",
                "tie_break": "physical",
                "policies": router_policies,
            },
            "paired_comparisons": [
                compare(prediction_rows, "shared_scaffold", "shared_plus_physical"),
                compare(prediction_rows, "shared_scaffold", "shared_plus_social"),
                compare(prediction_rows, "shared_plus_physical", "shared_plus_physical_social"),
                compare(prediction_rows, "shared_plus_social", "shared_plus_physical_social"),
                compare(prediction_rows, "shared_plus_physical", ROUTED_CONDITION),
                compare(prediction_rows, "shared_plus_physical_social", ROUTED_CONDITION),
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
            "spatial_social_executed": "strictly recomputed from the union of clean Physical and Social evidence",
            "shared_scaffold_conditions": "same query/rule scaffold in all four comparable component conditions",
            "oof_router": "episode-held-out diagnostic fitted on labeled folds; not a frozen hidden-test result",
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
    print(json.dumps({"conditions": condition_metrics}, indent=2))


if __name__ == "__main__":
    main()
