#!/usr/bin/env python3
"""Offline audit of legacy versus clean SSC answer inference on frozen traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "AutoToM/model", ROOT / "spatial_social_code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from clean_bayesian import score_clean_answers  # noqa: E402
from experiments.priority_ssc_v1.common import exact_mcnemar  # noqa: E402
from mmtom_visual import load_questions  # noqa: E402
from social_code import (  # noqa: E402
    SocialCodeResult,
    answer_posterior_probs,
    compile_native_bayesian,
    compile_query_candidates,
    compile_social_code,
)


REMOVED = {"query_candidate", "query_evidence", "posterior"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def without_legacy_query_evidence(question: str, social_trace: str) -> str:
    state = []
    for line in social_trace.splitlines():
        record = json.loads(line)
        if record.get("kind") not in REMOVED:
            state.append(record)
    frozen = SocialCodeResult(
        "frozen_state",
        "",
        "\n".join(json.dumps(record, ensure_ascii=False) for record in state),
        ("frozen_state",),
    )
    # Candidate registration is only an answer interface here.  The removed
    # heuristic is explicitly disabled.
    candidates = compile_query_candidates(question, include_heuristic_evidence=False)
    result = compile_native_bayesian(question, [frozen, candidates])
    posterior = answer_posterior_probs(result.trace)
    if posterior is None:
        raise RuntimeError("cleaned legacy Bayesian path produced no A/B posterior")
    return "a" if posterior["A"] >= posterior["B"] else "b"


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = str(row["question_type"])
        by_type[key][1] += 1
        by_type[key][0] += row[field] == row["answer"]
    correct = sum(values[0] for values in by_type.values())
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "by_question_type": {
            key: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for key, values in sorted(by_type.items())
        },
    }


def paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_only = sum(
        row[left] == row["answer"] and row[right] != row["answer"] for row in rows
    )
    right_only = sum(
        row[left] != row["answer"] and row[right] == row["answer"] for row in rows
    )
    return {
        "left": left,
        "right": right,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "MMToM-QA/Benchmark/questions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "workdirs/priority_ssc_v1/mmtom_clean_bayesian_rescore.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "reports/mmtom_clean_bayesian_rescore_summary.json",
    )
    args = parser.parse_args()

    source = read_jsonl(args.source)
    questions = load_questions(args.questions)
    output: list[dict[str, Any]] = []
    for row in source:
        index = int(row["index"])
        question = str(questions[index]["question"])
        stripped_prediction = without_legacy_query_evidence(question, row["social_trace"])
        decision = score_clean_answers(question, row["spatial_trace"], row["social_trace"])
        clean_state = compile_social_code(
            "bayesian",
            question,
            spatial_trace=row["spatial_trace"],
        )
        clean_state_decision = score_clean_answers(
            question,
            row["spatial_trace"],
            clean_state.trace,
        )
        output.append(
            {
                "index": index,
                "question_type": str(row["question_type"]),
                "answer": str(row["answer"]).lower(),
                "direct_prediction": str(row["direct_prediction"]).lower(),
                "legacy_prediction": str(row["prediction"]).lower(),
                "stripped_legacy_prediction": stripped_prediction,
                "clean_prediction": decision.choice.lower(),
                "clean_probabilities": decision.probabilities,
                "clean_scores": decision.scores,
                "clean_terms": decision.terms,
                "choice_hidden_rule_state_prediction": clean_state_decision.choice.lower(),
                "choice_hidden_rule_state_probabilities": clean_state_decision.probabilities,
                "choice_hidden_rule_state_scores": clean_state_decision.scores,
                "choice_hidden_rule_state_terms": clean_state_decision.terms,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    summary = {
        "schema_version": 1,
        "protocol": "offline_frozen_trace_clean_bayesian_audit",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "uses_gold_or_question_type_for_inference": False,
        "query_evidence_removed": True,
        "answer_directed_social_kinds_ignored": [
            "action_evidence",
            "posterior",
            "query_candidate",
            "query_evidence",
        ],
        "important_limitation": (
            "The frozen legacy Social traces were originally compiled with answer choices visible. "
            "The legacy-social rescore is diagnostic only. The choice-hidden-rule-state condition "
            "regenerates Social state without choices, but still reuses the shared frozen Physical "
            "trace and was developed after the benchmark labels had already been inspected; an "
            "independent frozen evaluation is required for a publishable gain."
        ),
        "metrics": {
            "direct": metric(output, "direct_prediction"),
            "legacy_saved": metric(output, "legacy_prediction"),
            "legacy_without_query_evidence": metric(output, "stripped_legacy_prediction"),
            "clean_fixed_bayesian_on_legacy_social": metric(output, "clean_prediction"),
            "clean_fixed_bayesian_on_choice_hidden_rule_state": metric(
                output, "choice_hidden_rule_state_prediction"
            ),
        },
        "legacy_social_clean_reasoner_flips_vs_legacy_saved": {
            "positive": sum(
                row["clean_prediction"] == row["answer"]
                and row["legacy_prediction"] != row["answer"]
                for row in output
            ),
            "negative": sum(
                row["clean_prediction"] != row["answer"]
                and row["legacy_prediction"] == row["answer"]
                for row in output
            ),
        },
        "paired_comparisons": {
            "direct_vs_choice_hidden_clean": paired(
                output, "direct_prediction", "choice_hidden_rule_state_prediction"
            ),
            "legacy_vs_choice_hidden_clean": paired(
                output, "legacy_prediction", "choice_hidden_rule_state_prediction"
            ),
        },
    }
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
