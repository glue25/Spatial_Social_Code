#!/usr/bin/env python3
"""Run the preregistered 2x2 Physical/Social human-correction experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for candidate in (ROOT / "MA-EgoQA", ROOT / "AutoToM/model", ROOT / "spatial_social_code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.priority_ssc_v1.common import FOUR_CELLS, atomic_json, atomic_jsonl, exact_mcnemar, forbidden_target_keys, format_question, parse_choice, read_jsonl, source_traces  # noqa: E402
from experiments.priority_ssc_v1.run_reasoning_baselines import make_client  # noqa: E402
from mmtom_visual import load_questions  # noqa: E402


FINAL_TEMPLATE = """Answer the multiple-choice question from the supplied Physical and agent-indexed Social states. Preserve who saw or heard each event. Missing state is unknown, not false. The states never contain a target answer. Compare every option and return JSON only: {{"answer":"A","reason":"one short sentence"}}.

Physical state:
{physical}

Social state:
{social}

Valid answer labels: {labels}

Question:
{question}"""


def one_call(client: Any, prompt: str) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    result = client.generate_parts_with_metadata([{"text": prompt}], max_output_tokens=1024, temperature=0.0)
    return str(result["text"]), {
        "calls": 1,
        "attempts": result.get("attempts"),
        "transport_retries": result.get("transport_retries", 0),
        "elapsed_seconds": result.get("elapsed_seconds", time.monotonic() - started),
        "usage": result.get("usage", {}),
    }


def paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_only = sum(row["correct"][left] and not row["correct"][right] for row in rows)
    right_only = sum(not row["correct"][left] and row["correct"][right] for row in rows)
    return {"n": len(rows), "left": left, "right": right, "left_only_correct": left_only, "right_only_correct": right_only, "net_right_wins": right_only - left_only, "mcnemar_exact_p": exact_mcnemar(left_only, right_only)}


def summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    complete = [row for row in rows if all(int(row.get("predictions", {}).get(cell, -1)) >= 0 for cell in FOUR_CELLS)]
    physical_correct = [row for row in complete if row["predicted_physical_confirmed_correct"]]
    return {
        "schema_version": 1,
        "protocol": "fixed_final_reasoner_2x2_human_state_correction",
        "model": model,
        "fully_complete": len(complete),
        "conditions": {cell: {"correct": sum(row["correct"][cell] for row in complete), "total": len(complete), "accuracy": sum(row["correct"][cell] for row in complete) / len(complete) if complete else None} for cell in FOUR_CELLS},
        "primary_social_correction_after_human_physical": paired(complete, "human_predicted", "human_human"),
        "social_correction_when_predicted_physical_was_confirmed_correct": paired(physical_correct, "predicted_predicted", "predicted_human"),
        "secondary_physical_correction_with_predicted_social": paired(complete, "predicted_predicted", "human_predicted"),
        "secondary_physical_correction_with_human_social": paired(complete, "predicted_human", "human_human"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mmtom", "maegoqa"), required=True)
    parser.add_argument("--backend", choices=("local-qwen32b", "jdcloud"), default="local-qwen32b")
    parser.add_argument("--verified", type=Path, default=ROOT / "workdirs/priority_ssc_v1/human_audit/verified_states.jsonl")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--credentials-file", type=Path, default=ROOT / "credentials.env")
    parser.add_argument("--local-model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--duty-cycle", type=float, default=1.0)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    args = parser.parse_args()
    if not args.verified.is_file():
        raise SystemExit(f"human-verified state file missing: {args.verified}")
    default_source = ROOT / ("spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl" if args.dataset == "mmtom" else "MA-EgoQA/data/baselines/video_balanced_role_full.jsonl")
    default_questions = ROOT / ("MMToM-QA/Benchmark/questions.json" if args.dataset == "mmtom" else "MA-EgoQA/data/MA-EgoQA.json")
    source_path, questions_path = args.source or default_source, args.questions or default_questions
    output = args.output or ROOT / f"workdirs/priority_ssc_v1/{args.dataset}_{args.backend}_four_cell.jsonl"
    source = {int(row["index"]): row for row in read_jsonl(source_path)}
    verified = [row for row in read_jsonl(args.verified) if row["dataset"] == args.dataset]
    if any(forbidden_target_keys(row) for row in verified):
        raise RuntimeError("verified state file contains target-answer fields")
    questions = load_questions(questions_path) if args.dataset == "mmtom" else json.loads(questions_path.read_text(encoding="utf-8"))
    existing = {int(row["index"]): row for row in read_jsonl(output)} if output.exists() else {}
    client = make_client(args)
    model_name = str(client.config.model)
    atomic_json(output.with_name(output.stem + "_manifest.json"), {"schema_version": 1, "dataset": args.dataset, "model": model_name, "conditions": list(FOUR_CELLS), "fixed_final_template_sha256": hashlib.sha256(FINAL_TEMPLATE.encode()).hexdigest(), "verified_states": str(args.verified), "verified_records_contain_target_answers": False})
    for human in verified:
        index = int(human["index"])
        question, labels, answer_index = format_question(args.dataset, questions[index])
        predicted_physical, predicted_social = source_traces(source[index])
        states = {
            "predicted_predicted": (predicted_physical, predicted_social),
            "human_predicted": (human["physical_trace"], predicted_social),
            "predicted_human": (predicted_physical, human["social_trace"]),
            "human_human": (human["physical_trace"], human["social_trace"]),
        }
        previous = existing.get(index, {})
        predictions, responses, costs = dict(previous.get("predictions", {})), dict(previous.get("responses", {})), dict(previous.get("call_metrics", {}))
        for cell, (physical, social) in states.items():
            if int(predictions.get(cell, -1)) >= 0:
                continue
            try:
                response, metric = one_call(client, FINAL_TEMPLATE.format(physical=physical, social=social, labels=", ".join(labels), question=question))
                predictions[cell] = parse_choice(response, labels)
                responses[cell], costs[cell] = response, metric
            except Exception as error:
                predictions[cell] = -1
                responses[cell] = f"ERROR: {type(error).__name__}: {error}"
        existing[index] = {"schema_version": 1, "dataset": args.dataset, "model": model_name, "index": index, "answer_index": answer_index, "predicted_physical_confirmed_correct": bool(human["predicted_physical_confirmed_correct"]), "predictions": predictions, "correct": {cell: int(predictions.get(cell, -1)) == answer_index for cell in FOUR_CELLS}, "responses": responses, "call_metrics": costs}
        atomic_jsonl(output, (existing[i] for i in sorted(existing)))
        atomic_json(output.with_name(output.stem + "_summary.json"), summary([existing[int(row["index"])] for row in verified if int(row["index"]) in existing], model_name))
        print(f"index={index} rows={len(existing)}/{len(verified)}", flush=True)


if __name__ == "__main__":
    main()
