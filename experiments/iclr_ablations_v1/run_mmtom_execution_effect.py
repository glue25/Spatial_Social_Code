#!/usr/bin/env python3
"""Paired raw-code versus executed-trace ablation on frozen MMToM code.

Inference never reads labels.  Both conditions use the same frozen Spatial and
Social programs, question, RGB frames, local Qwen checkpoint, prompt template,
and A/B logit decoder.  The only intervention is whether deterministic runtime
traces are exposed.  Scoring is a separate ``--score`` operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOTOM_MODEL = ROOT / "third_party/autotom" / "model"
if str(AUTOTOM_MODEL) not in sys.path:
    sys.path.insert(0, str(AUTOTOM_MODEL))

from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402


CONDITIONS = ("raw_code_only", "executed_trace")
PROMPT = """Answer the MMToM multiple-choice question from the ordered RGB
frames, problem statement, and supplied Spatial-Social program.  Compare both
choices.  Runtime records are derived deterministically from exactly the same
program and are not ground truth.  Return exactly one uppercase letter: A or B.

Spatial Code:
{spatial_code}

Social Code:
{social_code}

Executed Spatial Trace:
{spatial_trace}

Executed Social Trace:
{social_trace}

MMToM problem:
{question}

Answer:"""


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


def prompt_for(condition: str, question: str, cached: dict) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    show_trace = condition == "executed_trace"
    return PROMPT.format(
        spatial_code=str(cached["codes"]["spatial"]),
        social_code=str(cached["codes"]["social"]),
        spatial_trace=(str(cached["traces"]["spatial"]) if show_trace else "[not provided]"),
        social_trace=(str(cached["traces"]["social"]) if show_trace else "[not provided]"),
        question=question,
    )


def manifest(cached_path: Path, model_path: Path, examples: int) -> dict:
    return {
        "schema_version": 1,
        "status": "registered_not_run",
        "population": f"all {examples} aligned MMToM examples",
        "frozen_code_path": str(cached_path),
        "frozen_code_sha256": sha256(cached_path),
        "model": str(model_path),
        "conditions": list(CONDITIONS),
        "shared_inputs": "question, 8 ordered RGB frames, Spatial Code, Social Code, prompt, A/B logit decoder",
        "intervention": "executed_trace additionally exposes deterministic Spatial and Social runtime records",
        "new_extraction_calls": 0,
        "new_final_answer_calls": 2 * examples,
        "label_access_during_inference": False,
        "scoring": "separate --score pass after all predictions are frozen",
    }


def write_jsonl(path: Path, rows: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(rows[index], ensure_ascii=False) + "\n" for index in sorted(rows)),
        encoding="utf-8",
    )
    temporary.replace(path)


def predict(probabilities: dict[str, float]) -> str:
    return "a" if probabilities["A"] >= probabilities["B"] else "b"


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def score(output: Path, summary: Path, questions: list[dict], indices: list[int]) -> dict:
    predictions = load_jsonl(output)
    missing = [index for index in indices if index not in predictions]
    incomplete = [
        index for index in indices
        if index in predictions and any(name not in predictions[index].get("predictions", {}) for name in CONDITIONS)
    ]
    if missing or incomplete:
        raise ValueError(f"cannot score: missing={len(missing)} incomplete={len(incomplete)}")
    rows = []
    for index in indices:
        answer = str(questions[index]["answer"]).lower()
        row_predictions = {
            name: str(predictions[index]["predictions"][name]["prediction"]).lower()
            for name in CONDITIONS
        }
        rows.append(
            {
                "index": index,
                "episode": str(questions[index]["episode"]),
                "question_type": str(questions[index].get("question_type", "unknown")),
                "answer": answer,
                "predictions": row_predictions,
            }
        )
    metrics = {
        name: {
            "correct": sum(row["predictions"][name] == row["answer"] for row in rows),
            "total": len(rows),
        }
        for name in CONDITIONS
    }
    for value in metrics.values():
        value["accuracy"] = value["correct"] / value["total"]
    left_only = sum(
        row["predictions"]["raw_code_only"] == row["answer"]
        and row["predictions"]["executed_trace"] != row["answer"]
        for row in rows
    )
    right_only = sum(
        row["predictions"]["raw_code_only"] != row["answer"]
        and row["predictions"]["executed_trace"] == row["answer"]
        for row in rows
    )
    values = [
        int(row["predictions"]["executed_trace"] == row["answer"])
        - int(row["predictions"]["raw_code_only"] == row["answer"])
        for row in rows
    ]
    rng = random.Random(20260825)
    draws = 20000
    boot = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    by_episode: dict[str, list[int]] = {}
    for row, value in zip(rows, values):
        by_episode.setdefault(row["episode"], []).append(value)
    episode_ids = sorted(by_episode)
    cluster_boot = []
    for _ in range(draws):
        sampled = [episode_ids[rng.randrange(len(episode_ids))] for _ in episode_ids]
        sampled_values = [value for episode in sampled for value in by_episode[episode]]
        cluster_boot.append(sum(sampled_values) / len(sampled_values))
    cluster_boot.sort()
    by_question_type = {}
    for question_type in sorted({row["question_type"] for row in rows}):
        subset = [row for row in rows if row["question_type"] == question_type]
        by_question_type[question_type] = {
            name: {
                "correct": sum(row["predictions"][name] == row["answer"] for row in subset),
                "total": len(subset),
            }
            for name in CONDITIONS
        }
        for condition_metrics in by_question_type[question_type].values():
            condition_metrics["accuracy"] = condition_metrics["correct"] / condition_metrics["total"]
    report = {
        "schema_version": 1,
        "examples": len(rows),
        "conditions": metrics,
        "comparison": {
            "left": "raw_code_only",
            "right": "executed_trace",
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "accuracy_delta_right_minus_left": metrics["executed_trace"]["accuracy"]
            - metrics["raw_code_only"]["accuracy"],
            "paired_bootstrap_95_ci": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]],
            "episode_cluster_bootstrap_95_ci": [
                cluster_boot[int(0.025 * draws)],
                cluster_boot[int(0.975 * draws) - 1],
            ],
            "episode_clusters": len(episode_ids),
            "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
        },
        "by_question_type": by_question_type,
    }
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data" / "required_frames")
    parser.add_argument("--cached-code", type=Path, default=ROOT / "spatial_social_code" / "mmtom_candidate_aligned_full.jsonl")
    parser.add_argument("--model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--output", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1" / "mmtom_execution_effect_predictions.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1" / "mmtom_execution_effect_summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1" / "mmtom_execution_effect_manifest.json")
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    cached = load_jsonl(args.cached_code)
    indices = args.indices if args.indices else list(range(len(questions)))
    if sorted(cached) != list(range(600)) or len(questions) != 600:
        raise ValueError("registered full protocol requires 600 aligned examples")
    if any(index not in cached or not 0 <= index < len(questions) for index in indices):
        raise ValueError("requested index is not present in aligned inputs")
    protocol = manifest(args.cached_code, args.model_path, len(indices))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False))
        return
    if args.score:
        report = score(args.output, args.summary, questions, indices)
        protocol["status"] = "complete"
        protocol["completed_predictions"] = len(indices)
        protocol["summary_path"] = str(args.summary)
        args.manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return
    if not args.model_path.is_dir():
        parser.error(f"local model not found: {args.model_path}")

    os.environ["SPATIAL_SOCIAL_MODEL_PATH"] = str(args.model_path)
    os.environ["SPATIAL_SOCIAL_LOCAL_FILES_ONLY"] = "1"
    from local_backend import choice_probabilities, set_image_context

    existing = load_jsonl(args.output) if args.output.exists() else {}
    for index in indices:
        row = existing.get(
            index,
            {
                "schema_version": 1,
                "protocol": "paired_frozen_code_execution_effect_v1",
                "index": index,
                "episode": questions[index]["episode"],
                "predictions": {},
            },
        )
        image_paths, image_details = sample_rgb_frames(args.frames, questions[index], num_frames=8)
        set_image_context(image_paths)
        row["image_details"] = image_details
        for condition in CONDITIONS:
            if condition in row["predictions"]:
                continue
            probabilities = choice_probabilities(
                prompt_for(condition, str(questions[index]["question"]), cached[index]),
                ["A", "B"],
            )
            row["predictions"][condition] = {
                "prediction": predict(probabilities),
                "prob_a": probabilities["A"],
                "prob_b": probabilities["B"],
            }
            existing[index] = row
            write_jsonl(args.output, existing)
        print(f"completed={sum(i in existing for i in indices)}/{len(indices)} index={index}", flush=True)


if __name__ == "__main__":
    main()
