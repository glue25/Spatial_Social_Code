#!/usr/bin/env python3
"""Question-independent, compile-once MMToM multi-query pilot with GPT-5.6 Sol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "AutoToM/model", ROOT / "spatial_social_code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from runtime import execute_social, execute_spatial  # noqa: E402
from social_code import normalize_social_code  # noqa: E402
from experiments.cross_model_v1.run_mmtom_gpt56_paired import (  # noqa: E402
    ANSWER_SCHEMA,
    CODE_SCHEMA,
    call_codex,
    digest_paths,
    mcnemar_exact,
    normalize_spatial,
)


SPATIAL_PROMPT = """Compile query-independent Spatial Code for this scene window.
No question or answer choice is supplied. Record persistent entities, locations,
visibility/access conditions, and ordered physical events. Never infer an answer,
goal, belief, or intent. Return literal calls only in the required JSON code list:
S.fact(t, relation, subject, object, source="text_or_video", confidence=0.8)
S.event(t, actor, action, target, source="text_or_video", confidence=0.8)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)

Question-free scene description:
{scene}
"""

SOCIAL_PROMPT = """Compile query-independent, agent-indexed Social Code from the
question-free scene description and executed Spatial trace. Preserve who observed
which event, supported information states, goals, roles, and unknowns. A world fact
is not automatically an agent belief. Communication is an event, not an action plan.
Never infer an answer and never write a policy. Return literal calls only in the
required JSON code list using M.observe, M.belief, M.goal, M.intent, or M.unknown.

Question-free scene description:
{scene}

Executed Spatial trace:
{spatial_trace}
"""

DIRECT_PROMPT = """Answer this MMToM multiple-choice question from the fixed ordered
RGB frames and supplied problem. Compare A and B. Return the required JSON answer.

Problem:
{question}
"""

FINAL_PROMPT = """Answer this MMToM multiple-choice question from the fixed ordered
RGB frames and the immutable scene-level Spatial-Social Code compiled before any
question was shown. Missing records are unknown. Return the required JSON answer.

Spatial trace:
{spatial_trace}

Social trace:
{social_trace}

Problem:
{question}
"""


def scene_only(question: str) -> str:
    marker = "\nQuestion:"
    return question.split(marker, 1)[0].strip()


def groups_with_identical_evidence(
    questions: list[dict[str, Any]], frames: Path
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    paths_by_key: dict[tuple[str, tuple[str, ...]], list[Path]] = {}
    for index, row in enumerate(questions):
        try:
            paths, _ = sample_rgb_frames(frames, row, num_frames=16)
        except FileNotFoundError:
            continue
        key = (str(row["episode"]), tuple(str(path.resolve()) for path in paths))
        grouped[key].append(index)
        paths_by_key[key] = paths
    result = []
    for key, indices in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[1])):
        if len(indices) < 2:
            continue
        result.append(
            {
                "episode": key[0],
                "indices": indices,
                "paths": paths_by_key[key],
                "evidence_sha256": digest_paths(paths_by_key[key]),
            }
        )
    return result


def write_jsonl(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(rows[key], ensure_ascii=False) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(rows: dict[int, dict[str, Any]], selected: int, model: str) -> dict[str, Any]:
    complete = [rows[key] for key in sorted(rows) if "error" not in rows[key]]
    questions = [question for row in complete for question in row["questions"]]
    direct_only = sum(item["direct_correct"] and not item["ssc_correct"] for item in questions)
    ssc_only = sum(not item["direct_correct"] and item["ssc_correct"] for item in questions)
    usage: Counter[str] = Counter()
    calls = []
    for row in complete:
        calls.extend(row["calls"])
        for call in row["calls"]:
            usage.update(call["metadata"].get("usage", {}))
    return {
        "schema_version": 1,
        "status": "development_pilot_not_heldout_validation",
        "protocol": "question_independent_compile_once_multi_query_same_16_frames",
        "model": model,
        "selected_groups": selected,
        "complete_groups": len(complete),
        "questions": len(questions),
        "mean_questions_per_compilation": len(questions) / len(complete) if complete else None,
        "direct_accuracy": sum(item["direct_correct"] for item in questions) / len(questions) if questions else None,
        "ssc_accuracy": sum(item["ssc_correct"] for item in questions) / len(questions) if questions else None,
        "group_all_correct": {
            "direct": sum(all(item["direct_correct"] for item in row["questions"]) for row in complete) / len(complete) if complete else None,
            "ssc": sum(all(item["ssc_correct"] for item in row["questions"]) for row in complete) / len(complete) if complete else None,
        },
        "paired": {
            "direct_only_correct": direct_only,
            "ssc_only_correct": ssc_only,
            "mcnemar_exact_p": mcnemar_exact(direct_only, ssc_only),
        },
        "logical_calls": {
            "direct": sum(call["condition"] == "direct" for call in calls),
            "ssc_compile": sum(call["stage"] in {"spatial_compile", "social_compile"} for call in calls),
            "ssc_answer": sum(call["stage"] == "ssc_answer" for call in calls),
        },
        "usage": dict(usage),
        "claim_boundary": "Compiler sees scene evidence but no question or choices; development subset with repeated evidence, not held-out confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA/Benchmark/questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data/fair16_frames")
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "workdirs/cross_model_v1/mmtom_gpt56_compile_once20.jsonl")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    available = groups_with_identical_evidence(questions, args.frames)
    selected = available[: args.groups]
    if len(selected) != args.groups:
        raise ValueError(f"requested {args.groups} groups but only {len(selected)} available")
    manifest = {
        "schema_version": 1,
        "status": "registered_development_pilot_not_yet_a_paper_result",
        "protocol": "question_independent_compile_once_multi_query_same_16_frames",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "groups": [
            {"group": index, "episode": row["episode"], "indices": row["indices"], "evidence_sha256": row["evidence_sha256"]}
            for index, row in enumerate(selected)
        ],
        "compiler_visibility": "question-free scene prefix only; no question stem or choices",
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    work = args.output.parent / (args.output.stem + "_work")
    raw = work / "raw"
    schemas = work / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    answer_schema = schemas / "answer.json"
    code_schema = schemas / "code.json"
    answer_schema.write_text(json.dumps(ANSWER_SCHEMA, indent=2) + "\n", encoding="utf-8")
    code_schema.write_text(json.dumps(CODE_SCHEMA, indent=2) + "\n", encoding="utf-8")
    existing = {
        int(row["group"]): row
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    } if args.output.exists() else {}

    for group_index, group in enumerate(selected):
        if group_index in existing and "error" not in existing[group_index]:
            continue
        started = time.monotonic()
        calls: list[dict[str, Any]] = []
        # The compiler receives one canonical question-free description.  All
        # members share byte-identical frames and the same episode/time window.
        scene = scene_only(str(questions[group["indices"][0]]["question"]))
        try:
            spatial_payload, metadata = call_codex(
                prompt=SPATIAL_PROMPT.format(scene=scene), images=group["paths"],
                schema_path=code_schema, schema_name="code",
                output_path=raw / f"g{group_index:03d}_spatial.json", model=args.model,
                reasoning_effort=args.reasoning_effort, timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "ssc", "stage": "spatial_compile", "metadata": metadata})
            spatial_code = normalize_spatial("\n".join(spatial_payload["code"]))
            spatial_trace = execute_spatial(spatial_code)
            social_payload, metadata = call_codex(
                prompt=SOCIAL_PROMPT.format(scene=scene, spatial_trace=spatial_trace),
                images=group["paths"], schema_path=code_schema, schema_name="code",
                output_path=raw / f"g{group_index:03d}_social.json", model=args.model,
                reasoning_effort=args.reasoning_effort, timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "ssc", "stage": "social_compile", "metadata": metadata})
            social_code = normalize_social_code("\n".join(social_payload["code"])) or "M.unknown('agent', 'social_state')"
            social_trace = execute_social(social_code)
            question_results = []
            for question_index in group["indices"]:
                item = questions[question_index]
                question = str(item["question"])
                gold = str(item["answer"]).upper()
                direct, metadata = call_codex(
                    prompt=DIRECT_PROMPT.format(question=question), images=group["paths"],
                    schema_path=answer_schema, schema_name="answer",
                    output_path=raw / f"g{group_index:03d}_q{question_index:04d}_direct.json",
                    model=args.model, reasoning_effort=args.reasoning_effort,
                    timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
                )
                calls.append({"condition": "direct", "stage": "direct_answer", "metadata": metadata})
                final, metadata = call_codex(
                    prompt=FINAL_PROMPT.format(spatial_trace=spatial_trace, social_trace=social_trace, question=question),
                    images=group["paths"], schema_path=answer_schema, schema_name="answer",
                    output_path=raw / f"g{group_index:03d}_q{question_index:04d}_ssc.json",
                    model=args.model, reasoning_effort=args.reasoning_effort,
                    timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
                )
                calls.append({"condition": "ssc", "stage": "ssc_answer", "metadata": metadata})
                question_results.append({
                    "index": question_index, "answer": gold,
                    "direct_prediction": direct["answer"], "direct_correct": direct["answer"] == gold,
                    "ssc_prediction": final["answer"], "ssc_correct": final["answer"] == gold,
                })
            result = {
                "schema_version": 1, "group": group_index, "episode": group["episode"],
                "indices": group["indices"], "evidence_sha256": group["evidence_sha256"],
                "scene_digest": __import__("hashlib").sha256(scene.encode()).hexdigest(),
                "spatial_code": spatial_code, "spatial_trace": spatial_trace,
                "social_code": social_code, "social_trace": social_trace,
                "questions": question_results, "calls": calls,
                "elapsed_seconds": time.monotonic() - started,
            }
        except Exception as error:
            result = {
                "schema_version": 1, "group": group_index, "episode": group["episode"],
                "indices": group["indices"], "evidence_sha256": group["evidence_sha256"],
                "calls": calls, "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        existing[group_index] = result
        write_jsonl(args.output, existing)
        summary = summarize(existing, len(selected), args.model)
        args.output.with_name(args.output.stem + "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"group={group_index} complete={summary['complete_groups']}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
