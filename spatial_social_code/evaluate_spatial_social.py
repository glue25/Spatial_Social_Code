"""Evaluate the standalone Spatial-Social Code method on MMToM-QA.

Public conditions stay deliberately small: direct, spatial, social, and
spatial_social.  Social inference backends are implementation choices selected
with ``--social-source`` rather than being presented as separate methods.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SHARED_MODEL_CODE = ROOT / "third_party/autotom" / "model"
if str(SHARED_MODEL_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_MODEL_CODE))

from local_backend import choice_probabilities, generate, set_image_context  # noqa: E402
from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from runtime import UnsafeProgram, execute_spatial  # noqa: E402
from social_code import (
    SOCIAL_SOURCES,
    compile_social_code,
    question_without_options,
)


SCHEMA_VERSION = 7
CONDITIONS = ("direct", "spatial", "social", "spatial_social")


SPATIAL_PROMPT = """You are the Spatial Code compiler in a video understanding system.
Transcribe physically grounded facts from the option-free initial-state text, action history, and ordered RGB frames into an executable program. Record relevant objects, locations, stated initial locations, and observed actions. Keep world state separate from visibility. Never infer beliefs, goals, intentions, answer options, or an A/B answer. Missing records mean unknown, not false.
The runtime object S already exists. Use ONLY literal calls in these forms:
S.fact(t, relation, subject, object, source="text", confidence=1.0)
S.event(t, actor, action, target, source="text", confidence=1.0)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)
No assignments, variables, imports, loops, markdown, comments, or repeated calls. Output 4-12 complete calls and nothing else.

MMToM problem:
{question}
"""


def option_blind_spatial_prompt(question: str) -> str:
    """Build the Physical compiler prompt with all answer options removed."""
    return SPATIAL_PROMPT.format(question=question_without_options(question))


ANSWER_PROMPT = """Answer the MMToM question using the raw evidence and supplied executable understanding code. Compare both choices against the raw evidence before deciding. Treat unknown, omitted, or low-confidence records as uncertain; absence from code is not evidence that something did not happen. Raw evidence overrides intermediate code. Return exactly one uppercase letter: A or B.

In a Social Trace, observations are perceptual evidence and beliefs are the agent's persistent updated state. The trace is answer-independent: do not look for option scores inside it. Explicit contradictions mean that the weaker or older claim was discounted. External action_likelihood records, when present, are supporting evidence rather than answer labels.

{code_context}

MMToM problem:
{question}

Answer:"""


def normalize_spatial_code(text: str, max_lines: int = 16) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    kept = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        try:
            parsed = ast.parse(line)
        except SyntaxError:
            continue
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
            continue
        call = parsed.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "S":
            continue
        if call.func.attr not in {"fact", "event", "visible", "unknown"}:
            continue
        try:
            execute_spatial(line)
        except UnsafeProgram:
            continue
        seen.add(line)
        kept.append(line)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def execute_spatial_or_error(code: str) -> str:
    try:
        return execute_spatial(code)
    except (UnsafeProgram, NameError, AttributeError, TypeError, ValueError) as error:
        return json.dumps({"execution_error": str(error)}, ensure_ascii=False)


def select_indices(rows: list[dict], samples_per_type: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_type"])].append(index)
    return [
        candidates[offset]
        for offset in range(samples_per_type)
        for _, candidates in sorted(grouped.items())
        if offset < len(candidates)
    ]


def answer_probs(question: str, code_context: str) -> dict[str, float]:
    prompt = ANSWER_PROMPT.format(
        question=question,
        code_context=code_context or "Intermediate code: NONE (direct baseline)",
    )
    return choice_probabilities(prompt, ["A", "B"])


def predict(probabilities: dict[str, float]) -> str:
    return "a" if probabilities["A"] >= probabilities["B"] else "b"


def summarize(records: Iterable[dict]) -> dict:
    rows = list(records)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method": "Spatial-Social Code",
        "social_design": "option_blind_unweighted_state_v2",
        "completed_examples": len(rows),
        "conditions": {},
    }
    for condition in CONDITIONS:
        available = [row for row in rows if condition in row.get("predictions", {})]
        correct = sum(
            row["predictions"][condition]["prediction"] == row["answer"]
            for row in available
        )
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in available:
            grouped[str(row["question_type"])].append(row)
        by_type = {}
        for question_type, group in sorted(grouped.items()):
            type_correct = sum(
                row["predictions"][condition]["prediction"] == row["answer"]
                for row in group
            )
            by_type[question_type] = {
                "correct": type_correct,
                "total": len(group),
                "accuracy": type_correct / len(group),
            }
        summary["conditions"][condition] = {
            "correct": correct,
            "total": len(available),
            "accuracy": correct / len(available) if available else None,
            "by_question_type": by_type,
        }
    return summary


def write_checkpoint(path: Path, records: dict[int, dict]) -> None:
    """Atomically publish a complete JSONL checkpoint for live readers."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for saved_index in sorted(records):
            output.write(json.dumps(records[saved_index], ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=ROOT / "MMToM-QA-data" / "required_frames",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--social-source", choices=SOCIAL_SOURCES, default="hybrid")
    parser.add_argument(
        "--include-autotom-evidence",
        action="store_true",
        help="Legacy/ablation-only external evidence; rejected by clean neural and hybrid paths.",
    )
    parser.add_argument(
        "--hybrid-include-bayesian",
        dest="include_autotom_evidence",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autotom-model-prefix", default="automated_qwen3vl_mm_required_aligned"
    )
    parser.add_argument(
        "--autotom-fallback-prefix",
        action="append",
        default=["automated_qwen3vl_mm_required", "automated"],
    )
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_spatial_social_hybrid.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / "spatial_social_code"
        / "mmtom_spatial_social_hybrid_summary.json",
    )
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_id < num_shards")

    rows = load_questions(args.questions)
    indices = args.indices if args.indices else select_indices(rows, args.samples_per_type)
    indices = [index for index in indices if index % args.num_shards == args.shard_id]

    records: dict[int, dict] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"legacy record in {args.output}; use a new output path")
            if record.get("social_source") != args.social_source:
                raise ValueError(
                    f"record source {record.get('social_source')} != {args.social_source}"
                )
            if bool(record.get("external_bayesian_evidence", False)) != bool(
                args.include_autotom_evidence
            ):
                raise ValueError(
                    "existing record uses a different external Bayesian evidence setting"
                )
            records[int(record["index"])] = record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in indices:
        row = rows[index]
        record = records.get(
            index,
            {
                "schema_version": SCHEMA_VERSION,
                "method": "Spatial-Social Code",
                "social_design": "option_blind_unweighted_state_v2",
                "social_source": args.social_source,
                "social_components": [],
                "social_reasoning": {},
                "external_bayesian_evidence": args.include_autotom_evidence,
                "index": index,
                "episode": row["episode"],
                "question_type": str(row["question_type"]),
                "answer": row["answer"].lower(),
                "codes": {},
                "traces": {},
                "predictions": {},
            },
        )
        needed = [name for name in args.conditions if name not in record["predictions"]]
        if not needed:
            print(f"index={index} already complete")
            continue

        image_paths, image_details = sample_rgb_frames(
            args.frames, row, num_frames=args.num_frames
        )
        set_image_context(image_paths)
        record["image_details"] = image_details
        question = row["question"]

        needs_spatial = any(name in needed for name in ("spatial", "spatial_social"))
        if needs_spatial and "spatial" not in record["codes"]:
            record["codes"]["spatial"] = normalize_spatial_code(
                generate(option_blind_spatial_prompt(question), max_new_tokens=320)
            )
            record["traces"]["spatial"] = execute_spatial_or_error(
                record["codes"]["spatial"]
            )

        needs_social = any(name in needed for name in ("social", "spatial_social"))
        if needs_social and "social" not in record["codes"]:
            prefixes = [args.autotom_model_prefix, *args.autotom_fallback_prefix]

            def bayesian_loader() -> tuple[str, str]:
                from autotom_adapter import load_autotom_social

                code, trace, prediction = load_autotom_social(index, question, prefixes)
                record["bayesian_prediction"] = prediction
                return code, trace

            result = compile_social_code(
                args.social_source,
                question_without_options(question),
                generate=generate,
                spatial_trace=record["traces"].get("spatial", ""),
                bayesian_loader=bayesian_loader,
                include_external_bayesian=args.include_autotom_evidence,
            )
            record["codes"]["social"] = result.code
            record["codes"]["social_raw_evidence"] = result.raw_code
            record["traces"]["social"] = result.trace
            record["social_components"] = list(result.components)
            record["social_reasoning"] = result.diagnostics

        contexts = {
            "direct": "",
            "spatial": (
                "Spatial Code:\n"
                + record["codes"].get("spatial", "")
                + "\nExecuted Spatial Trace:\n"
                + record["traces"].get("spatial", "")
            ),
            "social": (
                "Social Code:\n"
                + record["codes"].get("social", "")
                + "\nExecuted Social Trace:\n"
                + record["traces"].get("social", "")
            ),
            "spatial_social": (
                "Spatial Code:\n"
                + record["codes"].get("spatial", "")
                + "\nExecuted Spatial Trace:\n"
                + record["traces"].get("spatial", "")
                + "\n\nSocial Code:\n"
                + record["codes"].get("social", "")
                + "\nExecuted Social Trace:\n"
                + record["traces"].get("social", "")
            ),
        }
        for condition in needed:
            probabilities = answer_probs(question, contexts[condition])
            record["predictions"][condition] = {
                "prediction": predict(probabilities),
                "prob_a": probabilities["A"],
                "prob_b": probabilities["B"],
            }

        records[index] = record
        write_checkpoint(args.output, records)
        current = summarize(records[i] for i in indices if i in records)
        write_summary(args.summary, current)
        short = {
            name: record["predictions"][name]["prediction"]
            for name in args.conditions
            if name in record["predictions"]
        }
        print(
            f"index={index} type={row['question_type']} source={args.social_source} "
            f"answer={row['answer']} predictions={short}"
        )

    final = summarize(records[i] for i in indices if i in records)
    write_summary(args.summary, final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
