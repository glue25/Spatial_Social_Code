"""Evaluate code-mediated MMToM understanding with a shared local VLM.

This is a deliberately controlled ablation.  Every condition uses the same
Qwen3-VL backbone and the same sampled RGB frames.  Conditions differ only in
the intermediate representation supplied before the final A/B decision:

* direct: no intermediate code
* spatial: CaP-X-inspired spatial/world-model code
* autotom: the existing AutoToM posterior
* autotom_rerank: AutoToM social code plus one final VLM decision
* spatial_social: spatial code plus executable AutoToM social code

The generated programs are compact Python-like intermediate representations.
The first pilot tests whether code mediation helps; grounding their API calls
in executable perception and probabilistic runtimes is the next experiment.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
AUTOTOM_MODEL = ROOT / "AutoToM" / "model"
if str(AUTOTOM_MODEL) not in sys.path:
    sys.path.insert(0, str(AUTOTOM_MODEL))

from local_backend import (  # noqa: E402
    choice_probabilities,
    generate,
    set_image_context,
)
from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from runtime import UnsafeProgram, execute_social, execute_spatial  # noqa: E402


CONDITIONS = (
    "direct",
    "spatial",
    "autotom",
    "autotom_rerank",
    "spatial_social",
)


SPATIAL_PROMPT = """You are the Spatial Code module, inspired by CaP-X code-as-policy but adapted to code-as-world-model for video understanding.
Transcribe physically grounded facts from the initial-state text and ordered RGB frames into an executable program. First identify the two candidate objects/locations in the final question. Include their explicitly stated initial locations, followed by every observed action relevant to distinguishing the candidates. Ignore all other objects unless they are a direct action target. Prioritize action events over irrelevant scene inventory. Never mark a fact unknown when the text explicitly states it.
The runtime object S already exists. Use ONLY literal calls in these exact forms:
S.fact(t, relation, subject, object, source="text", confidence=1.0)
S.event(t, actor, action, target, source="text", confidence=1.0)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)
Allowed relations include at, inside, on, near, open. Keep true world state separate from visibility. Never infer beliefs, goals, intentions, or an A/B answer. No assignments, variables, comments, imports, loops, markdown, or repeated calls. Output 4-12 complete calls and nothing else.

MMToM problem:
{question}
"""


ANSWER_PROMPT = """Answer the MMToM question using the raw evidence and any supplied executable understanding code. Treat code facts marked unknown or low-confidence as uncertain. AutoToM hypotheses may be ordered differently from the A/B options: match them by semantic content, not list position. Return exactly one uppercase letter: A or B.

{code_context}

MMToM problem:
{question}

Answer:"""


def _normalize_code(text: str, max_lines: int = 16) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    unique = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        try:
            parsed_line = ast.parse(line)
        except SyntaxError:
            continue
        if (
            len(parsed_line.body) != 1
            or not isinstance(parsed_line.body[0], ast.Expr)
            or not isinstance(parsed_line.body[0].value, ast.Call)
        ):
            continue
        seen.add(line)
        unique.append(line)
        if len(unique) >= max_lines:
            break
    return "\n".join(unique)


def execute_or_error(code: str, kind: str) -> str:
    try:
        return execute_spatial(code) if kind == "spatial" else execute_social(code)
    except (UnsafeProgram, NameError, AttributeError, TypeError, ValueError) as error:
        return json.dumps({"execution_error": str(error)}, ensure_ascii=False)


def _option_texts(question: str) -> dict[str, str]:
    match = re.search(
        r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*Please respond with either a or b\.?\s*$",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("could not extract A/B option text from MMToM question")
    return {"a": match.group(1).strip(), "b": match.group(2).strip()}


def _semantic_text(text: str) -> str:
    replacements = (
        ("refrigerator", "fridge"),
        ("bag of chips", "chips"),
        ("bottle of wine", "wine"),
        ("dishbowl", "dish bowl"),
        ("1st", "first"),
        ("2nd", "second"),
        ("3rd", "third"),
        ("4th", "fourth"),
        ("5th", "fifth"),
        ("6th", "sixth"),
    )
    normalized = text.lower()
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _autotom_paths(index: int, prefixes: list[str]) -> tuple[str, Path, Path]:
    parsed_root = ROOT / "AutoToM" / "results" / "parsed_result"
    metrics_root = ROOT / "AutoToM" / "results" / "metrics"
    for prefix in prefixes:
        parsed_path = parsed_root / f"{prefix}_MMToM-QA_{index}.json"
        candidates = list(
            metrics_root.glob(
                f"{prefix}_MMToM-QA_{index}_back1_reduce1_seed42_metrics*.json"
            )
        )
        if parsed_path.is_file() and candidates:
            metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
            return prefix, parsed_path, metrics_path
    raise FileNotFoundError(
        f"missing AutoToM result for index {index}; tried prefixes {prefixes}"
    )


def load_autotom_social(
    index: int,
    question: str,
    prefixes: list[str],
) -> tuple[str, str, dict[str, float | str]]:
    source_prefix, parsed_path, metrics_path = _autotom_paths(index, prefixes)
    parsed = json.loads(parsed_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    choices = parsed["choices"]
    probs = [float(value) for value in metrics["probs"]]
    if len(choices) != 2 or len(probs) != 2:
        raise ValueError(f"index {index}: expected exactly two AutoToM choices/probabilities")

    raw_options = _option_texts(question)
    normalized_options = {
        label: _semantic_text(option) for label, option in raw_options.items()
    }
    mapped: dict[str, float] = {}
    for choice, probability in zip(choices, probs, strict=True):
        normalized_choice = _semantic_text(str(choice))
        labels = [
            label
            for label, normalized_option in normalized_options.items()
            if normalized_choice == normalized_option
        ]
        if len(labels) != 1:
            raise ValueError(
                f"index {index}: AutoToM choice does not align with raw A/B options: "
                f"{choice!r} vs {raw_options!r}"
            )
        mapped[labels[0]] = probability
    if set(mapped) != {"a", "b"}:
        raise ValueError(f"index {index}: incomplete AutoToM A/B probability mapping")
    agent = str(parsed["inf_agent_name"])
    variable = str(parsed["inf_var_name"])
    evidence = "AutoToM Bayesian inverse-planning posterior from observed actions"
    lines = []
    for choice, probability in zip(choices, probs, strict=True):
        if variable == "Belief":
            lines.append(
                f"M.belief(-1, {agent!r}, {str(choice)!r}, {probability!r}, {evidence!r})"
            )
        elif variable == "Goal":
            lines.append(
                f"M.goal({agent!r}, {str(choice)!r}, {probability!r}, {evidence!r})"
            )
        else:
            lines.append(f"M.unknown({agent!r}, {variable!r})")
        lines.append(
            f"M.likelihood({agent!r}, 'observed action sequence', {str(choice)!r}, {probability!r})"
        )
    code = "\n".join(lines)
    trace = execute_or_error(code, "social")

    predicted = "a" if mapped["a"] >= mapped["b"] else "b"
    prediction = {
        "prediction": predicted,
        "prob_a": mapped["a"],
        "prob_b": mapped["b"],
        "source_prefix": source_prefix,
    }
    return code, trace, prediction


def select_indices(rows: list[dict], samples_per_type: int) -> list[int]:
    """Take a deterministic, interleaved sample from every question subtype."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_type"])].append(index)
    selected: list[int] = []
    for offset in range(samples_per_type):
        for question_type in sorted(grouped):
            candidates = grouped[question_type]
            if offset < len(candidates):
                selected.append(candidates[offset])
    return selected


def answer_probs(question: str, code_context: str) -> dict[str, float]:
    prompt = ANSWER_PROMPT.format(
        question=question,
        code_context=code_context or "Intermediate code: NONE (direct baseline)",
    )
    return choice_probabilities(prompt, ["A", "B"])


def predict(probs: dict[str, float]) -> str:
    return "a" if probs["A"] >= probs["B"] else "b"


def summarize(records: Iterable[dict]) -> dict:
    rows = list(records)
    summary: dict[str, object] = {"completed_examples": len(rows), "conditions": {}}
    for condition in CONDITIONS:
        available = [row for row in rows if condition in row.get("predictions", {})]
        correct = sum(
            row["predictions"][condition]["prediction"] == row["answer"]
            for row in available
        )
        by_type: dict[str, dict[str, int | float]] = {}
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in available:
            grouped[str(row["question_type"])].append(row)
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
    parser.add_argument(
        "--autotom-model-prefix",
        default="automated",
        help="Primary AutoToM result prefix.",
    )
    parser.add_argument(
        "--autotom-fallback-prefix",
        action="append",
        default=[],
        help="Fallback AutoToM result prefix; may be repeated.",
    )
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_pilot.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_pilot_summary.json",
    )
    args = parser.parse_args()

    rows = load_questions(args.questions)
    indices = args.indices if args.indices else select_indices(rows, args.samples_per_type)
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_id < num_shards")
    indices = [index for index in indices if index % args.num_shards == args.shard_id]
    existing: dict[int, dict] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[int(record["index"])] = record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = dict(existing)
    for index in indices:
        row = rows[index]
        record = records.get(
            index,
            {
                "index": index,
                "episode": row["episode"],
                "question_type": str(row["question_type"]),
                "answer": row["answer"].lower(),
                "codes": {},
                "traces": {},
                "predictions": {},
            },
        )
        record.setdefault("codes", {})
        record.setdefault("traces", {})
        record.setdefault("predictions", {})
        needed = [name for name in args.conditions if name not in record["predictions"]]
        if not needed:
            print(f"index={index} already complete")
            continue

        image_paths, image_details = sample_rgb_frames(
            args.frames, row, num_frames=args.num_frames
        )
        set_image_context(image_paths)
        question = row["question"]
        record["image_details"] = image_details

        needs_spatial = any(name in needed for name in ("spatial", "spatial_social"))
        if needs_spatial and "spatial" not in record["codes"]:
            record["codes"]["spatial"] = _normalize_code(
                generate(SPATIAL_PROMPT.format(question=question), max_new_tokens=320)
            )
        if needs_spatial and "spatial" not in record["traces"]:
            record["traces"]["spatial"] = execute_or_error(
                record["codes"]["spatial"], "spatial"
            )

        needs_autotom = any(
            name in needed
            for name in ("autotom", "autotom_rerank", "spatial_social")
        )
        if needs_autotom and "autotom" not in record["codes"]:
            social_code, social_trace, autotom_prediction = load_autotom_social(
                index,
                question,
                [args.autotom_model_prefix, *args.autotom_fallback_prefix],
            )
            record["codes"]["autotom"] = social_code
            record["traces"]["autotom"] = social_trace
            record["predictions"]["autotom"] = autotom_prediction

        contexts = {
            "direct": "",
            "spatial": (
                "Spatial Code:\n" + record["codes"].get("spatial", "")
                + "\nExecuted Spatial Trace:\n" + record["traces"].get("spatial", "")
            ),
            "autotom": "",
            "autotom_rerank": (
                "AutoToM Social Code:\n"
                + record["codes"].get("autotom", "")
                + "\n\nExecuted AutoToM Social Trace:\n"
                + record["traces"].get("autotom", "")
            ),
            "spatial_social": (
                "Spatial Code:\n"
                + record["codes"].get("spatial", "")
                + "\n\nExecuted Spatial Trace:\n"
                + record["traces"].get("spatial", "")
                + "\n\nAutoToM Social Code:\n"
                + record["codes"].get("autotom", "")
                + "\n\nExecuted AutoToM Social Trace:\n"
                + record["traces"].get("autotom", "")
            ),
        }
        for condition in needed:
            if condition == "autotom":
                continue
            probs = answer_probs(question, contexts[condition])
            record["predictions"][condition] = {
                "prediction": predict(probs),
                "prob_a": probs["A"],
                "prob_b": probs["B"],
            }

        records[index] = record
        with args.output.open("w", encoding="utf-8") as output:
            for saved_index in sorted(records):
                output.write(json.dumps(records[saved_index], ensure_ascii=False) + "\n")
        current = summarize(records[index] for index in indices if index in records)
        args.summary.write_text(json.dumps(current, indent=2) + "\n")
        short = {
            name: record["predictions"][name]["prediction"]
            for name in args.conditions
            if name in record["predictions"]
        }
        print(
            f"index={index} type={row['question_type']} answer={row['answer']} "
            f"predictions={short}"
        )

    final = summarize(records[index] for index in indices if index in records)
    args.summary.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
