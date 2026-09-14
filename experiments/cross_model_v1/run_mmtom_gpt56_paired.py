#!/usr/bin/env python3
"""Run a strict same-evidence Direct-vs-SSC MMToM pilot with GPT-5.6 Sol.

The Spatial and Social compilers never receive answer choices.  Direct and
SSC receive the same fixed 16 RGB frames.  Codex CLI is used only as a
multimodal model transport; all tools and repository instructions are disabled.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
for import_path in (ROOT / "AutoToM/model", ROOT / "spatial_social_code", ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from runtime import UnsafeProgram, execute_social, execute_spatial  # noqa: E402
from social_code import (  # noqa: E402
    NEURAL_SOCIAL_PROMPT,
    compile_spatial_bridge,
    normalize_social_code,
    question_without_options,
)


SCHEMA_VERSION = 1
MODEL = "gpt-5.6-sol"

DIRECT_PROMPT = """Answer this MMToM multiple-choice question using only the
attached fixed ordered RGB frames and the question text. Compare both choices.
Return the required JSON answer and one short evidence-based reason.

MMToM problem:
{question}
"""

SPATIAL_PROMPT = """You are the Spatial Code compiler in a video understanding
system. Transcribe only grounded physical facts from the initial-state text,
action history, and fixed ordered RGB frames. Answer choices are deliberately
hidden. Keep world state separate from visibility. Never infer beliefs, goals,
intentions, or an answer. Missing records mean unknown, not false.

Return 4-12 literal calls using only:
S.fact(t, relation, subject, object, source="text_or_video", confidence=0.8)
S.event(t, actor, action, target, source="text_or_video", confidence=0.8)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)

No assignments, imports, loops, comments, answer labels, or prose inside the
code strings.

Answer-blind MMToM context:
{question_without_options}
"""

FINAL_PROMPT = """Answer the MMToM multiple-choice question using the attached
fixed ordered RGB frames and the validated executable traces. Raw evidence
overrides a trace; missing records mean unknown, not false. Compare A and B.
Return the required JSON answer and one short evidence-based reason.

Executed Spatial trace:
{spatial_trace}

Executed Social trace:
{social_trace}

MMToM problem:
{question}
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B"]},
        "reason": {"type": "string"},
    },
    "required": ["answer", "reason"],
    "additionalProperties": False,
}

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 16,
        }
    },
    "required": ["code"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def balanced_sample(
    indices: Iterable[int], strata: dict[int, str], total: int, seed: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[strata[index]].append(index)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[int] = []
    offset = 0
    names = sorted(groups)
    while len(selected) < total:
        added = False
        for name in names:
            if offset < len(groups[name]):
                selected.append(groups[name][offset])
                added = True
                if len(selected) == total:
                    break
        if not added:
            break
        offset += 1
    if len(selected) != total:
        raise ValueError(f"requested {total} examples but selected {len(selected)}")
    return selected


def normalize_spatial(text: str, max_lines: int = 16) -> str:
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line in seen:
            continue
        try:
            parsed = ast.parse(line)
            call = parsed.body[0].value
            valid = (
                len(parsed.body) == 1
                and isinstance(parsed.body[0], ast.Expr)
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "S"
                and call.func.attr in {"fact", "event", "visible", "unknown"}
            )
            if not valid:
                continue
            execute_spatial(line)
        except (SyntaxError, IndexError, UnsafeProgram, TypeError, ValueError):
            continue
        kept.append(line)
        seen.add(line)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept) or "S.unknown(0, 'relevant_relation', 'scene', 'target')"


def merge_social_code(neural_code: str, bridge_code: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in (neural_code + "\n" + bridge_code).splitlines():
        line = raw_line.strip()
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
    return "\n".join(lines) or "M.unknown('agent', 'social_state')"


def parse_events(stdout: str) -> tuple[dict[str, int], str | None]:
    usage: dict[str, int] = {}
    response_text: str | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                str(key): int(value)
                for key, value in event["usage"].items()
                if isinstance(value, int)
            }
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
        ):
            response_text = str(item.get("text", ""))
    return usage, response_text


def validate_payload(payload: dict[str, Any], schema_name: str) -> None:
    if schema_name == "answer":
        if set(payload) != {"answer", "reason"} or payload["answer"] not in {"A", "B"}:
            raise ValueError(f"invalid answer payload: {payload}")
        if not isinstance(payload["reason"], str):
            raise ValueError("answer reason is not a string")
    elif schema_name == "code":
        if set(payload) != {"code"} or not isinstance(payload["code"], list):
            raise ValueError(f"invalid code payload: {payload}")
        if not payload["code"] or not all(isinstance(line, str) for line in payload["code"]):
            raise ValueError("code payload has no string calls")
    else:
        raise ValueError(schema_name)


def call_codex(
    *,
    prompt: str,
    images: list[Path],
    schema_path: Path,
    schema_name: str,
    output_path: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = output_path.resolve()
    schema_path = schema_path.resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.is_file() and metadata_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        validate_payload(payload, schema_name)
        return payload, json.loads(metadata_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    overall_started = time.monotonic()
    last_error = "unknown error"
    for attempt in range(1, max_retries + 2):
        attempt_output = output_path.with_suffix(f".attempt{attempt}.json")
        event_path = output_path.with_suffix(f".attempt{attempt}.events.jsonl")
        command = [
            "codex",
            "exec",
            prompt,
            "--ephemeral",
            "--ignore-rules",
            "--disable",
            "apps",
            "--disable",
            "goals",
            "--disable",
            "shell_snapshot",
            "--disable",
            "plugins",
            "--disable",
            "unified_exec",
            "--disable",
            "code_mode_host",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            "/tmp",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(attempt_output),
        ]
        for image in images:
            command.extend(["--image", str(image.resolve())])

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            event_path.write_text(completed.stdout, encoding="utf-8")
            usage, event_text = parse_events(completed.stdout)
            attempt_record = {
                "attempt": attempt,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "usage": usage,
                "stderr_tail": completed.stderr[-1000:],
                "event_log": str(event_path.relative_to(ROOT)),
            }
            attempts.append(attempt_record)
            if completed.returncode == 0:
                raw = (
                    attempt_output.read_text(encoding="utf-8")
                    if attempt_output.is_file()
                    else event_text or ""
                )
                payload = json.loads(raw)
                validate_payload(payload, schema_name)
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metadata = {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "attempts": attempts,
                    "transport_retries": attempt - 1,
                    "elapsed_seconds": time.monotonic() - overall_started,
                    "usage": usage,
                    "output": str(output_path.relative_to(ROOT)),
                }
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return payload, metadata
            last_error = completed.stderr[-2000:] or completed.stdout[-2000:]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": None,
                    "elapsed_seconds": time.monotonic() - started,
                    "usage": {},
                    "error": last_error,
                }
            )
        if attempt <= max_retries:
            time.sleep(min(60, 2 ** (attempt - 1)))
    raise RuntimeError(
        f"Codex call failed after {max_retries + 1} attempts: {last_error}"
    )


def mcnemar_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def write_jsonl(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(rows[index], ensure_ascii=False) + "\n" for index in sorted(rows)),
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(
    rows: dict[int, dict[str, Any]], selected: list[int], model: str, effort: str
) -> dict[str, Any]:
    complete = [rows[index] for index in selected if index in rows and "error" not in rows[index]]
    direct_only = sum(row["direct_correct"] and not row["ssc_correct"] for row in complete)
    ssc_only = sum(not row["direct_correct"] and row["ssc_correct"] for row in complete)

    def condition_summary(name: str) -> dict[str, Any]:
        condition_calls = [
            call for row in complete for call in row["calls"] if call["condition"] == name
        ]
        usage_totals: Counter[str] = Counter()
        for call in condition_calls:
            usage_totals.update(call["metadata"].get("usage", {}))
        return {
            "correct": sum(bool(row[f"{name}_correct"]) for row in complete),
            "total": len(complete),
            "accuracy": (
                sum(bool(row[f"{name}_correct"]) for row in complete) / len(complete)
                if complete
                else None
            ),
            "logical_calls": len(condition_calls),
            "request_attempts": sum(
                len(call["metadata"].get("attempts", [])) for call in condition_calls
            ),
            "transport_retries": sum(
                int(call["metadata"].get("transport_retries", 0))
                for call in condition_calls
            ),
            "elapsed_seconds": sum(
                float(call["metadata"].get("elapsed_seconds", 0))
                for call in condition_calls
            ),
            "usage": dict(usage_totals),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "development_pilot_not_heldout_validation",
        "protocol": "paired_same_model_same_16_frames_answer_blind_state_compilers",
        "model": model,
        "reasoning_effort": effort,
        "selected": len(selected),
        "complete": len(complete),
        "errors": len(selected) - len(complete),
        "conditions": {
            "direct": condition_summary("direct"),
            "ssc": condition_summary("ssc"),
        },
        "paired": {
            "direct_only_correct": direct_only,
            "ssc_only_correct": ssc_only,
            "net_ssc_wins": ssc_only - direct_only,
            "mcnemar_exact_p": mcnemar_exact(direct_only, ssc_only),
        },
        "interpretation_limit": (
            "Pilot estimates transfer under a stronger backbone; it is not a held-out "
            "confirmatory result and does not equalize inference call count."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA/Benchmark/questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data/fair16_frames")
    parser.add_argument("--output", type=Path, default=ROOT / "workdirs/cross_model_v1/mmtom_gpt56_paired20.jsonl")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional prior JSONL whose completed rows seed this run without new calls.",
    )
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260905)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.total < 1 or args.limit < 0 or args.max_retries < 0:
        parser.error("total must be positive; limit and max-retries cannot be negative")

    args.questions = args.questions.resolve()
    args.frames = args.frames.resolve()
    args.output = args.output.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()

    questions = load_questions(args.questions)
    strata = {index: str(row["question_type"]) for index, row in enumerate(questions)}
    selected = balanced_sample(range(len(questions)), strata, args.total, args.sample_seed)
    if args.limit:
        selected = selected[: args.limit]

    work_root = args.output.parent / (args.output.stem + "_work")
    raw_root = work_root / "raw"
    schema_root = work_root / "schemas"
    schema_root.mkdir(parents=True, exist_ok=True)
    answer_schema_path = schema_root / "answer.json"
    code_schema_path = schema_root / "code.json"
    answer_schema_path.write_text(json.dumps(ANSWER_SCHEMA, indent=2) + "\n", encoding="utf-8")
    code_schema_path.write_text(json.dumps(CODE_SCHEMA, indent=2) + "\n", encoding="utf-8")

    evidence: dict[int, dict[str, Any]] = {}
    for index in selected:
        paths, details = sample_rgb_frames(args.frames, questions[index], num_frames=16)
        evidence[index] = {
            "frame_numbers": details["frame_numbers"],
            "paths": [str(path.relative_to(ROOT)) for path in paths],
            "sha256": digest_paths(paths),
        }
    manifest_path = args.manifest or args.output.with_name(args.output.stem + "_manifest.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "registered_development_pilot_not_yet_a_paper_result",
        "protocol": "paired_same_model_same_16_frames_answer_blind_state_compilers",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sample_seed": args.sample_seed,
        "selected_indices": selected,
        "by_question_type": dict(sorted(Counter(strata[index] for index in selected).items())),
        "conditions": ["direct", "ssc"],
        "planned_model_calls_per_sample": {"direct": 1, "ssc": 3},
        "question_file": {"path": str(args.questions.relative_to(ROOT)), "sha256": sha256(args.questions)},
        "fixed_evidence": evidence,
        "answer_visibility": {
            "direct": "full question including A/B",
            "ssc_spatial_compile": "answer choices hidden",
            "ssc_social_compile": "answer choices hidden",
            "ssc_final": "full question including A/B",
        },
        "transport": "Codex CLI multimodal inference; tools and repo instructions disabled",
        "resume_from": str(args.resume_from.relative_to(ROOT)) if args.resume_from else None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False))
        return

    existing: dict[int, dict[str, Any]] = {}
    if args.output.is_file():
        existing = {
            int(row["index"]): row
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
    if args.resume_from and args.resume_from.is_file():
        for line in args.resume_from.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.setdefault(int(row["index"]), row)
    pending = [index for index in selected if index not in existing or "error" in existing[index]]
    print(
        f"model={args.model} effort={args.reasoning_effort} selected={len(selected)} "
        f"resumed={len(selected)-len(pending)} pending={len(pending)}",
        flush=True,
    )

    for position, index in enumerate(pending, start=1):
        row = questions[index]
        paths, details = sample_rgb_frames(args.frames, row, num_frames=16)
        question = str(row["question"])
        answer = str(row["answer"]).upper()
        calls: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            direct, direct_meta = call_codex(
                prompt=DIRECT_PROMPT.format(question=question),
                images=paths,
                schema_path=answer_schema_path.resolve(),
                schema_name="answer",
                output_path=raw_root / f"{index:04d}_direct.json",
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "direct", "stage": "answer", "metadata": direct_meta})

            blind_question = question_without_options(question)
            spatial_payload, spatial_meta = call_codex(
                prompt=SPATIAL_PROMPT.format(question_without_options=blind_question),
                images=paths,
                schema_path=code_schema_path.resolve(),
                schema_name="code",
                output_path=raw_root / f"{index:04d}_ssc_spatial.json",
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "ssc", "stage": "spatial_compile", "metadata": spatial_meta})
            spatial_raw = "\n".join(spatial_payload["code"])
            spatial_code = normalize_spatial(spatial_raw)
            spatial_trace = execute_spatial(spatial_code)

            social_prompt = NEURAL_SOCIAL_PROMPT.format(evidence=blind_question)
            social_payload, social_meta = call_codex(
                prompt=social_prompt,
                images=paths,
                schema_path=code_schema_path.resolve(),
                schema_name="code",
                output_path=raw_root / f"{index:04d}_ssc_social.json",
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "ssc", "stage": "social_compile", "metadata": social_meta})
            social_raw = "\n".join(social_payload["code"])
            neural_social_code = normalize_social_code(social_raw)
            if not neural_social_code:
                neural_social_code = "M.unknown('agent', 'social_state')"
            bridge = compile_spatial_bridge(question, spatial_trace)
            social_code = merge_social_code(neural_social_code, bridge.code)
            social_trace = execute_social(social_code)

            final, final_meta = call_codex(
                prompt=FINAL_PROMPT.format(
                    spatial_trace=spatial_trace,
                    social_trace=social_trace,
                    question=question,
                ),
                images=paths,
                schema_path=answer_schema_path.resolve(),
                schema_name="answer",
                output_path=raw_root / f"{index:04d}_ssc_final.json",
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            calls.append({"condition": "ssc", "stage": "answer", "metadata": final_meta})
            result = {
                "schema_version": SCHEMA_VERSION,
                "index": index,
                "episode": row["episode"],
                "question_type": str(row["question_type"]),
                "answer": answer,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "frame_numbers": details["frame_numbers"],
                "evidence_sha256": digest_paths(paths),
                "direct_prediction": direct["answer"],
                "direct_correct": direct["answer"] == answer,
                "direct_reason": direct["reason"],
                "spatial_code": spatial_code,
                "spatial_trace": spatial_trace,
                "social_code": social_code,
                "social_trace": social_trace,
                "ssc_prediction": final["answer"],
                "ssc_correct": final["answer"] == answer,
                "ssc_reason": final["reason"],
                "calls": calls,
                "elapsed_seconds": time.monotonic() - started,
            }
            state = "complete"
        except Exception as error:
            result = {
                "schema_version": SCHEMA_VERSION,
                "index": index,
                "episode": row["episode"],
                "question_type": str(row["question_type"]),
                "answer": answer,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "frame_numbers": details["frame_numbers"],
                "evidence_sha256": digest_paths(paths),
                "calls": calls,
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            }
            state = "error"
        existing[index] = result
        write_jsonl(args.output, existing)
        summary = summarize(existing, selected, args.model, args.reasoning_effort)
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{position}/{len(pending)}] index={index} state={state} "
            f"complete={summary['complete']}/{len(selected)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
