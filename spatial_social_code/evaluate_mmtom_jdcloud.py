"""Paired Gemini evaluation of direct and Spatial-Social MMToM answering."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
AUTOTOM_MODEL = ROOT / "AutoToM" / "model"
MA_EGOQA = ROOT / "MA-EgoQA"
for import_path in (AUTOTOM_MODEL, MA_EGOQA, ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from evaluate_spatial_social import (  # noqa: E402
    execute_spatial_or_error,
    normalize_spatial_code,
    option_blind_spatial_prompt,
)
from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from social_code import compile_social_code, question_without_options  # noqa: E402


SCHEMA_VERSION = 5
DEFAULT_OUTPUT = ROOT / "spatial_social_code" / "mmtom_option_blind_gemini3pro.jsonl"
DIRECT_PROMPT = """Answer the MMToM multiple-choice question from the initial
state, action history, and ordered RGB frames. Compare both choices against the
evidence. Return exactly one uppercase letter: A or B.

MMToM problem:
{question}

Answer:"""

DIRECT_REPAIR_PROMPT = """Answer the MMToM problem from the ordered RGB frames.
You must output exactly one character, A or B. Do not explain your answer.

MMToM problem:
{question}

Answer:"""

SPATIAL_REPAIR_PROMPT = """Compile additional Spatial Code only for the
uncovered claims. Use directly grounded initial-state text and ordered RGB
frames; do not select A or B. Include claim_ids in every grounded call. Do not
repeat existing records. Output only literal S.fact/S.event/S.visible calls.

Uncovered claims:
{uncovered}

Existing Spatial trace:
{spatial_trace}

MMToM problem:
{question}
"""

FINAL_PROMPT = """Answer the MMToM multiple-choice question from the ordered RGB
frames and the option-blind Spatial and Social traces. Raw evidence overrides
code, and a missing record means unknown rather than false. Compare both
choices. Return exactly one uppercase letter: A or B.

Spatial trace:
{spatial_trace}

Social trace:
{social_trace}

MMToM problem:
{question}

Answer:
"""

FINAL_REPAIR_PROMPT = """Return exactly one character, A or B, for the MMToM
question below. Do not explain your answer.

Spatial trace:
{spatial_trace}

Social trace:
{social_trace}

MMToM problem:
{question}

Answer:
"""


def image_parts(paths: Iterable[str | Path]) -> list[dict]:
    parts = []
    for path_like in paths:
        path = Path(path_like)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append(
            {
                "inline_data": {
                    "mimeType": mime,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            }
        )
    return parts


def parse_ab(text: str) -> str:
    stripped = text.strip().upper()
    match = re.search(r"(?:ANSWER\s*[:=]\s*)?\b([AB])\b", stripped)
    return match.group(1).lower() if match else ""


def covered_claims(trace: str, valid_claim_ids: set[str]) -> set[str]:
    covered: set[str] = set()
    for line in trace.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = str(record.get("values", {}).get("claim_ids", ""))
        covered.update(value.strip() for value in raw.split(",") if value.strip() in valid_claim_ids)
    return covered


def mcnemar_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


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


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {
        int(row["index"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def process_one(
    index: int,
    row: dict,
    frames_root: Path,
    num_frames: int,
    client: JDCloudGeminiClient,
) -> dict:
    started = time.monotonic()
    paths, image_details = sample_rgb_frames(frames_root, row, num_frames=num_frames)
    media = image_parts(paths)
    question = str(row["question"])
    option_blind_question = question_without_options(question)

    direct_raw = client.generate_parts(
        media + [{"text": DIRECT_PROMPT.format(question=question)}],
        max_output_tokens=1024,
    )
    direct_prediction = parse_ab(direct_raw)
    direct_retry_used = not direct_prediction
    if direct_retry_used:
        direct_raw = client.generate_parts(
            media + [{"text": DIRECT_REPAIR_PROMPT.format(question=question)}],
            max_output_tokens=1024,
        )
        direct_prediction = parse_ab(direct_raw)
    if not direct_prediction:
        raise RuntimeError("direct baseline did not return A or B after retry")

    spatial_raw = client.generate_parts(
        media + [{"text": option_blind_spatial_prompt(question)}],
        max_output_tokens=4096,
    )
    spatial_code = normalize_spatial_code(spatial_raw)
    spatial_trace = execute_spatial_or_error(spatial_code)

    def generate_with_frames(prompt: str, max_new_tokens: int = 512) -> str:
        return client.generate_parts(
            media + [{"text": prompt}],
            max_output_tokens=max(2048, max_new_tokens),
        )

    social = compile_social_code(
        "hybrid",
        option_blind_question,
        generate=generate_with_frames,
        spatial_trace=spatial_trace,
        include_external_bayesian=False,
    )
    final_raw = client.generate_parts(
        media
        + [
            {
                "text": FINAL_PROMPT.format(
                    spatial_trace=spatial_trace,
                    social_trace=social.trace,
                    question=question,
                )
            }
        ],
        max_output_tokens=1024,
    )
    method_prediction = parse_ab(final_raw)
    final_retry_used = not method_prediction
    if final_retry_used:
        final_raw = client.generate_parts(
            media
            + [
                {
                    "text": FINAL_REPAIR_PROMPT.format(
                        spatial_trace=spatial_trace,
                        social_trace=social.trace,
                        question=question,
                    )
                }
            ],
            max_output_tokens=1024,
        )
        method_prediction = parse_ab(final_raw)
    if not method_prediction:
        raise RuntimeError("final reasoner did not return A or B after retry")
    answer = str(row["answer"]).lower()
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "Option-blind Spatial-Social Code",
        "model": client.config.model,
        "index": index,
        "episode": row["episode"],
        "question_type": str(row["question_type"]),
        "answer": answer,
        "image_details": image_details,
        "direct_raw": direct_raw,
        "direct_retry_used": direct_retry_used,
        "direct_prediction": direct_prediction,
        "direct_correct": direct_prediction == answer,
        "spatial_raw": spatial_raw,
        "spatial_code": spatial_code,
        "spatial_trace": spatial_trace,
        "social_code": social.code,
        "social_trace": social.trace,
        "social_components": list(social.components),
        "social_reasoning": social.diagnostics,
        "final_raw": final_raw,
        "final_retry_used": final_retry_used,
        "answer_reasoning": {"name": "same_backbone_final_reasoner"},
        "prediction": method_prediction,
        "correct": method_prediction == answer,
        "elapsed_seconds": time.monotonic() - started,
    }


def metrics(rows: list[dict], field: str) -> dict:
    correct = sum(bool(row[field]) for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None}


def summarize(records: Iterable[dict], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "correct" in row and "direct_correct" in row]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["question_type"])].append(row)
    baseline_only = sum(row["direct_correct"] and not row["correct"] for row in rows)
    method_only = sum(not row["direct_correct"] and row["correct"] for row in rows)
    both = sum(row["direct_correct"] and row["correct"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": "paired_same_model_same_rgb_frames",
        "model": model,
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "overall": {
            "direct_baseline": metrics(rows, "direct_correct"),
            "standalone_spatial_social": metrics(rows, "correct"),
        },
        "contingency": {
            "both_correct": both,
            "baseline_only_correct": baseline_only,
            "standalone_only_correct": method_only,
            "both_wrong": len(rows) - both - baseline_only - method_only,
            "net_standalone_wins": method_only - baseline_only,
            "mcnemar_exact_p": mcnemar_exact(baseline_only, method_only),
        },
        "by_question_type": {
            name: {
                "direct_baseline": metrics(group, "direct_correct"),
                "standalone_spatial_social": metrics(group, "correct"),
            }
            for name, group in sorted(by_type.items())
        },
    }


def write_outputs(path: Path, records: dict[int, dict], selected: list[int], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)
    report = summarize([records[i] for i in selected if i in records], model)
    summary_path = path.with_name(path.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary_summary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data" / "required_frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--samples-per-type", type=int, default=100)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--keep-errors", action="store_true")
    args = parser.parse_args()

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(
        config,
        timeout_seconds=600,
        max_attempts=5,
        min_request_interval_seconds=args.min_request_interval,
    )
    rows = load_questions(args.questions)
    selected = args.indices if args.indices else select_indices(rows, args.samples_per_type)
    existing = load_jsonl(args.output)
    pending = [
        i
        for i in selected
        if i not in existing
        or ("error" in existing[i] and not args.keep_errors)
        or not str(existing[i].get("direct_prediction", ""))
    ]
    print(
        f"model={config.model!r} selected={len(selected)} resumed={len(selected)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, i, rows[i], args.frames, args.num_frames, client): i
            for i in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "correct" if record["correct"] else "wrong"
            except Exception as error:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "model": config.model,
                    "index": index,
                    "episode": rows[index]["episode"],
                    "question_type": str(rows[index]["question_type"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            existing[index] = record
            write_outputs(args.output, existing, selected, config.model)
            report = summarize([existing[i] for i in selected if i in existing], config.model)
            print(
                f"index={index} type={rows[index]['question_type']} state={state} "
                f"done={report['completed']}/{len(selected)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
