"""Strong no-code baselines for the frozen MA-EgoQA component sample.

All conditions use the same model, question, choices, and saved raw-video
clips as the component ablation.  They never receive Spatial/Social Code.
The runner is checkpointed per example and records logical calls and latency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
MA_ROOT = ROOT / "MA-EgoQA"
if str(MA_ROOT) not in sys.path:
    sys.path.insert(0, str(MA_ROOT))

from egomas.src.evaluate_video_component_ablation_jdcloud import (  # noqa: E402
    DEFAULT_BENCHMARK,
    DEFAULT_FFMPEG,
    DEFAULT_PAIRED,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    load_jsonl,
)
from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.run_video_grounded_v2_jdcloud import _answer_index  # noqa: E402
from egomas.src.run_video_spatial_social_jdcloud import video_parts  # noqa: E402
from egomas.utils.prompt_helpers import build_question_prompt  # noqa: E402


DEFAULT_COMPONENT_OUTPUT = (
    MA_ROOT / "data/baselines/video_component_ablation_balanced100.jsonl"
)
DEFAULT_MANIFEST = ROOT / "workdirs/iclr_ablations_v1/ma_component_ablation_manifest.json"
DEFAULT_OUTPUT = ROOT / "workdirs/iclr_ablations_v1/ma_reasoning_baselines.jsonl"
SCHEMA_VERSION = 1
NEW_CONDITIONS = (
    "option_decomposition",
    "natural_language_scratchpad",
    "three_call_natural_language",
)


OPTION_DECOMPOSITION_PROMPT = """Answer the multiple-choice question using
only the supplied raw videos. Check every option separately against visible or
audible evidence before choosing. Do not use program code, a symbolic state,
or an external answer. Return JSON only:
{{"answer":"A","reason":"one short sentence",
 "option_checks":[{{"option":"A","assessment":"short evidence check"}}]}}

Question:
{question}
"""


SCRATCHPAD_PROMPT = """Answer the multiple-choice question using only the
supplied raw videos. First form a concise natural-language scratchpad with the
chronology, who observed each event, and what remains unknown. Then compare
the options. Do not write program code or a symbolic Spatial/Social trace.
Return JSON only:
{{"answer":"A","reason":"one short sentence",
 "scratchpad":"concise natural-language notes"}}

Question:
{question}
"""


NOTES_PROMPT = """Study the supplied raw videos for the question below.
Write neutral natural-language notes covering relevant events, chronology,
people, observations, and uncertainties. The notes may be task-conditioned,
but do not choose, rank, or recommend an answer and do not write program code
or a symbolic schema.

Question:
{question}
"""


ANALYSIS_PROMPT = """Using the supplied raw videos and the neutral notes,
compare every answer option in natural language. State evidence for and
against each option. Do not use program code or a symbolic schema, and do not
give a final answer yet.

Neutral notes:
{notes}

Question:
{question}
"""


FINAL_PROMPT = """Choose the answer using the supplied raw videos and the
natural-language notes and option analysis. These notes are fallible; direct
video evidence overrides them. Return JSON only:
{{"answer":"A","reason":"one short sentence"}}

Neutral notes:
{notes}

Option analysis:
{analysis}

Question:
{question}
"""


def selected_indices(manifest: Path) -> list[int]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected = payload.get("selected_indices")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"no frozen selected_indices in {manifest}")
    return [int(index) for index in selected]


def generate(
    client: JDCloudGeminiClient,
    media: list[dict[str, Any]],
    prompt: str,
    max_output_tokens: int = 1024,
) -> str:
    return client.generate_parts(media + [{"text": prompt}], max_output_tokens=max_output_tokens)


def process_one(
    index: int,
    benchmark_row: dict,
    paired_row: dict,
    component_row: dict,
    client: JDCloudGeminiClient,
    video_root: Path,
    proxy_root: Path,
    ffmpeg: Path,
    existing_row: dict | None = None,
) -> dict:
    started = time.monotonic()
    clips = [dict(clip) for clip in paired_row["video_clips"]]
    media = video_parts(clips, video_root, proxy_root, ffmpeg)
    question = build_question_prompt(benchmark_row["question"], benchmark_row["options"])

    previous = existing_row or {}
    raw_responses = dict(previous.get("raw_responses", {}))
    predictions = {
        name: int(previous.get("predictions", {}).get(name, -1))
        for name in NEW_CONDITIONS
    }
    attempted_calls = Counter(previous.get("attempted_calls", {}))
    if not attempted_calls and previous.get("logical_calls"):
        attempted_calls.update(previous["logical_calls"])
    condition_errors: dict[str, str] = {}

    if predictions["option_decomposition"] < 0:
        try:
            response = generate(
                client,
                media,
                OPTION_DECOMPOSITION_PROMPT.format(question=question),
                max_output_tokens=2048,
            )
            attempted_calls["option_decomposition"] += 1
            raw_responses["option_decomposition"] = response
            predictions["option_decomposition"] = _answer_index(response)[0]
        except Exception as error:
            condition_errors["option_decomposition"] = f"{type(error).__name__}: {error}"

    if predictions["natural_language_scratchpad"] < 0:
        try:
            response = generate(
                client,
                media,
                SCRATCHPAD_PROMPT.format(question=question),
                # Gemini 3 Pro uses part of this budget for hidden reasoning.
                # A 2k cap occasionally truncates before the answer field.
                max_output_tokens=8192,
            )
            attempted_calls["natural_language_scratchpad"] += 1
            raw_responses["natural_language_scratchpad"] = response
            predictions["natural_language_scratchpad"] = _answer_index(response)[0]
        except Exception as error:
            condition_errors["natural_language_scratchpad"] = (
                f"{type(error).__name__}: {error}"
            )

    if predictions["three_call_natural_language"] < 0:
        try:
            notes = generate(
                client,
                media,
                NOTES_PROMPT.format(question=question),
                max_output_tokens=2048,
            )
            analysis = generate(
                client,
                media,
                ANALYSIS_PROMPT.format(notes=notes, question=question),
                max_output_tokens=2048,
            )
            final_response = generate(
                client,
                media,
                FINAL_PROMPT.format(notes=notes, analysis=analysis, question=question),
                max_output_tokens=1536,
            )
            attempted_calls["three_call_natural_language"] += 3
            raw_responses.update(
                {
                    "three_call_notes": notes,
                    "three_call_analysis": analysis,
                    "three_call_natural_language": final_response,
                }
            )
            predictions["three_call_natural_language"] = _answer_index(final_response)[0]
        except Exception as error:
            condition_errors["three_call_natural_language"] = (
                f"{type(error).__name__}: {error}"
            )
    for name, prediction in component_row["predictions"].items():
        predictions[f"saved_{name}"] = int(prediction)
    answer_index = benchmark_row["options"].index(benchmark_row["answer"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "MA-EgoQA strong no-code reasoning baselines",
        "model": client.config.model,
        "index": index,
        "category": benchmark_row["category"],
        "subcategory": benchmark_row["subcategory"],
        "answer_index": answer_index,
        "video_clip_ids": [str(clip["id"]) for clip in clips],
        "predictions": predictions,
        "correct": {name: value == answer_index for name, value in predictions.items()},
        "logical_calls": {
            "option_decomposition": 1,
            "natural_language_scratchpad": 1,
            "three_call_natural_language": 3,
        },
        "response_characters": {
            name: len(response) for name, response in raw_responses.items()
        },
        "attempted_calls": dict(attempted_calls),
        "condition_errors": condition_errors,
        "raw_responses": raw_responses,
        "elapsed_seconds": time.monotonic() - started,
    }


def record_complete(row: dict) -> bool:
    predictions = row.get("predictions", {})
    return all(int(predictions.get(name, -1)) >= 0 for name in NEW_CONDITIONS)


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def metrics(rows: list[dict], condition: str) -> dict:
    valid = [row for row in rows if row["predictions"].get(condition, -1) >= 0]
    correct = sum(row["correct"][condition] for row in valid)
    return {
        "correct": correct,
        "valid": len(valid),
        "total": len(rows),
        "accuracy": correct / len(valid) if valid else None,
    }


def comparison(rows: list[dict], left: str, right: str) -> dict:
    valid = [
        row
        for row in rows
        if row["predictions"].get(left, -1) >= 0
        and row["predictions"].get(right, -1) >= 0
    ]
    left_only = sum(row["correct"][left] and not row["correct"][right] for row in valid)
    right_only = sum(not row["correct"][left] and row["correct"][right] for row in valid)
    return {
        "left": left,
        "right": right,
        "paired_valid": len(valid),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_right_wins": right_only - left_only,
        "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
    }


def summarize(records: Iterable[dict], selected: list[int], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if record_complete(row)]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    saved = (
        "saved_raw_only_rerun",
        "saved_raw_spatial",
        "saved_raw_social",
        "saved_raw_spatial_social",
        "saved_trace_only_spatial_social",
        "saved_saved_direct",
        "saved_saved_full_primary",
    )
    conditions = (*NEW_CONDITIONS, *saved)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "MA-EgoQA",
        "protocol": (
            "frozen category-balanced 100; same model and saved raw clips; "
            "no Spatial/Social Code in new baselines"
        ),
        "model": model,
        "selected": len(selected),
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "logical_calls_per_question": {
            "option_decomposition": 1,
            "natural_language_scratchpad": 1,
            "three_call_natural_language": 3,
        },
        "conditions": {
            condition: {
                **metrics(rows, condition),
                "by_category": {
                    category: metrics(group, condition)
                    for category, group in sorted(by_category.items())
                },
            }
            for condition in conditions
        },
        "paired_comparisons": [
            comparison(rows, "saved_raw_only_rerun", condition)
            for condition in NEW_CONDITIONS
        ]
        + [
            comparison(rows, condition, "saved_raw_spatial_social")
            for condition in NEW_CONDITIONS
        ],
        "label_access_in_prompt": False,
        "task_question_conditioned": True,
        "test_set_diagnostic": True,
    }


def write_outputs(
    output: Path,
    existing: dict[int, dict],
    selected: list[int],
    model: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in sorted(existing):
            handle.write(json.dumps(existing[index], ensure_ascii=False) + "\n")
    temporary.replace(output)
    report = summarize((existing[index] for index in selected if index in existing), selected, model)
    summary_path = output.with_name(output.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--paired-source", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--component-output", type=Path, default=DEFAULT_COMPONENT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit must be non-negative")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    paired = load_jsonl(args.paired_source)
    component = load_jsonl(args.component_output)
    selected = selected_indices(args.manifest)
    if args.limit:
        selected = selected[: args.limit]
    missing = [index for index in selected if index not in paired or index not in component]
    if missing:
        raise ValueError(f"missing paired/component rows for indices: {missing[:10]}")
    clip_mismatches = [
        index
        for index in selected
        if [str(item["id"]) for item in paired[index]["video_clips"]]
        != [str(item) for item in component[index]["video_clip_ids"]]
    ]
    if clip_mismatches:
        raise ValueError(f"clip mismatch for indices: {clip_mismatches[:10]}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected": len(selected),
                    "conditions": NEW_CONDITIONS,
                    "new_logical_calls": 5 * len(selected),
                    "by_category": dict(
                        Counter(str(benchmark[index]["category"]) for index in selected)
                    ),
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.ffmpeg.is_file():
        parser.error(f"ffmpeg not found: {args.ffmpeg}")

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    existing = load_jsonl(args.output)
    pending = [
        index
        for index in selected
        if index not in existing
        or (not record_complete(existing[index]) and not args.keep_errors)
    ]
    print(
        f"model={config.model!r} selected={len(selected)} "
        f"resumed={len(selected)-len(pending)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                benchmark[index],
                paired[index],
                component[index],
                client,
                args.video_root,
                args.proxy_root,
                args.ffmpeg,
                existing.get(index),
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "ok" if record_complete(record) else "partial"
            except Exception as error:
                record = {
                    **existing.get(index, {}),
                    "schema_version": SCHEMA_VERSION,
                    "index": index,
                    "category": benchmark[index]["category"],
                    "subcategory": benchmark[index]["subcategory"],
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            existing[index] = record
            write_outputs(args.output, existing, selected, config.model)
            done = sum(index in existing for index in selected)
            print(f"index={index} state={state} done={done}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
