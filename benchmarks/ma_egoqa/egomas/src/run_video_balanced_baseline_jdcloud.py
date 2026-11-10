"""Build a paired MA-EgoQA source with answer-symmetric video retrieval.

BM25 captions are used only to locate raw clips. The direct baseline receives
the same raw clips later consumed by the standalone Spatial-Social runner; no
caption text or baseline prediction is exposed to that method.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent.parent
for import_path in (ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.index_bm25 import BM25TextRetriever  # noqa: E402
from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.run_video_spatial_social_jdcloud import (  # noqa: E402
    BASELINE_PROMPT,
    DEFAULT_BENCHMARK,
    DEFAULT_FFMPEG,
    DEFAULT_INDEX,
    DEFAULT_PLANNER,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    MIN_SOURCE_VIDEO_BYTES,
    ask,
    load_jsonl,
    retrieve_video_ids,
    select_indices,
    source_video_path,
    video_parts,
)
from egomas.utils.parsing import get_prediction_index  # noqa: E402
from egomas.utils.prompt_helpers import build_question_prompt  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "video_balanced_paired_gemini3pro.jsonl"
SCHEMA_VERSION = 1
METHOD = "Answer-symmetric raw-video retrieval baseline"


def process_one(
    index: int,
    row: dict,
    planner: dict,
    retriever: BM25TextRetriever,
    client: JDCloudGeminiClient,
    video_root: Path,
    proxy_root: Path,
    ffmpeg: Path,
    top_k_per_agent: int,
    top_k_per_option: int,
    max_videos: int,
    score_threshold: float,
) -> dict:
    started = time.monotonic()
    candidates = retrieve_video_ids(
        retriever,
        planner.get("selection", []),
        top_k_per_agent,
        max_videos * 2,
        score_threshold,
        question=str(row["question"]),
        options=list(row["options"]),
        top_k_per_option=top_k_per_option,
    )
    clips = [
        clip
        for clip in candidates
        if source_video_path(video_root, clip["id"]).stat().st_size
        >= MIN_SOURCE_VIDEO_BYTES
    ][:max_videos]
    if not clips:
        raise RuntimeError("answer-symmetric retrieval produced no video clips")
    media = video_parts(clips, video_root, proxy_root, ffmpeg)
    question = build_question_prompt(row["question"], row["options"])
    answer_index = row["options"].index(row["answer"])
    baseline_raw = ask(
        client, media, BASELINE_PROMPT.format(question=question), max_tokens=256
    )
    baseline_prediction = get_prediction_index(baseline_raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": client.config.model,
        "index": index,
        "category": row["category"],
        "subcategory": row["subcategory"],
        "answer_index": answer_index,
        "selection": planner.get("selection", []),
        "video_clips": clips,
        "baseline_prediction_raw": baseline_raw,
        "baseline_prediction_index": baseline_prediction,
        "baseline_correct": baseline_prediction == answer_index,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_outputs(path: Path, records: dict[int, dict], selected: list[int], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)
    rows = [records[index] for index in selected if index in records]
    successful = [row for row in rows if "baseline_correct" in row]
    correct = sum(bool(row["baseline_correct"]) for row in successful)
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": model,
        "completed": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "baseline": {
            "correct": correct,
            "total": len(successful),
            "accuracy": correct / len(successful) if successful else None,
        },
    }
    summary_path = path.with_name(path.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--planner-output", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument("--top-k-per-agent", type=int, default=5)
    parser.add_argument("--top-k-per-option", type=int, default=3)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=10.0)
    parser.add_argument("--keep-errors", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.top_k_per_agent, args.top_k_per_option, args.max_videos) < 1:
        parser.error("worker and retrieval limits must be positive")
    if not args.ffmpeg.is_file():
        parser.error(f"ffmpeg not found: {args.ffmpeg}")

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    planner = load_jsonl(args.planner_output)
    selected = select_indices(rows, planner, args)
    existing = load_jsonl(args.output)
    pending = [
        index for index in selected
        if index not in existing or ("error" in existing[index] and not args.keep_errors)
    ]
    retriever = BM25TextRetriever.load_vectorized_format(args.bm25_index)
    print(
        f"model={config.model!r} selected={len(selected)} "
        f"resumed={len(selected)-len(pending)} pending={len(pending)} "
        f"workers={args.workers} max_videos={args.max_videos}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one, index, rows[index], planner[index], retriever, client,
                args.video_root, args.proxy_root, args.ffmpeg, args.top_k_per_agent,
                args.top_k_per_option, args.max_videos, args.score_threshold,
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "correct" if record["baseline_correct"] else "wrong"
            except Exception as error:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "method": METHOD,
                    "model": config.model,
                    "index": index,
                    "category": rows[index]["category"],
                    "subcategory": rows[index]["subcategory"],
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            existing[index] = record
            write_outputs(args.output, existing, selected, config.model)
            completed = [existing[i] for i in selected if i in existing]
            valid = [row for row in completed if "baseline_correct" in row]
            accuracy = (
                sum(bool(row["baseline_correct"]) for row in valid) / len(valid)
                if valid else None
            )
            print(
                f"index={index} state={state} done={len(completed)}/{len(selected)} "
                f"baseline={accuracy}",
                flush=True,
            )


if __name__ == "__main__":
    main()
