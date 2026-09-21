"""Paired raw-video evaluation of EgoMAS and Spatial-Social Code on MA-EgoQA.

The existing EgoMAS planner selections and BM25 index are used only to locate
the relevant 30-second source clips.  Answering and code compilation receive
the original MP4 content (vision, time, and audio), not the stored captions.
Both paired conditions receive exactly the same clips.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
for import_path in (ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.src.jdcloud_gemini import (
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.run_spatial_social_jdcloud import (
    _mcnemar_exact,
    normalize_social_code,
    normalize_spatial_code,
    question_premise,
)
from egomas.utils.parsing import get_prediction_index
from egomas.utils.prompt_helpers import build_question_prompt
from runtime import execute_social, execute_spatial


DEFAULT_BENCHMARK = ROOT / "data" / "MA-EgoQA.json"
DEFAULT_INDEX = ROOT / "data" / "30sec_bm25.pkl"
DEFAULT_PLANNER = ROOT / "data" / "baselines" / "egomas_gemini3pro.jsonl"
DEFAULT_VIDEO_ROOT = ROOT / "data" / "egolife_raw"
DEFAULT_PROXY_ROOT = ROOT / "data" / "egolife_proxy_384p"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "video_spatial_social_gemini3pro.jsonl"
DEFAULT_FFMPEG = Path(
    os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"
)

SCHEMA_VERSION = 1
METHOD = "Raw-video Spatial-Social Code for MA-EgoQA"
MIN_SOURCE_VIDEO_BYTES = 1_000_000
_proxy_locks: dict[Path, threading.Lock] = {}
_proxy_locks_guard = threading.Lock()


BASELINE_PROMPT = """You are the direct raw-video baseline for MA-EgoQA.
Watch and listen to every supplied egocentric clip. Use visual actions, spoken
dialogue, speaker perspective, clip labels, and chronology. The clips were
retrieved for the question and may include distractors. Do not use outside
knowledge and do not explain your answer.

Question:
{question}

Return exactly one uppercase option letter: A, B, C, D, or E.
"""


SPATIAL_PROMPT = """You are the Spatial Code compiler for raw multi-person
egocentric videos. Watch and listen to every supplied clip. Extract only
grounded visible/audible facts, object interactions, locations, speakers, and
time-ordered events relevant to the question premise. Do not infer beliefs,
goals, intentions, or an answer. Answer choices are hidden.

The runtime object S already exists. Output ONLY 4-18 literal calls chosen from:
S.fact(t, relation, subject, object, source="video", confidence=0.8)
S.event(t, actor, action, target, source="video", confidence=0.8)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)

No assignments, imports, loops, markdown, comments, answer letters, or prose.

Question premise:
{question}
"""


SOCIAL_PROMPT = """You are the Social Code compiler for raw multi-person
egocentric videos. Watch and listen to every supplied clip. Infer relevant
observations, beliefs, goals, intents, coordination roles, and action evidence.
Do not choose an answer. Answer choices are hidden.

The runtime object M already exists. Output ONLY 4-18 literal calls chosen from:
M.observe(t, agent, fact, source="video_or_spatial", confidence=0.8)
M.belief(t, agent, fact, confidence=0.7, evidence="...")
M.goal(agent, goal, confidence=0.7, evidence="...")
M.intent(t, agent, intent, confidence=0.7, evidence="...")
M.action_evidence(t, agent, action, hypothesis, confidence=0.7, evidence="...")
M.social_goal(agent, target_agent, goal, confidence=0.7, evidence="...")
M.contradiction(agent, claim, source_a, source_b)
M.unknown(agent, variable)

No assignments, imports, loops, markdown, comments, answer letters, or prose.

Question premise:
{question}

Executed Spatial Trace:
{spatial_trace}
"""


ANSWER_PROMPT = """Answer the MA-EgoQA multiple-choice question after watching
and listening to every supplied raw egocentric clip. Use the executable
Spatial-Social records as structured assistance, while checking them against
the videos. Do not explain your answer.

Spatial Code:
{spatial_code}

Executed Spatial Trace:
{spatial_trace}

Social Code:
{social_code}

Executed Social Trace:
{social_trace}

Question:
{question}

Return exactly one uppercase option letter: A, B, C, D, or E.
"""


def load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["index"])] = row
    return records


def retrieve_video_ids(
    retriever: BM25TextRetriever,
    selections: list[dict],
    top_k_per_agent: int,
    max_videos: int,
    score_threshold: float,
    question: str = "",
    options: list[str] | None = None,
    top_k_per_option: int = 3,
) -> list[dict]:
    """Retrieve clips without letting one plausible answer monopolize context.

    Planner queries retain person-specific localization.  In parallel, each
    answer alternative receives an independent question+option retrieval
    pool.  Selection is round-robin across option pools before filling the
    remaining budget from all pools.  Captions are retrieval keys only and
    are never returned to either answering condition.
    """
    planner_pools: list[list[dict]] = []
    for selection in selections:
        name = str(selection.get("name", "")).strip().upper()
        query = str(selection.get("query", "")).strip()
        if not name or not query:
            continue
        ids, _, scores = retriever.retrieve(query, top_k=100, return_scores=True)
        named = [
            {"id": clip_id, "score": float(score), "agent": name, "query": query}
            for clip_id, score in zip(ids, scores)
            if name in clip_id.upper()
        ]
        if len(named) < top_k_per_agent:
            ids, _, scores = retriever.retrieve(query, top_k=1000, return_scores=True)
            named = [
                {"id": clip_id, "score": float(score), "agent": name, "query": query}
                for clip_id, score in zip(ids, scores)
                if name in clip_id.upper()
            ]
        accepted = [row for row in named[:top_k_per_agent] if row["score"] >= score_threshold]
        if not accepted and named:
            accepted = named[:1]
        for row in accepted:
            row["retrieval_source"] = "planner"
        if accepted:
            planner_pools.append(accepted)

    option_pools: list[list[dict]] = []
    for option_index, option in enumerate(options or []):
        query = " ".join(part for part in (question.strip(), str(option).strip()) if part)
        if not query:
            continue
        ids, _, scores = retriever.retrieve(query, top_k=max(20, top_k_per_option), return_scores=True)
        pool = [
            {
                "id": clip_id,
                "score": float(score),
                "agent": "",
                "query": query,
                "retrieval_source": "option",
                "option": "ABCDE"[option_index],
            }
            for clip_id, score in zip(ids, scores)
            if float(score) >= score_threshold
        ][:top_k_per_option]
        if not pool and ids:
            pool = [{
                "id": ids[0],
                "score": float(scores[0]),
                "agent": "",
                "query": query,
                "retrieval_source": "option",
                "option": "ABCDE"[option_index],
            }]
        if pool:
            option_pools.append(pool)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_first_unseen(pool: list[dict]) -> None:
        for candidate in pool:
            if candidate["id"] not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate["id"])
                return

    # Every alternative and every planner query receives representation before
    # score-based filling. This is answer-symmetric and dataset-agnostic.
    for pool in option_pools:
        if len(selected) >= max_videos:
            break
        add_first_unseen(pool)
    for pool in planner_pools:
        if len(selected) >= max_videos:
            break
        add_first_unseen(pool)

    remaining = sorted(
        (row for pool in option_pools + planner_pools for row in pool),
        key=lambda row: (-row["score"], row["id"]),
    )
    for row in remaining:
        if len(selected) >= max_videos:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    return selected


def source_video_path(video_root: Path, clip_id: str) -> Path:
    parts = clip_id.split("_")
    if len(parts) < 4 or not clip_id.endswith(".mp4"):
        raise ValueError(f"unexpected EgoLife clip id: {clip_id}")
    day = parts[0]
    participant = "_".join(parts[1:3])
    path = video_root / participant / day / clip_id
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def proxy_video_path(proxy_root: Path, clip_id: str) -> Path:
    parts = clip_id.split("_")
    return proxy_root / "_".join(parts[1:3]) / parts[0] / clip_id


def ensure_proxy(source: Path, destination: Path, ffmpeg: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with _proxy_locks_guard:
        lock = _proxy_locks.setdefault(destination, threading.Lock())
    with lock:
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.mp4")
        command = [
            str(ffmpeg), "-y", "-loglevel", "error", "-i", str(source),
            "-vf", "fps=8,scale=384:384", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "30", "-c:a", "aac", "-b:a", "48k", "-ac", "1", str(temporary),
        ]
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no proxy for {source}")
        os.replace(temporary, destination)
    return destination


def video_parts(clips: list[dict], video_root: Path, proxy_root: Path, ffmpeg: Path) -> list[dict]:
    parts: list[dict] = []
    for position, clip in enumerate(clips, start=1):
        source = source_video_path(video_root, clip["id"])
        proxy = ensure_proxy(source, proxy_video_path(proxy_root, clip["id"]), ffmpeg)
        encoded = base64.b64encode(proxy.read_bytes()).decode("ascii")
        parts.append({"text": f"Clip {position}: {clip['id']} (includes audio)"})
        parts.append({"inline_data": {"mimeType": "video/mp4", "data": encoded}})
    return parts


def ask(client: JDCloudGeminiClient, media: list[dict], prompt: str, max_tokens: int) -> str:
    return client.generate_parts(media + [{"text": prompt}], max_output_tokens=max_tokens)


def normalize_video_spatial_code(text: str) -> str:
    """Convert Gemini timestamp strings in S calls to integer video seconds."""
    converted: list[str] = []
    for line in text.splitlines():
        try:
            tree = ast.parse(line.strip())
        except SyntaxError:
            converted.append(line)
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            converted.append(line)
            continue
        call = tree.body[0].value
        if not isinstance(call, ast.Call) or not call.args:
            converted.append(line)
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            match = re.search(r"(?:(\d+):)?(\d+):(\d+)", first.value)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2))
                seconds = int(match.group(3))
                call.args[0] = ast.Constant(hours * 3600 + minutes * 60 + seconds)
                line = ast.unparse(tree)
        converted.append(line)
    return normalize_spatial_code("\n".join(converted))


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
    )
    clips = [
        clip
        for clip in candidates
        if source_video_path(video_root, clip["id"]).stat().st_size
        >= MIN_SOURCE_VIDEO_BYTES
    ][:max_videos]
    if not clips:
        raise RuntimeError("planner/BM25 retrieval produced no video clips")
    media = video_parts(clips, video_root, proxy_root, ffmpeg)
    question = build_question_prompt(row["question"], row["options"])
    answer_index = row["options"].index(row["answer"])

    baseline_raw = ask(
        client, media, BASELINE_PROMPT.format(question=question), max_tokens=256
    )
    baseline_prediction = get_prediction_index(baseline_raw)

    premise = question_premise(row)
    spatial_raw = ask(
        client, media, SPATIAL_PROMPT.format(question=premise), max_tokens=4096
    )
    spatial_code = normalize_video_spatial_code(spatial_raw)
    if not spatial_code:
        spatial_code = "S.unknown(0, 'relevant_relation', 'question', 'video')"
    spatial_trace = execute_spatial(spatial_code)

    social_raw = ask(
        client,
        media,
        SOCIAL_PROMPT.format(question=premise, spatial_trace=spatial_trace),
        max_tokens=4096,
    )
    social_code = normalize_social_code(social_raw)
    if not social_code:
        social_code = "M.unknown('multi_agent_group', 'social_state')"
    social_trace = execute_social(social_code)

    prediction_raw = ask(
        client,
        media,
        ANSWER_PROMPT.format(
            spatial_code=spatial_code,
            spatial_trace=spatial_trace,
            social_code=social_code,
            social_trace=social_trace,
            question=question,
        ),
        max_tokens=256,
    )
    prediction = get_prediction_index(prediction_raw)
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
        "spatial_raw": spatial_raw,
        "spatial_code": spatial_code,
        "spatial_trace": spatial_trace,
        "social_raw": social_raw,
        "social_code": social_code,
        "social_trace": social_trace,
        "prediction_raw": prediction_raw,
        "prediction_index": prediction,
        "correct": prediction == answer_index,
        "elapsed_seconds": time.monotonic() - started,
    }


def metrics(rows: list[dict], field: str) -> dict:
    correct = sum(bool(row[field]) for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None}


def summarize(records: Iterable[dict], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "correct" in row and "baseline_correct" in row]
    grouped: dict[str, list[dict]] = {}
    subgroups: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)
        subgroups.setdefault(str(row["subcategory"]), []).append(row)

    def paired(group: list[dict]) -> dict:
        base = metrics(group, "baseline_correct")
        method = metrics(group, "correct")
        return {"baseline": base, "spatial_social": method, "delta_accuracy": method["accuracy"] - base["accuracy"] if group else None}

    baseline_only = sum(row["baseline_correct"] and not row["correct"] for row in rows)
    method_only = sum(not row["baseline_correct"] and row["correct"] for row in rows)
    both_correct = sum(row["baseline_correct"] and row["correct"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "comparison": "paired_same_planner_bm25_localization_and_raw_video_clips",
        "model": model,
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "overall": paired(rows),
        "contingency": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "spatial_social_only_correct": method_only,
            "both_wrong": len(rows) - both_correct - baseline_only - method_only,
            "net_spatial_social_wins": method_only - baseline_only,
            "mcnemar_exact_p": _mcnemar_exact(baseline_only, method_only),
        },
        "by_category": {name: paired(group) for name, group in sorted(grouped.items())},
        "by_subcategory": {name: paired(group) for name, group in sorted(subgroups.items())},
    }


def write_outputs(path: Path, records: dict[int, dict], selected: list[int], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)
    report = summarize([records[index] for index in selected if index in records], model)
    summary_path = path.with_name(path.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)


def select_indices(rows: list[dict], planner: dict[int, dict], args: argparse.Namespace) -> list[int]:
    eligible = sorted(index for index, record in planner.items() if "selection" in record and index < len(rows))
    if args.indices:
        wanted = set(args.indices)
        eligible = [index for index in eligible if index in wanted]
    if args.samples_per_category is not None:
        counts: Counter[str] = Counter()
        selected = []
        for index in eligible:
            category = str(rows[index]["category"])
            if counts[category] < args.samples_per_category:
                selected.append(index)
                counts[category] += 1
        eligible = selected
    if args.limit is not None:
        eligible = eligible[: args.limit]
    return eligible


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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument("--top-k-per-agent", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=10.0)
    parser.add_argument("--keep-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.top_k_per_agent < 1 or args.max_videos < 1:
        parser.error("workers, top-k-per-agent, and max-videos must be positive")
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
        f"model={config.model!r} selected={len(selected)} resumed={len(selected)-len(pending)} "
        f"pending={len(pending)} workers={args.workers} max_videos={args.max_videos}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one, index, rows[index], planner[index], retriever, client,
                args.video_root, args.proxy_root, args.ffmpeg, args.top_k_per_agent,
                args.max_videos, args.score_threshold,
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "correct" if record["correct"] else "wrong"
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
            report = summarize([existing[i] for i in selected if i in existing], config.model)
            print(
                f"index={index} category={rows[index]['category']} state={state} "
                f"done={report['completed']}/{len(selected)} "
                f"baseline={report['overall']['baseline']['accuracy']} "
                f"spatial_social={report['overall']['spatial_social']['accuracy']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
