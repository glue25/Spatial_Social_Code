"""Checkpointed EgoMAS baseline using the shared JD Cloud Gemini 3 service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.src.jdcloud_gemini import (
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.retrieval_helpers import retrieve_for_person
from egomas.utils.constants import PERSON_NAMES
from egomas.utils.parsing import get_prediction_index, parse_planner_response
from egomas.utils.prompt_helpers import (
    build_answer_prompt,
    build_planner_prompt,
    build_question_prompt,
    get_context_text,
)


DEFAULT_BENCHMARK = ROOT / "data" / "MA-EgoQA_bm25.json"
DEFAULT_INDEX = ROOT / "data" / "30sec_bm25.pkl"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "egomas_gemini3pro.jsonl"


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_indices(
    rows: list[dict],
    limit: int | None,
    samples_per_category: int | None,
) -> list[int]:
    if samples_per_category is not None:
        counts: Counter[str] = Counter()
        selected = []
        for index, row in enumerate(rows):
            category = str(row["category"])
            if counts[category] < samples_per_category:
                selected.append(index)
                counts[category] += 1
        return selected
    indices = list(range(len(rows)))
    return indices[:limit] if limit is not None else indices


def sanitize_selection(selection: object, question: str, index: int) -> list[dict]:
    clean = []
    if isinstance(selection, list):
        for item in selection:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip().title()
            query = str(item.get("query", "")).strip()
            if name in PERSON_NAMES and query:
                clean.append({"name": name, "query": query})
    if clean:
        return clean
    return [{"name": PERSON_NAMES[index % len(PERSON_NAMES)], "query": question}]


def process_one(
    index: int,
    row: dict,
    retriever: BM25TextRetriever,
    client: JDCloudGeminiClient,
) -> dict:
    started = time.monotonic()
    question_prompt = build_question_prompt(row["question"], row["options"])
    shared_contexts = row["bm25"][:20]
    shared_text = get_context_text(shared_contexts)
    planner_prompt = build_planner_prompt(shared_text, question_prompt)
    planner_raw = client.generate(planner_prompt, max_output_tokens=2048)
    parsed = parse_planner_response(planner_raw, row["question"])
    selection = sanitize_selection(parsed, row["question"], index)

    retrieved = [
        retrieve_for_person(retriever, item["name"], item["query"])
        for item in selection
    ]
    retrieved = [context for context in retrieved if context]
    answer_prompt = build_answer_prompt(shared_text, retrieved, question_prompt)
    answer_raw = client.generate(answer_prompt, max_output_tokens=256)
    prediction_index = get_prediction_index(answer_raw)
    answer_index = row["options"].index(row["answer"])
    return {
        "index": index,
        "category": row["category"],
        "subcategory": row["subcategory"],
        "question": row["question"],
        "options": row["options"],
        "answer": row["answer"],
        "answer_index": answer_index,
        "selection": selection,
        "planner_raw": planner_raw,
        "prediction_raw": answer_raw,
        "prediction_index": prediction_index,
        "correct": prediction_index == answer_index,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_checkpoint(path: Path, records: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for index in sorted(records):
            output.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(records: Iterable[dict], model: str) -> dict:
    rows = list(records)
    valid = [row for row in rows if "correct" in row]
    errors = [row for row in rows if "error" in row]

    def metrics(group: list[dict]) -> dict:
        correct = sum(bool(row["correct"]) for row in group)
        return {
            "correct": correct,
            "total": len(group),
            "accuracy": correct / len(group) if group else None,
        }

    by_category: dict[str, list[dict]] = defaultdict(list)
    by_subcategory: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        by_category[str(row["category"])].append(row)
        by_subcategory[str(row["subcategory"])].append(row)
    return {
        "method": "EgoMAS",
        "provider": "JD Cloud Gemini-compatible responses endpoint",
        "model": model,
        "completed": len(rows),
        "successful": len(valid),
        "errors": len(errors),
        "overall": metrics(valid),
        "by_category": {
            name: metrics(group) for name, group in sorted(by_category.items())
        },
        "by_subcategory": {
            name: metrics(group) for name, group in sorted(by_subcategory.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument(
        "--keep-errors",
        action="store_true",
        help="Treat checkpointed errors as complete instead of retrying them.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config)
    rows = load_records(args.benchmark)
    indices = select_indices(rows, args.limit, args.samples_per_category)
    existing: dict[int, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[int(record["index"])] = record
    pending = [
        index
        for index in indices
        if index not in existing or ("error" in existing[index] and not args.keep_errors)
    ]
    retriever = BM25TextRetriever.load_vectorized_format(args.bm25_index)
    print(
        f"model={config.model!r} selected={len(indices)} "
        f"resumed={len(indices) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, index, rows[index], retriever, client): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "correct" if record["correct"] else "wrong"
            except Exception as error:
                record = {
                    "index": index,
                    "category": rows[index]["category"],
                    "subcategory": rows[index]["subcategory"],
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            existing[index] = record
            write_checkpoint(args.output, existing)
            selected_records = [existing[i] for i in indices if i in existing]
            summary = summarize(selected_records, config.model)
            summary_path = args.output.with_name(args.output.stem + "_summary.json")
            temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(summary_path)
            print(
                f"index={index} category={rows[index]['category']} state={state} "
                f"accuracy={summary['overall']['accuracy']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
