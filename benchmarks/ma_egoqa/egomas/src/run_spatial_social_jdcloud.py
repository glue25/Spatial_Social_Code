"""Paired Spatial-Social Code evaluation on MA-EgoQA with JD Cloud Gemini 3.

The runner deliberately reuses each EgoMAS baseline record's planner selection
and deterministic BM25 retrieval.  The only experimental difference is the
addition of executable Spatial Code and Social Code before the final answer.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent.parent
for path in (ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime import UnsafeProgram, execute_social, execute_spatial

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.src.jdcloud_gemini import (
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.retrieval_helpers import retrieve_for_person
from egomas.utils.parsing import get_prediction_index
from egomas.utils.prompt_helpers import build_question_prompt, get_context_text


SCHEMA_VERSION = 2
METHOD = "Spatial-Social Code for MA-EgoQA"
DEFAULT_BENCHMARK = ROOT / "data" / "MA-EgoQA_bm25.json"
DEFAULT_INDEX = ROOT / "data" / "30sec_bm25.pkl"
DEFAULT_BASELINE = ROOT / "data" / "baselines" / "egomas_gemini3pro.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "spatial_social_gemini3pro_v2.jsonl"


SPATIAL_PROMPT = """You are the Spatial Code evidence compiler for multi-person long-horizon egocentric memory.
Extract only physically grounded facts, visible states, object interactions, locations, and time-ordered events from the evidence. Preserve the evidence chronology using integer timestamps in source order. Do not infer beliefs, goals, intentions, or an answer. The answer choices are intentionally hidden.

The runtime object S already exists. Output ONLY 4-18 literal calls chosen from:
S.fact(t, relation, subject, object, source="caption", confidence=0.8)
S.event(t, actor, action, target, source="caption", confidence=0.8)
S.visible(t, observer, object, value, source="caption", confidence=0.8)
S.unknown(t, relation, subject, object)

No assignments, imports, loops, markdown, comments, answer letters, or prose.

Question premise (answer choices hidden):
{question}

Retrieved multi-agent evidence:
{evidence}
"""


SOCIAL_PROMPT = """You are the Social Code evidence compiler for multi-person long-horizon egocentric memory.
Infer socially relevant observations, persistent beliefs, goals, immediate intents, coordination roles, and action evidence. A world fact is not automatically a person's belief: attribute a belief only when that person could observe supporting evidence. Preserve chronology with integer timestamps. Record contradictions when perspectives conflict. Do not choose an answer. The answer choices are intentionally hidden.

The runtime object M already exists. Output ONLY 4-18 literal calls chosen from:
M.observe(t, agent, fact, source="caption_or_spatial", confidence=0.8)
M.belief(t, agent, fact, confidence=0.7, evidence="...")
M.goal(agent, goal, confidence=0.7, evidence="...")
M.intent(t, agent, intent, confidence=0.7, evidence="...")
M.action_evidence(t, agent, action, hypothesis, confidence=0.7, evidence="...")
M.social_goal(agent, target_agent, goal, confidence=0.7, evidence="...")
M.contradiction(agent, claim, source_a, source_b)
M.unknown(agent, variable)

No assignments, imports, loops, markdown, comments, answer letters, or prose.

Question premise (answer choices hidden):
{question}

Retrieved multi-agent evidence:
{evidence}

Executed Spatial Trace:
{spatial_trace}
"""


ANSWER_PROMPT = """Answer the MA-EgoQA multiple-choice question from the retrieved evidence and the executed Spatial-Social understanding records.

Spatial records represent grounded facts, visibility, actions, and chronology. Social records distinguish observations from beliefs, goals, intents, coordination, and contradictions. Treat unknown and low-confidence records as uncertain. The code is derived from the same evidence and does not replace it.

Retrieved multi-agent evidence:
{evidence}

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
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def question_premise(row: dict) -> str:
    """Hide choices from compilers, including inline A)-E) temporal choices."""
    question = str(row["question"]).strip()
    options = [str(option).strip().upper() for option in row.get("options", [])]
    if options == list("ABCDE"):
        question = re.split(r"\s+A\)\s+", question, maxsplit=1)[0].strip()
    return question


def _normalize_code(
    text: str,
    variable: str,
    allowed: set[str],
    executor,
    max_lines: int = 18,
) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            continue
        call = tree.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name):
            continue
        if call.func.value.id != variable or call.func.attr not in allowed:
            continue
        try:
            executor(line)
        except (UnsafeProgram, NameError, AttributeError, TypeError, ValueError):
            continue
        seen.add(line)
        kept.append(line)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def normalize_spatial_code(text: str) -> str:
    return _normalize_code(
        text, "S", {"fact", "event", "visible", "unknown"}, execute_spatial
    )


def normalize_social_code(text: str) -> str:
    return _normalize_code(
        text,
        "M",
        {
            "observe",
            "belief",
            "goal",
            "intent",
            "action_evidence",
            "social_goal",
            "contradiction",
            "unknown",
        },
        execute_social,
    )


def build_evidence(row: dict, baseline: dict, retriever: BM25TextRetriever) -> str:
    shared_text = get_context_text(row["bm25"][:20])
    retrieved = [
        retrieve_for_person(retriever, item["name"], item["query"])
        for item in baseline["selection"]
    ]
    retrieved = [text for text in retrieved if text]
    sections = ["### Shared 10-minute memory\n" + shared_text]
    sections.extend(
        f"### Agent-specific 30-second memory {offset}\n{text}"
        for offset, text in enumerate(retrieved, start=1)
    )
    return "\n\n".join(sections)


def process_one(
    index: int,
    row: dict,
    baseline: dict,
    retriever: BM25TextRetriever,
    client: JDCloudGeminiClient,
) -> dict:
    started = time.monotonic()
    evidence = build_evidence(row, baseline, retriever)
    premise = question_premise(row)

    spatial_raw = client.generate(
        SPATIAL_PROMPT.format(question=premise, evidence=evidence),
        max_output_tokens=4096,
    )
    spatial_code = normalize_spatial_code(spatial_raw)
    if not spatial_code:
        spatial_code = "S.unknown(0, 'relevant_relation', 'question', 'evidence')"
    spatial_trace = execute_spatial(spatial_code)

    social_raw = client.generate(
        SOCIAL_PROMPT.format(
            question=premise,
            evidence=evidence,
            spatial_trace=spatial_trace,
        ),
        max_output_tokens=4096,
    )
    social_code = normalize_social_code(social_raw)
    if not social_code:
        social_code = "M.unknown('multi_agent_group', 'social_state')"
    social_trace = execute_social(social_code)

    question = build_question_prompt(row["question"], row["options"])
    prediction_raw = client.generate(
        ANSWER_PROMPT.format(
            evidence=evidence,
            spatial_code=spatial_code,
            spatial_trace=spatial_trace,
            social_code=social_code,
            social_trace=social_trace,
            question=question,
        ),
        max_output_tokens=256,
    )
    prediction_index = get_prediction_index(prediction_raw)
    answer_index = row["options"].index(row["answer"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": client.config.model,
        "index": index,
        "category": row["category"],
        "subcategory": row["subcategory"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_index": answer_index,
        "baseline_prediction_index": baseline["prediction_index"],
        "baseline_correct": baseline["correct"],
        "selection": baseline["selection"],
        "spatial_raw": spatial_raw,
        "spatial_code": spatial_code,
        "spatial_trace": spatial_trace,
        "social_raw": social_raw,
        "social_code": social_code,
        "social_trace": social_trace,
        "prediction_raw": prediction_raw,
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


def _metrics(rows: list[dict], field: str) -> dict:
    correct = sum(bool(row[field]) for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else None,
    }


def _mcnemar_exact(baseline_only: int, method_only: int) -> float | None:
    discordant = baseline_only + method_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, i) for i in range(min(baseline_only, method_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(records: Iterable[dict], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "correct" in row]
    errors = [row for row in all_rows if "error" in row]
    both_correct = sum(row["baseline_correct"] and row["correct"] for row in rows)
    baseline_only = sum(row["baseline_correct"] and not row["correct"] for row in rows)
    method_only = sum(not row["baseline_correct"] and row["correct"] for row in rows)
    both_wrong = len(rows) - both_correct - baseline_only - method_only

    grouped: dict[str, list[dict]] = defaultdict(list)
    subgroups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
        subgroups[str(row["subcategory"])].append(row)

    def paired(group: list[dict]) -> dict:
        return {
            "baseline": _metrics(group, "baseline_correct"),
            "spatial_social": _metrics(group, "correct"),
            "delta_accuracy": (
                _metrics(group, "correct")["accuracy"]
                - _metrics(group, "baseline_correct")["accuracy"]
                if group
                else None
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "comparison": "paired_same_planner_and_bm25_evidence",
        "model": model,
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(errors),
        "overall": paired(rows),
        "contingency": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "spatial_social_only_correct": method_only,
            "both_wrong": both_wrong,
            "net_spatial_social_wins": method_only - baseline_only,
            "mcnemar_exact_p": _mcnemar_exact(baseline_only, method_only),
        },
        "by_category": {name: paired(group) for name, group in sorted(grouped.items())},
        "by_subcategory": {
            name: paired(group) for name, group in sorted(subgroups.items())
        },
    }


def select_indices(
    rows: list[dict],
    baseline: dict[int, dict],
    indices: list[int] | None,
    limit: int | None,
    samples_per_category: int | None,
) -> list[int]:
    eligible = sorted(
        index
        for index, record in baseline.items()
        if "correct" in record and 0 <= index < len(rows)
    )
    if indices:
        requested = set(indices)
        eligible = [index for index in eligible if index in requested]
    if samples_per_category is not None:
        counts: Counter[str] = Counter()
        selected = []
        for index in eligible:
            category = str(rows[index]["category"])
            if counts[category] < samples_per_category:
                selected.append(index)
                counts[category] += 1
        eligible = selected
    if limit is not None:
        eligible = eligible[:limit]
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--baseline-output", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument("--keep-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config)
    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    baseline = load_jsonl(args.baseline_output)
    indices = select_indices(
        rows, baseline, args.indices, args.limit, args.samples_per_category
    )
    existing = load_jsonl(args.output)
    for record in existing.values():
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"legacy or incompatible record in {args.output}")
        if record.get("model") != config.model:
            raise ValueError(
                f"existing model {record.get('model')!r} != requested {config.model!r}"
            )
    pending = [
        index
        for index in indices
        if index not in existing or ("error" in existing[index] and not args.keep_errors)
    ]
    retriever = BM25TextRetriever.load_vectorized_format(args.bm25_index)
    print(
        f"model={config.model!r} paired_baseline={len(baseline)} "
        f"selected={len(indices)} resumed={len(indices) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one, index, rows[index], baseline[index], retriever, client
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
                    "baseline_correct": baseline[index]["correct"],
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
                f"baseline={summary['overall']['baseline']['accuracy']} "
                f"spatial_social={summary['overall']['spatial_social']['accuracy']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
