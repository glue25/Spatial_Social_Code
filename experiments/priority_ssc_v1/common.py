"""Shared, label-safe utilities for the priority SSC experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("cot", "natural_language_scratchpad", "pddl_mind_adapted", "spatial_only", "ssc")
FOUR_CELLS = ("predicted_predicted", "human_predicted", "predicted_human", "human_human")

# These records directly score answer options or recommend an action.  They are
# deliberately excluded from frozen Social state and from every human form.
TARGET_DIRECTED_SOCIAL_KINDS = {
    "posterior",
    "query_candidate",
    "query_evidence",
    "action_evidence",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def trace_records(trace: str, *, social: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in str(trace or "").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
            continue
        if social and record["kind"] in TARGET_DIRECTED_SOCIAL_KINDS:
            continue
        records.append(record)
    return records


def clean_trace(trace: str, *, social: bool = False) -> str:
    return "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in trace_records(trace, social=social)
    )


def parse_choice(text: str, labels: str) -> int:
    stripped = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = {}
    answer = str(payload.get("answer", "")).strip().upper() if isinstance(payload, dict) else ""
    if answer and answer[0] in labels:
        return labels.index(answer[0])
    match = re.search(r"(?i)[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-E])\b", text)
    return labels.index(match.group(1).upper()) if match and match.group(1).upper() in labels else -1


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    n = left_only + right_only
    if not n:
        return None
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def source_traces(row: dict[str, Any]) -> tuple[str, str]:
    outputs = row.get("raw_outputs", {}).get("ssc", {})
    physical = row.get("spatial_trace") or row.get("traces", {}).get("spatial") or outputs.get("spatial_trace", "")
    social = row.get("social_trace") or row.get("traces", {}).get("social") or outputs.get("social_trace", "")
    return clean_trace(str(physical)), clean_trace(str(social), social=True)


def format_question(dataset: str, row: dict[str, Any]) -> tuple[str, str, int]:
    if dataset == "mmtom":
        labels = "AB"
        answer = labels.index(str(row["answer"]).strip().upper())
        return str(row["question"]), labels, answer
    options = list(row["options"])
    labels = "ABCDE"[: len(options)]
    question = str(row["question"]) + "\n" + "\n".join(
        f"{labels[index]}. {option}" for index, option in enumerate(options)
    )
    return question, labels, options.index(row["answer"])


def forbidden_target_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"answer", "answer_index", "answer_label", "gold", "label", "prediction", "correct"}:
                found.append(str(key))
            found.extend(forbidden_target_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_target_keys(item))
    return found
