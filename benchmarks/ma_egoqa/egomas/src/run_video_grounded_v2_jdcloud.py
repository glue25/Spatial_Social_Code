"""Grounded, paired raw-video evaluation for MA-EgoQA.

Version 2 keeps a direct-video answer as a conservative anchor and builds an
independent Spatial-Social candidate from clip-level, citation-bearing
evidence.  When the candidates disagree, a final verifier re-opens only the
cited videos and adjudicates the answer.  This avoids merging unrelated local
timelines and prevents unsupported intermediate records from silently steering
the final prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
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
from egomas.src.run_spatial_social_jdcloud import _mcnemar_exact, question_premise
from egomas.src.run_video_spatial_social_jdcloud import (
    DEFAULT_FFMPEG,
    DEFAULT_INDEX,
    DEFAULT_PLANNER,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    MIN_SOURCE_VIDEO_BYTES,
    retrieve_video_ids,
    source_video_path,
    video_parts,
)
from egomas.utils.prompt_helpers import build_question_prompt


DEFAULT_BENCHMARK = ROOT / "data" / "MA-EgoQA.json"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "video_grounded_v2_gemini3pro.jsonl"
SCHEMA_VERSION = 2
METHOD = "Grounded Spatial-Social Video v2"
OPTION_LETTERS = "ABCDE"
DEFAULT_STRUCTURED_CATEGORIES = {
    "theory_of_mind",
}


DIRECT_PROMPT = """You are the direct-video candidate for MA-EgoQA.
Watch and listen to the supplied clips and answer the multiple-choice question.
The clips are retrieval results and may contain distractors. Cite the exact
clip filename and local time span that most strongly supports the answer. Do
not invent a citation. Return JSON only:
{"answer":"A", "clip_id":"...mp4", "time_span":"00:00-00:05",
 "evidence":"short visible action or exact audible quote"}

Question:
{question}
"""


CLIP_EVIDENCE_PROMPT = """Extract question-relevant grounded evidence from ONE
egocentric video clip. Answer choices are hidden. Do not answer the question
and do not infer beliefs, goals, intent, or causality. Report only directly
visible actions/states or directly audible speech. If the clip is irrelevant,
return {"records":[]}.

Every record must use a local 00:00-00:30 time span and a concise claim. For
speech, include an exact short quote when intelligible. Return JSON only:
{"records":[{"kind":"event|speech|visibility|state",
 "time_span":"00:00-00:05", "subject":"person or object",
 "claim":"direct observation", "grounding":"visible|audible|both",
 "quote":"exact quote or empty string", "confidence":0.0}]}

Clip ID: {clip_id}
Wearer/observer: {observer}
Question premise: {premise}
"""


SOCIAL_PROMPT = """Build a minimal Social trace from grounded video records.
Answer choices are hidden. Every inference must cite one or more existing
evidence_id values. Do not add events, dialogue, or people absent from the
grounded records. It is valid to return no records. Return JSON only:
{"records":[{"kind":"observation|belief|goal|intent|coordination|contradiction",
 "agent":"person", "claim":"concise inference",
 "evidence_ids":["C1R1"], "confidence":0.0}]}

Question premise:
{premise}

Grounded records:
{grounded}
"""


STRUCTURED_ANSWER_PROMPT = """Answer the MA-EgoQA question using only the
grounded records and Social trace below. Retrieval can contain distractors.
Prefer direct speech/action over inferred beliefs. Cite the evidence_id values
that determine the answer. Return JSON only:
{"answer":"A", "evidence_ids":["C1R1"], "reason":"one sentence"}

Grounded records:
{grounded}

Social trace:
{social}

Question:
{question}
"""


VERIFY_PROMPT = """You are the conservative adjudicator for two disagreeing
MA-EgoQA candidates. Re-watch the cited clips and decide the question fresh.
Do not prefer the structured candidate merely because it is more detailed.
Switch away from the direct-video candidate only when a visible action or
audible statement clearly supports another option. Return JSON only:
{"answer":"A", "favored":"direct|structured|neither",
 "clip_id":"...mp4", "time_span":"00:00-00:05",
 "evidence":"short decisive observation"}

Direct candidate:
{direct}

Structured candidate:
{structured}

Structured candidate's cited grounded records:
{cited_records}

Question:
{question}
"""


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPAN_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})\s*[-–—]\s*(?:(\d+):)?(\d{1,2}):(\d{2})"
)
_CLIP_TIME_RE = re.compile(r"_(\d{8})\.mp4$")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["index"])] = row
    return records


def _json_value(text: str) -> object | None:
    """Return the first JSON object/array embedded in a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    starts = [position for position, char in enumerate(stripped) if char in "{["]
    for position in starts:
        try:
            value, _ = decoder.raw_decode(stripped[position:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _answer_index(text: str) -> tuple[int, dict]:
    payload = _json_value(text)
    data = payload if isinstance(payload, dict) else {}
    answer = str(data.get("answer", "")).strip().upper()
    if answer and answer[0] in OPTION_LETTERS:
        return OPTION_LETTERS.index(answer[:1]), data
    match = re.search(
        r"(?i)[\"']?(?:answer|option)[\"']?\s*[:=]?\s*[\"']?([A-E])\b",
        text,
    )
    if match:
        answer = match.group(1).upper()
        recovered = {"answer": answer}
        clip_match = re.search(r"DAY\d+_A\d+_[A-Z]+_\d{8}\.mp4", text, flags=re.IGNORECASE)
        if clip_match:
            recovered["clip_id"] = clip_match.group(0)
        span_match = _SPAN_RE.search(text)
        if span_match:
            recovered["time_span"] = span_match.group(0)
        evidence_ids = list(dict.fromkeys(re.findall(r"C\d+R\d+", text)))
        if evidence_ids:
            recovered["evidence_ids"] = evidence_ids
        return OPTION_LETTERS.index(answer), recovered
    plain = text.strip().upper()
    return (OPTION_LETTERS.index(plain[0]), data) if plain and plain[0] in OPTION_LETTERS else (-1, data)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def _clip_agent(clip_id: str) -> str:
    parts = clip_id.removesuffix(".mp4").split("_")
    return "_".join(parts[1:3]) if len(parts) >= 4 else clip_id


def _clip_observer(clip_id: str) -> str:
    agent = _clip_agent(clip_id)
    return agent.split("_", 1)[1].title() if "_" in agent else agent.title()


def _clip_start_seconds(clip_id: str) -> int:
    match = _CLIP_TIME_RE.search(clip_id)
    if not match:
        return 0
    value = match.group(1)
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:6])


def _span_seconds(value: object) -> tuple[int, int] | None:
    match = _SPAN_RE.search(str(value))
    if not match:
        return None
    start = int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    end = int(match.group(4) or 0) * 3600 + int(match.group(5)) * 60 + int(match.group(6))
    if start < 0 or end < start or start > 35 or end > 35:
        return None
    return start, end


def rerank_clips(
    candidates: list[dict],
    question: str,
    captions: dict[str, str],
    max_clips: int,
) -> list[dict]:
    """Question reranking with a small diversity penalty across wearer streams."""
    if not candidates:
        return []
    question_tokens = _tokens(question)
    raw_scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    low, high = min(raw_scores), max(raw_scores)
    scored: list[dict] = []
    for candidate in candidates:
        row = dict(candidate)
        caption_tokens = _tokens(captions.get(str(row["id"]), ""))
        overlap = len(question_tokens & caption_tokens) / math.sqrt(
            max(1, len(question_tokens) * len(caption_tokens))
        )
        normalized_bm25 = (
            (float(row.get("score", 0.0)) - low) / (high - low)
            if high > low else 1.0
        )
        row["rerank_score"] = normalized_bm25 + 1.75 * overlap
        row["question_overlap"] = overlap
        scored.append(row)

    selected: list[dict] = []
    remaining = scored[:]
    while remaining and len(selected) < max_clips:
        agent_counts = Counter(_clip_agent(row["id"]) for row in selected)
        best = max(
            remaining,
            key=lambda row: (
                row["rerank_score"] - 0.12 * agent_counts[_clip_agent(row["id"])],
                row["question_overlap"],
                row["id"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def normalize_grounded_records(text: str, clip: dict, clip_number: int) -> list[dict]:
    payload = _json_value(text)
    raw_records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        return []
    allowed_kinds = {"event", "speech", "visibility", "state"}
    allowed_grounding = {"visible", "audible", "both"}
    records: list[dict] = []
    for raw in raw_records[:8]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        span = _span_seconds(raw.get("time_span", ""))
        subject = str(raw.get("subject", "")).strip()[:120]
        claim = str(raw.get("claim", "")).strip()[:500]
        grounding = str(raw.get("grounding", "")).strip().lower()
        quote = str(raw.get("quote", "")).strip()[:300]
        if kind not in allowed_kinds or span is None or not subject or not claim:
            continue
        if grounding not in allowed_grounding:
            grounding = "audible" if kind == "speech" else "visible"
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(0.9 if quote else 0.8, max(0.0, confidence))
        start, end = span
        records.append(
            {
                "evidence_id": f"C{clip_number}R{len(records) + 1}",
                "kind": kind,
                "clip_id": clip["id"],
                "observer": _clip_observer(clip["id"]),
                "local_time_span": f"00:{start:02d}-00:{end:02d}",
                "absolute_start_seconds": _clip_start_seconds(clip["id"]) + start,
                "subject": subject,
                "claim": claim,
                "grounding": grounding,
                "quote": quote,
                "confidence": confidence,
            }
        )
    return records


def normalize_social_records(text: str, valid_evidence_ids: set[str]) -> list[dict]:
    payload = _json_value(text)
    raw_records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        return []
    allowed = {"observation", "belief", "goal", "intent", "coordination", "contradiction"}
    records: list[dict] = []
    for raw in raw_records[:12]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        agent = str(raw.get("agent", "")).strip()[:120]
        claim = str(raw.get("claim", "")).strip()[:500]
        ids = raw.get("evidence_ids", [])
        evidence_ids = list(dict.fromkeys(str(value) for value in ids if str(value) in valid_evidence_ids)) if isinstance(ids, list) else []
        if kind not in allowed or not agent or not claim or not evidence_ids:
            continue
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        records.append(
            {
                "kind": kind,
                "agent": agent,
                "claim": claim,
                "evidence_ids": evidence_ids,
                "confidence": min(0.75, max(0.0, confidence)),
            }
        )
    return records


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _cited_clip_ids(
    direct_data: dict,
    structured_data: dict,
    evidence_by_id: dict[str, dict],
    valid_clip_ids: set[str],
) -> list[str]:
    clip_ids: list[str] = []
    direct_clip = str(direct_data.get("clip_id", ""))
    if direct_clip in valid_clip_ids:
        clip_ids.append(direct_clip)
    structured_ids = structured_data.get("evidence_ids", [])
    if isinstance(structured_ids, list):
        for evidence_id in structured_ids:
            record = evidence_by_id.get(str(evidence_id))
            if record and record["clip_id"] not in clip_ids:
                clip_ids.append(record["clip_id"])
    return clip_ids


def should_switch_to_structured(
    category: str,
    subcategory: str,
    structured_prediction: int,
    verifier_prediction: int,
    structured_data: dict,
    verifier_data: dict,
) -> bool:
    """Apply conservative, auditable gates before overriding direct video."""
    if verifier_prediction != structured_prediction or structured_prediction < 0:
        return False
    if str(verifier_data.get("favored", "")).strip().lower() != "structured":
        return False
    evidence_ids = structured_data.get("evidence_ids", [])
    evidence_ids = set(map(str, evidence_ids)) if isinstance(evidence_ids, list) else set()
    if not evidence_ids:
        return False
    if category == "task_coordination":
        return subcategory == "single_span" and len(evidence_ids) >= 2
    return category in {"social_interaction", "theory_of_mind"}


def process_one(
    index: int,
    row: dict,
    planner: dict,
    retriever: BM25TextRetriever,
    captions: dict[str, str],
    client: JDCloudGeminiClient,
    video_root: Path,
    proxy_root: Path,
    ffmpeg: Path,
    top_k_per_agent: int,
    candidate_pool: int,
    direct_max_clips: int,
    max_clips: int,
    verifier_max_clips: int,
    score_threshold: float,
    structured_categories: set[str],
) -> dict:
    started = time.monotonic()
    candidates = retrieve_video_ids(
        retriever,
        planner.get("selection", []),
        top_k_per_agent,
        candidate_pool,
        score_threshold,
    )
    available = [
        clip for clip in candidates
        if source_video_path(video_root, clip["id"]).stat().st_size >= MIN_SOURCE_VIDEO_BYTES
    ]
    premise = question_premise(row)
    direct_clips = rerank_clips(available, premise, captions, direct_max_clips)
    structured_clips = direct_clips[:max_clips]
    if not direct_clips:
        raise RuntimeError("planner/BM25 retrieval produced no usable video clips")

    question = build_question_prompt(row["question"], row["options"])
    answer_index = row["options"].index(row["answer"])
    direct_raw = client.generate_parts(
        video_parts(direct_clips, video_root, proxy_root, ffmpeg)
        + [{"text": DIRECT_PROMPT.replace("{question}", question)}],
        max_output_tokens=2048,
    )
    direct_prediction, direct_data = _answer_index(direct_raw)

    grounded: list[dict] = []
    extraction_raw: list[dict] = []
    use_structured_route = str(row["category"]) in structured_categories
    if use_structured_route:
        for clip_number, clip in enumerate(structured_clips, start=1):
            prompt = (
                CLIP_EVIDENCE_PROMPT
                .replace("{clip_id}", clip["id"])
                .replace("{observer}", _clip_observer(clip["id"]))
                .replace("{premise}", premise)
            )
            raw = client.generate_parts(
                video_parts([clip], video_root, proxy_root, ffmpeg) + [{"text": prompt}],
                max_output_tokens=2048,
            )
            records = normalize_grounded_records(raw, clip, clip_number)
            grounded.extend(records)
            extraction_raw.append({"clip_id": clip["id"], "raw": raw, "accepted": len(records)})
    grounded.sort(key=lambda record: (record["absolute_start_seconds"], record["clip_id"], record["evidence_id"]))

    valid_evidence_ids = {record["evidence_id"] for record in grounded}
    if grounded:
        social_raw = client.generate(
            SOCIAL_PROMPT.replace("{premise}", premise).replace(
                "{grounded}", _compact_json(grounded)
            ),
            max_output_tokens=2048,
        )
        social = normalize_social_records(social_raw, valid_evidence_ids)
        structured_raw = client.generate(
            STRUCTURED_ANSWER_PROMPT
            .replace("{grounded}", _compact_json(grounded))
            .replace("{social}", _compact_json(social))
            .replace("{question}", question),
            max_output_tokens=2048,
        )
        structured_prediction, structured_data = _answer_index(structured_raw)
        cited_ids = structured_data.get("evidence_ids", [])
        cited_ids = [
            str(evidence_id) for evidence_id in cited_ids
            if str(evidence_id) in valid_evidence_ids
        ] if isinstance(cited_ids, list) else []
        structured_data["evidence_ids"] = list(dict.fromkeys(cited_ids))
        if not cited_ids:
            structured_prediction = -1
    else:
        social_raw = ""
        social = []
        structured_raw = ""
        structured_data = {}
        structured_prediction = -1

    verifier_triggered = (
        direct_prediction >= 0
        and structured_prediction >= 0
        and direct_prediction != structured_prediction
    )
    verifier_raw = ""
    verifier_data: dict = {}
    verifier_clip_ids: list[str] = []
    verifier_switch_accepted = False
    if direct_prediction < 0:
        prediction = structured_prediction
    elif structured_prediction < 0 or not verifier_triggered:
        prediction = direct_prediction
    else:
        evidence_by_id = {record["evidence_id"]: record for record in grounded}
        verifier_clip_ids = _cited_clip_ids(
            direct_data,
            structured_data,
            evidence_by_id,
            {clip["id"] for clip in direct_clips},
        )
        if not verifier_clip_ids:
            verifier_clip_ids = [clip["id"] for clip in structured_clips]
        verifier_clip_ids = verifier_clip_ids[:verifier_max_clips]
        verifier_clips = [clip for clip in direct_clips if clip["id"] in verifier_clip_ids]
        cited_records = [
            evidence_by_id[str(evidence_id)]
            for evidence_id in structured_data.get("evidence_ids", [])
            if str(evidence_id) in evidence_by_id
        ] if isinstance(structured_data.get("evidence_ids", []), list) else []
        verifier_raw = client.generate_parts(
            video_parts(verifier_clips, video_root, proxy_root, ffmpeg)
            + [{
                "text": VERIFY_PROMPT
                .replace(
                    "{direct}",
                    _compact_json(direct_data or {"answer_index": direct_prediction}),
                )
                .replace(
                    "{structured}",
                    _compact_json(structured_data or {"answer_index": structured_prediction}),
                )
                .replace("{cited_records}", _compact_json(cited_records))
                .replace("{question}", question)
            }],
            max_output_tokens=2048,
        )
        verifier_prediction, verifier_data = _answer_index(verifier_raw)
        verifier_switch_accepted = should_switch_to_structured(
            str(row["category"]),
            str(row["subcategory"]),
            structured_prediction,
            verifier_prediction,
            structured_data,
            verifier_data,
        )
        prediction = structured_prediction if verifier_switch_accepted else direct_prediction

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": client.config.model,
        "index": index,
        "category": row["category"],
        "subcategory": row["subcategory"],
        "answer_index": answer_index,
        "route": "grounded_spatial_social" if use_structured_route else "direct_video",
        "selection": planner.get("selection", []),
        "candidate_clips": candidates,
        "video_clips": direct_clips,
        "structured_video_clips": structured_clips if use_structured_route else [],
        "direct_raw": direct_raw,
        "direct_data": direct_data,
        "direct_prediction_index": direct_prediction,
        "direct_correct": direct_prediction == answer_index,
        "extraction_raw": extraction_raw,
        "grounded_records": grounded,
        "social_raw": social_raw,
        "social_records": social,
        "structured_raw": structured_raw,
        "structured_data": structured_data,
        "structured_prediction_index": structured_prediction,
        "structured_correct": structured_prediction == answer_index,
        "verifier_triggered": verifier_triggered,
        "verifier_clip_ids": verifier_clip_ids,
        "verifier_raw": verifier_raw,
        "verifier_data": verifier_data,
        "verifier_switch_accepted": verifier_switch_accepted,
        "prediction_index": prediction,
        "correct": prediction == answer_index,
        "elapsed_seconds": time.monotonic() - started,
    }


def _metrics(rows: list[dict], field: str) -> dict:
    correct = sum(bool(row[field]) for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None}


def summarize(records: Iterable[dict], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "correct" in row and "direct_correct" in row]

    def paired(group: list[dict]) -> dict:
        direct = _metrics(group, "direct_correct")
        structured_group = [
            row for row in group
            if row.get("route") == "grounded_spatial_social"
            and int(row.get("structured_prediction_index", -1)) >= 0
        ]
        structured = _metrics(structured_group, "structured_correct")
        method = _metrics(group, "correct")
        return {
            "direct": direct,
            "structured": structured,
            "grounded_v2": method,
            "delta_vs_direct": method["accuracy"] - direct["accuracy"] if group else None,
        }

    direct_only = sum(row["direct_correct"] and not row["correct"] for row in rows)
    method_only = sum(not row["direct_correct"] and row["correct"] for row in rows)
    both_correct = sum(row["direct_correct"] and row["correct"] for row in rows)
    by_category: dict[str, list[dict]] = {}
    by_subcategory: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)
        by_subcategory.setdefault(str(row["subcategory"]), []).append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "comparison": "paired_same_reranked_raw_video_clips_with_conservative_verifier",
        "model": model,
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "overall": paired(rows),
        "verifier": {
            "triggered": sum(bool(row.get("verifier_triggered")) for row in rows),
            "changed_from_direct": sum(
                row["prediction_index"] != row["direct_prediction_index"] for row in rows
            ),
        },
        "contingency": {
            "both_correct": both_correct,
            "direct_only_correct": direct_only,
            "grounded_v2_only_correct": method_only,
            "both_wrong": len(rows) - both_correct - direct_only - method_only,
            "net_grounded_v2_wins": method_only - direct_only,
            "mcnemar_exact_p": _mcnemar_exact(direct_only, method_only),
        },
        "by_category": {name: paired(group) for name, group in sorted(by_category.items())},
        "by_subcategory": {name: paired(group) for name, group in sorted(by_subcategory.items())},
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
    temporary_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)


def rescore_records(records: dict[int, dict], structured_categories: set[str]) -> None:
    """Reapply routing and verifier gates to checkpointed model responses."""
    for row in records.values():
        if "direct_prediction_index" not in row or "answer_index" not in row:
            continue
        category = str(row.get("category", ""))
        use_structured = category in structured_categories
        direct_prediction = int(row.get("direct_prediction_index", -1))
        structured_prediction = int(row.get("structured_prediction_index", -1))
        verifier_prediction, verifier_data = _answer_index(str(row.get("verifier_raw", "")))
        accepted = use_structured and should_switch_to_structured(
            category,
            str(row.get("subcategory", "")),
            structured_prediction,
            verifier_prediction,
            row.get("structured_data", {}),
            verifier_data,
        )
        if direct_prediction < 0 and use_structured and structured_prediction >= 0:
            prediction = structured_prediction
        else:
            prediction = structured_prediction if accepted else direct_prediction
        row["route"] = "grounded_spatial_social" if use_structured else "direct_video"
        row["verifier_triggered"] = (
            use_structured
            and direct_prediction >= 0
            and structured_prediction >= 0
            and direct_prediction != structured_prediction
        )
        row["verifier_switch_accepted"] = accepted
        row["prediction_index"] = prediction
        row["correct"] = prediction == int(row["answer_index"])


def select_indices(rows: list[dict], planner: dict[int, dict], args: argparse.Namespace) -> list[int]:
    eligible = sorted(index for index in planner if "selection" in planner[index] and index < len(rows))
    if args.exclude_output:
        excluded: set[int] = set()
        for path in args.exclude_output:
            excluded.update(load_jsonl(path))
        eligible = [index for index in eligible if index not in excluded]
    if args.indices:
        wanted = set(args.indices)
        eligible = [index for index in eligible if index in wanted]
    if args.samples_per_category is not None:
        if args.sample_seed is not None:
            random.Random(args.sample_seed).shuffle(eligible)
        counts: Counter[str] = Counter()
        selected: list[int] = []
        for index in eligible:
            category = str(rows[index]["category"])
            if counts[category] < args.samples_per_category:
                selected.append(index)
                counts[category] += 1
        eligible = selected
    if args.limit is not None:
        eligible = eligible[:args.limit]
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-category", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--exclude-output", type=Path, nargs="*")
    parser.add_argument("--top-k-per-agent", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=30)
    parser.add_argument("--direct-max-clips", type=int, default=10)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--verifier-max-clips", type=int, default=4)
    parser.add_argument(
        "--structured-categories",
        nargs="*",
        default=sorted(DEFAULT_STRUCTURED_CATEGORIES),
    )
    parser.add_argument("--score-threshold", type=float, default=10.0)
    parser.add_argument("--keep-errors", action="store_true")
    parser.add_argument(
        "--rescore-input",
        type=Path,
        nargs="+",
        help="Reapply the current route/gates to existing checkpoints without API calls.",
    )
    args = parser.parse_args()
    if min(
        args.workers,
        args.top_k_per_agent,
        args.candidate_pool,
        args.direct_max_clips,
        args.max_clips,
        args.verifier_max_clips,
    ) < 1:
        parser.error("worker, retrieval, and clip limits must be positive")
    if args.max_clips > args.direct_max_clips:
        parser.error("--max-clips cannot exceed --direct-max-clips")
    if args.rescore_input:
        rescored: dict[int, dict] = {}
        for path in args.rescore_input:
            overlap = set(rescored) & set(load_jsonl(path))
            if overlap:
                parser.error(f"rescore inputs contain overlapping indices: {sorted(overlap)[:5]}")
            rescored.update(load_jsonl(path))
        rescore_records(rescored, set(args.structured_categories))
        selected = sorted(rescored)
        model = next((str(row.get("model", "unknown")) for row in rescored.values()), "unknown")
        write_outputs(args.output, rescored, selected, model)
        report = summarize(rescored.values(), model)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
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
    captions = dict(zip(retriever.file_paths, retriever.captions))
    print(
        f"model={config.model!r} selected={len(selected)} resumed={len(selected)-len(pending)} "
        f"pending={len(pending)} workers={args.workers} max_clips={args.max_clips}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                rows[index],
                planner[index],
                retriever,
                captions,
                client,
                args.video_root,
                args.proxy_root,
                args.ffmpeg,
                args.top_k_per_agent,
                args.candidate_pool,
                args.direct_max_clips,
                args.max_clips,
                args.verifier_max_clips,
                args.score_threshold,
                set(args.structured_categories),
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
            overall = report["overall"]
            print(
                f"index={index} category={rows[index]['category']} state={state} "
                f"done={report['completed']}/{len(selected)} direct={overall['direct']['accuracy']} "
                f"structured={overall['structured']['accuracy']} v2={overall['grounded_v2']['accuracy']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
