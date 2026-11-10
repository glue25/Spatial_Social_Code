"""Paired provenance-constrained Spatial-Social Code evaluation on MA-EgoQA.

The runner reuses the exact raw-video clips and direct-video prediction from a
completed paired run. Spatial records must cite a clip and local time span;
Social records must cite valid Spatial evidence IDs. The direct-video baseline
is stored only for paired evaluation and never influences the method's answer.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent.parent
for import_path in (ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from egomas.src.run_spatial_social_jdcloud import _mcnemar_exact  # noqa: E402
from egomas.src.run_video_grounded_v2_jdcloud import _answer_index, _json_value  # noqa: E402
from egomas.src.run_video_spatial_social_jdcloud import (  # noqa: E402
    DEFAULT_FFMPEG,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    video_parts,
)
from egomas.utils.prompt_helpers import build_question_prompt  # noqa: E402
from runtime import UnsafeProgram, execute_social, execute_spatial  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "data" / "MA-EgoQA.json"
DEFAULT_PAIRED_SOURCE = (
    ROOT / "data" / "baselines" / "video_spatial_social_gemini3pro.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "baselines" / "video_provenance_robust_gemini3pro.jsonl"
)
SCHEMA_VERSION = 7
METHOD = "Standalone contrastive provenance Spatial-Social Video v4.1"


CLAIM_PLANNER_PROMPT = """Decompose each multiple-choice option into the
smallest concrete, video-checkable claims needed to distinguish it. Keep
people, actions, objects, dialogue, counts, duration, and ordering explicit.
Do not decide which option is correct and do not add unstated facts. Return
JSON only with at most 12 claims:
{"claims":[{"option":"A","claim":"concrete checkable statement"}]}

Question:
{question}
"""


SPATIAL_PROMPT = """Compile directly grounded Spatial Code from the supplied
raw egocentric clips. Use the complete question and choices only to identify
the concrete details that distinguish the alternatives. Extract positive and
negative evidence for those details when visible: people, objects, dialogue,
counts, duration, ordering, and interactions. Do not select, rank, copy, or
encode an answer choice. Do not infer beliefs, goals, intent, or causes.

Every grounded call must have a unique evidence_id (S1, S2, ...), an exact
clip_id from the manifest, the clip wearer as observer, and a local
evidence_span within 00:00-00:30. Put a short exact audible quote in quote when
available; otherwise use quote="". It is valid to emit only S.unknown when the
clips contain no relevant evidence. Do not fill a quota with weak evidence.

Use ONLY literal calls in these forms, replacing the first argument with a
concrete integer sequence number (1, 2, 3, ...). Never output the literal t:
S.fact(1, relation, subject, object, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="")
S.event(1, actor, action, target, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="...", event_role="executor", phase="during")
S.visible(1, observer, object, value, source="video", confidence=0.8, evidence_id="S1", clip_id="...", evidence_span="00:00-00:05", quote="")
S.unknown(0, relation, subject, object)

No assignments, imports, loops, markdown, comments, answer letters, or prose.
Output at most 16 calls. Omission is uncertainty, never evidence that an event
did not happen.
For S.event, event_role must be initiator, executor, speaker, reactor,
observer, participant, or unknown; phase must be before, during, after, or
unknown. Use unknown rather than guessing either field.

Clip manifest:
{manifest}

Complete question and choices:
{question}
"""


SPATIAL_REPAIR_PROMPT = """Add only directly visible or audible Spatial Code
evidence for the uncovered claims below. Existing evidence is supplied to
avoid repetition. Start evidence IDs at {next_evidence_id}. It is acceptable
to emit no calls when the clips do not contain relevant evidence. Do not infer
or select an answer. The runtime object S already exists. Output ONLY literal
calls in the following forms (not JSON, dictionaries, markdown, or prose):
S.fact(1, relation, subject, object, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="", claim_ids="C1")
S.event(1, actor, action, target, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="...", claim_ids="C1,C2", event_role="executor", phase="during")
S.visible(1, observer, object, value, source="video", confidence=0.8, evidence_id="S1", clip_id="...", evidence_span="00:00-00:05", quote="", claim_ids="C1")
Include valid clip_id, observer, local evidence_span, quote, and claim_ids.
Output at most 8 complete calls. Begin the response with S. or return NONE.
For S.event, use only the documented event roles and phases; use unknown when
the role or phase is not directly supported.

Clip manifest:
{manifest}

Uncovered claims:
{uncovered_claims}

Existing validated Spatial Code:
{spatial_code}
"""


SOCIAL_PROMPT = """Compile a minimal Social Code trace using ONLY the validated
Spatial trace below. Use the complete question and choices to identify which
social distinctions require evidence, but do not select, rank, copy, or encode
an answer choice. Every observation, belief, goal, intent, action evidence, or
social goal must cite one or more existing Spatial evidence IDs using
evidence_ids="S1,S2". Do not invent people, events, dialogue, or citations. A
world fact is not automatically an agent belief. Use M.unknown when the trace
does not support a social inference; missing evidence is not negative evidence.
An executor is not thereby the initiator, person responsible, decision maker,
cause, or person with the corresponding intent. Infer those variables only
from explicit speech or a directly grounded event with the matching role and
phase; otherwise keep them unknown.

Use ONLY literal calls in these forms:
M.observe(t, agent, fact, source="spatial_trace", confidence=0.8, evidence_ids="S1")
M.belief(t, agent, fact, confidence=0.7, evidence="...", evidence_ids="S1")
M.goal(agent, goal, confidence=0.7, evidence="...", evidence_ids="S1")
M.intent(t, agent, intent, confidence=0.7, evidence="...", evidence_ids="S1")
M.action_evidence(t, agent, action, hypothesis, confidence=0.7, evidence="...", evidence_ids="S1")
M.social_goal(agent, target_agent, goal, confidence=0.7, evidence="...", evidence_ids="S1")
M.unknown(agent, variable)

No assignments, imports, loops, markdown, comments, answer letters, or prose.
Output at most 10 calls. Output only M.unknown if no valid evidence IDs exist.

Complete question and choices:
{question}

Validated Spatial trace:
{spatial_trace}
"""


STRUCTURED_ANSWER_PROMPT = """Answer the multiple-choice question using the raw
videos and the validated executed traces. Independently compare every choice
against the raw videos before deciding. The traces are navigation assistance,
not ground truth: an omitted event is unknown, not absent, and raw video always
overrides a trace. Prefer directly visible/audible records over mental-state
inference. You must choose exactly one listed option A-E. Return JSON only:
{"answer":"A", "evidence_ids":["S1"], "reason":"one short sentence"}

Do not treat performing or observing an action as evidence that the person
initiated it, was responsible for it, intended it, or caused it. Those claims
require explicit matching evidence; when absent, keep them unknown and rely on
the raw video chronology.

Validated Spatial trace:
{spatial_trace}

Validated Social trace:
{social_trace}

Question:
{question}
"""


ANSWER_REPAIR_PROMPT = """Answer the multiple-choice question from the raw
videos. Your previous response did not contain a valid listed option. Compare
all choices and choose exactly one uppercase letter A, B, C, D, or E, even if
uncertain. Return JSON only:
{"answer":"A", "evidence_ids":[], "reason":"one short sentence"}

Question:
{question}
"""


_SPAN_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})\s*[-–—]\s*(?:(\d+):)?(\d{1,2}):(\d{2})$"
)
_EVIDENCE_ID_RE = re.compile(r"^S\d+$")


def load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["index"])] = row
    return records


def clip_observer(clip_id: str) -> str:
    parts = clip_id.removesuffix(".mp4").split("_")
    return parts[2].title() if len(parts) >= 4 else "Unknown"


def clip_manifest(clips: list[dict]) -> str:
    return "\n".join(
        f"Clip {offset}: clip_id={clip['id']}; observer={clip_observer(clip['id'])}"
        for offset, clip in enumerate(clips, start=1)
    )


def _span_is_local(value: object) -> bool:
    match = _SPAN_RE.fullmatch(str(value).strip())
    if not match:
        return False
    start = int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    end = int(match.group(4) or 0) * 3600 + int(match.group(5)) * 60 + int(match.group(6))
    return 0 <= start <= end <= 35


def normalize_spatial_code(
    text: str,
    valid_clip_ids: set[str],
    max_lines: int = 16,
    valid_claim_ids: set[str] | None = None,
    existing_evidence_ids: set[str] | None = None,
    fallback_unknown: bool = True,
) -> str:
    """Keep safe Spatial calls and require provenance on grounded records."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    kept: list[str] = []
    evidence_ids: set[str] = set(existing_evidence_ids or ())
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
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
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "S":
            continue
        method = call.func.attr
        if method not in {"fact", "event", "visible", "unknown"}:
            continue
        # Models occasionally copy the signature placeholder `t` verbatim.
        # It carries no semantic content here, so map it deterministically to
        # an integer rather than discarding otherwise valid grounded evidence.
        if call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "t":
            call.args[0] = ast.Constant(
                value=0 if method == "unknown" else len(evidence_ids) + 1
            )
            ast.fix_missing_locations(tree)
            line = ast.unparse(tree)
        keywords = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg and isinstance(keyword.value, ast.Constant)
        }
        if method != "unknown":
            evidence_id = str(keywords.get("evidence_id", ast.Constant("")).value).strip()
            clip_id = str(keywords.get("clip_id", ast.Constant("")).value).strip()
            evidence_span = str(keywords.get("evidence_span", ast.Constant("")).value).strip()
            claim_ids = {
                value.strip()
                for value in str(keywords.get("claim_ids", ast.Constant("")).value).split(",")
                if value.strip()
            }
            observer = (
                str(keywords.get("observer", ast.Constant("")).value).strip()
                if method != "visible"
                else str(call.args[1].value).strip()
                if len(call.args) > 1 and isinstance(call.args[1], ast.Constant)
                else ""
            )
            if (
                not _EVIDENCE_ID_RE.fullmatch(evidence_id)
                or evidence_id in evidence_ids
                or clip_id not in valid_clip_ids
                or observer.casefold() != clip_observer(clip_id).casefold()
                or not _span_is_local(evidence_span)
                or (valid_claim_ids is not None and (not claim_ids or not claim_ids <= valid_claim_ids))
            ):
                continue
            evidence_ids.add(evidence_id)
        try:
            execute_spatial(line)
        except (UnsafeProgram, ValueError, TypeError):
            continue
        kept.append(line)
        if len(kept) >= max_lines:
            break
    if not kept and not fallback_unknown:
        return ""
    if not kept:
        return "S.unknown(0, 'relevant_relation', 'question', 'video')"
    code = "\n".join(kept)
    try:
        execute_spatial(code)
    except UnsafeProgram:
        return "S.unknown(0, 'relevant_relation', 'question', 'video')"
    return code


def build_claim_plan(raw: str, options: list[str]) -> list[dict[str, str]]:
    payload = _json_value(raw)
    raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
    claims: list[dict[str, str]] = []
    for item in raw_claims if isinstance(raw_claims, list) else []:
        if not isinstance(item, dict):
            continue
        option = str(item.get("option", "")).strip().upper()[:1]
        claim = " ".join(str(item.get("claim", "")).split())
        if option in "ABCDE"[: len(options)] and claim:
            claims.append({"option": option, "claim": claim})
        if len(claims) >= 12:
            break
    if not claims:
        claims = [
            {"option": "ABCDE"[index], "claim": " ".join(option.split())}
            for index, option in enumerate(options)
        ]
    return [
        {"claim_id": f"C{index}", **claim}
        for index, claim in enumerate(claims, start=1)
    ]


def covered_claim_ids(records: dict[str, dict], valid_claim_ids: set[str]) -> set[str]:
    covered: set[str] = set()
    for record in records.values():
        raw = str(record.get("values", {}).get("claim_ids", ""))
        covered.update(value.strip() for value in raw.split(",") if value.strip() in valid_claim_ids)
    return covered


def final_answer_index(text: str) -> tuple[int, dict]:
    """Parse only an explicit final answer, never an option inside checks."""
    payload = _json_value(text)
    data = payload if isinstance(payload, dict) else {}
    answer = str(data.get("answer", "")).strip().upper()
    if answer and answer[0] in "ABCDE":
        return "ABCDE".index(answer[0]), data
    match = re.search(
        r"(?i)[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-E])\b",
        text,
    )
    if match:
        answer = match.group(1).upper()
        return "ABCDE".index(answer), {"answer": answer, "evidence_ids": []}
    plain = text.strip().upper()
    return ("ABCDE".index(plain), {}) if plain in "ABCDE" else (-1, {})


def spatial_records(trace: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in trace.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = record.get("values", {})
        evidence_id = str(values.get("evidence_id", ""))
        if _EVIDENCE_ID_RE.fullmatch(evidence_id):
            records[evidence_id] = record
    return records


def normalize_social_code(text: str, valid_evidence_ids: set[str], max_lines: int = 10) -> str:
    """Keep safe Social calls whose citations resolve to Spatial evidence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    allowed = {"observe", "belief", "goal", "intent", "action_evidence", "social_goal", "unknown"}
    kept: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
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
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "M":
            continue
        if call.func.attr not in allowed:
            continue
        try:
            execute_social(
                line,
                valid_evidence_ids=valid_evidence_ids,
                require_evidence=True,
            )
        except (UnsafeProgram, ValueError, TypeError):
            continue
        kept.append(line)
        if len(kept) >= max_lines:
            break
    if not kept:
        return "M.unknown('multi_agent_group', 'social_state')"
    code = "\n".join(kept)
    try:
        execute_social(
            code,
            valid_evidence_ids=valid_evidence_ids,
            require_evidence=True,
        )
    except UnsafeProgram:
        return "M.unknown('multi_agent_group', 'social_state')"
    return code


def _validated_citations(data: dict, valid_evidence_ids: set[str]) -> list[str]:
    raw = data.get("evidence_ids", [])
    if not isinstance(raw, list):
        return []
    return list(
        dict.fromkeys(str(value) for value in raw if str(value) in valid_evidence_ids)
    )


def process_one(
    index: int,
    row: dict,
    paired: dict,
    client: JDCloudGeminiClient,
    video_root: Path,
    proxy_root: Path,
    ffmpeg: Path,
) -> dict:
    started = time.monotonic()
    clips = list(paired.get("video_clips", []))
    if not clips:
        raise RuntimeError("paired source contains no video clips")
    baseline_prediction = int(paired.get("baseline_prediction_index", -1))
    question = build_question_prompt(row["question"], row["options"])
    answer_index = row["options"].index(row["answer"])
    media = video_parts(clips, video_root, proxy_root, ffmpeg)

    spatial_prompt = (
        SPATIAL_PROMPT.replace("{manifest}", clip_manifest(clips))
        .replace("{question}", question)
    )
    spatial_raw = client.generate_parts(
        media + [{"text": spatial_prompt}], max_output_tokens=4096
    )
    valid_clip_ids = {str(clip["id"]) for clip in clips}
    spatial_code = normalize_spatial_code(spatial_raw, valid_clip_ids)
    spatial_trace = execute_spatial(spatial_code)
    evidence_by_id = spatial_records(spatial_trace)
    claim_plan_raw = ""
    claim_plan: list[dict[str, str]] = []
    valid_claim_ids: set[str] = set()
    covered_before_repair: set[str] = set()
    spatial_repair_raw = ""
    spatial_repair_code = ""
    repair_triggered = False
    valid_evidence_ids = set(evidence_by_id)
    covered_after_repair = covered_claim_ids(evidence_by_id, valid_claim_ids)

    social_prompt = (
        SOCIAL_PROMPT.replace("{question}", question)
        .replace("{spatial_trace}", spatial_trace)
    )
    social_raw = client.generate(social_prompt, max_output_tokens=4096)
    social_code = normalize_social_code(social_raw, valid_evidence_ids)
    social_trace = execute_social(
        social_code,
        valid_evidence_ids=valid_evidence_ids,
        require_evidence=True,
    )

    structured_prompt = (
        STRUCTURED_ANSWER_PROMPT.replace("{spatial_trace}", spatial_trace)
        .replace("{social_trace}", social_trace)
        .replace("{question}", question)
    )
    structured_raw = client.generate_parts(
        media + [{"text": structured_prompt}], max_output_tokens=2048
    )
    structured_prediction, structured_data = final_answer_index(structured_raw)
    answer_retry_used = structured_prediction < 0
    structured_initial_raw = structured_raw if answer_retry_used else ""
    if answer_retry_used:
        repair_prompt = ANSWER_REPAIR_PROMPT.replace("{question}", question)
        structured_raw = client.generate_parts(
            media + [{"text": repair_prompt}], max_output_tokens=1024
        )
        structured_prediction, structured_data = final_answer_index(structured_raw)
    structured_ids = _validated_citations(structured_data, valid_evidence_ids)
    structured_data["evidence_ids"] = structured_ids
    # The baseline is retained only for paired evaluation. It never selects or
    # changes the standalone method's answer.
    prediction = structured_prediction

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": client.config.model,
        "index": index,
        "category": row["category"],
        "subcategory": row["subcategory"],
        "answer_index": answer_index,
        "video_clips": clips,
        "baseline_prediction_index": baseline_prediction,
        "baseline_correct": baseline_prediction == answer_index,
        "claim_plan_raw": claim_plan_raw,
        "claim_plan": claim_plan,
        "covered_claim_ids_before_repair": sorted(covered_before_repair),
        "covered_claim_ids": sorted(covered_after_repair),
        "repair_triggered": repair_triggered,
        "claim_coverage": (
            len(covered_after_repair) / len(valid_claim_ids) if valid_claim_ids else None
        ),
        "spatial_raw": spatial_raw,
        "spatial_repair_raw": spatial_repair_raw,
        "spatial_repair_code": spatial_repair_code,
        "spatial_code": spatial_code,
        "spatial_trace": spatial_trace,
        "valid_evidence_ids": sorted(valid_evidence_ids),
        "social_raw": social_raw,
        "social_code": social_code,
        "social_trace": social_trace,
        "structured_raw": structured_raw,
        "structured_initial_raw": structured_initial_raw,
        "answer_retry_used": answer_retry_used,
        "structured_data": structured_data,
        "structured_prediction_index": structured_prediction,
        "structured_correct": structured_prediction == answer_index,
        "structured_citation_valid": bool(structured_ids),
        "prediction_index": prediction,
        "correct": prediction == answer_index,
        "elapsed_seconds": time.monotonic() - started,
    }


def _metrics(rows: list[dict], field: str) -> dict:
    correct = sum(bool(row[field]) for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None}


def summarize(records: Iterable[dict], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "correct" in row and "baseline_correct" in row]
    measured_coverage = [
        float(row["claim_coverage"])
        for row in rows
        if isinstance(row.get("claim_coverage"), (int, float))
    ]

    def paired(group: list[dict]) -> dict:
        baseline = _metrics(group, "baseline_correct")
        method = _metrics(group, "correct")
        return {
            "baseline": baseline,
            "standalone_spatial_social": method,
            "delta_vs_baseline": method["accuracy"] - baseline["accuracy"] if group else None,
        }

    baseline_only = sum(row["baseline_correct"] and not row["correct"] for row in rows)
    method_only = sum(not row["baseline_correct"] and row["correct"] for row in rows)
    both_correct = sum(row["baseline_correct"] and row["correct"] for row in rows)
    by_category: dict[str, list[dict]] = {}
    by_subcategory: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)
        by_subcategory.setdefault(str(row["subcategory"]), []).append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "comparison": "paired_exact_same_raw_video_clips_and_direct_predictions",
        "model": model,
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "overall": paired(rows),
        "provenance": {
            "with_spatial_evidence": sum(bool(row.get("valid_evidence_ids")) for row in rows),
            "answer_parse_valid": sum(row.get("prediction_index", -1) >= 0 for row in rows),
            "citation_valid": sum(bool(row.get("structured_citation_valid")) for row in rows),
            "repair_triggered": sum(bool(row.get("repair_triggered")) for row in rows),
            "full_claim_coverage": sum(row.get("claim_coverage") == 1.0 for row in rows),
            "mean_claim_coverage": (
                sum(measured_coverage) / len(measured_coverage)
                if measured_coverage
                else None
            ),
        },
        "contingency": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "standalone_only_correct": method_only,
            "both_wrong": len(rows) - both_correct - baseline_only - method_only,
            "net_standalone_wins": method_only - baseline_only,
            "mcnemar_exact_p": _mcnemar_exact(baseline_only, method_only),
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
    temporary_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary_summary.replace(summary_path)


def select_indices(rows: list[dict], paired: dict[int, dict], args: argparse.Namespace) -> list[int]:
    eligible = sorted(index for index in paired if index < len(rows) and paired[index].get("video_clips"))
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
        eligible = eligible[: args.limit]
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--paired-source", type=Path, default=DEFAULT_PAIRED_SOURCE)
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
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--keep-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.ffmpeg.is_file():
        parser.error(f"ffmpeg not found: {args.ffmpeg}")

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    paired = load_jsonl(args.paired_source)
    selected = select_indices(rows, paired, args)
    existing = load_jsonl(args.output)
    pending = [
        index for index in selected
        if index not in existing or ("error" in existing[index] and not args.keep_errors)
    ]
    print(
        f"model={config.model!r} selected={len(selected)} resumed={len(selected)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                rows[index],
                paired[index],
                client,
                args.video_root,
                args.proxy_root,
                args.ffmpeg,
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
                f"done={report['completed']}/{len(selected)} "
                f"baseline={overall['baseline']['accuracy']} "
                f"standalone={overall['standalone_spatial_social']['accuracy']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
