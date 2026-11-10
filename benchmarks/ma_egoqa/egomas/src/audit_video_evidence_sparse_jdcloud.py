"""Independently audit evidence-sparse MA-EgoQA answers from raw video."""

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

from egomas.src.jdcloud_gemini import DEFAULT_CREDENTIALS_FILE, JDCloudGeminiClient, load_service_config
from egomas.src.run_video_grounded_v2_jdcloud import _json_value
from egomas.src.run_video_provenance_v3_jdcloud import (
    _span_is_local,
    clip_manifest,
    final_answer_index,
    load_jsonl,
    write_outputs,
)
from egomas.src.run_video_spatial_social_jdcloud import (
    DEFAULT_FFMPEG,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    video_parts,
)
from egomas.utils.prompt_helpers import build_question_prompt


SCHEMA_VERSION = 9
METHOD = "Evidence-audited standalone Spatial-Social Video"
DEFAULT_PRIMARY = ROOT / "data" / "baselines" / "video_balanced_role_full.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "video_evidence_audit_full.jsonl"

AUDIT_PROMPT = """Independently answer this multiple-choice question from the
supplied raw egocentric clips. No previous answer is provided. Compare every
choice against visible actions, audible dialogue, observer viewpoint, and
chronology. Similar-looking clips may be distractors. Do not infer initiation,
responsibility, intent, or causality merely from execution or observation.

Return JSON only:
{"answer":"A","confidence":0.90,"evidence":[{"clip_id":"exact manifest id","span":"00:00-00:05","observation":"short grounded fact"}],"reason":"short comparison"}

Use confidence above 0.85 only when the answer is supported by at least two
concrete video observations. Use exact clip IDs and local spans. Never mention
captions, retrieval, a previous answer, or a baseline.

Clip manifest:
{manifest}

Question:
{question}
"""


def needs_audit(record: dict) -> bool:
    return not record.get("structured_data", {}).get("evidence_ids", [])


def unchanged(primary: dict) -> dict:
    result = dict(primary)
    result.update({
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "audit_triggered": False,
        "primary_prediction_index": primary["prediction_index"],
        "primary_correct": bool(primary["correct"]),
    })
    return result


def valid_audit_evidence(payload: dict, valid_clip_ids: set[str]) -> list[dict]:
    raw = payload.get("evidence", []) if isinstance(payload, dict) else []
    valid: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("clip_id", "")).strip()
        span = str(item.get("span", "")).strip()
        observation = " ".join(str(item.get("observation", "")).split())
        key = (clip_id, span)
        if clip_id in valid_clip_ids and _span_is_local(span) and observation and key not in seen:
            valid.append({"clip_id": clip_id, "span": span, "observation": observation})
            seen.add(key)
    return valid


def audit_one(primary: dict, row: dict, client: JDCloudGeminiClient, video_root: Path, proxy_root: Path, ffmpeg: Path) -> dict:
    started = time.monotonic()
    clips = list(primary["video_clips"])
    media = video_parts(clips, video_root, proxy_root, ffmpeg)
    question = build_question_prompt(row["question"], row["options"])
    prompt = AUDIT_PROMPT.replace("{manifest}", clip_manifest(clips)).replace("{question}", question)
    raw = client.generate_parts(media + [{"text": prompt}], max_output_tokens=2048)
    proposed, parsed = final_answer_index(raw)
    payload = _json_value(raw)
    if not isinstance(payload, dict):
        payload = parsed if isinstance(parsed, dict) else {}
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    evidence = valid_audit_evidence(payload, {str(clip["id"]) for clip in clips})
    primary_prediction = int(primary["prediction_index"])
    flip_accepted = (
        proposed >= 0
        and proposed != primary_prediction
        and confidence >= 0.85
        and len(evidence) >= 2
    )
    prediction = proposed if flip_accepted else primary_prediction
    answer_index = row["options"].index(row["answer"])
    result = dict(primary)
    result.update({
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "audit_triggered": True,
        "primary_prediction_index": primary_prediction,
        "primary_correct": bool(primary["correct"]),
        "audit_raw": raw,
        "audit_prediction_index": proposed,
        "audit_confidence": confidence,
        "audit_evidence": evidence,
        "audit_flip_accepted": flip_accepted,
        "prediction_index": prediction,
        "correct": prediction == answer_index,
        "audit_seconds": time.monotonic() - started,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "MA-EgoQA.json")
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    args = parser.parse_args()

    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    primary = load_jsonl(args.primary)
    selected = sorted(primary)
    audited = [index for index in selected if needs_audit(primary[index])]
    existing = load_jsonl(args.output)
    for index in selected:
        if index not in audited and (index not in existing or "error" in existing[index]):
            existing[index] = unchanged(primary[index])
    config = load_service_config(args.credentials_file, args.model)
    write_outputs(args.output, existing, selected, config.model)
    pending = [i for i in audited if i not in existing or "error" in existing[i]]
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    print(f"selected={len(selected)} audited={len(audited)} resumed={len(selected)-len(pending)} pending={len(pending)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_one, primary[i], rows[i], client, args.video_root, args.proxy_root, args.ffmpeg): i
            for i in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                existing[index] = future.result()
                state = "correct" if existing[index]["correct"] else "wrong"
            except Exception as error:
                existing[index] = {"index": index, "category": rows[index]["category"], "subcategory": rows[index]["subcategory"], "error": f"{type(error).__name__}: {error}"}
                state = "error"
            write_outputs(args.output, existing, selected, config.model)
            print(f"index={index} state={state} done={len(existing)}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
