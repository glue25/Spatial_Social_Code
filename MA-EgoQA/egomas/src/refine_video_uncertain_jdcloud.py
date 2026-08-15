"""Refine evidence-sparse MA-EgoQA predictions without consulting the baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MA_ROOT.parent
for import_path in (MA_ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.jdcloud_gemini import DEFAULT_CREDENTIALS_FILE, JDCloudGeminiClient, load_service_config
from egomas.src.run_video_provenance_v3_jdcloud import (
    ANSWER_REPAIR_PROMPT,
    ROOT,
    SOCIAL_PROMPT,
    STRUCTURED_ANSWER_PROMPT,
    _validated_citations,
    clip_manifest,
    final_answer_index,
    load_jsonl,
    normalize_social_code,
    normalize_spatial_code,
    spatial_records,
    write_outputs,
)
from egomas.src.run_video_spatial_social_jdcloud import (
    DEFAULT_FFMPEG,
    DEFAULT_PROXY_ROOT,
    DEFAULT_VIDEO_ROOT,
    video_parts,
)
from egomas.utils.prompt_helpers import build_question_prompt
from runtime import execute_social, execute_spatial


SCHEMA_VERSION = 8
METHOD = "Uncertainty-refined provenance Spatial-Social Video"
DEFAULT_PRIMARY = ROOT / "data" / "baselines" / "video_provenance_robust_gemini3pro.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "video_uncertainty_gemini3pro.jsonl"

REOBSERVE_PROMPT = """Re-observe the raw egocentric clips to add only new,
directly visible or audible evidence relevant to the complete question. The
existing validated code contains too little evidence. Verify who acted or
spoke, who observed it, exact dialogue when audible, action order, counts, and
duration. Seek disconfirming as well as supporting evidence, but never select,
rank, or encode an answer option. Do not repeat existing records.

The runtime object S exists. Output ONLY complete literal calls:
S.fact(1, relation, subject, object, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="")
S.event(1, actor, action, target, source="video", confidence=0.8, evidence_id="S1", clip_id="...", observer="...", evidence_span="00:00-00:05", quote="...")
S.visible(1, observer, object, value, source="video", confidence=0.8, evidence_id="S1", clip_id="...", evidence_span="00:00-00:05", quote="")

Start new evidence IDs at {next_evidence_id}. Use only exact clip IDs from the
manifest and local spans within 00:00-00:30. Return NONE if no new grounded
evidence exists. No JSON, markdown, prose, variables, or answer letters.

Clip manifest:
{manifest}

Existing validated Spatial Code:
{spatial_code}

Complete question and choices:
{question}
"""


def needs_refinement(record: dict) -> bool:
    return len(record.get("valid_evidence_ids", [])) < 2


def unchanged(primary: dict) -> dict:
    result = dict(primary)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "uncertainty_triggered": False,
            "primary_prediction_index": primary["prediction_index"],
            "primary_correct": bool(primary["correct"]),
        }
    )
    return result


def refine_one(
    primary: dict,
    row: dict,
    client: JDCloudGeminiClient,
    video_root: Path,
    proxy_root: Path,
    ffmpeg: Path,
) -> dict:
    started = time.monotonic()
    clips = list(primary["video_clips"])
    media = video_parts(clips, video_root, proxy_root, ffmpeg)
    question = build_question_prompt(row["question"], row["options"])
    old_ids = set(primary.get("valid_evidence_ids", []))
    next_number = max(
        [int(value[1:]) for value in old_ids if value.startswith("S") and value[1:].isdigit()],
        default=0,
    ) + 1
    prompt = (
        REOBSERVE_PROMPT.replace("{next_evidence_id}", f"S{next_number}")
        .replace("{manifest}", clip_manifest(clips))
        .replace("{spatial_code}", str(primary.get("spatial_code", "")))
        .replace("{question}", question)
    )
    reobserve_raw = client.generate_parts(media + [{"text": prompt}], max_output_tokens=4096)
    valid_clip_ids = {str(clip["id"]) for clip in clips}
    reobserve_code = normalize_spatial_code(
        reobserve_raw,
        valid_clip_ids,
        max_lines=8,
        existing_evidence_ids=old_ids,
        fallback_unknown=False,
    )
    combined_code = "\n".join(
        value for value in (str(primary.get("spatial_code", "")), reobserve_code) if value
    )
    combined_trace = execute_spatial(combined_code)
    combined_records = spatial_records(combined_trace)
    new_ids = set(combined_records) - old_ids
    if not new_ids:
        result = unchanged(primary)
        result.update(
            {
                "uncertainty_triggered": True,
                "new_evidence_accepted": False,
                "reobserve_raw": reobserve_raw,
                "reobserve_code": reobserve_code,
                "refinement_seconds": time.monotonic() - started,
            }
        )
        return result

    valid_ids = set(combined_records)
    social_prompt = (
        SOCIAL_PROMPT.replace("{question}", question)
        .replace("{spatial_trace}", combined_trace)
    )
    social_raw = client.generate(social_prompt, max_output_tokens=4096)
    social_code = normalize_social_code(social_raw, valid_ids)
    social_trace = execute_social(
        social_code, valid_evidence_ids=valid_ids, require_evidence=True
    )
    answer_prompt = (
        STRUCTURED_ANSWER_PROMPT.replace("{spatial_trace}", combined_trace)
        .replace("{social_trace}", social_trace)
        .replace("{question}", question)
    )
    structured_raw = client.generate_parts(
        media + [{"text": answer_prompt}], max_output_tokens=2048
    )
    prediction, structured_data = final_answer_index(structured_raw)
    retry_used = prediction < 0
    initial_raw = structured_raw if retry_used else ""
    if retry_used:
        structured_raw = client.generate_parts(
            media + [{"text": ANSWER_REPAIR_PROMPT.replace("{question}", question)}],
            max_output_tokens=1024,
        )
        prediction, structured_data = final_answer_index(structured_raw)
    citations = _validated_citations(structured_data, valid_ids)
    structured_data["evidence_ids"] = citations
    answer_index = row["options"].index(row["answer"])
    proposed_prediction = prediction
    cited_new_ids = set(citations) & new_ids
    flip_accepted = proposed_prediction >= 0 and (
        proposed_prediction == int(primary["prediction_index"])
        or len(cited_new_ids) >= 2
    )
    if not flip_accepted:
        prediction = int(primary["prediction_index"])
    result = dict(primary)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "uncertainty_triggered": True,
            "new_evidence_accepted": True,
            "primary_prediction_index": primary["prediction_index"],
            "primary_correct": bool(primary["correct"]),
            "reobserve_raw": reobserve_raw,
            "reobserve_code": reobserve_code,
            "spatial_code": combined_code,
            "spatial_trace": combined_trace,
            "valid_evidence_ids": sorted(valid_ids),
            "social_raw": social_raw,
            "social_code": social_code,
            "social_trace": social_trace,
            "structured_raw": structured_raw,
            "structured_initial_raw": initial_raw,
            "answer_retry_used": retry_used,
            "structured_data": structured_data,
            "structured_prediction_index": proposed_prediction,
            "structured_correct": proposed_prediction == answer_index,
            "structured_citation_valid": bool(citations),
            "new_cited_evidence_ids": sorted(cited_new_ids),
            "refinement_flip_accepted": flip_accepted,
            "prediction_index": prediction,
            "correct": prediction == answer_index,
            "refinement_seconds": time.monotonic() - started,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "MA-EgoQA.json")
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--uncertain-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    args = parser.parse_args()

    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    primary = load_jsonl(args.primary)
    uncertain = [i for i in sorted(primary) if needs_refinement(primary[i])]
    selected = uncertain if args.uncertain_only else sorted(primary)
    if args.limit is not None:
        selected = selected[: args.limit]
    existing = load_jsonl(args.output)
    for index in selected:
        if index not in uncertain and (index not in existing or "error" in existing[index]):
            existing[index] = unchanged(primary[index])
    config = load_service_config(args.credentials_file, args.model)
    write_outputs(args.output, existing, selected, config.model)
    pending = [
        i for i in selected
        if i in uncertain and (i not in existing or "error" in existing[i])
    ]
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    print(
        f"selected={len(selected)} uncertain={len(uncertain)} resumed={len(selected)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index in pending:
            future = executor.submit(
                refine_one,
                primary[index],
                rows[index],
                client,
                args.video_root,
                args.proxy_root,
                args.ffmpeg,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            try:
                existing[index] = future.result()
                state = "correct" if existing[index]["correct"] else "wrong"
            except Exception as error:
                existing[index] = {
                    "index": index,
                    "category": rows[index]["category"],
                    "subcategory": rows[index]["subcategory"],
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            write_outputs(args.output, existing, selected, config.model)
            print(f"index={index} state={state} done={len(existing)}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
