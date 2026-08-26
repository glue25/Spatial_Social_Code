#!/usr/bin/env python3
"""Registered call-count-matched Direct baseline for frozen MMToM inputs.

The schedule is fixed before inference: two calls for every example and a
third call for 350 examples selected by SHA-256 rank.  This totals 1,550
logical model calls, matching the frozen Spatial-Social method's 1,200 base
calls plus 350 refinement calls.  Invalid outputs are retained and never
replaced outside this fixed budget.  Labels are read only by ``--score`` after
the prediction file is complete.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOTOM_MODEL = ROOT / "AutoToM" / "model"
MA_EGOQA = ROOT / "MA-EgoQA"
for import_path in (AUTOTOM_MODEL, MA_EGOQA, ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402


PROMPT = """Answer the MMToM multiple-choice question from the initial state,
action history, and ordered RGB frames. Compare both choices against the
evidence. Return exactly one uppercase letter: A or B.

This is independent inference replicate {replicate}; do not rely on any prior
response.

MMToM problem:
{question}

Answer:"""


def sha_rank(index: int) -> str:
    return hashlib.sha256(f"iclr-equal-call-direct-v1:{index}".encode()).hexdigest()


def call_schedule(indices: list[int]) -> dict[int, int]:
    third_call = set(sorted(indices, key=sha_rank)[:350])
    return {index: 3 if index in third_call else 2 for index in indices}


def image_parts(paths: list[Path]) -> list[dict]:
    parts = []
    for path in paths:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append(
            {"inline_data": {"mimeType": mime, "data": base64.b64encode(path.read_bytes()).decode("ascii")}}
        )
    return parts


def parse_ab(text: str) -> str:
    match = re.search(r"(?:ANSWER\s*[:=]\s*)?\b([AB])\b", text.strip().upper())
    return match.group(1).lower() if match else ""


def majority(predictions: list[str]) -> str:
    valid = [value for value in predictions if value in {"a", "b"}]
    if not valid:
        return ""
    counts = Counter(valid)
    if counts["a"] == counts["b"]:
        return valid[0]  # preregistered, label-independent tie break
    return "a" if counts["a"] > counts["b"] else "b"


def load_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {
        int(row["index"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def write_jsonl(path: Path, rows: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(rows[index], ensure_ascii=False) + "\n" for index in sorted(rows)),
        encoding="utf-8",
    )
    temporary.replace(path)


def infer_one(index: int, row: dict, calls: int, frames_root: Path, client: JDCloudGeminiClient) -> dict:
    frame_paths, details = sample_rgb_frames(frames_root, row, num_frames=8)
    media = image_parts([Path(path) for path in frame_paths])
    raw = []
    predictions = []
    for replicate in range(calls):
        response = client.generate_parts(
            media + [{"text": PROMPT.format(replicate=replicate + 1, question=row["question"])}],
            max_output_tokens=64,
        )
        raw.append(response)
        predictions.append(parse_ab(response))
    return {
        "schema_version": 1,
        "protocol": "preregistered_equal_logical_call_direct_v1",
        "index": index,
        "episode": row["episode"],
        "planned_calls": calls,
        "image_details": details,
        "replicate_predictions": predictions,
        "prediction": majority(predictions),
        "raw_responses": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data" / "required_frames")
    parser.add_argument("--output", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1" / "equal_call_direct_predictions.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "workdirs" / "iclr_ablations_v1" / "equal_call_direct_manifest.json")
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    rows = load_questions(args.questions)
    indices = list(range(len(rows)))
    if len(indices) != 600:
        raise ValueError(f"registered protocol requires 600 examples, found {len(indices)}")
    schedule = call_schedule(indices)
    manifest = {
        "schema_version": 1,
        "status": "registered_not_run",
        "selection": "all 600 examples; third-call allocation by SHA-256 rank, independent of labels and predictions",
        "aggregation": "majority vote; first valid response breaks a two-way tie",
        "model": args.model or "must resolve to the same Gemini 3-Pro endpoint as the frozen primary",
        "frames_per_example": 8,
        "total_logical_calls": sum(schedule.values()),
        "calls_by_example": {str(index): schedule[index] for index in indices},
        "label_access_during_inference": False,
        "caveat": "call-count parity is not token or FLOP parity",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"examples": len(indices), "total_calls": sum(schedule.values()), "two_call": 250, "three_call": 350}))
        return

    existing = load_existing(args.output)
    if args.score:
        if len(existing) != 600:
            raise ValueError("scoring requires all 600 predictions")
        valid = [index for index in indices if existing[index].get("prediction") in {"a", "b"}]
        correct = sum(existing[index]["prediction"] == str(rows[index]["answer"]).lower() for index in valid)
        score = {"valid": len(valid), "total": 600, "correct": correct, "accuracy": correct / len(valid) if valid else None}
        args.output.with_name("equal_call_direct_score.json").write_text(json.dumps(score, indent=2) + "\n")
        print(json.dumps(score))
        return

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    pending = [index for index in indices if index not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(infer_one, index, rows[index], schedule[index], args.frames, client): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                existing[index] = future.result()
            except Exception as error:
                existing[index] = {"index": index, "error": f"{type(error).__name__}: {error}"}
            write_jsonl(args.output, existing)
            print(f"completed={len(existing)}/600 index={index}", flush=True)


if __name__ == "__main__":
    main()
