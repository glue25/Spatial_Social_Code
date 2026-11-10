"""Paired MMToM VLM-only versus VLM+SAM Spatial grounding ablation."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import mimetypes
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "third_party/autotom/model", ROOT / "benchmarks/ma_egoqa", ROOT / "spatial_social_code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egomas.src.jdcloud_gemini import (  # noqa: E402
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)
from mmtom_visual import load_questions, sample_rgb_frames  # noqa: E402
from runtime import UnsafeProgram, execute_spatial  # noqa: E402
from social_code import answer_posterior_probs, compile_social_code  # noqa: E402


DEFAULT_BASELINE = ROOT / "spatial_social_code/mmtom_robust_gemini3pro.jsonl"
DEFAULT_OUTPUT = ROOT / "workdirs/iclr_ablations_v1/mmtom_sam_ablation.jsonl"
DEFAULT_CACHE = ROOT / "workdirs/iclr_ablations_v1/mmtom_sam_cache"
OBJECT_TERMS = (
    "bottle of wine",
    "condiment bottle",
    "remote control",
    "bag of chips",
    "wine glass",
    "water glass",
    "dish bowl",
    "dishbowl",
    "cupcake",
    "salmon",
    "plate",
    "apple",
    "book",
)
CONTAINER_PATTERNS = (
    r"[1-4](?:st|nd|rd|th) kitchen cabinet",
    r"bathroom cabinet",
    r"kitchen cabinet",
    r"coffee table",
    r"microwave",
    r"fridge",
    r"stove",
    r"cabinet",
)
COLORS = ("#ff1744", "#00c853", "#2979ff", "#ff9100")


SPATIAL_PROMPT = """You are the Spatial Code compiler in a video understanding system.
Transcribe physically grounded facts from the initial-state text and ordered RGB frames into an executable program. Use both answer choices only to identify the concrete facts and observations that distinguish them. Seek evidence for each alternative without selecting, copying, or encoding an answer. Identify relevant objects and locations, their stated initial locations, and observed actions. Keep world state separate from visibility. Never infer beliefs, goals, intentions, or an A/B answer. Missing records mean unknown, not false.
The runtime object S already exists. Use ONLY literal calls in these forms:
S.fact(t, relation, subject, object, source="text", confidence=1.0)
S.event(t, actor, action, target, source="text", confidence=1.0)
S.visible(t, observer, object, value, source="video", confidence=0.8)
S.unknown(t, relation, subject, object)
No assignments, variables, imports, loops, markdown, comments, or repeated calls. Output 4-12 complete calls and nothing else.

MMToM problem:
{question}
"""


SAM_SPATIAL_SUFFIX = """

SAM grounding aid:
The ordered frames contain colored boxes from a label-free, text-prompted SAM
proposal model. Labels were extracted symmetrically from all question choices.
A box is only a localization proposal, not proof of identity, containment,
visibility, belief, or intent. Verify every proposal against the pixels. Use it
only to improve directly grounded Spatial records and preserve unknowns.
Proposal terms: {terms}
"""


def image_parts(paths: Iterable[str | Path]) -> list[dict]:
    parts = []
    for path_like in paths:
        path = Path(path_like)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        parts.append(
            {
                "inline_data": {
                    "mimeType": mime,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            }
        )
    return parts


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    return {
        int(row["index"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def select_indices(rows: list[dict], samples_per_type: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_type"])].append(index)
    return [
        candidates[offset]
        for offset in range(samples_per_type)
        for _, candidates in sorted(grouped.items())
        if offset < len(candidates)
    ]


def normalize_spatial_code(text: str, max_lines: int = 16) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        try:
            parsed = ast.parse(line)
        except SyntaxError:
            continue
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
            continue
        call = parsed.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "S":
            continue
        if call.func.attr not in {"fact", "event", "visible", "unknown"}:
            continue
        try:
            execute_spatial(line)
        except UnsafeProgram:
            continue
        kept.append(line)
        seen.add(line)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def execute_spatial_or_error(code: str) -> str:
    try:
        return execute_spatial(code)
    except (UnsafeProgram, NameError, AttributeError, TypeError, ValueError) as error:
        return json.dumps({"execution_error": str(error)}, ensure_ascii=False)


def question_tail(question: str) -> str:
    return question.rsplit("Question:", 1)[-1].lower()


def extract_terms(question: str) -> list[str]:
    tail = question_tail(question)
    terms: list[str] = []
    normalized = tail.replace("dishbowl", "dish bowl")
    for term in OBJECT_TERMS:
        canonical = term.replace("dishbowl", "dish bowl")
        if canonical in normalized and canonical not in terms:
            terms.append(canonical)
    for pattern in CONTAINER_PATTERNS:
        for match in re.findall(pattern, tail):
            if match not in terms and not any(
                match in existing and match != existing for existing in terms
            ):
                terms.append(match)
    return terms[:4]


def encode_png(path: Path) -> str:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sam_request(image_path: Path, term: str, service_url: str) -> list[dict]:
    payload = json.dumps(
        {"image_base64": encode_png(image_path), "text_prompt": term}
    ).encode("utf-8")
    request = urllib.request.Request(
        service_url.rstrip("/") + "/segment",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    proposals = []
    for item in result.get("results", []):
        box = item.get("box") if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4:
            proposals.append(
                {
                    "box_xyxy": [float(value) for value in box],
                    "score": float(item.get("score", 0.0)),
                    "label": term,
                }
            )
    return sorted(proposals, key=lambda item: item["score"], reverse=True)


def cache_path(cache_root: Path, image_path: Path, term: str) -> Path:
    key = hashlib.sha256(f"{image_path.resolve()}\0{term}".encode()).hexdigest()
    return cache_root / "proposals" / f"{key}.json"


def cached_sam(
    cache_root: Path, image_path: Path, term: str, service_url: str
) -> list[dict]:
    path = cache_path(cache_root, image_path, term)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    proposals = sam_request(image_path, term, service_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(proposals) + "\n", encoding="utf-8")
    temporary.replace(path)
    return proposals


def render_overlay(
    output: Path, image_path: Path, terms: list[str], proposals: dict[str, list[dict]]
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for index, term in enumerate(terms):
        if not proposals.get(term):
            continue
        item = proposals[term][0]
        x1, y1, x2, y2 = item["box_xyxy"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            continue
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1 + 3, max(0, y1 - 14)), f"{term} {item['score']:.2f}", fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def prepare_overlays(
    index: int,
    row: dict,
    frames_root: Path,
    num_frames: int,
    cache_root: Path,
    service_url: str,
) -> tuple[list[Path], list[str], dict[str, Any]]:
    frame_paths, details = sample_rgb_frames(frames_root, row, num_frames=num_frames)
    frames = [Path(path) for path in frame_paths]
    terms = extract_terms(str(row["question"]))
    if not terms:
        raise ValueError(f"no symmetric object/container terms for index {index}")
    manifest: dict[str, Any] = {"terms": terms, "frames": []}
    overlays: list[Path] = []
    for frame_number, frame in enumerate(frames):
        proposals = {
            term: cached_sam(cache_root, frame, term, service_url) for term in terms
        }
        overlay = cache_root / "overlays" / str(index) / f"frame_{frame_number:02d}.png"
        render_overlay(overlay, frame, terms, proposals)
        overlays.append(overlay)
        manifest["frames"].append(
            {
                "source": str(frame),
                "overlay": str(overlay),
                "proposals": proposals,
            }
        )
    manifest["image_details"] = details
    return overlays, terms, manifest


def process_one(
    index: int,
    row: dict,
    baseline: dict,
    frames_root: Path,
    num_frames: int,
    cache_root: Path,
    service_url: str,
    client: JDCloudGeminiClient,
) -> dict:
    started = time.monotonic()
    overlays, terms, sam_manifest = prepare_overlays(
        index, row, frames_root, num_frames, cache_root, service_url
    )
    overlay_media = image_parts(overlays)
    raw_frames, _ = sample_rgb_frames(frames_root, row, num_frames=num_frames)
    raw_media = image_parts(raw_frames)
    question = str(row["question"])
    prompt = SPATIAL_PROMPT.format(question=question) + SAM_SPATIAL_SUFFIX.format(
        terms=", ".join(terms)
    )
    spatial_raw = client.generate_parts(
        overlay_media + [{"text": prompt}], max_output_tokens=4096
    )
    spatial_code = normalize_spatial_code(spatial_raw)
    spatial_trace = execute_spatial_or_error(spatial_code)

    def generate_with_raw_frames(prompt_text: str, max_new_tokens: int = 512) -> str:
        return client.generate_parts(
            raw_media + [{"text": prompt_text}],
            max_output_tokens=max(2048, max_new_tokens),
        )

    social = compile_social_code(
        "hybrid",
        question,
        generate=generate_with_raw_frames,
        spatial_trace=spatial_trace,
        include_external_bayesian=False,
    )
    posterior = answer_posterior_probs(social.trace)
    if posterior is None:
        raise RuntimeError("SAM-augmented trace contains no A/B posterior")
    prediction = "a" if posterior["A"] >= posterior["B"] else "b"
    answer = str(row["answer"]).lower()
    term_coverage = {
        term: term.lower() in spatial_trace.lower().replace("dishbowl", "dish bowl")
        for term in terms
    }
    return {
        "schema_version": 1,
        "method": "MMToM VLM+SAM Spatial grounding ablation",
        "model": client.config.model,
        "index": index,
        "episode": row["episode"],
        "question_type": str(row["question_type"]),
        "answer": answer,
        "vlm_only_prediction": baseline["prediction"],
        "vlm_only_correct": bool(baseline["correct"]),
        "sam_terms": terms,
        "sam_manifest": sam_manifest,
        "sam_spatial_raw": spatial_raw,
        "sam_spatial_code": spatial_code,
        "sam_spatial_trace": spatial_trace,
        "sam_term_coverage": term_coverage,
        "sam_social_code": social.code,
        "sam_social_trace": social.trace,
        "sam_prediction": prediction,
        "sam_correct": prediction == answer,
        "prob_a": posterior["A"],
        "prob_b": posterior["B"],
        "elapsed_seconds": time.monotonic() - started,
    }


def exact_mcnemar(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(records: Iterable[dict], selected: list[int], model: str) -> dict:
    all_rows = list(records)
    rows = [row for row in all_rows if "sam_correct" in row]
    baseline_only = sum(row["vlm_only_correct"] and not row["sam_correct"] for row in rows)
    sam_only = sum(not row["vlm_only_correct"] and row["sam_correct"] for row in rows)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)

    def accuracy(group: list[dict], field: str) -> float | None:
        return sum(bool(row[field]) for row in group) / len(group) if group else None

    coverage_values = [
        value for row in rows for value in row.get("sam_term_coverage", {}).values()
    ]
    return {
        "schema_version": 1,
        "benchmark": "MMToM",
        "protocol": "same frozen RGB frames and VLM; SAM overlays visible only to Spatial compiler",
        "model": model,
        "selected": len(selected),
        "completed": len(all_rows),
        "successful": len(rows),
        "errors": len(all_rows) - len(rows),
        "vlm_only_accuracy": accuracy(rows, "vlm_only_correct"),
        "vlm_sam_accuracy": accuracy(rows, "sam_correct"),
        "contingency": {
            "vlm_only_only_correct": baseline_only,
            "vlm_sam_only_correct": sam_only,
            "net_sam_wins": sam_only - baseline_only,
            "mcnemar_exact_p": exact_mcnemar(baseline_only, sam_only),
        },
        "sam_target_term_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        "by_question_type": {
            name: {
                "total": len(group),
                "vlm_only_accuracy": accuracy(group, "vlm_only_correct"),
                "vlm_sam_accuracy": accuracy(group, "sam_correct"),
            }
            for name, group in sorted(by_type.items())
        },
        "label_access_in_inference": False,
        "sam_terms_from_all_choices_symmetrically": True,
    }


def write_outputs(
    path: Path, records: dict[int, dict], selected: list[int], model: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    temporary.replace(path)
    summary = summarize((records[index] for index in selected if index in records), selected, model)
    summary_path = path.with_name(path.stem + "_summary.json")
    temporary_summary = summary_path.with_suffix(".tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA/Benchmark/questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data/required_frames")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sam-url", default="http://127.0.0.1:8114")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--samples-per-type", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_questions(args.questions)
    baseline = load_jsonl(args.baseline)
    selected = args.indices or select_indices(rows, args.samples_per_type)
    missing = [index for index in selected if index not in baseline]
    if missing:
        raise ValueError(f"missing VLM-only rows: {missing}")
    plan = {
        "selected": selected,
        "by_type": dict(Counter(str(rows[index]["question_type"]) for index in selected)),
        "terms": {str(index): extract_terms(rows[index]["question"]) for index in selected},
        "same_frames": True,
        "sam_visible_only_to_spatial_compiler": True,
        "label_access": False,
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False))
        return

    config = load_service_config(args.credentials_file, args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)
    existing = load_jsonl(args.output)
    pending = [
        index for index in selected if index not in existing or "sam_correct" not in existing[index]
    ]
    print(
        f"model={config.model!r} selected={len(selected)} "
        f"resumed={len(selected)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                rows[index],
                baseline[index],
                args.frames,
                args.num_frames,
                args.cache_root,
                args.sam_url,
                client,
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                state = "ok"
            except Exception as error:
                record = {
                    "schema_version": 1,
                    "index": index,
                    "episode": rows[index]["episode"],
                    "question_type": str(rows[index]["question_type"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                state = "error"
            existing[index] = record
            write_outputs(args.output, existing, selected, config.model)
            print(f"index={index} state={state}", flush=True)


if __name__ == "__main__":
    main()
