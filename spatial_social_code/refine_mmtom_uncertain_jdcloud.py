"""Refine only uncertain MMToM Spatial-Social predictions with denser evidence."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate_mmtom_jdcloud import (
    ROOT,
    image_parts,
    load_jsonl,
    summarize,
    write_outputs,
)
from evaluate_spatial_social import SPATIAL_PROMPT, execute_spatial_or_error, normalize_spatial_code
from mmtom_visual import load_questions, sample_rgb_frames
from social_code import answer_posterior_probs, compile_social_code

from egomas.src.jdcloud_gemini import (
    DEFAULT_CREDENTIALS_FILE,
    JDCloudGeminiClient,
    load_service_config,
)


SCHEMA_VERSION = 4
METHOD = "Uncertainty-refined Spatial-Social Code"
DEFAULT_PRIMARY = ROOT / "spatial_social_code" / "mmtom_robust_gemini3pro.jsonl"
DEFAULT_OUTPUT = ROOT / "spatial_social_code" / "mmtom_uncertainty_gemini3pro.jsonl"
CONFIDENCE_THRESHOLD = 0.70

SECOND_PASS_SUFFIX = """

This is an independent uncertainty-resolution pass over a denser frame sample.
Prioritize evidence about what each agent could observe, completed or incomplete
searches, object state changes, and action order. Do not infer or emit A/B.
"""


def confidence(record: dict) -> float:
    return max(float(record["prob_a"]), float(record["prob_b"]))


def refine_one(
    primary: dict,
    row: dict,
    frames_root: Path,
    dense_frames: int,
    client: JDCloudGeminiClient,
) -> dict:
    started = time.monotonic()
    paths, details = sample_rgb_frames(frames_root, row, num_frames=dense_frames)
    media = image_parts(paths)
    question = str(row["question"])
    spatial_prompt = SPATIAL_PROMPT.format(question=question) + SECOND_PASS_SUFFIX
    spatial_raw = client.generate_parts(
        media + [{"text": spatial_prompt}], max_output_tokens=4096
    )
    spatial_code = normalize_spatial_code(spatial_raw)
    spatial_trace = execute_spatial_or_error(spatial_code)

    def generate_dense(prompt: str, max_new_tokens: int = 512) -> str:
        return client.generate_parts(
            media + [{"text": prompt}],
            max_output_tokens=max(2048, max_new_tokens),
        )

    # This script reproduces the historical uncertainty-gated result.  New
    # evaluations use the clean state compiler and clean_bayesian reasoner.
    social = compile_social_code(
        "legacy_hybrid",
        question,
        generate=generate_dense,
        spatial_trace=spatial_trace,
        include_external_bayesian=False,
    )
    second = answer_posterior_probs(social.trace)
    if second is None:
        raise RuntimeError("second Spatial-Social trace contains no A/B posterior")
    fused_a = (float(primary["prob_a"]) + second["A"]) / 2.0
    fused_b = (float(primary["prob_b"]) + second["B"]) / 2.0
    prediction = "a" if fused_a >= fused_b else "b"
    answer = str(primary["answer"]).lower()
    result = dict(primary)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "uncertainty_triggered": True,
            "primary_prediction": primary["prediction"],
            "primary_correct": bool(primary["correct"]),
            "primary_prob_a": float(primary["prob_a"]),
            "primary_prob_b": float(primary["prob_b"]),
            "second_image_details": details,
            "second_spatial_raw": spatial_raw,
            "second_spatial_code": spatial_code,
            "second_spatial_trace": spatial_trace,
            "second_social_code": social.code,
            "second_social_trace": social.trace,
            "second_prob_a": second["A"],
            "second_prob_b": second["B"],
            "prediction": prediction,
            "correct": prediction == answer,
            "prob_a": fused_a,
            "prob_b": fused_b,
            "refinement_seconds": time.monotonic() - started,
        }
    )
    return result


def unchanged(primary: dict) -> dict:
    result = dict(primary)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "uncertainty_triggered": False,
            "primary_prediction": primary["prediction"],
            "primary_correct": bool(primary["correct"]),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "MMToM-QA-data" / "required_frames")
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dense-frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--uncertain-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--model")
    args = parser.parse_args()

    rows = load_questions(args.questions)
    primary = load_jsonl(args.primary)
    uncertain = [i for i in sorted(primary) if confidence(primary[i]) < CONFIDENCE_THRESHOLD]
    selected = uncertain if args.uncertain_only else sorted(primary)
    if args.limit is not None:
        selected = selected[: args.limit]
    existing = load_jsonl(args.output)
    for index in selected:
        if index not in uncertain and (index not in existing or "error" in existing[index]):
            existing[index] = unchanged(primary[index])
    write_outputs(args.output, existing, selected, load_service_config(args.credentials_file, args.model).model)
    pending = [
        i for i in selected
        if i in uncertain and (i not in existing or "error" in existing[i])
    ]
    config = load_service_config(args.credentials_file, args.model)
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
                refine_one, primary[index], rows[index], args.frames, args.dense_frames, client
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            try:
                existing[index] = future.result()
                state = "correct" if existing[index]["correct"] else "wrong"
            except Exception as error:
                existing[index] = {"index": index, "error": f"{type(error).__name__}: {error}"}
                state = "error"
            write_outputs(args.output, existing, selected, config.model)
            print(f"index={index} state={state} done={len(existing)}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
