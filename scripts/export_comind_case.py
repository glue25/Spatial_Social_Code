#!/usr/bin/env python3
"""Export one paired CoMind case with exact predictions, code, labels, and frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_social_code.comind import box_iou, load_annotations, score_sample
from spatial_social_code.run_comind_codex_pilot import evidence_prompt


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--image-indices", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--image-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in manifest_rows
        if row["task"] == args.task
        and row["recording_id"] == args.recording_id
        and row["sample_id"] == args.sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row, found {len(matches)}")
    manifest = matches[0]

    ground_truth = load_annotations(args.annotation_root, args.task).get(
        args.recording_id, {}
    ).get(args.sample_id)
    if not isinstance(ground_truth, dict):
        raise ValueError("Labeled-train ground truth not found")

    stem = f"{args.task}_{args.recording_id}_{args.sample_id}"
    source_files = {
        phase: args.raw_root / f"{stem}_{phase}.json"
        for phase in ("direct", "spatial", "social", "answer")
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing paired outputs: {missing}")
    direct = load_json(source_files["direct"])
    spatial = load_json(source_files["spatial"])
    social = load_json(source_files["social"])
    spatial_social = load_json(source_files["answer"])
    direct_scores = score_sample(args.task, ground_truth, direct)
    spatial_social_scores = score_sample(args.task, ground_truth, spatial_social)
    positive_flips = [
        metric
        for metric in direct_scores
        if not direct_scores[metric] and spatial_social_scores[metric]
    ]
    regressions = [
        metric
        for metric in direct_scores
        if direct_scores[metric] and not spatial_social_scores[metric]
    ]
    if not positive_flips:
        raise ValueError("Requested case is not a Direct-to-SSC positive flip")

    args.image_output_dir.mkdir(parents=True, exist_ok=True)
    selected_images = []
    for index in args.image_indices:
        image = manifest["images"][index]
        source = Path(image["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        approximate_frame = round(float(image["time"]) * 30)
        destination = args.image_output_dir / (
            f"comind_{args.task}_{args.sample_id}_{image['role']}_"
            f"f{approximate_frame:06d}_{image['kind']}{source.suffix.lower()}"
        )
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Copied image hash mismatch: {source}")
        selected_images.append(
            {
                "role": image["role"],
                "kind": image["kind"],
                "time_seconds": image["time"],
                "approximate_frame_at_30fps": approximate_frame,
                "public_path": repo_path(destination),
                "sha256": sha256(destination),
                "exact_copy_of_source_frame": True,
            }
        )

    annotation_file = {
        "joint_attention": "dataset_joint_attention_consolidated.json",
        "scoia": "dataset_scoia_consolidated.json",
        "handover": "dataset_handover_consolidated.json",
    }[args.task]
    artifact = {
        "schema_version": "comind_paired_case_v1",
        "case_scope": "illustrative labeled-train per-field positive flip",
        "model": args.model,
        "task": args.task,
        "recording_id": args.recording_id,
        "sample_id": args.sample_id,
        "prediction_time_seconds": manifest["prediction_time"],
        "query": {
            "description": "Predict the upcoming SCOIA object interaction at the prediction timestamp.",
            "shared_evidence_prompt": evidence_prompt(manifest),
            "fields": ["bbox", "cue_type", "action_verb", "action_noun_l1"],
            "transcript_10s": manifest["transcript_10s"],
            "same_ordered_frames_for_direct_and_spatial_social": True,
        },
        "direct_answer": direct,
        "spatial_social_answer": spatial_social,
        "ground_truth": ground_truth,
        "spatial_code": spatial["spatial_code"],
        "social_code": social["social_code"],
        "score_audit": {
            "direct": direct_scores,
            "spatial_social": spatial_social_scores,
            "positive_flips": positive_flips,
            "regressions": regressions,
            "bbox_iou": {
                "direct": box_iou(direct.get("bbox"), ground_truth.get("bbox")),
                "spatial_social": box_iou(
                    spatial_social.get("bbox"), ground_truth.get("bbox")
                ),
                "threshold": 0.5,
            },
        },
        "selected_original_frames": selected_images,
        "all_input_frames": [
            {
                "role": image["role"],
                "kind": image["kind"],
                "time_seconds": image["time"],
                "source_sha256": sha256(Path(image["path"])),
            }
            for image in manifest["images"]
        ],
        "protocol_boundary": {
            "split": "labeled CoMind train subset",
            "label_used_during_generation": False,
            "label_used_only_for_post_generation_scoring_and_case_selection": True,
            "aggregate_claim": "The 300-example paired experiment does not show an overall SSC improvement; this is an illustrative positive-flip case only.",
        },
        "source_provenance": {
            "manifest": {
                "path": repo_path(args.manifest),
                "sha256": sha256(args.manifest),
            },
            "annotation": {
                "path": repo_path(args.annotation_root / annotation_file),
                "sha256": sha256(args.annotation_root / annotation_file),
            },
            "paired_outputs": {
                phase: {"path": repo_path(path), "sha256": sha256(path)}
                for phase, path in source_files.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
