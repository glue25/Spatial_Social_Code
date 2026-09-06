#!/usr/bin/env python3
"""Prepare answer-blind, two-reviewer state audit packets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.priority_ssc_v1.common import atomic_json, atomic_jsonl, forbidden_target_keys, read_jsonl, source_traces, trace_records


FIELDS = [
    "sample_id",
    "claim_id",
    "state_correct",
    "citation_support",
    "corrected_claim_json",
    "notes",
]


def balanced_unique(rows: list[dict[str, Any]], dataset: str, total: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("question_type") if dataset == "mmtom" else row.get("category"))
        groups[key].append(row)
    for group in groups.values():
        rng.shuffle(group)
    cursors = {key: 0 for key in groups}
    used_episodes: set[str] = set()
    used_clips: set[str] = set()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < total:
        progressed = False
        for key in sorted(groups):
            group = groups[key]
            while cursors[key] < len(group):
                row = group[cursors[key]]
                cursors[key] += 1
                if dataset == "mmtom":
                    episode = str(row["episode"])
                    if episode in used_episodes:
                        continue
                    used_episodes.add(episode)
                else:
                    clips = {str(clip["id"]) for clip in row.get("video_clips", [])}
                    if not clips or clips & used_clips:
                        continue
                    used_clips.update(clips)
                chosen.append(row)
                progressed = True
                break
            if len(chosen) == total:
                break
        if not progressed:
            break
    if len(chosen) != total:
        raise ValueError(f"could select only {len(chosen)} independent {dataset} units; requested {total}")
    return chosen


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmtom-source", type=Path, default=ROOT / "spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl")
    parser.add_argument("--ma-source", type=Path, default=ROOT / "MA-EgoQA/data/baselines/video_balanced_role_full.jsonl")
    parser.add_argument("--mmtom-questions", type=Path, default=ROOT / "MMToM-QA/Benchmark/questions.json")
    parser.add_argument("--ma-questions", type=Path, default=ROOT / "MA-EgoQA/data/MA-EgoQA.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workdirs/priority_ssc_v1/human_audit")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    mmtom_questions = [json.loads(line) for line in args.mmtom_questions.open(encoding="utf-8") if line.strip()]
    ma_questions = json.loads(args.ma_questions.read_text(encoding="utf-8"))
    packets: list[dict[str, Any]] = []
    annotations: list[dict[str, str]] = []
    manifest_samples: list[dict[str, Any]] = []
    for dataset, source_path, questions in (
        ("mmtom", args.mmtom_source, mmtom_questions),
        ("maegoqa", args.ma_source, ma_questions),
    ):
        chosen = balanced_unique(read_jsonl(source_path), dataset, args.per_dataset, args.seed)
        for row in chosen:
            index = int(row["index"])
            sample_id = f"{dataset}:{index}"
            physical, social = source_traces(row)
            physical_records = trace_records(physical)
            social_records = trace_records(social, social=True)
            claims = []
            for layer, records in (("physical", physical_records), ("social", social_records)):
                for offset, claim in enumerate(records, start=1):
                    claim_id = f"{sample_id}:{'P' if layer == 'physical' else 'S'}{offset:03d}"
                    claims.append({"claim_id": claim_id, "layer": layer, "claim": claim})
                    annotations.append({name: "" for name in FIELDS} | {"sample_id": sample_id, "claim_id": claim_id})
            question_row = questions[index]
            evidence = row.get("image_details", []) if dataset == "mmtom" else row.get("video_clips", [])
            packet = {
                "schema_version": 1,
                "sample_id": sample_id,
                "dataset": dataset,
                "index": index,
                "unit_id": str(row.get("episode", index)),
                "question": question_row["question"],
                "options": question_row.get("options"),
                "evidence": evidence,
                "claims": claims,
                "instructions": {
                    "physical": "For P claims, state_correct means the physical fact/event is supported.",
                    "social": "For S claims, state_correct means the named agent truly saw/heard/knew the stated content.",
                    "citation": "citation_support means the cited frame/clip/span actually supports the assertion.",
                    "labels": "Use yes, no, uncertain, or na. Reviewers work independently.",
                },
            }
            forbidden = forbidden_target_keys(packet)
            if forbidden:
                raise RuntimeError(f"target answer leaked into audit packet {sample_id}: {forbidden}")
            packets.append(packet)
            manifest_samples.append({"sample_id": sample_id, "dataset": dataset, "index": index, "unit_id": packet["unit_id"], "claims": len(claims)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "packets.jsonl", packets)
    for reviewer in ("reviewer_a", "reviewer_b"):
        write_csv(args.output_dir / f"{reviewer}.csv", annotations, FIELDS)
    write_csv(
        args.output_dir / "adjudication.csv",
        [row | {"final_keep": "", "adjudicated_claim_json": "", "adjudication_notes": ""} for row in annotations],
        FIELDS + ["final_keep", "adjudicated_claim_json", "adjudication_notes"],
    )
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "awaiting_two_independent_reviewers",
            "seed": args.seed,
            "samples_per_dataset": args.per_dataset,
            "independence": {"mmtom": "one QA per unique episode", "maegoqa": "no exact source clip reused across selected QA"},
            "target_answers_present": False,
            "target_directed_social_records_removed": True,
            "samples": manifest_samples,
        },
    )
    print(f"prepared samples={len(packets)} claims={len(annotations)} at {args.output_dir}")


if __name__ == "__main__":
    main()
