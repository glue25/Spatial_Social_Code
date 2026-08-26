#!/usr/bin/env python3
"""Re-score frozen CoMind predictions under the correct task coordinates.

This never mutates source predictions or legacy reports. Joint attention and
handover predictions were requested in individual 1408x1408 ego frames while
their public train boxes use a 1408x704 half-scale horizontal mosaic. SCOIA
already uses the individual helper frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("comparison", HERE / "compare_balanced_pilot.py")
CMP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CMP)
V2 = CMP.V2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    annotations = V2.load_annotations(args.annotations)
    raw = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    newest = {
        (str(row.get("condition")), str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row
        for row in raw
    }
    records = []
    for task, rid, sid in sorted({key[1:] for key in newest}):
        gt = annotations[task].get(rid, {}).get(sid)
        direct = newest.get(("direct", task, rid, sid))
        spatial = newest.get(("spatial_social", task, rid, sid))
        if not isinstance(gt, dict) or (task == "handover" and gt.get("skip")):
            continue
        if not direct or not spatial or direct.get("error") or spatial.get("error"):
            continue
        if not isinstance(direct.get("prediction"), dict) or not isinstance(spatial.get("prediction"), dict):
            continue
        predictions = {
            "direct": CMP.coordinate_correct_original(task, direct["prediction"]),
            "spatial_social": CMP.coordinate_correct_original(task, spatial["prediction"]),
        }
        records.append({
            "task": task, "recording_id": rid, "sample_id": sid,
            "scores": {name: V2.score_prediction(task, pred, gt) for name, pred in predictions.items()},
        })
    result = {
        "source_predictions": str(args.predictions),
        "protocol": "paired successful intersection; corrected task-specific coordinate spaces; frozen source predictions",
        "coordinate_protocol": {
            "joint_attention": "leader [x/2,y/2,w/2,h/2]; helper [x/2+704,y/2,w/2,h/2]",
            "scoia": "individual helper 1408x1408 unchanged",
            "handover": "half-scale mosaic side selected from each condition's predicted delivery_flow",
        },
        "paired_total": len(records), "variants": {}, "paired": {},
    }
    for variant in ("direct", "spatial_social"):
        result["variants"][variant] = {}
        for task in V2.TASK_FILES:
            scores = [row["scores"][variant] for row in records if row["task"] == task]
            result["variants"][variant][task] = V2.metric_summary(scores)
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        rows = [row for row in records if row["task"] == task]
        result["paired"][task] = {
            metric: V2.exact_mcnemar(
                [row["scores"]["spatial_social"][metric] for row in rows],
                [row["scores"]["direct"][metric] for row in rows],
            )
            for metric in metrics
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# CoMind coordinate-corrected re-score", "",
        f"Frozen source: `{args.predictions}`; paired n={len(records)}. Legacy reports are unchanged.", "",
        "| Metric | Direct | Spatial-Social | S-S only / Direct only | McNemar p |", "|---|---:|---:|---:|---:|",
    ]
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        for metric in metrics:
            direct = 100 * result["variants"]["direct"][task]["metrics"][metric]
            spatial = 100 * result["variants"]["spatial_social"][task]["metrics"][metric]
            paired = result["paired"][task][metric]
            lines.append(f"| {task}.{metric} | {direct:.2f}% | {spatial:.2f}% | {paired['candidate_only_correct']}/{paired['baseline_only_correct']} | {paired['mcnemar_exact_p']:.4g} |")
    lines.extend(["", "This is a train-only corrected re-score, not hidden-test performance."])
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "paired_total": len(records)}, indent=2))


if __name__ == "__main__":
    main()
