#!/usr/bin/env python3
"""Evaluate already-generated GPU grounding candidates on labeled train only."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("comind_v2", HERE / "comind_v2.py")
V2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V2)


def individual_xyxy_to_joint_mosaic_xywh(value, role: str):
    """Map a 1408-square ego frame box to the official 1408x704 mosaic.

    CoMind joint-attention annotations concatenate half-scale leader and
    helper frames horizontally.  They are not coordinates in either saved
    1408x1408 ego frame.
    """
    box = V2.convert_box(value, "xyxy")
    if box is None:
        return None
    x, y, width, height = box
    return [x / 2 + (704 if role == "helper" else 0), y / 2, width / 2, height / 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--output", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_evaluation.json"))
    args = parser.parse_args()
    annotations = V2.load_annotations(args.annotations)
    rows = V2.load_jsonl(args.predictions)
    scored = defaultdict(list)
    for row in rows:
        # Resumable runners retain failed attempts for diagnosis. They are not
        # predictions and must not make a partial pilot impossible to score.
        if row.get("error") or not isinstance(row.get("prediction"), dict):
            continue
        task, rid, sid = row["task"], str(row["recording_id"]), str(row["sample_id"])
        gt = annotations[task].get(rid, {}).get(sid)
        if not isinstance(gt, dict) or (task == "handover" and gt.get("skip")):
            continue
        pred = row["prediction"]
        converted = {"category_l1": pred.get("object_identity")}
        if task == "joint_attention":
            converted["left_bbox"] = individual_xyxy_to_joint_mosaic_xywh(
                pred.get("left_bbox_xyxy"), "leader"
            )
            converted["right_bbox"] = individual_xyxy_to_joint_mosaic_xywh(
                pred.get("right_bbox_xyxy"), "helper"
            )
            metrics = ("left_iou_at_0_5", "right_iou_at_0_5", "category_l1")
        else:
            if task == "handover" and pred.get("selected_view") in {"leader", "helper"}:
                converted["bbox"] = individual_xyxy_to_joint_mosaic_xywh(
                    pred.get("bbox_xyxy"), pred["selected_view"]
                )
            else:
                # SCOIA labels and predictions both use the individual helper
                # 1408x1408 view; no half-scale mosaic transform applies.
                converted["bbox"] = V2.convert_box(pred.get("bbox_xyxy"), "xyxy")
            metrics = ("iou_at_0_5", "category_l1" if task == "handover" else "action_noun_l1")
            if task == "scoia":
                converted["action_noun_l1"] = pred.get("object_identity")
        scored[task].append({metric: V2.field_correct(task, metric, converted.get(V2.FIELD_TO_PREDICTION[task][metric]), gt) for metric in metrics})
    report = {"boundary": "post-generation scoring on labeled train only; no hidden test", "tasks": {task: V2.metric_summary(items) for task, items in scored.items()}}
    V2.write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
