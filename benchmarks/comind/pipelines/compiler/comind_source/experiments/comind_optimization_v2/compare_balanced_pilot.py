#!/usr/bin/env python3
"""Strict same-sample comparison for the balanced CoMind two-stage pilot."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("comind_v2", HERE / "comind_v2.py")
V2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V2)


def per_view_xywh_to_mosaic(value, role):
    box = V2.finite_box(value)
    if box is None:
        return None
    x, y, width, height = box
    return [x / 2 + (704 if role == "helper" else 0), y / 2, width / 2, height / 2]


def per_view_xyxy_to_mosaic(value, role):
    box = V2.convert_box(value, "xyxy")
    return per_view_xywh_to_mosaic(box, role)


def flow_side(value):
    token = V2.norm_text(value).replace(" ", "")
    return "helper" if token in {"rtl", "righttoleft", "helpertotheleader"} else "leader"


def coordinate_correct_original(task, prediction):
    result = copy.deepcopy(prediction)
    if task == "joint_attention":
        result["left_bbox"] = per_view_xywh_to_mosaic(result.get("left_bbox"), "leader")
        result["right_bbox"] = per_view_xywh_to_mosaic(result.get("right_bbox"), "helper")
    elif task == "handover":
        result["bbox"] = per_view_xywh_to_mosaic(result.get("bbox"), flow_side(result.get("delivery_flow")))
    return result


def optimize_spatial(task, spatial, two_stage):
    result = coordinate_correct_original(task, spatial)
    candidate = two_stage["prediction"]
    if task == "joint_attention":
        result["left_bbox"] = per_view_xyxy_to_mosaic(candidate.get("left_bbox_xyxy"), "leader")
        result["right_bbox"] = per_view_xyxy_to_mosaic(candidate.get("right_bbox_xyxy"), "helper")
        result["category_l1"] = candidate.get("object_identity")
    elif task == "scoia":
        result["bbox"] = V2.convert_box(candidate.get("bbox_xyxy"), "xyxy")
        result["action_noun_l1"] = candidate.get("object_identity")
    else:
        # Preserve the two-stage runner's label-free SAM-score view choice.
        # The original SSC delivery-flow prediction is retained as a separate
        # field, but is not allowed to retroactively choose the bbox proposal.
        role = candidate.get("selected_view")
        result["bbox"] = per_view_xyxy_to_mosaic(candidate.get("bbox_xyxy"), role)
        result["category_l1"] = candidate.get("object_identity")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("pipelines/comind_optimization_v2/gpu_grounding_manifest.jsonl"))
    parser.add_argument("--base", type=Path, default=Path("pipelines/spatial_social_optimization_v1/comind_codex_300_predictions.jsonl"))
    parser.add_argument("--two-stage", type=Path, default=Path("pipelines/comind_optimization_v2/two_stage_grounding_predictions.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--per-task", type=int, default=10, help="First N frozen rows per task; 0 uses every completed two-stage row")
    parser.add_argument("--baseline-label", default="GPT-5.6 baseline")
    parser.add_argument("--output", type=Path, default=Path("pipelines/comind_optimization_v2/balanced30_comparison.json"))
    parser.add_argument("--report", type=Path, default=Path("pipelines/comind_optimization_v2/balanced30_report.md"))
    args = parser.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = None
    if args.per_task > 0:
        selected = {
            (task, str(row["recording_id"]), str(row["sample_id"]))
            for task in V2.TASK_FILES
            for row in [item for item in manifest if item["task"] == task][: args.per_task]
        }
    base_rows = [json.loads(line) for line in args.base.read_text().splitlines() if line.strip()]
    base = {
        (str(row["condition"]), str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row
        for row in base_rows
        if not row.get("error") and isinstance(row.get("prediction"), dict)
    }
    two_rows = [json.loads(line) for line in args.two_stage.read_text().splitlines() if line.strip() and not json.loads(line).get("error")]
    two = {(str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row for row in two_rows}
    if selected is None:
        selected = set(two)
    annotations = V2.load_annotations(args.annotations)
    variants = ("direct", "spatial_social", "optimized_spatial_social")
    records = []
    for task, rid, sid in sorted(selected):
        direct_row = base.get(("direct", task, rid, sid))
        spatial_row = base.get(("spatial_social", task, rid, sid))
        candidate = two.get((task, rid, sid))
        gt = annotations[task].get(rid, {}).get(sid)
        if not direct_row or not spatial_row or not candidate or not isinstance(gt, dict):
            continue
        direct = coordinate_correct_original(task, direct_row["prediction"])
        spatial = coordinate_correct_original(task, spatial_row["prediction"])
        optimized = optimize_spatial(task, spatial_row["prediction"], candidate)
        predictions = {"direct": direct, "spatial_social": spatial, "optimized_spatial_social": optimized}
        records.append({
            "task": task, "recording_id": rid, "sample_id": sid,
            "scores": {name: V2.score_prediction(task, value, gt) for name, value in predictions.items()},
        })

    result = {
        "protocol": "strict successful-ID intersection; coordinate-corrected; no per-sample gold routing",
        "baseline_label": args.baseline_label,
        "coordinate_protocol": {
            "joint_attention": "leader/helper 1408 ego boxes -> half-scale 1408x704 mosaic",
            "scoia": "individual helper 1408x1408 unchanged",
            "handover": "half-scale mosaic side selected label-free by the two-stage SAM proposal score",
        },
        "samples": dict(Counter(row["task"] for row in records)),
        "variants": {}, "paired": {},
    }
    for variant in variants:
        result["variants"][variant] = {}
        for task in V2.TASK_FILES:
            scores = [row["scores"][variant] for row in records if row["task"] == task]
            result["variants"][variant][task] = V2.metric_summary(scores)
    for baseline in ("direct", "spatial_social"):
        result["paired"][f"optimized_vs_{baseline}"] = {}
        for task, metrics in V2.FIELD_TO_PREDICTION.items():
            rows = [row for row in records if row["task"] == task]
            result["paired"][f"optimized_vs_{baseline}"][task] = {}
            for metric in metrics:
                candidate = [row["scores"]["optimized_spatial_social"][metric] for row in rows]
                reference = [row["scores"][baseline][metric] for row in rows]
                result["paired"][f"optimized_vs_{baseline}"][task][metric] = V2.exact_mcnemar(candidate, reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    labels = {"direct": "Direct", "spatial_social": "原 SSC", "optimized_spatial_social": "优化 SSC"}
    lines = [
        "# CoMind two-stage comparison", "",
        f"Baseline: {args.baseline_label}. 严格共同 ID；优化 SSC 保留原 SSC 的其他字段，只替换 bbox 与 concrete identity。所有 bbox 均按任务官方坐标空间转换。", "",
        "| 任务字段 | Direct | 原 SSC | 优化 SSC | 优化 vs Direct (win/loss) | 优化 vs 原 SSC (win/loss) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        for metric in metrics:
            values = [100 * result["variants"][variant][task]["metrics"][metric] for variant in variants]
            pairs = []
            for baseline in ("direct", "spatial_social"):
                item = result["paired"][f"optimized_vs_{baseline}"][task][metric]
                pairs.append(f"{item['candidate_only_correct']}/{item['baseline_only_correct']}")
            lines.append(f"| {task}.{metric} | {values[0]:.0f}% | {values[1]:.0f}% | {values[2]:.0f}% | {pairs[0]} | {pairs[1]} |")
    lines.extend(["", f"样本数：{result['samples']}。这是 train-only 机制实验，不是独立 held-out 或官方 test。JA 缺失 right gold 仍按现有 scorer 计为未命中。"])
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "samples": result["samples"]}, indent=2))


if __name__ == "__main__":
    main()
