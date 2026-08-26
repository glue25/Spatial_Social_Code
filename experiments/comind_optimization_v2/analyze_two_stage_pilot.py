#!/usr/bin/env python3
"""Create numerical and visual diagnostics for a two-stage JA pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("comind_v2", HERE / "comind_v2.py")
V2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V2)


def xyxy_to_xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def to_mosaic(box, role):
    value = xyxy_to_xywh(box)
    return [value[0] / 2 + (704 if role == "helper" else 0), value[1] / 2, value[2] / 2, value[3] / 2]


def gold_to_individual(box, role):
    offset = 704 if role == "helper" else 0
    return [(box[0] - offset) * 2, box[1] * 2, (box[0] + box[2] - offset) * 2, (box[1] + box[3]) * 2]


def geometry(pred, gold):
    if gold is None:
        return None
    pc = (pred[0] + pred[2] / 2, pred[1] + pred[3] / 2)
    gc = (gold[0] + gold[2] / 2, gold[1] + gold[3] / 2)
    return {
        "iou": V2.iou(pred, gold),
        "center_error_px": math.dist(pc, gc),
        "area_ratio": (pred[2] * pred[3]) / (gold[2] * gold[3]),
        "hit_at_0_5": V2.iou(pred, gold) > 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("workdirs/comind_optimization_v2/two_stage_grounding_predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--output-root", type=Path, default=Path("workdirs/comind_optimization_v2/two_stage_diagnostics"))
    args = parser.parse_args()
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    manifests = {(str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row for row in map(json.loads, args.manifest.open())}
    annotations = V2.load_annotations(args.annotations)["joint_attention"]
    rows = []
    thumbs = []
    for row in predictions:
        if row.get("error") or row["task"] != "joint_attention":
            continue
        key = (row["task"], str(row["recording_id"]), str(row["sample_id"]))
        manifest = manifests[key]
        gt = annotations[key[1]][key[2]]
        gold_boxes = gt.get("bounding_boxes") or []
        pred = row["prediction"]
        frame_paths = {image["role"]: Path(image["path"]) for image in manifest["images"] if image["kind"] == "prediction"}
        details = {"sample_id": key[2], "gold_identity": gt.get("object_category_level_1"), "predicted_identity": pred.get("object_identity"), "gold_box_count": len(gold_boxes), "views": {}}
        view_images = []
        for index, (role, field) in enumerate((("leader", "left_bbox_xyxy"), ("helper", "right_bbox_xyxy"))):
            pred_xyxy = pred[field]
            pred_mosaic = to_mosaic(pred_xyxy, role)
            gold = gold_boxes[index] if len(gold_boxes) > index else None
            details["views"][role] = {"pred_mosaic_xywh": pred_mosaic, "gold_mosaic_xywh": gold, "geometry": geometry(pred_mosaic, gold)}
            image = Image.open(frame_paths[role]).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle(pred_xyxy, outline="red", width=8)
            if gold is not None:
                draw.rectangle(gold_to_individual(gold, role), outline="lime", width=8)
            image.thumbnail((352, 352))
            view_images.append(image.copy())
        strip = Image.new("RGB", (704, 382), "white")
        strip.paste(view_images[0], (0, 30)); strip.paste(view_images[1], (352, 30))
        ImageDraw.Draw(strip).text((5, 5), f"{key[2]} gold={details['gold_identity']} pred={details['predicted_identity']} | green=GT red=SAM", fill="black")
        thumbs.append(strip)
        rows.append(details)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contact = Image.new("RGB", (704, 382 * len(thumbs)), "white")
    for index, image in enumerate(thumbs):
        contact.paste(image, (0, index * 382))
    contact.save(args.output_root / "joint10_contact_sheet.jpg", quality=90)
    summary = {
        "coordinate_protocol": "saved ego frames are 1408x1408; official JA labels are a 1408x704 mosaic: leader/2 on left, helper/2 plus x=704 on right",
        "samples": len(rows),
        "identity_correct": sum(V2.single_match(row["predicted_identity"], row["gold_identity"]) for row in rows),
        "left_hits": sum(bool(row["views"]["leader"]["geometry"] and row["views"]["leader"]["geometry"]["hit_at_0_5"]) for row in rows),
        "right_hits": sum(bool(row["views"]["helper"]["geometry"] and row["views"]["helper"]["geometry"]["hit_at_0_5"]) for row in rows),
        "right_labeled": sum(row["gold_box_count"] > 1 for row in rows),
        "rows": rows,
    }
    (args.output_root / "joint10_geometry.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
