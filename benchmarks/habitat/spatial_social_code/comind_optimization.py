"""Leakage-safe CoMind bbox sanitation and recording-level field fusion.

This module treats the frozen predictions as immutable inputs.  In ``auto``
mode an ambiguous four-number list remains xywh; it is interpreted as xyxy
only when xywh is geometrically impossible but xyxy is valid.  No annotation
is consulted by normalization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spatial_social_code.comind import TASK_FILES, load_annotations, score_sample


FRAME_SIZE = (1408.0, 1408.0)
BBOX_FIELDS = {
    "joint_attention": ("left_bbox", "right_bbox"),
    "scoia": ("bbox",),
    "handover": ("bbox",),
}
METRIC_FIELDS = {
    "joint_attention": {
        "left_iou_at_0_5": "left_bbox",
        "right_iou_at_0_5": "right_bbox",
        "cue_type": "cue_type",
        "category_l1": "category_l1",
    },
    "scoia": {
        "iou_at_0_5": "bbox",
        "cue_type": "cue_type",
        "action_verb": "action_verb",
        "action_noun_l1": "action_noun_l1",
    },
    "handover": {
        "iou_at_0_5": "bbox",
        "delivery_flow": "delivery_flow",
        "initiator": "initiator",
        "initiation_type": "initiation_type",
        "category_l1": "category_l1",
        "tth_within_0_25s": "time_to_handover",
    },
}


def _finite_box(value: Any) -> tuple[list[float], str] | None:
    convention = "list"
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value[key] for key in ("x1", "y1", "x2", "y2")]
            convention = "explicit_xyxy"
        elif all(key in value for key in ("x", "y", "w", "h")):
            value = [value[key] for key in ("x", "y", "w", "h")]
            convention = "explicit_xywh"
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    return box, convention


def sanitize_bbox(
    value: Any,
    *,
    frame_size: tuple[float, float] = FRAME_SIZE,
    source_size: tuple[float, float] | None = None,
    convention: str = "auto",
) -> tuple[list[float] | None, str]:
    """Return clipped xywh pixels and an annotation-independent action label.

    Supported coordinate spaces are explicit source pixels and normalized
    [0,1] coordinates.  ``auto`` never changes an ambiguous valid xywh list.
    """
    parsed = _finite_box(value)
    if parsed is None:
        return None, "malformed"
    box, declared = parsed
    width, height = frame_size
    if source_size is not None:
        source_width, source_height = source_size
        if source_width <= 0 or source_height <= 0:
            return None, "invalid_source_size"
        box[0] *= width / source_width
        box[2] *= width / source_width
        box[1] *= height / source_height
        box[3] *= height / source_height
        scaling = "scaled_source_pixels"
    elif all(0.0 <= item <= 1.0 for item in box) and max(box) > 0:
        box[0] *= width
        box[2] *= width
        box[1] *= height
        box[3] *= height
        scaling = "scaled_normalized"
    else:
        scaling = "native_pixels"

    x, y, third, fourth = box
    xywh_valid = x >= 0 and y >= 0 and third > 0 and fourth > 0 and x < width and y < height and x + third <= width and y + fourth <= height
    xyxy_valid = 0 <= x < third <= width and 0 <= y < fourth <= height
    interpret_xyxy = declared == "explicit_xyxy" or convention == "xyxy"
    if convention == "auto" and declared == "list" and not xywh_valid and xyxy_valid:
        interpret_xyxy = True
    if declared == "explicit_xywh":
        interpret_xyxy = False
    if interpret_xyxy:
        third, fourth = third - x, fourth - y
        action = "xyxy_to_xywh"
    else:
        action = "kept_xywh"

    x1 = min(width, max(0.0, x))
    y1 = min(height, max(0.0, y))
    x2 = min(width, max(0.0, x + third))
    y2 = min(height, max(0.0, y + fourth))
    if x2 <= x1 or y2 <= y1:
        return None, "degenerate_after_clip"
    result = [x1, y1, x2 - x1, y2 - y1]
    clipped = any(abs(a - b) > 1e-9 for a, b in zip(result, [x, y, third, fourth]))
    suffix = "_clipped" if clipped else ""
    return result, f"{scaling}:{action}{suffix}"


def normalize_prediction(task: str, prediction: dict, convention: str = "auto") -> tuple[dict, Counter]:
    result = copy.deepcopy(prediction)
    actions: Counter = Counter()
    for field in BBOX_FIELDS[task]:
        normalized, action = sanitize_bbox(result.get(field), convention=convention)
        actions[action] += 1
        result[field] = normalized
    return result, actions


def _paired_rows(rows: list[dict]) -> list[dict]:
    newest = {}
    for row in rows:
        key = (str(row.get("condition")), str(row["task"]), str(row["recording_id"]), str(row["sample_id"]))
        newest[key] = row
    pairs = []
    identities = sorted({key[1:] for key in newest})
    for task, recording_id, sample_id in identities:
        direct = newest.get(("direct", task, recording_id, sample_id))
        spatial = newest.get(("spatial_social", task, recording_id, sample_id))
        if not direct or not spatial or direct.get("error") or spatial.get("error"):
            continue
        if not isinstance(direct.get("prediction"), dict) or not isinstance(spatial.get("prediction"), dict):
            continue
        pairs.append({
            "task": task,
            "recording_id": recording_id,
            "sample_id": sample_id,
            "direct": direct["prediction"],
            "spatial_social": spatial["prediction"],
        })
    return pairs


def _fold(recording_id: str, folds: int) -> int:
    digest = hashlib.sha256(recording_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _metric_summary(scored: list[dict[str, bool]]) -> dict:
    if not scored:
        return {"samples": 0, "metrics": {}}
    total = len(scored)
    correct = {key: sum(row[key] for row in scored) for key in scored[0]}
    return {
        "samples": total,
        "correct": correct,
        "metrics": {key: value / total for key, value in correct.items()},
        "wilson_95_ci": {key: _wilson(value, total) for key, value in correct.items()},
    }


def _wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def _paired_test(candidate: list[dict[str, bool]], baseline: list[dict[str, bool]], metric: str) -> dict:
    candidate_only = sum(c[metric] and not b[metric] for c, b in zip(candidate, baseline))
    baseline_only = sum(b[metric] and not c[metric] for c, b in zip(candidate, baseline))
    discordant = candidate_only + baseline_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, baseline_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {"candidate_only_correct": candidate_only, "baseline_only_correct": baseline_only, "mcnemar_exact_p": p_value}


def evaluate_optimization(annotation_root: Path, prediction_path: Path, folds: int = 5) -> dict:
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = _paired_rows(rows)
    annotations = {task: load_annotations(annotation_root, task) for task in TASK_FILES}
    prepared = []
    normalization_actions: dict[str, Counter] = defaultdict(Counter)
    for pair in pairs:
        task = pair["task"]
        ground_truth = annotations[task].get(pair["recording_id"], {}).get(pair["sample_id"])
        if not isinstance(ground_truth, dict) or (task == "handover" and ground_truth.get("skip")):
            continue
        item = {**pair, "ground_truth": ground_truth, "fold": _fold(pair["recording_id"], folds)}
        for condition in ("direct", "spatial_social"):
            auto, actions = normalize_prediction(task, pair[condition], "auto")
            forced, forced_actions = normalize_prediction(task, pair[condition], "xyxy")
            item[f"{condition}_normalized"] = auto
            item[f"{condition}_forced_xyxy"] = forced
            normalization_actions[f"{condition}_auto"].update(actions)
            normalization_actions[f"{condition}_forced_xyxy"].update(forced_actions)
        prepared.append(item)

    variants = ("direct", "spatial_social", "direct_normalized", "spatial_social_normalized", "direct_forced_xyxy", "spatial_social_forced_xyxy")
    scored_by_variant: dict[str, dict[str, list]] = {variant: defaultdict(list) for variant in variants}
    for item in prepared:
        for variant in variants:
            scored_by_variant[variant][item["task"]].append(score_sample(item["task"], item["ground_truth"], item[variant]))
    result_variants = {
        variant: {task: _metric_summary(scored_by_variant[variant][task]) for task in TASK_FILES}
        for variant in variants
    }

    fusion_scored: dict[str, list] = defaultdict(list)
    fusion_reference_scored: dict[str, dict[str, list]] = {
        "direct_normalized": defaultdict(list),
        "spatial_social_normalized": defaultdict(list),
    }
    choices = []
    for fold in range(folds):
        train = [item for item in prepared if item["fold"] != fold]
        test = [item for item in prepared if item["fold"] == fold]
        fold_choices = {}
        for task, metric_fields in METRIC_FIELDS.items():
            task_choices = {}
            task_train = [item for item in train if item["task"] == task]
            for metric, field in metric_fields.items():
                totals = {}
                for condition in ("direct_normalized", "spatial_social_normalized"):
                    totals[condition] = sum(score_sample(task, item["ground_truth"], item[condition])[metric] for item in task_train)
                # Stable conservative tie break retains Direct.
                choice = max(("direct_normalized", "spatial_social_normalized"), key=lambda name: (totals[name], name == "direct_normalized"))
                task_choices[metric] = {"field": field, "condition": choice, "train_correct": totals, "train_samples": len(task_train)}
            fold_choices[task] = task_choices
        choices.append({"fold": fold, "choices": fold_choices})
        for item in test:
            task = item["task"]
            prediction = {}
            for metric, selection in fold_choices[task].items():
                field = selection["field"]
                prediction[field] = item[selection["condition"]].get(field)
            fusion_scored[task].append(score_sample(task, item["ground_truth"], prediction))
            for baseline in fusion_reference_scored:
                fusion_reference_scored[baseline][task].append(
                    score_sample(task, item["ground_truth"], item[baseline])
                )

    result_variants["recording_cv_field_fusion"] = {task: _metric_summary(fusion_scored[task]) for task in TASK_FILES}
    paired_tests = {}
    for baseline in ("direct_normalized", "spatial_social_normalized"):
        paired_tests[f"recording_cv_field_fusion_vs_{baseline}"] = {
            task: {
                metric: _paired_test(fusion_scored[task], fusion_reference_scored[baseline][task], metric)
                for metric in METRIC_FIELDS[task]
            }
            for task in TASK_FILES
        }
    return {
        "protocol": "paired successful intersection; deterministic SHA-256 recording-level 5-fold CV",
        "normalization_policy": "auto converts list xyxy only when declared xywh is invalid; forced_xyxy is diagnostic only",
        "paired_total": len(prepared),
        "paired_by_task": dict(Counter(item["task"] for item in prepared)),
        "recordings": len({item["recording_id"] for item in prepared}),
        "normalization_actions": {key: dict(value) for key, value in normalization_actions.items()},
        "variants": result_variants,
        "fold_choices": choices,
        "paired_tests": paired_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_optimization(args.annotations, args.predictions, args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "paired_total": report["paired_total"]}, indent=2))


if __name__ == "__main__":
    main()
