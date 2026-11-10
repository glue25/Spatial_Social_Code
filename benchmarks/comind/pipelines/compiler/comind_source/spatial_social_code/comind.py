"""CoMind data adapters and local metrics for labeled train recordings.

The public CoMind annotation files expose train labels and test sample IDs. Test
values are null, so official test evaluation must go through the benchmark
submission forms. This module deliberately refuses to treat null test entries
as ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_FILES = {
    "joint_attention": "dataset_joint_attention_consolidated.json",
    "scoia": "dataset_scoia_consolidated.json",
    "handover": "dataset_handover_consolidated.json",
}

PAPER_QWEN3_VL_32B_FT = {
    "joint_attention": {
        "left_iou_at_0_5": 0.1562,
        "right_iou_at_0_5": 0.1496,
        "cue_type": 0.4572,
        "category_l1": 0.3323,
    },
    "scoia": {
        "iou_at_0_5": 0.2513,
        "cue_type": 0.5503,
        "action_verb": 0.6508,
        "action_noun_l1": 0.4312,
    },
    "handover": {
        "iou_at_0_5": 0.0130,
        "delivery_flow": 0.6143,
        "initiator": 0.7286,
        "initiation_type": 0.3714,
        "category_l1": 0.3619,
        "tth_within_0_25s": 0.0905,
    },
}


def _text(value: Any) -> str:
    return re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())


def _values(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {_text(item) for item in value if _text(item)}
    text = _text(value)
    return {text} if text else set()


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x", "y", "w", "h")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def box_iou(first: Any, second: Any) -> float:
    """Return IoU for CoMind ``[x, y, width, height]`` boxes."""
    left = _box(first)
    right = _box(second)
    if left is None or right is None:
        return 0.0
    ax1, ay1, aw, ah = left
    bx1, by1, bw, bh = right
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def load_annotations(annotation_root: str | Path, task: str) -> dict[str, dict]:
    if task not in TASK_FILES:
        raise ValueError(f"unknown CoMind task: {task}")
    path = Path(annotation_root) / TASK_FILES[task]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"]


def iter_labeled_samples(
    annotation_root: str | Path,
    task: str,
    recording_ids: Iterable[str] | None = None,
) -> Iterable[tuple[str, str, dict]]:
    selected = set(recording_ids or [])
    for recording_id, samples in load_annotations(annotation_root, task).items():
        if selected and recording_id not in selected:
            continue
        for sample_id, ground_truth in samples.items():
            if not isinstance(ground_truth, dict):
                continue
            if task == "handover" and ground_truth.get("skip"):
                continue
            yield recording_id, sample_id, ground_truth


def _single_match(predicted: Any, accepted: Any) -> bool:
    prediction = _text(predicted)
    return bool(prediction) and prediction in _values(accepted)


def _cue_match(predicted: Any, expected: Any, aliases: dict[str, str]) -> bool:
    normalize = lambda item: aliases.get(_text(item), _text(item))
    return {normalize(item) for item in _values(predicted)} == {
        normalize(item) for item in _values(expected)
    }


def score_sample(task: str, ground_truth: dict, prediction: dict) -> dict[str, bool]:
    """Score one structured prediction with the paper's reported metrics."""
    if task == "joint_attention":
        boxes = ground_truth.get("bounding_boxes") or []
        left = boxes[0] if len(boxes) > 0 else None
        right = boxes[1] if len(boxes) > 1 else None
        return {
            "left_iou_at_0_5": box_iou(prediction.get("left_bbox"), left) > 0.5,
            "right_iou_at_0_5": box_iou(prediction.get("right_bbox"), right) > 0.5,
            "cue_type": _cue_match(
                prediction.get("cue_type"),
                ground_truth.get("cue_types"),
                {"communication": "verbal", "gesture": "gestural"},
            ),
            "category_l1": _single_match(
                prediction.get("category_l1"),
                ground_truth.get("object_category_level_1"),
            ),
        }
    if task == "scoia":
        cue = {
            "A": ["verbal"],
            "G": ["gestural"],
            "AG": ["gestural", "verbal"],
            "O": ["implicit"],
        }.get(str(ground_truth.get("annot_type", "")).upper(), [])
        return {
            "iou_at_0_5": box_iou(
                prediction.get("bbox"), ground_truth.get("bbox")
            )
            > 0.5,
            "cue_type": _cue_match(prediction.get("cue_type"), cue, {}),
            "action_verb": _single_match(
                prediction.get("action_verb"), ground_truth.get("verbs")
            ),
            "action_noun_l1": _single_match(
                prediction.get("action_noun_l1"), ground_truth.get("nouns")
            ),
        }
    if task == "handover":
        flow = _text(prediction.get("delivery_flow"))
        flow = {
            "left to right": "ltr",
            "right to left": "rtl",
            "leader to helper": "ltr",
            "helper to leader": "rtl",
        }.get(flow, flow.replace(" ", ""))
        try:
            tth_correct = abs(
                float(prediction.get("time_to_handover"))
                - (
                    float(ground_truth["end_time"])
                    - float(ground_truth["start_time"])
                )
            ) <= 0.25
        except (KeyError, TypeError, ValueError):
            tth_correct = False
        return {
            "iou_at_0_5": box_iou(
                prediction.get("bbox"), ground_truth.get("bbox")
            )
            > 0.5,
            "delivery_flow": flow == _text(ground_truth.get("delivering_flow")),
            "initiator": _single_match(
                prediction.get("initiator"), ground_truth.get("initiator")
            ),
            "initiation_type": _cue_match(
                prediction.get("initiation_type"),
                ground_truth.get("initiation_type"),
                {"gesture": "gestural", "communication": "verbal"},
            ),
            "category_l1": _single_match(
                prediction.get("category_l1"),
                ground_truth.get("object_category_level_1"),
            ),
            "tth_within_0_25s": tth_correct,
        }
    raise ValueError(f"unknown CoMind task: {task}")


def evaluate_rows(annotation_root: str | Path, rows: Iterable[dict]) -> dict:
    annotations = {
        task: load_annotations(annotation_root, task) for task in TASK_FILES
    }
    grouped: dict[tuple[str, str], list[dict[str, bool]]] = defaultdict(list)
    missing_ground_truth = 0
    failed_predictions = 0
    for row in rows:
        if row.get("error") or not isinstance(row.get("prediction"), dict):
            failed_predictions += 1
            continue
        task = str(row["task"])
        recording_id = str(row["recording_id"])
        sample_id = str(row["sample_id"])
        ground_truth = annotations[task].get(recording_id, {}).get(sample_id)
        if not isinstance(ground_truth, dict):
            missing_ground_truth += 1
            continue
        if task == "handover" and ground_truth.get("skip"):
            continue
        condition = str(row.get("condition", "spatial_social"))
        grouped[(condition, task)].append(
            score_sample(task, ground_truth, row.get("prediction", {}))
        )

    conditions: dict[str, dict] = defaultdict(dict)
    for (condition, task), scored in sorted(grouped.items()):
        metrics = {
            metric: sum(item[metric] for item in scored) / len(scored)
            for metric in scored[0]
        }
        conditions[condition][task] = {
            "samples": len(scored),
            "metrics": metrics,
            "paper_qwen3_vl_32b_ft": PAPER_QWEN3_VL_32B_FT[task],
        }
    return {
        "conditions": dict(conditions),
        "missing_or_hidden_ground_truth": missing_ground_truth,
        "failed_predictions_excluded": failed_predictions,
    }


def evaluate_paired_rows(annotation_root: str | Path, rows: Iterable[dict]) -> dict:
    """Compare Direct and Spatial-Social on their exact successful intersection."""
    annotations = {
        task: load_annotations(annotation_root, task) for task in TASK_FILES
    }
    newest: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("condition", "spatial_social")),
            str(row["task"]),
            str(row["recording_id"]),
            str(row["sample_id"]),
        )
        newest[key] = row

    identities = sorted({key[1:] for key in newest})
    scored: dict[tuple[str, str], list[dict[str, bool]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_ground_truth = 0
    for task, recording_id, sample_id in identities:
        totals[task] += 1
        ground_truth = annotations[task].get(recording_id, {}).get(sample_id)
        if not isinstance(ground_truth, dict):
            missing_ground_truth += 1
            continue
        if task == "handover" and ground_truth.get("skip"):
            continue
        pair: dict[str, dict] = {}
        for condition in ("direct", "spatial_social"):
            row = newest.get((condition, task, recording_id, sample_id))
            if row is None or row.get("error") or not isinstance(row.get("prediction"), dict):
                failures[condition][task] += 1
            else:
                pair[condition] = row["prediction"]
        if len(pair) != 2:
            continue
        for condition, prediction in pair.items():
            scored[(condition, task)].append(
                score_sample(task, ground_truth, prediction)
            )

    conditions: dict[str, dict] = defaultdict(dict)
    paired_samples: dict[str, int] = {}
    for task in TASK_FILES:
        direct_scored = scored.get(("direct", task), [])
        paired_samples[task] = len(direct_scored)
        for condition in ("direct", "spatial_social"):
            values = scored.get((condition, task), [])
            metrics = (
                {
                    metric: sum(item[metric] for item in values) / len(values)
                    for metric in values[0]
                }
                if values
                else {}
            )
            conditions[condition][task] = {
                "samples": len(values),
                "metrics": metrics,
                "paper_qwen3_vl_32b_ft": PAPER_QWEN3_VL_32B_FT[task],
            }
    return {
        "protocol": "paired_successful_intersection",
        "samples_seen": dict(totals),
        "paired_samples": paired_samples,
        "failed_or_missing_by_condition": {
            condition: dict(by_task) for condition, by_task in failures.items()
        },
        "conditions": dict(conditions),
        "missing_or_hidden_ground_truth": missing_ground_truth,
    }


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("external_datasets/comind/annotations"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="score Direct and Spatial-Social only on their successful intersection",
    )
    args = parser.parse_args()
    rows = load_jsonl(args.predictions)
    report = (
        evaluate_paired_rows(args.annotations, rows)
        if args.paired
        else evaluate_rows(args.annotations, rows)
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
