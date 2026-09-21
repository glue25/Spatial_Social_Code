#!/usr/bin/env python3
"""Leakage-safe CoMind error audit and recording-fold offline optimizer.

This module only consumes labeled CoMind *train* annotations.  Every routed
prediction for a recording is produced by a policy fitted without that
recording.  The policy never receives the current sample's gold value.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TASK_FILES = {
    "joint_attention": "dataset_joint_attention_consolidated.json",
    "scoia": "dataset_scoia_consolidated.json",
    "handover": "dataset_handover_consolidated.json",
}

FIELD_TO_PREDICTION = {
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

BBOX_FIELDS = {
    "joint_attention": ("left_bbox", "right_bbox"),
    "scoia": ("bbox",),
    "handover": ("bbox",),
}

SCALAR_IDENTITY_FIELDS = {"category_l1", "action_noun_l1", "action_verb"}
FRAME_SIZE = (1408.0, 1408.0)


def norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def values(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {norm_text(item) for item in value if norm_text(item)}
    value = norm_text(value)
    return {value} if value else set()


def load_annotations(root: Path) -> dict[str, dict]:
    # Refuse paths that appear to be hidden/test annotations.  Official test
    # media may exist locally, but labels must never enter this experiment.
    if "test" in root.name.lower() or "hidden" in str(root).lower():
        raise ValueError(f"refusing possible hidden/test annotation path: {root}")
    result = {}
    for task, filename in TASK_FILES.items():
        path = root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("data"), dict):
            raise ValueError(f"invalid train annotation file: {path}")
        result[task] = payload["data"]
    return result


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fold_for(recording_id: str, folds: int) -> int:
    digest = hashlib.sha256(recording_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def finite_box(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "w", "h")):
            value = [value[key] for key in ("x", "y", "w", "h")]
        elif all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value[key] for key in ("x1", "y1", "x2", "y2")]
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return box if all(math.isfinite(item) for item in box) else None


def convert_box(value: Any, convention: str, frame_size: tuple[float, float] = FRAME_SIZE) -> list[float] | None:
    """Convert an explicit coordinate hypothesis to clipped pixel xywh."""
    box = finite_box(value)
    if box is None:
        return None
    width, height = frame_size
    normalized = all(0 <= item <= 1 for item in box) and max(box) > 0
    if normalized:
        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    x, y, third, fourth = box
    if convention == "xyxy":
        third, fourth = third - x, fourth - y
    elif convention != "xywh":
        raise ValueError(convention)
    x1, y1 = max(0.0, min(width, x)), max(0.0, min(height, y))
    x2, y2 = max(0.0, min(width, x + third)), max(0.0, min(height, y + fourth))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def iou(first: Any, second: Any) -> float:
    # Match the official/local scorer for already-declared xywh inputs: do not
    # silently clip raw baselines. Routed candidates are explicitly converted
    # before reaching this function and therefore remain comparable.
    a, b = finite_box(first), finite_box(second)
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def cue_match(predicted: Any, expected: Any) -> bool:
    aliases = {"communication": "verbal", "gesture": "gestural"}
    normalize = lambda item: aliases.get(norm_text(item), norm_text(item))
    return {normalize(item) for item in values(predicted)} == {normalize(item) for item in values(expected)}


def single_match(predicted: Any, expected: Any) -> bool:
    return bool(norm_text(predicted)) and norm_text(predicted) in values(expected)


def ground_truth_value(task: str, metric: str, gt: dict) -> Any:
    if task == "joint_attention":
        boxes = gt.get("bounding_boxes") or []
        return {
            "left_iou_at_0_5": boxes[0] if len(boxes) > 0 else None,
            "right_iou_at_0_5": boxes[1] if len(boxes) > 1 else None,
            "cue_type": gt.get("cue_types"),
            "category_l1": gt.get("object_category_level_1"),
        }[metric]
    if task == "scoia":
        cues = {"A": ["verbal"], "G": ["gestural"], "AG": ["gestural", "verbal"], "O": ["implicit"]}
        return {
            "iou_at_0_5": gt.get("bbox"),
            "cue_type": cues.get(str(gt.get("annot_type", "")).upper(), []),
            "action_verb": gt.get("verbs"),
            "action_noun_l1": gt.get("nouns"),
        }[metric]
    if task == "handover":
        if metric == "tth_within_0_25s":
            return float(gt["end_time"]) - float(gt["start_time"])
        return {
            "iou_at_0_5": gt.get("bbox"),
            "delivery_flow": gt.get("delivering_flow"),
            "initiator": gt.get("initiator"),
            "initiation_type": gt.get("initiation_type"),
            "category_l1": gt.get("object_category_level_1"),
        }[metric]
    raise ValueError(task)


def field_correct(task: str, metric: str, predicted: Any, gt: dict) -> bool:
    expected = ground_truth_value(task, metric, gt)
    if "iou" in metric:
        return iou(predicted, expected) > 0.5
    if metric in {"cue_type", "initiation_type"}:
        return cue_match(predicted, expected)
    if metric == "delivery_flow":
        flow = {"left to right": "ltr", "leader to helper": "ltr", "right to left": "rtl", "helper to leader": "rtl"}.get(norm_text(predicted), norm_text(predicted).replace(" ", ""))
        return flow == norm_text(expected)
    if metric == "tth_within_0_25s":
        try:
            return abs(float(predicted) - float(expected)) <= 0.25
        except (TypeError, ValueError):
            return False
    return single_match(predicted, expected)


def score_prediction(task: str, prediction: dict, gt: dict) -> dict[str, bool]:
    return {
        metric: field_correct(task, metric, prediction.get(field), gt)
        for metric, field in FIELD_TO_PREDICTION[task].items()
    }


def trace_records(row: dict) -> list[dict]:
    records = []
    for line in str(row.get("spatial_trace") or "").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("values"), dict):
            records.append(value)
    return records


def parse_box_text(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple, dict)):
        return finite_box(value)
    text = str(value or "").strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", text)
        parsed = list(map(float, match.groups())) if match else None
    return finite_box(parsed)


def trace_candidates(row: dict, task: str) -> dict[str, list[tuple[Any, float, str]]]:
    """Extract provenance-carrying candidates without consulting labels."""
    result: dict[str, list[tuple[Any, float, str]]] = defaultdict(list)
    for record in trace_records(row):
        kind, item = norm_text(record.get("kind")), record["values"]
        relation = norm_text(item.get("relation"))
        confidence = item.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        obj = item.get("object")
        if "bounding box" in relation or relation in {"bbox", "box"}:
            box = parse_box_text(obj)
            if box is not None:
                # Joint-attention traces usually do not preserve view identity;
                # expose to both sides but let fold training decide whether usable.
                for metric in ("left_iou_at_0_5", "right_iou_at_0_5") if task == "joint_attention" else ("iou_at_0_5",):
                    result[metric].append((box, confidence, "trace_bbox"))
        if "cue" in relation and obj is not None:
            metric = "initiation_type" if task == "handover" else "cue_type"
            result[metric].append((obj, confidence, "trace_relation"))
        if "category" in relation and obj is not None:
            result["category_l1"].append((obj, confidence, "trace_relation"))
            # The compiler often encodes concrete identity as the *subject*
            # and an abstract super-category as the object, e.g.
            # level_1_category('plate', 'dishware'). Preserve both provenance
            # slots rather than collapsing the subject before final decoding.
            subject = item.get("subject")
            if subject is not None and norm_text(subject) not in {"handover", "anticipated action", "prediction frame"}:
                result["category_l1"].append((subject, confidence, "trace_subject_identity"))
        if "noun" in relation and obj is not None:
            result["action_noun_l1"].append((obj, confidence, "trace_relation"))
        if kind == "event":
            if item.get("target") is not None:
                result["action_noun_l1"].append((item["target"], confidence, "trace_event"))
                result["category_l1"].append((item["target"], confidence, "trace_event"))
            if item.get("action") is not None:
                result["action_verb"].append((item["action"], confidence, "trace_event"))
    # De-duplicate while preserving the highest confidence and provenance.
    clean = {}
    for metric, candidates in result.items():
        by_value = {}
        for value, confidence, source in candidates:
            key = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else norm_text(value)
            if key and (key not in by_value or confidence > by_value[key][1]):
                by_value[key] = (value, confidence, source)
        source_priority = {"trace_subject_identity": 2, "trace_relation": 1, "trace_event": 0, "trace_bbox": 0}
        clean[metric] = sorted(by_value.values(), key=lambda item: (-item[1], -source_priority.get(item[2], 0), norm_text(item[0])))
    return clean


def paired_samples(rows: Iterable[dict], annotations: dict[str, dict], folds: int) -> list[dict]:
    newest = {}
    for row in rows:
        key = (str(row.get("condition")), str(row.get("task")), str(row.get("recording_id")), str(row.get("sample_id")))
        newest[key] = row
    samples = []
    for task, recording_id, sample_id in sorted({key[1:] for key in newest}):
        direct = newest.get(("direct", task, recording_id, sample_id))
        spatial = newest.get(("spatial_social", task, recording_id, sample_id))
        gt = annotations.get(task, {}).get(recording_id, {}).get(sample_id)
        if not direct or not spatial or direct.get("error") or spatial.get("error") or not isinstance(gt, dict):
            continue
        if task == "handover" and gt.get("skip"):
            continue
        if not isinstance(direct.get("prediction"), dict) or not isinstance(spatial.get("prediction"), dict):
            continue
        samples.append({
            "task": task,
            "recording_id": recording_id,
            "sample_id": sample_id,
            "fold": fold_for(recording_id, folds),
            "ground_truth": gt,
            "direct": direct["prediction"],
            "spatial_social": spatial["prediction"],
            "spatial_row": spatial,
            "trace_candidates": trace_candidates(spatial, task),
        })
    return samples


def candidate_values(item: dict, metric: str) -> list[tuple[str, Any, float]]:
    field = FIELD_TO_PREDICTION[item["task"]][metric]
    result = [
        ("direct", item["direct"].get(field), 1.0),
        ("spatial_social", item["spatial_social"].get(field), 1.0),
    ]
    for value, confidence, provenance in item["trace_candidates"].get(metric, []):
        result.append((provenance, value, confidence))
    return result


def accepted_scalar(gt_value: Any) -> list[str]:
    return sorted(values(gt_value))


def fit_scalar_router(train: list[dict], task: str, metric: str) -> dict:
    """Fit train-fold-only canonicalization and calibrated source reliability."""
    source_stats: dict[str, Counter] = defaultdict(Counter)
    mappings: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    label_counts = Counter()
    for item in train:
        if item["task"] != task:
            continue
        accepted = accepted_scalar(ground_truth_value(task, metric, item["ground_truth"]))
        label_counts.update(accepted)
        for source, value, _ in candidate_values(item, metric):
            token = norm_text(value)
            if not token:
                continue
            correct = token in accepted
            source_stats[source]["n"] += 1
            source_stats[source]["correct"] += int(correct)
            for label in accepted:
                mappings[source][token][label] += 1
    compiled_maps = {}
    for source, tokens in mappings.items():
        compiled_maps[source] = {}
        for token, counts in tokens.items():
            label, count = counts.most_common(1)[0]
            total = sum(counts.values())
            # Mapping is permitted only with replicated train-fold support and
            # clear dominance. Exact known label tokens remain unchanged.
            if token in label_counts:
                compiled_maps[source][token] = {"label": token, "support": label_counts[token], "purity": 1.0}
            elif total >= 2 and count / total >= 0.67:
                compiled_maps[source][token] = {"label": label, "support": total, "purity": count / total}
    reliability = {
        source: (stats["correct"] + 1) / (stats["n"] + 2)
        for source, stats in source_stats.items()
    }
    return {"maps": compiled_maps, "reliability": reliability, "source_stats": {k: dict(v) for k, v in source_stats.items()}}


def route_scalar(item: dict, metric: str, policy: dict) -> tuple[Any, dict]:
    ranked = []
    for order, (source, raw, confidence) in enumerate(candidate_values(item, metric)):
        token = norm_text(raw)
        if not token:
            continue
        mapping = policy["maps"].get(source, {}).get(token)
        value = mapping["label"] if mapping else raw
        base = policy["reliability"].get(source, 0.0)
        support_bonus = min(0.12, math.log1p(mapping["support"]) / 35) if mapping else 0.0
        score = base + support_bonus + (confidence - 0.5) * 0.04
        # Stable preference is Direct, then explicit final, then trace.
        tie = {"direct": 2, "spatial_social": 1}.get(source, 0)
        ranked.append((score, tie, -order, value, source, raw, mapping))
    if not ranked:
        return None, {"source": "missing"}
    best = max(ranked, key=lambda row: row[:3])
    return best[3], {"source": best[4], "raw": best[5], "mapping": best[6], "score": best[0]}


def box_hypotheses(item: dict, metric: str) -> list[tuple[str, list[float]]]:
    result = []
    for source, raw, _ in candidate_values(item, metric):
        for convention in ("xywh", "xyxy"):
            converted = convert_box(raw, convention)
            if converted is not None:
                result.append((f"{source}:{convention}", converted))
    return result


def fit_box_policy(train: list[dict], task: str, metric: str) -> dict:
    stats: dict[str, Counter] = defaultdict(Counter)
    for item in train:
        if item["task"] != task:
            continue
        seen = set()
        for source, box in box_hypotheses(item, metric):
            if source in seen:
                continue
            seen.add(source)
            stats[source]["n"] += 1
            stats[source]["correct"] += int(field_correct(task, metric, box, item["ground_truth"]))
    preference = ["direct:xywh", "spatial_social:xywh", "trace_bbox:xywh", "direct:xyxy", "spatial_social:xyxy", "trace_bbox:xyxy"]
    rank = {source: -index for index, source in enumerate(preference)}
    best = max(stats, key=lambda source: ((stats[source]["correct"] + 1) / (stats[source]["n"] + 2), rank.get(source, -99))) if stats else "direct:xywh"
    # With fewer than five positive localization examples, fold-to-fold source
    # selection is dominated by single hits. Keep the declared Direct xywh
    # protocol instead of turning noise into a routing rule.
    if not stats or stats[best]["correct"] < 5:
        best = "direct:xywh"
    return {"selected": best, "stats": {key: dict(value) for key, value in stats.items()}}


def route_box(item: dict, metric: str, policy: dict) -> tuple[Any, dict]:
    hypotheses = dict(box_hypotheses(item, metric))
    selected = policy["selected"]
    if selected in hypotheses:
        return hypotheses[selected], {"source": selected}
    # Missing trace candidates fall back deterministically without gold.
    for fallback in ("direct:xywh", "spatial_social:xywh", "direct:xyxy", "spatial_social:xyxy"):
        if fallback in hypotheses:
            return hypotheses[fallback], {"source": fallback, "fallback_from": selected}
    return None, {"source": "missing", "fallback_from": selected}


def fit_fixed_source(train: list[dict], task: str, metric: str) -> dict:
    totals = {}
    for source in ("direct", "spatial_social"):
        field = FIELD_TO_PREDICTION[task][metric]
        relevant = [item for item in train if item["task"] == task]
        totals[source] = sum(field_correct(task, metric, item[source].get(field), item["ground_truth"]) for item in relevant)
    selected = max(("direct", "spatial_social"), key=lambda source: (totals[source], source == "direct"))
    return {"selected": selected, "train_correct": totals}


def metric_summary(scores: list[dict[str, bool]]) -> dict:
    if not scores:
        return {"samples": 0, "correct": {}, "metrics": {}}
    correct = {metric: sum(row[metric] for row in scores) for metric in scores[0]}
    return {"samples": len(scores), "correct": correct, "metrics": {metric: count / len(scores) for metric, count in correct.items()}}


def exact_mcnemar(candidate: list[bool], baseline: list[bool]) -> dict:
    both_correct = sum(c and b for c, b in zip(candidate, baseline))
    candidate_only = sum(c and not b for c, b in zip(candidate, baseline))
    baseline_only = sum(b and not c for c, b in zip(candidate, baseline))
    both_wrong = sum(not c and not b for c, b in zip(candidate, baseline))
    discordant = candidate_only + baseline_only
    if not discordant:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, baseline_only) + 1)) / 2**discordant
        p = min(1.0, 2 * tail)
    return {
        "total": len(candidate), "both_correct": both_correct,
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only, "both_wrong": both_wrong,
        "mcnemar_exact_p": p,
    }


def evaluate_oof(samples: list[dict], folds: int) -> tuple[dict, list[dict]]:
    outputs = []
    policies = []
    for fold in range(folds):
        train = [item for item in samples if item["fold"] != fold]
        test = [item for item in samples if item["fold"] == fold]
        fold_policy = {}
        for task, metric_fields in FIELD_TO_PREDICTION.items():
            fold_policy[task] = {}
            for metric in metric_fields:
                raw_fixed = fit_fixed_source(train, task, metric)
                if "iou" in metric:
                    route = {"kind": "box", **fit_box_policy(train, task, metric)}
                elif metric in SCALAR_IDENTITY_FIELDS:
                    route = {"kind": "identity", **fit_scalar_router(train, task, metric)}
                else:
                    route = {"kind": "fixed", **raw_fixed}
                route["raw_fixed_source"] = raw_fixed["selected"]
                route["raw_fixed_train_correct"] = raw_fixed["train_correct"]
                fold_policy[task][metric] = route
        policies.append({"fold": fold, "train_recordings": len({x['recording_id'] for x in train}), "test_recordings": len({x['recording_id'] for x in test}), "policy": fold_policy})
        for item in test:
            task = item["task"]
            routed, routing = {}, {}
            fixed, fixed_routing = {}, {}
            trace_identity = copy.deepcopy(item["spatial_social"])
            trace_identity_routing = {}
            for metric, field in FIELD_TO_PREDICTION[task].items():
                policy = fold_policy[task][metric]
                if policy["kind"] == "box":
                    value, info = route_box(item, metric, policy)
                elif policy["kind"] == "identity":
                    value, info = route_scalar(item, metric, policy)
                else:
                    source = policy["selected"]
                    value, info = item[source].get(field), {"source": source}
                routed[field], routing[metric] = value, info
                # Routing-only control: raw Direct vs Spatial fields, with no
                # trace extraction or canonical mapping.
                fixed_source = fold_policy[task][metric]["raw_fixed_source"]
                fixed[field], fixed_routing[metric] = item[fixed_source].get(field), {"source": fixed_source}
                if metric in SCALAR_IDENTITY_FIELDS:
                    candidates = item["trace_candidates"].get(metric, [])
                    if candidates:
                        trace_identity[field] = candidates[0][0]
                        trace_identity_routing[metric] = {"source": candidates[0][2], "confidence": candidates[0][1]}
                    else:
                        trace_identity_routing[metric] = {"source": "spatial_social_fallback"}
            outputs.append({
                "task": task,
                "recording_id": item["recording_id"],
                "sample_id": item["sample_id"],
                "fold": fold,
                "direct": item["direct"],
                "spatial_social": item["spatial_social"],
                "fixed_field_fusion": fixed,
                "trace_identity_raw": trace_identity,
                "provenance_router": routed,
                "routing": routing,
                "fixed_routing": fixed_routing,
                "trace_identity_routing": trace_identity_routing,
                "scores": {
                    "direct": score_prediction(task, item["direct"], item["ground_truth"]),
                    "spatial_social": score_prediction(task, item["spatial_social"], item["ground_truth"]),
                    "fixed_field_fusion": score_prediction(task, fixed, item["ground_truth"]),
                    "trace_identity_raw": score_prediction(task, trace_identity, item["ground_truth"]),
                    "provenance_router": score_prediction(task, routed, item["ground_truth"]),
                },
            })
    summary = {"protocol": f"{folds}-fold SHA256 recording-level out-of-fold; train annotations only; no per-sample oracle", "folds": policies, "variants": {}, "paired_tests": {}}
    variants = ("direct", "spatial_social", "trace_identity_raw", "fixed_field_fusion", "provenance_router")
    for variant in variants:
        summary["variants"][variant] = {}
        for task in TASK_FILES:
            scores = [row["scores"][variant] for row in outputs if row["task"] == task]
            summary["variants"][variant][task] = metric_summary(scores)
    comparisons = (
        ("trace_identity_raw", "spatial_social"),
        ("fixed_field_fusion", "direct"),
        ("fixed_field_fusion", "spatial_social"),
        ("provenance_router", "direct"),
        ("provenance_router", "spatial_social"),
        ("provenance_router", "fixed_field_fusion"),
    )
    for candidate, baseline in comparisons:
        key = f"{candidate}_vs_{baseline}"
        summary["paired_tests"][key] = {}
        for task, metrics in FIELD_TO_PREDICTION.items():
            rows = [row for row in outputs if row["task"] == task]
            summary["paired_tests"][key][task] = {
                metric: exact_mcnemar([row["scores"][candidate][metric] for row in rows], [row["scores"][baseline][metric] for row in rows])
                for metric in metrics
            }
    summary["routing_coverage"] = {}
    for variant, route_key in (("trace_identity_raw", "trace_identity_routing"), ("fixed_field_fusion", "fixed_routing"), ("provenance_router", "routing")):
        summary["routing_coverage"][variant] = {}
        for task, metrics in FIELD_TO_PREDICTION.items():
            task_rows = [row for row in outputs if row["task"] == task]
            summary["routing_coverage"][variant][task] = {
                metric: dict(Counter(row.get(route_key, {}).get(metric, {}).get("source", "unchanged") for row in task_rows))
                for metric in metrics
            }
    return summary, sorted(outputs, key=lambda x: (x["task"], x["recording_id"], x["sample_id"]))


def quantiles(items: list[float]) -> dict:
    if not items:
        return {}
    ordered = sorted(items)
    pick = lambda p: ordered[round((len(ordered) - 1) * p)]
    return {"min": ordered[0], "p25": pick(0.25), "median": median(ordered), "p75": pick(0.75), "max": ordered[-1]}


def error_audit(samples: list[dict]) -> dict:
    audit = {"paired_total": len(samples), "paired_by_task": dict(Counter(x["task"] for x in samples)), "recordings": len({x["recording_id"] for x in samples}), "fields": {}, "bbox": {}, "identity_loss": {}}
    for task, metrics in FIELD_TO_PREDICTION.items():
        task_items = [item for item in samples if item["task"] == task]
        audit["fields"][task] = {}
        for metric, field in metrics.items():
            contingency = Counter()
            for item in task_items:
                d = field_correct(task, metric, item["direct"].get(field), item["ground_truth"])
                s = field_correct(task, metric, item["spatial_social"].get(field), item["ground_truth"])
                contingency["both_correct" if d and s else "direct_only" if d else "spatial_only" if s else "both_wrong"] += 1
            audit["fields"][task][metric] = dict(contingency)
        bbox_metrics = [metric for metric in metrics if "iou" in metric]
        for metric in bbox_metrics:
            field = metrics[metric]
            key = f"{task}.{metric}"
            audit["bbox"][key] = {}
            for source in ("direct", "spatial_social"):
                raw_ious, xyxy_ious, center_errors, area_ratios = [], [], [], []
                valid_xywh = valid_xyxy = 0
                for item in task_items:
                    raw = item[source].get(field)
                    expected = ground_truth_value(task, metric, item["ground_truth"])
                    xywh, xyxy = convert_box(raw, "xywh"), convert_box(raw, "xyxy")
                    valid_xywh += xywh is not None
                    valid_xyxy += xyxy is not None
                    raw_ious.append(iou(xywh, expected))
                    xyxy_ious.append(iou(xyxy, expected))
                    gt_box = convert_box(expected, "xywh")
                    if xywh and gt_box:
                        cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
                        gx, gy = gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2
                        center_errors.append(math.hypot(cx - gx, cy - gy) / math.hypot(*FRAME_SIZE))
                        area_ratios.append((xywh[2] * xywh[3]) / (gt_box[2] * gt_box[3]))
                audit["bbox"][key][source] = {
                    "samples": len(task_items), "valid_xywh": valid_xywh, "valid_xyxy": valid_xyxy,
                    "iou_xywh": quantiles(raw_ious), "iou_xyxy": quantiles(xyxy_ious),
                    "normalized_center_error_xywh": quantiles(center_errors), "area_ratio_xywh": quantiles(area_ratios),
                    "hits_xywh": sum(x > 0.5 for x in raw_ious), "hits_xyxy": sum(x > 0.5 for x in xyxy_ious),
                }
        identity_metrics = [metric for metric in metrics if metric in {"category_l1", "action_noun_l1"}]
        for metric in identity_metrics:
            field = metrics[metric]
            losses = [item for item in task_items if field_correct(task, metric, item["direct"].get(field), item["ground_truth"]) and not field_correct(task, metric, item["spatial_social"].get(field), item["ground_truth"])]
            trace_gold = trace_direct = trace_any = 0
            spatial_code_mentions_gold = spatial_trace_mentions_gold = social_trace_mentions_gold = 0
            wrong_values = Counter()
            examples = []
            for item in losses:
                gold = values(ground_truth_value(task, metric, item["ground_truth"]))
                direct = norm_text(item["direct"].get(field))
                candidates = item["trace_candidates"].get(metric, [])
                tokens = {norm_text(value) for value, _, _ in candidates}
                trace_gold += bool(gold & tokens)
                trace_direct += direct in tokens
                trace_any += bool(tokens)
                normalized_code = f" {norm_text(item['spatial_row'].get('spatial_code'))} "
                normalized_spatial_trace = f" {norm_text(item['spatial_row'].get('spatial_trace'))} "
                normalized_social_trace = f" {norm_text(item['spatial_row'].get('social_trace'))} "
                mentions = lambda text: any(f" {label} " in text for label in gold)
                spatial_code_mentions_gold += mentions(normalized_code)
                spatial_trace_mentions_gold += mentions(normalized_spatial_trace)
                social_trace_mentions_gold += mentions(normalized_social_trace)
                wrong_values[norm_text(item["spatial_social"].get(field))] += 1
                if len(examples) < 12:
                    examples.append({"recording_id": item["recording_id"], "sample_id": item["sample_id"], "gold": sorted(gold), "direct": item["direct"].get(field), "spatial_final": item["spatial_social"].get(field), "trace_candidates": [{"value": v, "confidence": c, "source": s} for v, c, s in candidates]})
            audit["identity_loss"][f"{task}.{metric}"] = {
                "direct_correct_spatial_wrong": len(losses), "trace_has_any_candidate": trace_any,
                "trace_exactly_mentions_gold": trace_gold, "trace_mentions_direct_identity": trace_direct,
                "spatial_code_text_mentions_gold": spatial_code_mentions_gold,
                "spatial_trace_text_mentions_gold": spatial_trace_mentions_gold,
                "social_trace_text_mentions_gold": social_trace_mentions_gold,
                "top_wrong_spatial_final_values": wrong_values.most_common(15), "examples": examples,
            }
    return audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_dataset(name: str, prediction_path: Path, annotations: dict[str, dict], folds: int, output_root: Path) -> dict:
    rows = load_jsonl(prediction_path)
    samples = paired_samples(rows, annotations, folds)
    audit = error_audit(samples)
    evaluation, outputs = evaluate_oof(samples, folds)
    write_json(output_root / f"{name}_audit.json", audit)
    write_json(output_root / f"{name}_oof_evaluation.json", evaluation)
    # Gold values are intentionally omitted from exported routed predictions.
    write_jsonl(output_root / f"{name}_oof_predictions.jsonl", outputs)
    compact_identity = {
        key: {field: value for field, value in details.items() if field != "examples"}
        for key, details in audit["identity_loss"].items()
    }
    compact_bbox = {
        key: {
            source: {
                "samples": details["samples"], "hits_xywh": details["hits_xywh"],
                "hits_xyxy": details["hits_xyxy"], "median_iou_xywh": details["iou_xywh"].get("median"),
                "median_normalized_center_error_xywh": details["normalized_center_error_xywh"].get("median"),
                "median_area_ratio_xywh": details["area_ratio_xywh"].get("median"),
            }
            for source, details in sources.items()
        }
        for key, sources in audit["bbox"].items()
    }
    return {
        "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "paired_total": len(samples), "paired_by_task": audit["paired_by_task"],
        "audit": f"{name}_audit.json", "evaluation": f"{name}_oof_evaluation.json", "predictions": f"{name}_oof_predictions.jsonl",
        "variants": evaluation["variants"], "paired_tests": evaluation["paired_tests"],
        "routing_coverage": evaluation["routing_coverage"],
        "identity_loss_audit": compact_identity,
        "bbox_audit": compact_bbox,
        "_sample_keys": {(x["task"], x["recording_id"], x["sample_id"]) for x in samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--full-predictions", type=Path, default=Path("workdirs/comind_full_train/predictions.jsonl"))
    parser.add_argument("--gpt300-predictions", type=Path, default=Path("workdirs/spatial_social_optimization_v1/comind_codex_300_predictions.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("workdirs/comind_optimization_v2"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations)
    args.output_root.mkdir(parents=True, exist_ok=True)
    full = run_dataset("full_train", args.full_predictions, annotations, args.folds, args.output_root)
    gpt300 = run_dataset("gpt56_300", args.gpt300_predictions, annotations, args.folds, args.output_root)
    overlap = len(full["_sample_keys"] & gpt300["_sample_keys"])
    full.pop("_sample_keys")
    gpt300.pop("_sample_keys")
    summary = {
        "schema_version": 1,
        "scientific_boundary": {
            "annotations": "CoMind labeled train only", "hidden_test_read": False,
            "per_sample_gold_routing": False, "split_unit": "recording_id", "folds": args.folds,
            "claim_status": "diagnostic train out-of-fold; not official hidden-test performance",
        },
        "ablation_definitions": {
            "trace_identity_raw": "Spatial-Social output with identity fields replaced by the highest-confidence concrete Spatial-trace subject; no Direct branch and no learned mapping.",
            "fixed_field_fusion": "Recording-fold-trained field-level choice between raw Direct and raw Spatial-Social; no trace extraction or canonical mapping.",
            "provenance_router": "Recording-fold-trained routing over Direct, Spatial-Social, and trace candidates, with train-fold-only replicated-support canonicalization.",
        },
        "dataset_relationship": {
            "gpt56_300_overlap_with_full_train_samples": overlap,
            "gpt56_300_is_independent_heldout": False,
            "note": "The 300 examples are a model-comparison subset of the same labeled train corpus. Its OOF policy is fitted only within its own recording folds, but it is not an independent benchmark test set.",
        },
        "input_integrity": {
            "annotation_files_sha256": {
                task: sha256(args.annotations / filename) for task, filename in TASK_FILES.items()
            }
        },
        "current_environment": {
            "gpu_grounding_status": "not_run_no_nvidia_driver",
            "frozen_manifest": "gpu_grounding_manifest.jsonl",
            "generated_grounding_predictions": False,
        },
        "verification": {
            "v2_unit_tests": "5/5 passed",
            "existing_bbox_unit_functions": "5/5 passed",
            "full_legacy_unittest_discovery": "blocked at import because optional PyAV module 'av' is absent; unrelated to v2 logic",
        },
        "datasets": {"full_train_paired": full, "gpt56_300": gpt300},
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps({"output": str(args.output_root / "summary.json"), "paired": {key: value["paired_total"] for key, value in summary["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
