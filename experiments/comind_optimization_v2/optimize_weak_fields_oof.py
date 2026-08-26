#!/usr/bin/env python3
"""Nested recording-CV optimization for weak CoMind semantic fields.

Outer folds provide the reported held-out predictions.  For every outer fold
and task/field, the strategy is selected by inner recording-fold validation on
the remaining recordings, then refit without the outer fold.  No held-out
label is used for strategy selection, fitting, or prediction.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V2 = load_module("comind_v2_weak", HERE / "comind_v2.py")
CMP = load_module("comind_cmp_weak", HERE / "compare_balanced_pilot.py")

STRATEGIES = (
    "direct",
    "spatial_social",
    "provenance_router",
    "grounding_raw",
    "train_prior",
    "candidate_tuple_map",
    "candidate_value_map",
    "candidate_categorical_nb",
    "grounding_map",
    "transcript_nb",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple[str, str, str]:
    return str(row["task"]), str(row["recording_id"]), str(row["sample_id"])


def norm(value: Any) -> str:
    return V2.norm_text(value)


def gold_options(task: str, metric: str, gold: dict) -> list[Any]:
    value = V2.ground_truth_value(task, metric, gold)
    if metric in {"cue_type", "initiation_type"}:
        return [sorted(V2.values(value))]
    if metric == "tth_within_0_25s":
        return [float(value)]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if norm(item)]
    return [value]


def label_key(metric: str, value: Any) -> str:
    if metric in {"cue_type", "initiation_type"}:
        return "|".join(sorted(V2.values(value)))
    if metric == "tth_within_0_25s":
        return f"{float(value):.6f}"
    return norm(value)


def decode_label(metric: str, value: str) -> Any:
    if metric in {"cue_type", "initiation_type"}:
        return [item for item in value.split("|") if item]
    if metric == "tth_within_0_25s":
        return float(value)
    return value


def raw_prediction(row: dict, strategy: str, field: str) -> Any:
    if strategy in {"direct", "spatial_social", "provenance_router"}:
        return row[strategy].get(field)
    if strategy == "grounding_raw":
        return row.get("grounding_identity")
    raise ValueError(strategy)


def feature_tuple(row: dict, field: str) -> str:
    return json.dumps(
        [norm(row[source].get(field)) for source in ("direct", "spatial_social", "provenance_router")],
        separators=(",", ":"),
    )


def tokens(row: dict, field: str) -> list[str]:
    text = " ".join(
        [
            str(row.get("transcript", "")),
            f" grounding_{norm(row.get('grounding_identity'))}",
            *(f" {source}_{norm(row[source].get(field))}" for source in ("direct", "spatial_social", "provenance_router")),
        ]
    ).lower()
    words = re.findall(r"[a-z0-9]+", text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def best_constant(train: list[dict], task: str, metric: str) -> Any:
    if metric == "tth_within_0_25s":
        values = [float(option) for row in train for option in gold_options(task, metric, row["gold"])]
        if not values:
            return 0.0
        candidates = sorted(set(values))
        return max(candidates, key=lambda candidate: (sum(abs(candidate - value) <= 0.25 for value in values), -candidate))
    counts = Counter(
        label_key(metric, option)
        for row in train
        for option in gold_options(task, metric, row["gold"])
        if label_key(metric, option)
    )
    return decode_label(metric, counts.most_common(1)[0][0]) if counts else ""


def fit_map(train: list[dict], task: str, metric: str, feature_name: str) -> dict[str, Any]:
    field = V2.FIELD_TO_PREDICTION[task][metric]
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        feature = feature_tuple(row, field) if feature_name == "tuple" else norm(row.get("grounding_identity"))
        if not feature:
            continue
        for option in gold_options(task, metric, row["gold"]):
            option_key = label_key(metric, option)
            if option_key:
                counts[feature][option_key] += 1
    result = {}
    for feature, options in counts.items():
        label, support = options.most_common(1)[0]
        total = sum(options.values())
        if total >= 3 and support / total >= 0.5:
            result[feature] = decode_label(metric, label)
    return result


def fit_value_maps(train: list[dict], task: str, metric: str) -> dict[str, dict[str, tuple[Any, int, float]]]:
    """Fit per-source vocabulary projections using training recordings only."""
    field = V2.FIELD_TO_PREDICTION[task][metric]
    source_counts: dict[str, dict[str, Counter]] = {
        source: defaultdict(Counter) for source in ("direct", "spatial_social", "provenance_router")
    }
    for row in train:
        labels = [label_key(metric, option) for option in gold_options(task, metric, row["gold"])]
        labels = [label for label in labels if label]
        if not labels:
            continue
        for source in source_counts:
            feature = norm(row[source].get(field))
            if not feature:
                continue
            source_counts[source][feature].update(labels)
    result = {}
    for source, features in source_counts.items():
        result[source] = {}
        for feature, counts in features.items():
            label, support = counts.most_common(1)[0]
            total = sum(counts.values())
            result[source][feature] = (decode_label(metric, label), total, support / total)
    return result


def predict_value_map(model: dict[str, dict[str, tuple[Any, int, float]]], row: dict, field: str, fallback: Any) -> Any:
    candidates = []
    for source in ("direct", "spatial_social", "provenance_router"):
        feature = norm(row[source].get(field))
        if feature in model[source]:
            value, total, confidence = model[source][feature]
            # Confidence first; support breaks ties, then prefer the routed candidate.
            candidates.append((confidence, total, source == "provenance_router", value))
    return max(candidates, key=lambda item: item[:3])[3] if candidates else fallback


def fit_categorical_nb(train: list[dict], task: str, metric: str) -> dict:
    """Naive Bayes over the three model outputs, used as a label-vocabulary adapter."""
    field = V2.FIELD_TO_PREDICTION[task][metric]
    class_docs = Counter()
    feature_counts: dict[str, dict[str, Counter]] = {
        source: defaultdict(Counter) for source in ("direct", "spatial_social", "provenance_router")
    }
    vocabularies = {source: set() for source in feature_counts}
    for row in train:
        labels = [label_key(metric, option) for option in gold_options(task, metric, row["gold"])]
        labels = [label for label in labels if label]
        for label in labels:
            class_docs[label] += 1
            for source in feature_counts:
                feature = norm(row[source].get(field)) or "<empty>"
                feature_counts[source][label][feature] += 1
                vocabularies[source].add(feature)
    return {
        "class_docs": class_docs,
        "feature_counts": feature_counts,
        "vocab_sizes": {source: max(1, len(values) + 1) for source, values in vocabularies.items()},
    }


def predict_categorical_nb(model: dict, row: dict, field: str, metric: str, fallback: Any) -> Any:
    if not model["class_docs"]:
        return fallback
    labels = list(model["class_docs"])
    total_docs = sum(model["class_docs"].values())
    alpha = 1.0
    scores = {}
    for label in labels:
        score = math.log((model["class_docs"][label] + alpha) / (total_docs + alpha * len(labels)))
        for source in ("direct", "spatial_social", "provenance_router"):
            feature = norm(row[source].get(field)) or "<empty>"
            count = model["feature_counts"][source][label][feature]
            denom = model["class_docs"][label] + alpha * model["vocab_sizes"][source]
            score += math.log((count + alpha) / denom)
        scores[label] = score
    return decode_label(metric, max(scores, key=scores.get))


def fit_nb(train: list[dict], task: str, metric: str) -> dict:
    field = V2.FIELD_TO_PREDICTION[task][metric]
    class_docs = Counter()
    word_counts: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    vocabulary = set()
    for row in train:
        row_tokens = tokens(row, field)
        vocabulary.update(row_tokens)
        for option in gold_options(task, metric, row["gold"]):
            label = label_key(metric, option)
            if not label:
                continue
            class_docs[label] += 1
            word_counts[label].update(row_tokens)
            totals[label] += len(row_tokens)
    return {"class_docs": class_docs, "word_counts": word_counts, "totals": totals, "vocab": max(1, len(vocabulary))}


def predict_nb(model: dict, row: dict, field: str, metric: str, fallback: Any) -> Any:
    if not model["class_docs"]:
        return fallback
    total_docs = sum(model["class_docs"].values())
    labels = list(model["class_docs"])
    scores = {}
    for label in labels:
        score = math.log(model["class_docs"][label] / total_docs)
        denom = model["totals"][label] + model["vocab"]
        for token in tokens(row, field):
            score += math.log((model["word_counts"][label][token] + 1) / denom)
        scores[label] = score
    return decode_label(metric, max(scores, key=scores.get))


def predict_strategy(strategy: str, train: list[dict], test: list[dict], task: str, metric: str) -> list[Any]:
    field = V2.FIELD_TO_PREDICTION[task][metric]
    if strategy in {"direct", "spatial_social", "provenance_router", "grounding_raw"}:
        return [raw_prediction(row, strategy, field) for row in test]
    prior = best_constant(train, task, metric)
    if strategy == "train_prior":
        return [prior for _ in test]
    if strategy == "candidate_value_map":
        mapping = fit_value_maps(train, task, metric)
        return [predict_value_map(mapping, row, field, prior) for row in test]
    if strategy == "candidate_categorical_nb":
        if metric == "tth_within_0_25s":
            return [prior for _ in test]
        model = fit_categorical_nb(train, task, metric)
        return [predict_categorical_nb(model, row, field, metric, prior) for row in test]
    if strategy in {"candidate_tuple_map", "grounding_map"}:
        feature_name = "tuple" if strategy == "candidate_tuple_map" else "grounding"
        mapping = fit_map(train, task, metric, feature_name)
        values = []
        for row in test:
            feature = feature_tuple(row, field) if feature_name == "tuple" else norm(row.get("grounding_identity"))
            values.append(mapping.get(feature, prior))
        return values
    if strategy == "transcript_nb":
        if metric == "tth_within_0_25s":
            return [prior for _ in test]
        model = fit_nb(train, task, metric)
        return [predict_nb(model, row, field, metric, prior) for row in test]
    raise ValueError(strategy)


def score_values(rows: list[dict], task: str, metric: str, values: list[Any]) -> int:
    return sum(V2.field_correct(task, metric, value, row["gold"]) for row, value in zip(rows, values))


def apply_grounding_bbox(task: str, prediction: dict, grounding: dict) -> dict:
    result = copy.deepcopy(prediction)
    candidate = grounding["prediction"]
    if task == "joint_attention":
        result["left_bbox"] = CMP.per_view_xyxy_to_mosaic(candidate.get("left_bbox_xyxy"), "leader")
        result["right_bbox"] = CMP.per_view_xyxy_to_mosaic(candidate.get("right_bbox_xyxy"), "helper")
    elif task == "scoia":
        result["bbox"] = V2.convert_box(candidate.get("bbox_xyxy"), "xyxy")
    else:
        result["bbox"] = CMP.per_view_xyxy_to_mosaic(candidate.get("bbox_xyxy"), candidate.get("selected_view"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    routed = {key(row): row for row in read_jsonl(args.routed)}
    manifest = {key(row): row for row in read_jsonl(args.manifest)}
    grounding = {key(row): row for row in read_jsonl(args.grounding) if not row.get("error")}
    annotations = V2.load_annotations(args.annotations)
    records = []
    for item_key in sorted(set(routed) & set(manifest) & set(grounding)):
        task, rid, sid = item_key
        gold = annotations[task].get(rid, {}).get(sid)
        if not isinstance(gold, dict):
            continue
        row = routed[item_key]
        records.append(
            {
                "key": item_key,
                "task": task,
                "fold": int(row["fold"]),
                "gold": gold,
                "transcript": manifest[item_key].get("transcript_10s", ""),
                "grounding_identity": grounding[item_key]["prediction"].get("object_identity"),
                "grounding": grounding[item_key],
                "direct": row["direct"],
                "spatial_social": row["spatial_social"],
                "provenance_router": row["provenance_router"],
            }
        )

    per_strategy_scores: dict[str, dict[str, dict[str, list[bool]]]] = {
        strategy: defaultdict(lambda: defaultdict(list)) for strategy in STRATEGIES
    }
    nested_scores: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    final_rows = []
    policies = []

    for outer_fold in range(5):
        outer_train = [row for row in records if row["fold"] != outer_fold]
        outer_test = [row for row in records if row["fold"] == outer_fold]
        for task, metrics in V2.FIELD_TO_PREDICTION.items():
            train_task = [row for row in outer_train if row["task"] == task]
            test_task = [row for row in outer_test if row["task"] == task]
            for metric in metrics:
                if "iou" in metric:
                    continue
                inner_scores = Counter()
                inner_total = 0
                for inner_fold in range(5):
                    if inner_fold == outer_fold:
                        continue
                    inner_train = [row for row in train_task if row["fold"] != inner_fold]
                    inner_test = [row for row in train_task if row["fold"] == inner_fold]
                    inner_total += len(inner_test)
                    for strategy in STRATEGIES:
                        predicted = predict_strategy(strategy, inner_train, inner_test, task, metric)
                        inner_scores[strategy] += score_values(inner_test, task, metric, predicted)
                # Prefer the simpler frozen/raw candidate on exact ties.
                selected = max(STRATEGIES, key=lambda strategy: (inner_scores[strategy], -STRATEGIES.index(strategy)))
                policies.append(
                    {
                        "outer_fold": outer_fold,
                        "task": task,
                        "metric": metric,
                        "selected": selected,
                        "inner_n": inner_total,
                        "inner_correct": dict(inner_scores),
                    }
                )
                nested_values = predict_strategy(selected, train_task, test_task, task, metric)
                for row, value in zip(test_task, nested_values):
                    row.setdefault("nested_values", {})[metric] = value
                    nested_scores[task][metric].append(V2.field_correct(task, metric, value, row["gold"]))
                for strategy in STRATEGIES:
                    predicted = predict_strategy(strategy, train_task, test_task, task, metric)
                    per_strategy_scores[strategy][task][metric].extend(
                        V2.field_correct(task, metric, value, row["gold"])
                        for row, value in zip(test_task, predicted)
                    )

    for row in records:
        prediction = apply_grounding_bbox(row["task"], row["provenance_router"], row["grounding"])
        for metric, value in row.get("nested_values", {}).items():
            prediction[V2.FIELD_TO_PREDICTION[row["task"]][metric]] = value
        scores = V2.score_prediction(row["task"], prediction, row["gold"])
        final_rows.append(
            {
                "task": row["task"],
                "recording_id": row["key"][1],
                "sample_id": row["key"][2],
                "fold": row["fold"],
                "prediction": prediction,
                "scores": scores,
            }
        )

    result = {
        "protocol": "nested 5-fold recording-level OOF strategy selection; outer recording labels never used in fitting or selection",
        "samples": dict(Counter(row["task"] for row in records)),
        "policies": policies,
        "strategies": {},
        "nested_selected": {},
        "combined_grounding_nested": {},
    }
    for strategy in STRATEGIES:
        result["strategies"][strategy] = {
            task: {
                metric: {"samples": len(values), "correct": sum(values), "accuracy": sum(values) / len(values) if values else 0.0}
                for metric, values in metrics.items()
            }
            for task, metrics in per_strategy_scores[strategy].items()
        }
    result["nested_selected"] = {
        task: {
            metric: {"samples": len(values), "correct": sum(values), "accuracy": sum(values) / len(values) if values else 0.0}
            for metric, values in metrics.items()
        }
        for task, metrics in nested_scores.items()
    }
    for task in V2.TASK_FILES:
        task_scores = [row["scores"] for row in final_rows if row["task"] == task]
        result["combined_grounding_nested"][task] = V2.metric_summary(task_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.predictions.write_text(
        "".join(json.dumps({key: value for key, value in row.items() if key != "scores"}) + "\n" for row in final_rows)
    )

    lines = [
        "# CoMind weak-field nested recording-CV optimization",
        "",
        result["protocol"],
        "",
        "| Field | Direct | SSC | Provenance | Train prior | Nested selected | Final combined |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        for metric in metrics:
            if "iou" in metric:
                continue
            values = []
            for strategy in ("direct", "spatial_social", "provenance_router", "train_prior"):
                values.append(100 * result["strategies"][strategy][task][metric]["accuracy"])
            nested = 100 * result["nested_selected"][task][metric]["accuracy"]
            final = 100 * result["combined_grounding_nested"][task]["metrics"][metric]
            lines.append(
                f"| {task}.{metric} | {values[0]:.1f}% | {values[1]:.1f}% | {values[2]:.1f}% | {values[3]:.1f}% | {nested:.1f}% | {final:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Final combined uses two-stage grounding for every bbox and the nested-selected strategy for every non-bbox field.",
            "This is labeled-train recording-heldout analysis, not official hidden-test performance.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"samples": result["samples"], "output": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
