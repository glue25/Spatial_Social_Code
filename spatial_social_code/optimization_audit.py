"""Reproduce frozen results and run label-free gate diagnostics.

Gold labels are read only after each gate decision has been made.  The gate
functions accept a single prediction record and never accept gold, category,
question type, saved correctness, or another row's outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spatial_social_code.comind import box_iou, load_annotations, score_sample
from spatial_social_code.comind_optimization import normalize_prediction


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def paired_summary(rows: list[dict], direct_key: str, spatial_key: str) -> dict:
    both = direct_only = spatial_only = neither = 0
    for row in rows:
        direct, spatial = bool(row[direct_key]), bool(row[spatial_key])
        if direct and spatial:
            both += 1
        elif direct:
            direct_only += 1
        elif spatial:
            spatial_only += 1
        else:
            neither += 1
    return {
        "total": len(rows),
        "direct_correct": both + direct_only,
        "spatial_social_correct": both + spatial_only,
        "both_correct": both,
        "direct_only_correct": direct_only,
        "spatial_social_only_correct": spatial_only,
        "both_wrong": neither,
        "oracle_union_correct": both + direct_only + spatial_only,
        "oracle_union_accuracy": (both + direct_only + spatial_only) / len(rows),
        "mcnemar_exact_p": exact_mcnemar(spatial_only, direct_only),
    }


def trace_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return all(isinstance(json.loads(line), dict) for line in value.splitlines() if line.strip())
    except json.JSONDecodeError:
        return False


def trace_records(value: Any) -> list[dict]:
    if not trace_valid(value):
        return []
    return [json.loads(line) for line in value.splitlines() if line.strip()]


def mmtom_gate_decision(row: dict) -> tuple[str, dict]:
    """Frozen label-free gate; deliberately has no correctness/type inputs."""
    records = trace_records(row.get("social_trace"))
    kinds = Counter(item.get("kind") for item in records)
    prob_a, prob_b = row.get("prob_a"), row.get("prob_b")
    probabilities_valid = all(isinstance(value, (int, float)) and math.isfinite(value) for value in (prob_a, prob_b))
    margin = abs(float(prob_a) - float(prob_b)) if probabilities_valid else 0.0
    candidates = kinds.get("query_candidate", 0)
    evidence = kinds.get("query_evidence", 0) + kinds.get("action_evidence", 0) + kinds.get("observation", 0)
    execution_valid = trace_valid(row.get("spatial_trace")) and bool(records)
    answer_valid = str(row.get("prediction", "")).lower() in {"a", "b"}
    evidence_covered = candidates >= 2 and evidence >= 1 and kinds.get("posterior", 0) >= 2
    high_confidence = margin >= 0.20
    uncertainty_ok = not row.get("uncertainty_triggered") or margin >= 0.50
    use_spatial = execution_valid and answer_valid and evidence_covered and high_confidence and uncertainty_ok
    return ("spatial_social" if use_spatial else "direct"), {
        "execution_valid": execution_valid,
        "answer_valid": answer_valid,
        "candidate_count": candidates,
        "evidence_count": evidence,
        "evidence_covered": evidence_covered,
        "posterior_margin": margin,
        "uncertainty_ok": uncertainty_ok,
    }


def ma_gate_decision(row: dict) -> tuple[str, dict]:
    """Transfer the same validity/evidence/uncertainty policy without MA labels."""
    spatial_valid = trace_valid(row.get("spatial_trace"))
    social_valid = trace_valid(row.get("social_trace"))
    prediction = row.get("prediction_index")
    answer_valid = isinstance(prediction, int) and 0 <= prediction <= 4
    cited = row.get("structured_data", {}).get("evidence_ids", [])
    valid_ids = set(row.get("valid_evidence_ids") or [])
    citation_valid = bool(cited) and all(item in valid_ids for item in cited) and bool(row.get("structured_citation_valid"))
    evidence_covered = spatial_valid and social_valid and bool(valid_ids)
    execution_valid = spatial_valid and social_valid
    # MA has no calibrated posterior. Missing margin is treated as unavailable,
    # not fabricated; citation support supplies the confidence requirement.
    use_spatial = execution_valid and answer_valid and evidence_covered and citation_valid and not row.get("audit_triggered")
    return ("spatial_social" if use_spatial else "direct"), {
        "execution_valid": execution_valid,
        "answer_valid": answer_valid,
        "evidence_covered": evidence_covered,
        "citation_valid": citation_valid,
        "posterior_margin": None,
        "uncertainty_ok": not row.get("audit_triggered"),
    }


def evaluate_gate(rows: list[dict], dataset: str) -> dict:
    decisions = []
    feature_counts: Counter = Counter()
    for row in rows:
        condition, features = mmtom_gate_decision(row) if dataset == "mmtom" else ma_gate_decision(row)
        feature_counts.update({key: 1 for key, value in features.items() if value is True})
        if dataset == "mmtom":
            prediction = row["prediction"] if condition == "spatial_social" else row["direct_prediction"]
            gold = row["answer"]
            direct_correct = row["direct_prediction"] == gold
            spatial_correct = row["prediction"] == gold
        else:
            prediction = row["prediction_index"] if condition == "spatial_social" else row["baseline_prediction_index"]
            gold = row["answer_index"]
            direct_correct = row["baseline_prediction_index"] == gold
            spatial_correct = row["prediction_index"] == gold
        decisions.append({
            "index": row["index"],
            "condition": condition,
            "prediction": prediction,
            "correct": prediction == gold,
            "direct_correct": direct_correct,
            "spatial_correct": spatial_correct,
            "features": features,
        })
    correct = sum(item["correct"] for item in decisions)
    direct_correct = sum(item["direct_correct"] for item in decisions)
    spatial_correct = sum(item["spatial_correct"] for item in decisions)
    gate_only = sum(item["correct"] and not item["spatial_correct"] for item in decisions)
    spatial_only = sum(item["spatial_correct"] and not item["correct"] for item in decisions)
    return {
        "policy": "fixed label-free evidence/validity/margin gate",
        "dataset_status": "diagnostic_repeatedly_inspected" if dataset == "mmtom" else "frozen_transfer_no_MA_label_tuning",
        "total": len(rows),
        "selected_spatial_social": sum(item["condition"] == "spatial_social" for item in decisions),
        "selected_direct": sum(item["condition"] == "direct" for item in decisions),
        "direct": {"correct": direct_correct, "accuracy": direct_correct / len(rows), "wilson_95_ci": wilson(direct_correct, len(rows))},
        "spatial_social": {"correct": spatial_correct, "accuracy": spatial_correct / len(rows), "wilson_95_ci": wilson(spatial_correct, len(rows))},
        "gate": {"correct": correct, "accuracy": correct / len(rows), "wilson_95_ci": wilson(correct, len(rows))},
        "gate_vs_spatial": {"gate_only_correct": gate_only, "spatial_only_correct": spatial_only, "mcnemar_exact_p": exact_mcnemar(gate_only, spatial_only)},
        "feature_pass_counts": dict(feature_counts),
        "decisions": decisions,
    }


def group_audit(rows: list[dict], group_key: str, direct_key: str, spatial_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {key: paired_summary(group, direct_key, spatial_key) for key, group in sorted(groups.items())}


def representative_cases(rows: list[dict], dataset: str, limit: int = 6) -> dict:
    if dataset == "mmtom":
        direct_key, spatial_key, group_key = "direct_correct", "correct", "question_type"
        fields = ("index", "episode", "question_type", "answer", "direct_prediction", "prediction", "prob_a", "prob_b", "uncertainty_triggered")
    else:
        direct_key, spatial_key, group_key = "baseline_correct", "correct", "category"
        fields = ("index", "category", "subcategory", "answer_index", "baseline_prediction_index", "prediction_index", "structured_citation_valid", "valid_evidence_ids")
    result = {}
    for name, predicate in {
        "direct_only": lambda row: row[direct_key] and not row[spatial_key],
        "spatial_social_only": lambda row: row[spatial_key] and not row[direct_key],
        "both_wrong": lambda row: not row[direct_key] and not row[spatial_key],
    }.items():
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(key=lambda row: (str(row[group_key]), int(row["index"])))
        result[name] = [{field: row.get(field) for field in fields} for row in candidates[:limit]]
    return result


def comind_error_audit(prediction_path: Path, annotation_root: Path) -> dict:
    rows = load_jsonl(prediction_path)
    newest = {(r.get("condition"), r["task"], r["recording_id"], r["sample_id"]): r for r in rows}
    annotations = {task: load_annotations(annotation_root, task) for task in ("joint_attention", "scoia", "handover")}
    metric_contingencies = defaultdict(lambda: defaultdict(Counter))
    bbox_examples = []
    for condition, task, recording_id, sample_id in sorted(newest):
        if condition != "direct":
            continue
        direct = newest[(condition, task, recording_id, sample_id)]
        spatial = newest.get(("spatial_social", task, recording_id, sample_id))
        if not spatial or direct.get("error") or spatial.get("error"):
            continue
        gt = annotations[task].get(recording_id, {}).get(sample_id)
        if not isinstance(gt, dict) or (task == "handover" and gt.get("skip")):
            continue
        direct_norm, _ = normalize_prediction(task, direct["prediction"])
        spatial_norm, _ = normalize_prediction(task, spatial["prediction"])
        direct_scores = score_sample(task, gt, direct["prediction"])
        spatial_scores = score_sample(task, gt, spatial["prediction"])
        direct_norm_scores = score_sample(task, gt, direct_norm)
        spatial_norm_scores = score_sample(task, gt, spatial_norm)
        for metric in direct_scores:
            key = "both_correct" if direct_scores[metric] and spatial_scores[metric] else "direct_only" if direct_scores[metric] else "spatial_only" if spatial_scores[metric] else "both_wrong"
            metric_contingencies[task][metric][key] += 1
        bbox_metrics = [metric for metric in direct_scores if "iou" in metric]
        if len(bbox_examples) < 12 and any(direct_norm_scores[m] and not direct_scores[m] for m in bbox_metrics):
            bbox_examples.append({"task": task, "recording_id": recording_id, "sample_id": sample_id, "condition": "direct", "raw_prediction": direct["prediction"], "normalized_prediction": direct_norm, "ground_truth_bbox": gt.get("bounding_boxes", gt.get("bbox")), "raw_scores": direct_scores, "normalized_scores": direct_norm_scores})
        if len(bbox_examples) < 12 and any(spatial_norm_scores[m] and not spatial_scores[m] for m in bbox_metrics):
            bbox_examples.append({"task": task, "recording_id": recording_id, "sample_id": sample_id, "condition": "spatial_social", "raw_prediction": spatial["prediction"], "normalized_prediction": spatial_norm, "ground_truth_bbox": gt.get("bounding_boxes", gt.get("bbox")), "raw_scores": spatial_scores, "normalized_scores": spatial_norm_scores})
    rendered = {task: {metric: dict(counts) for metric, counts in metrics.items()} for task, metrics in metric_contingencies.items()}
    return {"metric_contingencies": rendered, "bbox_normalization_rescues": bbox_examples}


def run(args: argparse.Namespace) -> None:
    mmtom = load_jsonl(args.mmtom)
    ma = load_jsonl(args.ma)
    comind_summary = json.loads(args.comind_summary.read_text(encoding="utf-8"))
    audit = {
        "frozen_file_hashes_sha256": {str(path): sha256(path) for path in (args.mmtom, args.mmtom_summary, args.ma, args.ma_summary, args.comind_predictions, args.comind_summary)},
        "completion_markers": {str(path): path.exists() for path in (Path("workdirs/comind_full_train/DOWNLOAD_COMPLETE"), Path("workdirs/comind_full_train/PIPELINE_COMPLETE"), Path("external_datasets/comind/TEST_EGO_MP4_TRANSCRIPTS_COMPLETE"))},
        "mmtom": {
            "recomputed": paired_summary(mmtom, "direct_correct", "correct"),
            "by_question_type": group_audit(mmtom, "question_type", "direct_correct", "correct"),
            "representative_cases": representative_cases(mmtom, "mmtom"),
            "oracle_note": "diagnostic only; MMToM has been repeatedly inspected",
        },
        "ma_egoqa": {
            "recomputed": paired_summary(ma, "baseline_correct", "correct"),
            "by_category": group_audit(ma, "category", "baseline_correct", "correct"),
            "representative_cases": representative_cases(ma, "ma"),
            "oracle_note": "diagnostic only; not an inference-time method",
        },
        "comind": {
            "frozen_summary": comind_summary,
            "recomputed_error_audit": comind_error_audit(args.comind_predictions, args.annotations),
        },
    }
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gates = {"mmtom": evaluate_gate(mmtom, "mmtom"), "ma_egoqa": evaluate_gate(ma, "ma")}
    args.gate_output.write_text(json.dumps(gates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(args.audit_output), "gate": str(args.gate_output), "mmtom_gate_correct": gates["mmtom"]["gate"]["correct"], "ma_gate_correct": gates["ma_egoqa"]["gate"]["correct"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmtom", type=Path, default=Path("spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl"))
    parser.add_argument("--mmtom-summary", type=Path, default=Path("spatial_social_code/mmtom_dense_authoritative_gemini3pro_summary.json"))
    parser.add_argument("--ma", type=Path, default=Path("benchmarks/ma_egoqa/data/baselines/video_evidence_audit_full.jsonl"))
    parser.add_argument("--ma-summary", type=Path, default=Path("benchmarks/ma_egoqa/data/baselines/video_evidence_audit_full_summary.json"))
    parser.add_argument("--comind-predictions", type=Path, default=Path("workdirs/comind_full_train/predictions.jsonl"))
    parser.add_argument("--comind-summary", type=Path, default=Path("workdirs/comind_full_train/comparison_paired.json"))
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
