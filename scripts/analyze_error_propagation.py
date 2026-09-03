#!/usr/bin/env python3
"""Cross-environment error propagation and uncertainty audit.

This is a post-hoc diagnostic.  It never calls a model and never reads hidden
test labels.  Where an intermediate representation has no gold annotation,
the output calls the measured quantity a proxy rather than a true stage error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def accuracy(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return {"n": len(values), "correct": sum(values), "rate": ratio(sum(values), len(values))}


def transition(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if all(row.get(field) is not None for field in fields):
            counts["→".join("T" if row[field] else "F" for field in fields)] += 1
    return dict(sorted(counts.items()))


def stage_change(rows: list[dict[str, Any]], before: str, after: str) -> dict[str, Any]:
    before_wrong = [row for row in rows if not row[before]]
    before_right = [row for row in rows if row[before]]
    rescues = sum(bool(row[after]) for row in before_wrong)
    regressions = sum(not bool(row[after]) for row in before_right)
    return {
        "input_errors": len(before_wrong), "errors_corrected": rescues,
        "error_correction_rate": ratio(rescues, len(before_wrong)),
        "input_correct": len(before_right), "new_errors_introduced": regressions,
        "regression_rate": ratio(regressions, len(before_right)),
        "net_correct": rescues - regressions,
    }


def calibration(confidences: list[float], outcomes: list[bool], bins: int = 10) -> dict[str, Any]:
    assert len(confidences) == len(outcomes)
    rows = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidences) if low <= value < high or (index == bins - 1 and value == 1.0)]
        if not selected:
            continue
        avg_conf = mean(confidences[i] for i in selected)
        avg_acc = mean(outcomes[i] for i in selected)
        ece += len(selected) / len(confidences) * abs(avg_conf - avg_acc)
        rows.append({"low": low, "high": high, "n": len(selected), "mean_confidence": avg_conf, "accuracy": avg_acc})
    return {
        "n": len(confidences),
        "ece_10_bin": ece,
        "brier_correctness": mean((confidence - int(outcome)) ** 2 for confidence, outcome in zip(confidences, outcomes)),
        "bins": rows,
        "warning": "Model-reported/derived confidence; calibration does not establish epistemic certainty.",
    }


def contains_unknown(value: Any) -> bool:
    return bool(re.search(r"\bunknown\b|UNKNOWN_", str(value or ""), flags=re.IGNORECASE))


def trace_has_confidence(value: Any) -> bool:
    for line in str(value or "").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("values"), dict) and "confidence" in record["values"]:
            return True
    return False


def short(value: Any, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def analyze_mmtom(cases: list[dict[str, Any]]) -> dict[str, Any]:
    path = ROOT / "spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl"
    rows = read_jsonl(path)
    confidences = [max(float(row["prob_a"]), float(row["prob_b"])) for row in rows]
    outcomes = [bool(row["correct"]) for row in rows]
    triggered = [row for row in rows if row.get("uncertainty_triggered")]
    primary_wrong = [row for row in rows if not row.get("primary_correct")]
    rescues = [row for row in triggered if not row.get("primary_correct") and row.get("correct")]
    regressions = [row for row in triggered if row.get("primary_correct") and not row.get("correct")]
    spatial_unknown = [contains_unknown(row.get("spatial_code")) for row in rows]
    social_unknown = [contains_unknown(row.get("social_code")) for row in rows]

    for row in rescues[:5]:
        cases.append({"environment": "MMToM", "case_type": "uncertainty_rescue", "id": row["index"], "episode": row["episode"], "question_type": row["question_type"], "direct": row["direct_prediction"], "primary": row["primary_prediction"], "final": row["prediction"], "gold": row["answer"], "confidence": max(row["prob_a"], row["prob_b"]), "spatial_code_excerpt": short(row["spatial_code"]), "social_code_excerpt": short(row["social_code"])})
    for row in regressions[:5]:
        cases.append({"environment": "MMToM", "case_type": "correct_intermediate_broken_by_refinement", "id": row["index"], "episode": row["episode"], "primary": row["primary_prediction"], "final": row["prediction"], "gold": row["answer"], "confidence": max(row["prob_a"], row["prob_b"]), "spatial_code_excerpt": short(row["spatial_code"]), "social_code_excerpt": short(row["social_code"])})

    unknown_groups = {}
    for label, flags in (("spatial", spatial_unknown), ("social", social_unknown)):
        unknown_groups[label] = {
            "with_unknown": {"n": sum(flags), "accuracy": ratio(sum(ok and flag for ok, flag in zip(outcomes, flags)), sum(flags))},
            "without_unknown": {"n": len(flags) - sum(flags), "accuracy": ratio(sum(ok and not flag for ok, flag in zip(outcomes, flags)), len(flags) - sum(flags))},
        }
    return {
        "status": "main result; 600 questions",
        "source": str(path.relative_to(ROOT)),
        "stages": {"direct": accuracy(rows, "direct_correct"), "primary_code_answer": accuracy(rows, "primary_correct"), "final_after_uncertainty": accuracy(rows, "correct")},
        "propagation_direct_primary_final": transition(rows, ("direct_correct", "primary_correct", "correct")),
        "stage_changes": {"direct_to_primary": stage_change(rows, "direct_correct", "primary_correct"), "primary_to_final": stage_change(rows, "primary_correct", "correct")},
        "uncertainty_gate": {
            "triggered": len(triggered),
            "primary_errors": len(primary_wrong),
            "errors_flagged": sum(not row["primary_correct"] for row in triggered),
            "error_detection_precision": ratio(sum(not row["primary_correct"] for row in triggered), len(triggered)),
            "error_detection_recall": ratio(sum(not row["primary_correct"] for row in triggered), len(primary_wrong)),
            "wrong_to_right": len(rescues),
            "right_to_wrong": len(regressions),
            "net_correct": len(rescues) - len(regressions),
        },
        "final_calibration": calibration(confidences, outcomes),
        "high_confidence_wrong": {">=0.8": sum(not ok and conf >= 0.8 for conf, ok in zip(confidences, outcomes)), ">=0.9": sum(not ok and conf >= 0.9 for conf, ok in zip(confidences, outcomes))},
        "unknown_semantics": unknown_groups,
        "known_but_final_wrong": {"strict_observable_proxy": len(regressions), "definition": "Primary answer was exactly correct, but uncertainty refinement changed it to a wrong answer."},
        "first_failure_interpretation": "No gold labels exist for individual Spatial/Social facts. Unknown incidence and answer-stage transitions are observable proxies, not true perception-layer accuracy.",
    }


def evidence_count(row: dict[str, Any]) -> int:
    return len(row.get("valid_evidence_ids") or [])


def analyze_maegoqa(cases: list[dict[str, Any]]) -> dict[str, Any]:
    path = ROOT / "MA-EgoQA/data/baselines/video_evidence_audit_full.jsonl"
    rows = read_jsonl(path)
    triggered = [row for row in rows if row.get("audit_triggered")]
    rescues = [row for row in triggered if not row.get("primary_correct") and row.get("correct")]
    regressions = [row for row in triggered if row.get("primary_correct") and not row.get("correct")]
    evidence_bands = [("0", lambda n: n == 0), ("1", lambda n: n == 1), ("2", lambda n: n == 2), (">=3", lambda n: n >= 3)]
    evidence_result = {}
    for label, predicate in evidence_bands:
        selected = [row for row in rows if predicate(evidence_count(row))]
        evidence_result[label] = {"n": len(selected), "accuracy": ratio(sum(bool(row["correct"]) for row in selected), len(selected))}
    for row in rescues[:5]:
        cases.append({"environment": "MA-EgoQA", "case_type": "audit_rescue", "id": row["index"], "category": row["category"], "subcategory": row["subcategory"], "direct": row["baseline_prediction_index"], "primary": row["primary_prediction_index"], "final": row["prediction_index"], "gold": row["answer_index"], "evidence_ids": row.get("valid_evidence_ids"), "spatial_code_excerpt": short(row["spatial_code"]), "social_code_excerpt": short(row["social_code"])})
    for row in regressions[:5]:
        cases.append({"environment": "MA-EgoQA", "case_type": "correct_intermediate_broken_by_audit", "id": row["index"], "category": row["category"], "primary": row["primary_prediction_index"], "final": row["prediction_index"], "gold": row["answer_index"], "evidence_ids": row.get("valid_evidence_ids"), "spatial_code_excerpt": short(row["spatial_code"]), "social_code_excerpt": short(row["social_code"])})
    valid = [row for row in rows if row.get("structured_citation_valid")]
    invalid = [row for row in rows if not row.get("structured_citation_valid")]
    strong_proxy = [row for row in rows if row.get("structured_citation_valid") and evidence_count(row) >= 2 and not contains_unknown(row.get("spatial_code")) and not contains_unknown(row.get("social_code"))]
    return {
        "status": "main result; 1,741 questions",
        "source": str(path.relative_to(ROOT)),
        "stages": {"direct": accuracy(rows, "baseline_correct"), "structured_primary": accuracy(rows, "primary_correct"), "final_after_audit": accuracy(rows, "correct")},
        "propagation_direct_primary_final": transition(rows, ("baseline_correct", "primary_correct", "correct")),
        "stage_changes": {"direct_to_primary": stage_change(rows, "baseline_correct", "primary_correct"), "primary_to_final": stage_change(rows, "primary_correct", "correct")},
        "audit_gate": {"triggered": len(triggered), "primary_errors": sum(not row["primary_correct"] for row in rows), "errors_flagged": sum(not row["primary_correct"] for row in triggered), "error_detection_precision": ratio(sum(not row["primary_correct"] for row in triggered), len(triggered)), "error_detection_recall": ratio(sum(not row["primary_correct"] for row in triggered), sum(not row["primary_correct"] for row in rows)), "wrong_to_right": len(rescues), "right_to_wrong": len(regressions), "wrong_unchanged": sum(not row["primary_correct"] and not row["correct"] for row in triggered), "net_correct": len(rescues) - len(regressions)},
        "evidence_count_vs_accuracy": evidence_result,
        "citation_validity": {"valid": {"n": len(valid), "accuracy": ratio(sum(row["correct"] for row in valid), len(valid))}, "invalid": {"n": len(invalid), "accuracy": ratio(sum(row["correct"] for row in invalid), len(invalid))}},
        "unknown_semantics": {
            stage: {
                "with_unknown": {"n": sum(contains_unknown(row[field]) for row in rows), "accuracy": ratio(sum(row["correct"] and contains_unknown(row[field]) for row in rows), sum(contains_unknown(row[field]) for row in rows))},
                "without_unknown": {"n": sum(not contains_unknown(row[field]) for row in rows), "accuracy": ratio(sum(row["correct"] and not contains_unknown(row[field]) for row in rows), sum(not contains_unknown(row[field]) for row in rows))},
            }
            for stage, field in (("spatial", "spatial_code"), ("social", "social_code"))
        },
        "known_but_final_wrong": {"strict_observable_proxy": len(regressions), "strong_evidence_proxy_n": len(strong_proxy), "strong_evidence_proxy_wrong": sum(not row["correct"] for row in strong_proxy), "definition": "Strong proxy = valid citation, >=2 evidence IDs, and no literal unknown in either code. It does not prove the cited facts are correct."},
        "instrumentation_gaps": {"missing_social_confidence": sum(not trace_has_confidence(row.get("social_trace")) for row in rows), "repair_triggered": sum(bool(row.get("repair_triggered")) for row in rows), "claim_coverage_non_null": sum(row.get("claim_coverage") is not None for row in rows)},
        "first_failure_interpretation": "Evidence absence/invalid citation can be localized. Semantic correctness of Spatial/Social facts cannot be localized without fact-level annotation.",
    }


def action_type(action: str) -> str:
    match = re.match(r"\s*([A-Za-z_]+)", action or "")
    return match.group(1) if match else "invalid"


def action_failed(item: dict[str, Any]) -> bool:
    result = str(item.get("result") or item.get("observation") or "").lower()
    return "unexpected failure" in result or result.startswith("failed") or "execution failed" in result


def habitat_log(summary_path: Path) -> dict[str, Any] | None:
    logs = list(summary_path.parent.rglob("planner-log/*.json"))
    if not logs:
        return None
    return json.loads(max(logs, key=lambda path: path.stat().st_mtime).read_text(encoding="utf-8"))


def collect_habitat() -> list[dict[str, Any]]:
    roots: list[tuple[int, str, Path]] = [
        # EnactToM calls private-information execution "standard" and its
        # shared-information comparison "baseline".  The former is the Direct
        # condition paired with our private agent-indexed code.
        (1, "direct", ROOT / "workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_official_baselines_full12/standard"),
    ]
    first = ROOT / "workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_v2/results"
    roots += [(1, condition, first / condition) for condition in ("physical_only", "social_only", "physical_social")]
    repeats = ROOT / "workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_repeats/results"
    roots += [(seed, condition, repeats / f"repeat_{seed}" / condition) for seed in (2, 3) for condition in ("direct", "physical_only", "social_only", "physical_social")]
    rows = []
    for seed, condition, root in roots:
        for path in sorted(root.rglob("benchmark_summary.json")) if root.exists() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = (payload.get("results") or [{}])[0]
            log = habitat_log(path) or result
            history = log.get("action_history") or []
            failures = [item for item in history if action_failed(item)]
            actions = [str(item.get("action") or item.get("action_taken") or "") for item in history]
            failed_actions = [str(item.get("action") or item.get("action_taken") or "") for item in failures]
            instruction = "\n".join(str(value) for value in (log.get("instruction") or {}).values())
            evaluation = log.get("evaluation") or result.get("evaluation") or {}
            rows.append({
                "seed": seed, "condition": condition, "task_id": log.get("task_id") or result.get("task_id"),
                "success": bool(log.get("success", result.get("success"))), "progress": float(evaluation.get("percent_complete", log.get("percent_complete", 0.0)) or 0.0),
                "timeout": not bool(log.get("done", result.get("done"))) and not bool(log.get("episode_over", result.get("episode_over"))),
                "steps": int(log.get("steps", result.get("steps", 0)) or 0), "action_count": len(history), "failed_action_count": len(failures),
                "first_failed_action": action_type(failed_actions[0]) if failed_actions else None,
                "repeated_failed_actions": sum(count - 1 for count in Counter(failed_actions).values() if count > 1),
                "unknown_actions": sum(contains_unknown(action) for action in actions), "unknown_prompt_mentions": len(re.findall(r"UNKNOWN_", instruction)),
                "summary_path": str(path.relative_to(ROOT)),
            })
    return rows


def analyze_habitat(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = collect_habitat()
    conditions = {}
    for condition in ("direct", "physical_only", "social_only", "physical_social"):
        selected = [row for row in rows if row["condition"] == condition]
        first_failures = Counter(row["first_failed_action"] for row in selected if row["first_failed_action"])
        conditions[condition] = {
            "n": len(selected), "success": sum(row["success"] for row in selected), "success_rate": ratio(sum(row["success"] for row in selected), len(selected)),
            "mean_completion": mean(row["progress"] for row in selected), "timeouts": sum(row["timeout"] for row in selected), "mean_steps": mean(row["steps"] for row in selected),
            "runs_with_action_failure": sum(row["failed_action_count"] > 0 for row in selected), "failed_actions": sum(row["failed_action_count"] for row in selected),
            "first_failed_action_type": dict(first_failures), "repeated_failed_actions": sum(row["repeated_failed_actions"] for row in selected),
            "unknown_actions": sum(row["unknown_actions"] for row in selected), "mean_unknown_prompt_mentions": mean(row["unknown_prompt_mentions"] for row in selected),
            "zero_unknown_but_failed": sum(row["unknown_prompt_mentions"] == 0 and not row["success"] for row in selected),
        }
    indexed = {(row["seed"], row["task_id"], row["condition"]): row for row in rows}
    paired = Counter()
    progress_direction = Counter()
    deltas = []
    for seed, task_id, _ in {(row["seed"], row["task_id"], "") for row in rows}:
        direct = indexed.get((seed, task_id, "direct"))
        full = indexed.get((seed, task_id, "physical_social"))
        if not direct or not full:
            continue
        paired[("T" if direct["success"] else "F") + "→" + ("T" if full["success"] else "F")] += 1
        delta = full["progress"] - direct["progress"]
        deltas.append(delta)
        progress_direction["improved" if delta > 1e-12 else "degraded" if delta < -1e-12 else "unchanged"] += 1
        if not direct["success"] and full["success"] and len([case for case in cases if case["environment"] == "Habitat" and case["case_type"] == "direct_failure_rescued_by_ps"]) < 5:
            cases.append({"environment": "Habitat", "case_type": "direct_failure_rescued_by_ps", "seed": seed, "task_id": task_id, "direct_completion": direct["progress"], "ps_completion": full["progress"], "direct_first_failed_action": direct["first_failed_action"], "ps_first_failed_action": full["first_failed_action"]})
    return {
        "status": "main controlled task set; 12 tasks × 3 seeds × 4 conditions",
        "conditions": conditions,
        "paired_direct_to_physical_social": dict(sorted(paired.items())),
        "paired_completion_direction": dict(progress_direction),
        "paired_mean_completion_delta": mean(deltas) if deltas else None,
        "first_failure_interpretation": "The first failed simulator action is directly observed. A timeout with zero UNKNOWN markers is only a known-state planning-failure proxy; prompts are not fact-level gold annotations.",
    }


def finite_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def xyxy_to_xywh(value: Any) -> list[float] | None:
    box = finite_box(value)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def ego_to_joint(value: Any, role: str) -> list[float] | None:
    box = xyxy_to_xywh(value)
    if box is None:
        return None
    x, y, width, height = box
    return [x / 2 + (704 if role == "helper" else 0), y / 2, width / 2, height / 2]


def iou(first: Any, second: Any) -> float:
    a, b = finite_box(first), finite_box(second)
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def analyze_comind(cases: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_path = ROOT / "workdirs/comind_optimization_v2/full_train_oof_predictions.jsonl"
    grounding_path = ROOT / "workdirs/comind_optimization_v2/two_stage_grounding_full_predictions.jsonl"
    gate_path = ROOT / "workdirs/comind_optimization_v2/two_stage_full_oof_gate.json"
    rows = read_jsonl(prediction_path)
    ground_rows = read_jsonl(grounding_path)
    common_keys = {(row["task"], str(row["recording_id"]), str(row["sample_id"])) for row in rows}
    original_by_key = {(row["task"], str(row["recording_id"]), str(row["sample_id"])): row for row in rows}
    annotations = {}
    gate_summary = json.loads(gate_path.read_text(encoding="utf-8"))
    filenames = {"joint_attention": "dataset_joint_attention_consolidated.json", "scoia": "dataset_scoia_consolidated.json", "handover": "dataset_handover_consolidated.json"}
    for task, filename in filenames.items():
        annotations[task] = json.loads((ROOT / "external_datasets/comind/annotations" / filename).read_text(encoding="utf-8"))["data"]

    metric_names = {"joint_attention": ("left_iou_at_0_5", "right_iou_at_0_5"), "scoia": ("iou_at_0_5",), "handover": ("iou_at_0_5",)}
    contingencies: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        task = row["task"]
        for metric in metric_names[task]:
            direct = bool(row.get("scores", {}).get("direct", {}).get(metric))
            spatial = bool(row.get("scores", {}).get("spatial_social", {}).get(metric))
            contingencies[f"{task}.{metric}"][("T" if spatial else "F") + "→" + ("T" if direct else "F")] += 1

    scores, outcomes, per_task = [], [], defaultdict(lambda: {"n": 0, "hit": 0})
    grounding_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ground_rows:
        task, rid, sid = row.get("task"), str(row.get("recording_id")), str(row.get("sample_id"))
        if (task, rid, sid) not in common_keys:
            continue
        gt = annotations.get(task, {}).get(rid, {}).get(sid)
        pred = row.get("prediction") or {}
        if not isinstance(gt, dict) or (task == "handover" and gt.get("skip")):
            continue
        boxes: list[tuple[Any, Any, float, str]] = []
        if task == "joint_attention":
            props = row.get("sam3_proposals") or {}
            gold_boxes = gt.get("bounding_boxes") or []
            for role, key, index in (("leader", "left_bbox_xyxy", 0), ("helper", "right_bbox_xyxy", 1)):
                proposal_scores = [float(item["score"]) for item in props.get(role, []) if item.get("score") is not None]
                boxes.append((ego_to_joint(pred.get(key), role), gold_boxes[index] if index < len(gold_boxes) else None, max(proposal_scores) if proposal_scores else 0.0, role))
        else:
            role = "helper" if task == "scoia" else pred.get("selected_view")
            proposal_scores = [float(item["score"]) for item in (row.get("sam3_proposals") or {}).get(role, []) if item.get("score") is not None]
            converted = xyxy_to_xywh(pred.get("bbox_xyxy")) if task == "scoia" else ego_to_joint(pred.get("bbox_xyxy"), role)
            boxes.append((converted, gt.get("bbox"), max(proposal_scores) if proposal_scores else 0.0, str(role)))
        for predicted_box, gold_box, score, role in boxes:
            hit = iou(predicted_box, gold_box) > 0.5
            scores.append(score); outcomes.append(hit)
            key = f"{task}.{role}"
            per_task[key]["n"] += 1; per_task[key]["hit"] += hit
            metric = "left_iou_at_0_5" if task == "joint_attention" and role == "leader" else "right_iou_at_0_5" if task == "joint_attention" else "iou_at_0_5"
            original_hit = bool(original_by_key[(task, rid, sid)].get("scores", {}).get("spatial_social", {}).get(metric))
            grounding_transitions[f"{task}.{metric}"][("T" if original_hit else "F") + "→" + ("T" if hit else "F")] += 1
            if score >= 0.8 and not hit and len([case for case in cases if case["environment"] == "CoMind" and case["case_type"] == "high_score_grounding_miss"]) < 5:
                cases.append({"environment": "CoMind", "case_type": "high_score_grounding_miss", "task": task, "recording_id": rid, "sample_id": sid, "view": role, "sam_score": score, "iou": iou(predicted_box, gold_box), "predicted_box": predicted_box, "gold_box": gold_box})
    return {
        "status": "diagnostic only: labeled train, 5-fold recording-heldout OOF; not hidden test",
        "sources": [str(prediction_path.relative_to(ROOT)), str(grounding_path.relative_to(ROOT)), str(gate_path.relative_to(ROOT))],
        "raw_spatial_social_vs_direct_bbox_correctness": {key: dict(value) for key, value in sorted(contingencies.items())},
        "sam_grounding": {"overall": {"n": len(outcomes), "iou_at_0.5": ratio(sum(outcomes), len(outcomes))}, "by_task_view": {key: {**value, "rate": ratio(value["hit"], value["n"])} for key, value in sorted(per_task.items())}, "raw_spatial_social_to_sam_recomputed": {key: dict(value) for key, value in sorted(grounding_transitions.items())}, "published_oof_gate_vs_spatial_social": gate_summary.get("paired_gate_vs_spatial_social"), "score_calibration": calibration(scores, outcomes)},
        "first_failure_interpretation": "BBox has gold labels, so grounding is a true localized error. Other Social fields have field-level labels in the OOF scorer, but the SAM score applies only to grounding.",
    }


def analyze_socialnav(cases: list[dict[str, Any]]) -> dict[str, Any]:
    root = ROOT / "workdirs/socialnav_sub/router_v3_heldout/jdcloud_pilot"
    rows = read_jsonl(root / "predictions.jsonl")
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["scene_id"], row["question_key"])][row["condition"]] = row
    transitions = Counter()
    agreements = defaultdict(list)
    consensus_bands: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for key, pair in by_key.items():
        if "direct" not in pair or "visual_predicted_ps" not in pair:
            continue
        flags = []
        for condition in ("direct", "visual_predicted_ps"):
            row = pair[condition]
            distribution = row.get("human_distribution") or {}
            mode = max(distribution, key=distribution.get) if distribution else None
            correct = row.get("answer") == mode
            flags.append(correct)
            agreements[condition].append(float(distribution.get(row.get("answer"), 0.0)))
            consensus = max(distribution.values()) if distribution else 0.0
            band = "<0.6" if consensus < 0.6 else "0.6-0.79" if consensus < 0.8 else ">=0.8"
            consensus_bands[condition][band].append(correct)
        transitions[("T" if flags[0] else "F") + "→" + ("T" if flags[1] else "F")] += 1
        if flags[0] and not flags[1] and len([case for case in cases if case["environment"] == "SocialNav-SUB"]) < 5:
            cases.append({"environment": "SocialNav-SUB", "case_type": "direct_correct_code_wrong", "scene_id": key[0], "question_key": key[1], "question": pair["direct"]["question"], "direct": pair["direct"]["answer"], "physical_social": pair["visual_predicted_ps"]["answer"], "human_distribution": pair["direct"]["human_distribution"]})
    compiled = read_jsonl(root / "visual_compiled.jsonl")
    return {
        "status": "held-out pilot; 120 questions",
        "source": str(root.relative_to(ROOT)),
        "paired_human_mode_transition": dict(sorted(transitions.items())),
        "mean_human_probability_agreement": {condition: mean(values) for condition, values in agreements.items()},
        "human_consensus_vs_mode_accuracy": {condition: {band: {"n": len(values), "accuracy": mean(values)} for band, values in sorted(bands.items())} for condition, bands in consensus_bands.items()},
        "compiler_valid": {"n": len(compiled), "valid": sum(bool(row.get("valid")) for row in compiled)},
        "uncertainty_note": "Human answer distributions quantify label ambiguity. They are not model confidence. No fact-level code labels are available, so code-field errors remain proxies.",
    }


def analyze_storyvideo(cases: list[dict[str, Any]]) -> dict[str, Any]:
    episode_roots = []
    for version in ("indexed_v3_heldout", "indexed_v4_heldout2", "indexed_v5_heldout3"):
        episode_roots.extend(sorted((ROOT / "workdirs/storyvideoqa" / version).glob("BigBang-*")))
    direct, code = {}, {}
    for episode_root in episode_roots:
        for row in read_jsonl(episode_root / "jdcloud/predictions.jsonl"):
            direct[(row["episode"], row["id"])] = row
        for row in read_jsonl(episode_root / "indexed/predictions.jsonl"):
            code[(row["episode"], row["id"])] = row
    transitions = Counter()
    view_bands = defaultdict(list)
    for key in sorted(direct.keys() & code.keys()):
        first, second = direct[key], code[key]
        transitions[("T" if first["correct"] else "F") + "→" + ("T" if second["correct"] else "F")] += 1
        count = len(second.get("view_event_ids") or [])
        band = "0" if count == 0 else "1-9" if count < 10 else "10-39" if count < 40 else ">=40"
        view_bands[band].append(bool(second["correct"]))
        if first["correct"] and not second["correct"] and len([case for case in cases if case["environment"] == "StoryVideoQA"]) < 5:
            cases.append({"environment": "StoryVideoQA", "case_type": "direct_correct_code_wrong", "episode": key[0], "id": key[1], "question": second["question"], "gold": second["gold"], "direct": first["prediction"], "story_code": second["prediction"], "view_event_count": count, "view_event_ids": second.get("view_event_ids")})
    return {
        "status": "held-out pilot; 12 episodes, 240 questions",
        "paired_direct_to_story_code": dict(sorted(transitions.items())),
        "stages": {"direct": {"n": len(direct), "correct": sum(row["correct"] for row in direct.values()), "rate": ratio(sum(row["correct"] for row in direct.values()), len(direct))}, "story_code": {"n": len(code), "correct": sum(row["correct"] for row in code.values()), "rate": ratio(sum(row["correct"] for row in code.values()), len(code))}},
        "retrieved_event_count_vs_accuracy": {band: {"n": len(values), "accuracy": mean(values)} for band, values in sorted(view_bands.items())},
        "uncertainty_note": "No model probabilities or fact-level code labels are stored. Event count measures retrieval coverage, not evidence correctness or confidence.",
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def render_report(summary: dict[str, Any]) -> str:
    mm, ma, habitat, co, sn, story = (summary[key] for key in ("mmtom", "ma_egoqa", "habitat", "comind", "socialnav_sub", "storyvideoqa"))
    lines = [
        "# Spatial–Social Code 误差传播与不确定性分析", "",
        "> 本报告由 `scripts/analyze_error_propagation.py` 从冻结结果自动生成。中间层没有人工真值时只报告代理指标，不把模型自报 confidence 写成真实确定性。", "",
        "## 一页结论", "",
        f"- **MMToM**：Direct {pct(mm['stages']['direct']['rate'])} → primary {pct(mm['stages']['primary_code_answer']['rate'])} → final {pct(mm['stages']['final_after_uncertainty']['rate'])}。不确定性门修复 {mm['uncertainty_gate']['wrong_to_right']} 题、破坏 {mm['uncertainty_gate']['right_to_wrong']} 题，净增 {mm['uncertainty_gate']['net_correct']}；但仍有 {mm['high_confidence_wrong']['>=0.9']} 个置信度≥0.9的错误。",
        f"- **MA-EgoQA**：Direct {pct(ma['stages']['direct']['rate'])} → structured {pct(ma['stages']['structured_primary']['rate'])} → audit final {pct(ma['stages']['final_after_audit']['rate'])}。audit 只净增 {ma['audit_gate']['net_correct']} 题，说明主要增益来自结构化回答之前，而不是事后审计。",
        f"- **Habitat**：P+S 的成功率 {pct(habitat['conditions']['physical_social']['success_rate'])}、平均完成度 {pct(habitat['conditions']['physical_social']['mean_completion'])}；相对 Direct 的配对平均完成度变化 {100 * habitat['paired_mean_completion_delta']:+.2f} pp。这里可直接定位首个 simulator action failure。",
        f"- **CoMind**：SAM grounding 的 {co['sam_grounding']['overall']['n']} 个 bbox/view 判断中 IoU@0.5 为 {pct(co['sam_grounding']['overall']['iou_at_0.5'])}。这是最清楚的真值定位：原始文本 bbox 几乎全错，SAM 大幅修复，但 grounding 仍是主要瓶颈。",
        f"- **SocialNav-SUB / StoryVideoQA**：仍是负向或中性 pilot。SocialNav P+S agreement {pct(sn['mean_human_probability_agreement']['visual_predicted_ps'])} vs Direct {pct(sn['mean_human_probability_agreement']['direct'])}；Story Code {pct(story['stages']['story_code']['rate'])} vs Direct {pct(story['stages']['direct']['rate'])}。", "",
        "## MMToM：答案阶段传播和校准", "",
        "|阶段|正确/总数|准确率|", "|---|---:|---:|",
    ]
    for label, key in (("Direct", "direct"), ("Code primary", "primary_code_answer"), ("Uncertainty final", "final_after_uncertainty")):
        item = mm["stages"][key]; lines.append(f"|{label}|{item['correct']}/{item['n']}|{pct(item['rate'])}|")
    gate = mm["uncertainty_gate"]
    mm_dp, mm_pf = mm["stage_changes"]["direct_to_primary"], mm["stage_changes"]["primary_to_final"]
    lines += ["", f"传播拆解：Code primary 修复 Direct 错误 {mm_dp['errors_corrected']}/{mm_dp['input_errors']}（{pct(mm_dp['error_correction_rate'])}），同时把原本正确的答案改错 {mm_dp['new_errors_introduced']}/{mm_dp['input_correct']}（{pct(mm_dp['regression_rate'])}），净增 {mm_dp['net_correct']}。随后 uncertainty 阶段修复 {mm_pf['errors_corrected']}/{mm_pf['input_errors']}，又破坏 {mm_pf['new_errors_introduced']} 个正确答案。", "", f"不确定性 gate 触发 {gate['triggered']} 题；错误检测 precision={pct(gate['error_detection_precision'])}，recall={pct(gate['error_detection_recall'])}。ECE={mm['final_calibration']['ece_10_bin']:.4f}，Brier={mm['final_calibration']['brier_correctness']:.4f}。这里最重要的风险是 **{mm['high_confidence_wrong']['>=0.8']} 个 confidence≥0.8 的错误**。", "", f"Spatial Code 中 literal unknown 为 {mm['unknown_semantics']['spatial']['with_unknown']['n']}；Social Code 保留 unknown 的 {mm['unknown_semantics']['social']['with_unknown']['n']} 题准确率为 {pct(mm['unknown_semantics']['social']['with_unknown']['accuracy'])}，不含 unknown 的准确率为 {pct(mm['unknown_semantics']['social']['without_unknown']['accuracy'])}。这支持“允许 unknown 比强行填值更安全”，也暴露 Spatial 层过度声明。", "",
        "## MA-EgoQA：证据覆盖、引用和审计", "", "|有效 evidence 数|题数|准确率|", "|---|---:|---:|"]
    for label, item in ma["evidence_count_vs_accuracy"].items(): lines.append(f"|{label}|{item['n']}|{pct(item['accuracy'])}|")
    ma_dp, ma_pf = ma["stage_changes"]["direct_to_primary"], ma["stage_changes"]["primary_to_final"]
    lines += ["", f"传播拆解：structured stage 修复 Direct 错误 {ma_dp['errors_corrected']}/{ma_dp['input_errors']}，但引入 {ma_dp['new_errors_introduced']}/{ma_dp['input_correct']} 个新错，净增 {ma_dp['net_correct']}。audit 只修复 {ma_pf['errors_corrected']} 个 primary 错误并破坏 {ma_pf['new_errors_introduced']} 个正确答案；其错误检测 precision={pct(ma['audit_gate']['error_detection_precision'])}、recall={pct(ma['audit_gate']['error_detection_recall'])}。", "", f"citation valid 时准确率 {pct(ma['citation_validity']['valid']['accuracy'])}，invalid 时 {pct(ma['citation_validity']['invalid']['accuracy'])}。但“引用有效”仅表示引用 ID 合法，不表示证据语义正确。强证据代理集合有 {ma['known_but_final_wrong']['strong_evidence_proxy_n']} 题，其中 {ma['known_but_final_wrong']['strong_evidence_proxy_wrong']} 题最终仍错；不能把它直接叫作“模型已经知道答案”。", "", f"Instrumentation 缺口：{ma['instrumentation_gaps']['missing_social_confidence']} 题 Social trace 没有 confidence；repair 实际触发 {ma['instrumentation_gaps']['repair_triggered']} 次；claim coverage 非空 {ma['instrumentation_gaps']['claim_coverage_non_null']} 题。", "",
        "## Habitat：动作级错误传播", "", "|条件|成功率|平均完成度|有动作失败的 run|失败动作数|timeout|", "|---|---:|---:|---:|---:|---:|"]
    for condition in ("direct", "physical_only", "social_only", "physical_social"):
        item = habitat["conditions"][condition]
        lines.append(f"|{condition}|{pct(item['success_rate'])}|{pct(item['mean_completion'])}|{item['runs_with_action_failure']}/{item['n']}|{item['failed_actions']}|{item['timeouts']}|")
    lines += ["", f"Direct→P+S success 配对转移：`{habitat['paired_direct_to_physical_social']}`；完成度逐 run 比较为 `{habitat['paired_completion_direction']}`。P+S 首个失败动作类型：`{habitat['conditions']['physical_social']['first_failed_action_type']}`。P+S 中执行了含 UNKNOWN 的动作 {habitat['conditions']['physical_social']['unknown_actions']} 次，说明表示中的未决变量确实会传播到 action 层。", "",
        "## CoMind：可被真值定位的 grounding 误差", "", "|任务/视角|样本|IoU@0.5|", "|---|---:|---:|"]
    for key, item in co["sam_grounding"]["by_task_view"].items(): lines.append(f"|{key}|{item['n']}|{pct(item['rate'])}|")
    lines += ["", f"SAM proposal score 对最终 bbox hit 的 ECE={co['sam_grounding']['score_calibration']['ece_10_bin']:.4f}。这个 score 明显不是校准后的正确概率；高 score 错框案例已写入 cases 文件。完整的 raw SSC→SAM 配对 contingency（含所有 Social 字段）保存在汇总 JSON 的 `published_oof_gate_vs_spatial_social`。", "",
        "## 两个扩展 pilot", "", f"SocialNav-SUB 的 Direct→P+S human-mode 转移为 `{sn['paired_human_mode_transition']}`。由于只有问题级人类分布，没有 code 字段真值，只能定位“用了 code 后答案变坏”，不能断言究竟是哪条 Spatial/Social fact 错。", "", f"StoryVideoQA 的 Direct→Story Code 转移为 `{story['paired_direct_to_story_code']}`。检索 event 数与准确率是覆盖代理，不是证据正确率。", "",
        "## 论文中建议采用的错误分类", "",
        "1. **Representation omission/unknown**：应有事实缺失或保留 unknown。", "2. **Representation hallucination**：需要 fact-level 人工标注才能严格确认；现阶段不要用最终答错反推 code 错。", "3. **Grounding error**：CoMind bbox 有真值，可以严格定位。", "4. **Reasoning/fusion error**：中间答案正确而后处理改错，或强证据代理下仍答错。", "5. **Execution error**：Habitat simulator 明确返回失败。", "6. **Calibration failure**：高置信错误、gate 对错误的 precision/recall 不足。", "",
        "## 科学边界", "",
        "- MMToM/MA-EgoQA 没有逐条 Spatial/Social fact 的人工真值；报告中的 unknown、citation 和 evidence coverage 都是代理。", "- unknown 与准确率的关系是相关性分析，不能解释成 unknown 本身造成提升或下降。", "- CoMind 分析是 labeled-train recording-heldout OOF，不是官方 hidden test。", "- SocialNav-SUB 与 StoryVideoQA 是 pilot，不进入主结果主张。", "- Habitat 的首个失败动作不一定是最早的认知错误；计划错误可能在此前已经发生，只是在执行时才可观察。", "- 若论文要正式声称“知道但答不出”，下一步需要抽样人工标注 code 是否包含足够且正确的 gold facts。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/error_propagation_summary.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/error_propagation_analysis_zh.md")
    parser.add_argument("--cases", type=Path, default=ROOT / "reports/error_propagation_cases.jsonl")
    args = parser.parse_args()
    cases: list[dict[str, Any]] = []
    summary = {
        "schema_version": 1,
        "scope": "Post-hoc error propagation and uncertainty audit; no model calls",
        "mmtom": analyze_mmtom(cases),
        "ma_egoqa": analyze_maegoqa(cases),
        "habitat": analyze_habitat(cases),
        "comind": analyze_comind(cases),
        "socialnav_sub": analyze_socialnav(cases),
        "storyvideoqa": analyze_storyvideo(cases),
    }
    for path in (args.output, args.report, args.cases): path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(summary), encoding="utf-8")
    args.cases.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(json.dumps({"summary": str(args.output), "report": str(args.report), "cases": str(args.cases), "case_count": len(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
