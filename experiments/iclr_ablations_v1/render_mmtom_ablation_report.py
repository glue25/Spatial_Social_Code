#!/usr/bin/env python3
"""Render the completed MMToM ablation bundle into publication artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "workdirs" / "iclr_ablations_v1"
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mcnemar(left: list[bool], right: list[bool]) -> dict:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": p_value,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    component_path = ROOT / "spatial_social_code" / "mmtom_component_ablation_offline_summary.json"
    strict_path = WORK / "summary.json"
    execution_path = WORK / "mmtom_execution_effect_summary.json"
    equal_score_path = WORK / "equal_call_direct_score.json"
    equal_predictions_path = WORK / "equal_call_direct_predictions.jsonl"
    method_predictions_path = ROOT / "spatial_social_code" / "mmtom_dense_authoritative_gemini3pro.jsonl"

    required = [component_path, strict_path, execution_path, method_predictions_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MMToM ablation bundle is incomplete: {missing}")

    component = load_json(component_path)
    strict = load_json(strict_path)
    execution = load_json(execution_path)
    method_rows = load_jsonl(method_predictions_path)
    method_by_index = {int(row["index"]): row for row in method_rows}
    if set(method_by_index) != set(range(600)):
        raise ValueError("MMToM method index set must be exactly 0..599")

    main_conditions = component["conditions"]
    paired_comparisons = component["paired_comparisons"]
    strict_conditions = strict["strict_cached_trace_ablation"]["conditions"]
    method = main_conditions["full_hybrid_dense_authoritative"]
    direct = main_conditions["direct"]
    direct_vs_method = next(
        row
        for row in paired_comparisons
        if row["left"] == "direct" and row["right"] == "full_hybrid_dense_authoritative"
    )
    equal_call = {
        "status": "not_run",
        "reason": "The existing paired Gemini Direct baseline is used; an additional resampled call-count-matched control was not requested.",
    }
    equal_comparison = None
    if equal_score_path.is_file() and equal_predictions_path.is_file():
        equal_score = load_json(equal_score_path)
        equal_rows = load_jsonl(equal_predictions_path)
        if equal_score.get("valid") != 600 or equal_score.get("total") != 600 or len(equal_rows) != 600:
            raise ValueError("equal-call Direct must contain 600 valid paired predictions")
        equal_by_index = {int(row["index"]): row for row in equal_rows}
        if set(equal_by_index) != set(range(600)):
            raise ValueError("equal-call Direct index set must be exactly 0..599")
        equal_correct = []
        method_correct = []
        for index in range(600):
            answer = str(method_by_index[index]["answer"]).lower()
            equal_correct.append(str(equal_by_index[index]["prediction"]).lower() == answer)
            method_correct.append(str(method_by_index[index]["prediction"]).lower() == answer)
        paired = mcnemar(equal_correct, method_correct)
        equal_call = equal_score
        equal_comparison = {
            "equal_call_accuracy": equal_score["accuracy"],
            "method_accuracy": method["accuracy"],
            "delta_method_minus_equal_call": method["accuracy"] - equal_score["accuracy"],
            **paired,
        }
        required.extend([equal_score_path, equal_predictions_path])

    summary = {
        "schema_version": 1,
        "benchmark": "MMToM",
        "status": "complete",
        "examples": 600,
        "main_component_ablation": main_conditions,
        "main_component_paired_comparisons": paired_comparisons,
        "strict_cached_trace_ablation": strict_conditions,
        "execution_ablation": execution,
        "equal_call_direct": equal_call,
        "equal_call_direct_vs_method": equal_comparison,
        "protocol": {
            "equal_call_logical_calls_if_run": 1550,
            "same_frozen_gemini_endpoint": True,
            "label_access_during_inference": False,
            "call_count_is_not_token_or_flop_parity": True,
            "diagnostic_not_clean_heldout": True,
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required},
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "mmtom_iclr_ablation_summary.json"
    markdown_path = REPORTS / "mmtom_iclr_ablation_results_zh.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    main_order = [
        ("direct", "Direct"),
        ("query_only", "Query only"),
        ("query_rules", "+ rules"),
        ("query_rules_spatial", "+ Spatial Code"),
        ("full_hybrid_sparse", "+ Social Code (sparse)"),
        ("full_hybrid_avg_dense", "+ dense average"),
        ("full_hybrid_dense_authoritative", "完整方法"),
    ]
    lines = [
        "# MMToM ICLR 消融结果",
        "",
        "## 主组件消融",
        "",
        "| 条件 | 正确数 | 准确率 |",
        "|---|---:|---:|",
    ]
    for key, label in main_order:
        row = main_conditions[key]
        lines.append(f"| {label} | {row['correct']}/{row['total']} | {pct(row['accuracy'])} |")

    lines.extend([
        "",
        "## 冻结 trace 表示消融",
        "",
        "| 条件 | 正确数 | 准确率 |",
        "|---|---:|---:|",
    ])
    for key, label in [
        ("spatial_only_executed", "Spatial only"),
        ("social_only_executed", "Social only"),
        ("spatial_social_executed_recomputed", "Spatial + Social"),
    ]:
        row = strict_conditions[key]
        lines.append(f"| {label} | {row['correct']}/{row['total']} | {pct(row['accuracy'])} |")

    raw = execution["conditions"]["raw_code_only"]
    executed = execution["conditions"]["executed_trace"]
    lines.extend([
        "",
        "## 执行消融与 Direct 对照",
        "",
        f"- Raw code only：{raw['correct']}/{raw['total']}（{pct(raw['accuracy'])}）。",
        f"- Executed trace：{executed['correct']}/{executed['total']}（{pct(executed['accuracy'])}）；提升 {100.0 * execution['comparison']['accuracy_delta_right_minus_left']:.2f} pp，McNemar p={execution['comparison']['mcnemar_exact_p']:.4g}。",
        f"- 已有配对 Gemini Direct：{direct['correct']}/{direct['total']}（{pct(direct['accuracy'])}）。",
        f"- 完整方法：{method['correct']}/{method['total']}（{pct(method['accuracy'])}）；提升 {100.0 * (method['accuracy'] - direct['accuracy']):.2f} pp，McNemar p={direct_vs_method['mcnemar_exact_p']:.4g}。",
        "",
        "## 结论边界",
        "",
        "组件条件使用冻结的模型输出做离线重算；已有 Direct 与完整方法使用同一冻结 Gemini 3-Pro endpoint、同一 600 题和相同视觉输入，但调用次数并未匹配。额外的 1,550-call Direct 重采样未运行，因此不能声称计算预算完全匹配。MMToM 已在开发过程中被查看，这些消融属于诊断性全量评测，不包装成未见测试集结果。推理阶段不读取标签，标签仅在预测完成后用于评分。",
        "",
    ])
    if equal_comparison is not None:
        lines[-2] = (
            f"等调用 Direct 为 {equal_call['correct']}/{equal_call['valid']}（{pct(equal_call['accuracy'])}），"
            f"完整方法高 {100.0 * equal_comparison['delta_method_minus_equal_call']:.2f} pp，"
            f"McNemar p={equal_comparison['mcnemar_exact_p']:.4g}。组件条件使用冻结输出做离线重算；"
            "MMToM 已在开发过程中被查看，因此属于诊断性全量评测。"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(json_path), "report": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
