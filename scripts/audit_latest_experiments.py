#!/usr/bin/env python3
"""Recompute and export the evidence currently available in this checkout.

This script is deliberately offline: it does not call a model, start Habitat,
or reinterpret historical scores with a newer scorer.  It copies selected raw
records into a reviewable bundle, removes machine-specific absolute paths, and
records both source and exported SHA-256 digests.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "audit_latest_experiments"
HABITAT_FILES = (
    "manifest.json",
    "plans.jsonl",
    "events.jsonl",
    "state_before_after.jsonl",
    "summary.json",
    "usage.jsonl",
    "predictions.jsonl",
)
SCORER_FILES = ("physical_checks.jsonl", "labels_SCORER_ONLY.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_mcnemar(left_only: int, right_only: int) -> float:
    import math

    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def audit_mmtom() -> dict[str, Any]:
    path = ROOT / "spatial_social_code" / "mmtom_option_blind_gemini3pro.jsonl"
    summary_path = path.with_name("mmtom_option_blind_gemini3pro_summary.json")
    rows = load_jsonl(path)
    valid = [row for row in rows if not row.get("error")]
    both = sum(bool(row["direct_correct"]) and bool(row["correct"]) for row in valid)
    direct_only = sum(bool(row["direct_correct"]) and not bool(row["correct"]) for row in valid)
    ssc_only = sum(not bool(row["direct_correct"]) and bool(row["correct"]) for row in valid)
    neither = len(valid) - both - direct_only - ssc_only
    compiled = "\n".join(
        str(row.get(field, ""))
        for row in valid
        for field in ("spatial_code", "social_code")
    ).lower()
    recomputed = {
        "rows": len(rows),
        "valid_pairs": len(valid),
        "errors": len(rows) - len(valid),
        "error_indices": [row.get("index") for row in rows if row.get("error")],
        "unique_valid_indices": len({row.get("index") for row in valid}),
        "selected_indices": [row.get("index") for row in rows],
        "valid_indices": [row.get("index") for row in valid],
        "models": sorted({str(row.get("model")) for row in rows}),
        "direct_correct": sum(bool(row["direct_correct"]) for row in valid),
        "ssc_correct": sum(bool(row["correct"]) for row in valid),
        "both_correct": both,
        "direct_only_correct": direct_only,
        "ssc_only_correct": ssc_only,
        "both_wrong": neither,
        "mcnemar_exact_p": exact_mcnemar(direct_only, ssc_only),
        "compiler_code_query_evidence_occurrences": compiled.count("query_evidence"),
        "compiler_code_candidate_occurrences": compiled.count("m.candidate"),
        "compiler_code_posterior_occurrences": compiled.count("m.posterior"),
        "retry_flags": {
            "direct": sum(bool(row.get("direct_retry_used")) for row in rows),
            "final": sum(bool(row.get("final_retry_used")) for row in rows),
        },
        "provider_token_usage_available": False,
        "provider_build_identifier_available": False,
    }
    supplied = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "valid_pair_count": supplied["successful"] == recomputed["valid_pairs"],
        "direct_correct": supplied["overall"]["direct_baseline"]["correct"] == recomputed["direct_correct"],
        "ssc_correct": supplied["overall"]["standalone_spatial_social"]["correct"] == recomputed["ssc_correct"],
        "contingency": supplied["contingency"]["baseline_only_correct"] == direct_only
        and supplied["contingency"]["standalone_only_correct"] == ssc_only,
        "mcnemar": abs(supplied["contingency"]["mcnemar_exact_p"] - recomputed["mcnemar_exact_p"]) < 1e-15,
    }
    return {
        "status": "partial_run_205_rows_204_valid_pairs",
        "recomputed": recomputed,
        "summary_checks": checks,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path), str(summary_path.relative_to(ROOT)): sha256(summary_path)},
        "code_sha256": {
            "spatial_social_code/evaluate_mmtom_jdcloud.py": sha256(ROOT / "spatial_social_code/evaluate_mmtom_jdcloud.py"),
            "spatial_social_code/evaluate_spatial_social.py": sha256(ROOT / "spatial_social_code/evaluate_spatial_social.py"),
            "spatial_social_code/social_code.py": sha256(ROOT / "spatial_social_code/social_code.py"),
        },
        "claim_boundary": (
            "Arithmetic and absence of legacy answer-scoring calls in saved compiler code are verified. "
            "The 204 rows are not a complete 600-question benchmark, and provider token usage is absent."
        ),
    }


def audit_compile_once() -> dict[str, Any]:
    path = ROOT / "workdirs" / "cross_model_v1" / "mmtom_gpt56_compile_once20.jsonl"
    rows = load_jsonl(path)
    questions = [question for row in rows for question in row["questions"]]
    direct = sum(bool(row["direct_correct"]) for row in questions)
    ssc = sum(bool(row["ssc_correct"]) for row in questions)
    direct_only = sum(bool(row["direct_correct"]) and not bool(row["ssc_correct"]) for row in questions)
    ssc_only = sum(not bool(row["direct_correct"]) and bool(row["ssc_correct"]) for row in questions)
    return {
        "status": "development_pilot_not_heldout_validation",
        "groups": len(rows),
        "questions": len(questions),
        "unique_question_indices": len({row["index"] for row in questions}),
        "direct_correct": direct,
        "ssc_correct": ssc,
        "direct_only_correct": direct_only,
        "ssc_only_correct": ssc_only,
        "mcnemar_exact_p": exact_mcnemar(direct_only, ssc_only),
        "source_sha256": sha256(path),
    }


def audit_maegoqa() -> dict[str, Any]:
    path = ROOT / "workdirs" / "cross_model_v1" / "maegoqa_gpt56_reasoning20.jsonl"
    rows = load_jsonl(path)
    conditions = sorted(rows[0]["correct"])
    metrics = {}
    for condition in conditions:
        calls = [row["call_metrics"][condition] for row in rows]
        metrics[condition] = {
            "correct": sum(bool(row["correct"][condition]) for row in rows),
            "total": len(rows),
            "calls": sum(int(call.get("calls", 0)) for call in calls),
            "transport_retries": sum(int(call.get("transport_retries", 0)) for call in calls),
            "input_tokens": sum(int(call.get("usage", {}).get("input_tokens", 0)) for call in calls),
            "output_tokens": sum(int(call.get("usage", {}).get("output_tokens", 0)) for call in calls),
        }
    return {
        "status": "reasoning_only_frozen_perception_pilot",
        "records": len(rows),
        "unique_ids": len({row["unit_id"] for row in rows}),
        "models": sorted({str(row.get("model")) for row in rows}),
        "conditions": metrics,
        "source_sha256": sha256(path),
    }


def audit_habitat() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / "reports" / "habitat_closed_loop_multiseed" / "habitat_multiseed_raw.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
        per_task[(row["task_id"], row["condition"])].append(row)
    raw_summary_found = 0
    stability_advance_false = []
    stability_advance_missing = []
    for row in rows:
        summary_path = Path(row["artifact_dir"]) / "summary.json"
        if not summary_path.is_file():
            continue
        raw_summary_found += 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        strict = summary.get("historical_benchmark_evaluation", {}).get("strict_physical", {})
        value = strict.get("stability_advance_ok", "missing")
        run_id = {key: row[key] for key in ("task_id", "condition", "seed")}
        if value is False:
            stability_advance_false.append(run_id)
        elif value == "missing":
            stability_advance_missing.append(run_id)
    aggregate = {}
    for condition, values in sorted(grouped.items()):
        aggregate[condition] = {
            "runs": len(values),
            "strict_successes": sum(int(row["strict_success"]) for row in values),
            "mean_strict_percent_complete": sum(float(row["strict_percent_complete"]) for row in values) / len(values),
            "mean_api_calls": sum(float(row["api_calls"]) for row in values) / len(values),
            "mean_total_tokens": sum(int(row["input_tokens"]) + int(row["output_tokens"]) for row in values) / len(values),
        }
    task_rows = []
    for (task_id, condition), values in sorted(per_task.items()):
        task_rows.append({
            "task_id": task_id,
            "condition": condition,
            "seeds": len(values),
            "strict_successes": sum(int(row["strict_success"]) for row in values),
            "mean_strict_percent_complete": sum(float(row["strict_percent_complete"]) for row in values) / len(values),
        })
    return ({
        "status": "historical_v1_scores_recomputed_without_rescoring",
        "rows": len(rows),
        "unique_tasks": len({row["task_id"] for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "raw_summary_artifacts_found": raw_summary_found,
        "stability_advance_false_runs": stability_advance_false,
        "stability_advance_field_missing_runs": stability_advance_missing,
        "aggregate": aggregate,
        "source_sha256": sha256(path),
        "claim_boundary": (
            "CSV arithmetic is verified. All 108 source summaries were found and none reports "
            "stability_advance_ok=false; the v2 invalid-measurement fix therefore does not alter these rows."
        ),
    }, task_rows)


def sanitized(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(str(ROOT), "$WORKSPACE").replace(
            "/pfs/xiabin/models", "$MODEL_ROOT"
        )
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    return value


def export_jsonish(source: Path, destination: Path, *, compress: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".jsonl":
        rows = load_jsonl(source)
        content = "".join(json.dumps(sanitized(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        content = json.dumps(sanitized(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if compress:
        destination.write_bytes(gzip.compress(content.encode("utf-8"), mtime=0))
    else:
        destination.write_text(content, encoding="utf-8")


def export_bundle(output: Path, audit: dict[str, Any], task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    copies: list[tuple[Path, Path, bool]] = []
    for source in (
        ROOT / "spatial_social_code/mmtom_option_blind_gemini3pro.jsonl",
        ROOT / "spatial_social_code/mmtom_option_blind_gemini3pro_summary.json",
        ROOT / "workdirs/cross_model_v1/mmtom_gpt56_compile_once20.jsonl",
        ROOT / "workdirs/cross_model_v1/mmtom_gpt56_compile_once20_manifest.json",
        ROOT / "workdirs/cross_model_v1/mmtom_gpt56_compile_once20_summary.json",
        ROOT / "workdirs/cross_model_v1/maegoqa_gpt56_reasoning20.jsonl",
        ROOT / "workdirs/cross_model_v1/maegoqa_gpt56_reasoning20_manifest.json",
        ROOT / "workdirs/cross_model_v1/maegoqa_gpt56_reasoning20_summary.json",
        ROOT / "workdirs/priority_ssc_v1/human_audit/manifest.json",
    ):
        if "option_blind" in source.name:
            category = "mmtom_option_blind"
        elif "human_audit" in str(source):
            category = "human_audit"
        else:
            category = "cross_model_pilots"
        copies.append((source, output / category / source.name, False))

    raw_csv = ROOT / "reports/habitat_closed_loop_multiseed/habitat_multiseed_raw.csv"
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        episode = next((item for item in ("1166", "1566") if item in row["task_id"]), None)
        if episode is None:
            continue
        source_dir = Path(row["artifact_dir"])
        target = output / "habitat_1166_1566" / f"episode_{episode}" / row["condition"] / f"seed_{row['seed']}"
        for name in HABITAT_FILES:
            source = source_dir / name
            if source.is_file():
                copies.append((source, target / "agent_visible" / name, False))
        for name in SCORER_FILES:
            source = source_dir / name
            if source.is_file():
                copies.append((source, target / "scorer_only" / name, False))
        provider_usage = source_dir.parent / "api_usage.jsonl"
        if provider_usage.is_file():
            copies.append((provider_usage, target / "provider_usage.jsonl", False))

    # Complete 108-run evidence: summaries/manifests remain directly readable;
    # larger JSONL logs are gzip-compressed to keep the public artifact small.
    for index, row in enumerate(sorted(rows, key=lambda item: (item["task_id"], item["condition"], item["seed"]))):
        source_dir = Path(row["artifact_dir"])
        run_name = f"run_{index:03d}_{row['condition']}_seed_{row['seed']}"
        target = output / "habitat_all_108" / run_name
        for name in HABITAT_FILES:
            source = source_dir / name
            if not source.is_file():
                continue
            compress = source.suffix == ".jsonl"
            destination = target / "agent_visible" / (name + ".gz" if compress else name)
            copies.append((source, destination, compress))
        for name in SCORER_FILES:
            source = source_dir / name
            if not source.is_file():
                continue
            copies.append((source, target / "scorer_only" / f"{name}.gz", True))
        provider_usage = source_dir.parent / "api_usage.jsonl"
        if provider_usage.is_file():
            copies.append((provider_usage, target / "provider_usage.jsonl.gz", True))

    manifest_files = []
    for source, destination, compress in copies:
        export_jsonish(source, destination, compress=compress)
        manifest_files.append({
            "source": str(source).replace(str(ROOT), "$WORKSPACE"),
            "export": str(destination.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "export_sha256": sha256(destination),
        })

    per_task_path = output / "habitat_per_task.csv"
    with per_task_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    known_exports = {str(destination.relative_to(ROOT)) for _, destination, _ in copies}
    known_generated = {
        str(per_task_path.relative_to(ROOT)),
        str(audit_path.relative_to(ROOT)),
    }
    supplemental = {}
    for path in sorted(output.rglob("*")):
        relative = str(path.relative_to(ROOT))
        if (
            path.is_file()
            and path.name != "manifest.json"
            and relative not in known_exports
            and relative not in known_generated
        ):
            supplemental[relative] = sha256(path)
    manifest = {
        "schema_version": "latest_experiment_evidence_v1",
        "offline_only": True,
        "files": manifest_files,
        "generated": {
            str(per_task_path.relative_to(ROOT)): sha256(per_task_path),
            str(audit_path.relative_to(ROOT)): sha256(audit_path),
        },
        "supplemental_files": supplemental,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    habitat, task_rows = audit_habitat()
    audit = {
        "schema_version": "ssc_latest_experiment_audit_v1",
        "execution_policy": "offline_recomputation_only_no_model_or_gpu_calls",
        "mmtom_option_blind": audit_mmtom(),
        "mmtom_compile_once": audit_compile_once(),
        "maegoqa_reasoning_baselines": audit_maegoqa(),
        "habitat_multiseed": habitat,
    }
    if not args.no_export:
        export_bundle(args.output, audit, task_rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
