#!/usr/bin/env python3
"""Recording-fold OOF identity gate for CoMind two-stage grounding.

Bounding boxes always come from the label-free two-stage grounder.  For each
task and held-out recording fold, identity is either preserved from the frozen
SSC prediction or replaced by the two-stage identity.  That one-bit policy is
chosen using only the other recording folds.  Replacement must clear a frozen
minimum practical gain; otherwise the conservative default is to preserve.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V2 = load_module("comind_v2_gate", HERE / "comind_v2.py")
CMP = load_module("compare_two_stage_gate", HERE / "compare_balanced_pilot.py")
IDENTITY_METRIC = {
    "joint_attention": "category_l1",
    "scoia": "action_noun_l1",
    "handover": "category_l1",
}
IDENTITY_FIELD = IDENTITY_METRIC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("workdirs/comind_full_train/predictions.jsonl"))
    parser.add_argument("--two-stage", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("external_datasets/comind/annotations"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-identity-gain", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    fold_by_key = {
        (str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): int(row["recording_fold"])
        for row in manifest_rows
    }
    base_rows = [json.loads(line) for line in args.base.read_text().splitlines() if line.strip()]
    base = {
        (str(row["condition"]), str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row
        for row in base_rows if not row.get("error") and isinstance(row.get("prediction"), dict)
    }
    two_rows = [json.loads(line) for line in args.two_stage.read_text().splitlines() if line.strip()]
    two = {
        (str(row["task"]), str(row["recording_id"]), str(row["sample_id"])): row
        for row in two_rows if not row.get("error") and isinstance(row.get("prediction"), dict)
    }
    annotations = V2.load_annotations(args.annotations)

    records = []
    for key in sorted(set(fold_by_key) & set(two)):
        task, rid, sid = key
        direct_row = base.get(("direct", task, rid, sid))
        spatial_row = base.get(("spatial_social", task, rid, sid))
        gold = annotations[task].get(rid, {}).get(sid)
        if not direct_row or not spatial_row or not isinstance(gold, dict):
            continue
        direct = CMP.coordinate_correct_original(task, direct_row["prediction"])
        spatial = CMP.coordinate_correct_original(task, spatial_row["prediction"])
        unconditional = CMP.optimize_spatial(task, spatial_row["prediction"], two[key])
        records.append({
            "key": key,
            "task": task,
            "fold": fold_by_key[key],
            "gold": gold,
            "predictions": {"direct": direct, "spatial_social": spatial, "unconditional": unconditional},
        })

    policies = []
    for fold in range(args.folds):
        for task, metric in IDENTITY_METRIC.items():
            train = [row for row in records if row["task"] == task and row["fold"] != fold]
            preserve = sum(V2.score_prediction(task, row["predictions"]["spatial_social"], row["gold"])[metric] for row in train)
            replace = sum(V2.score_prediction(task, row["predictions"]["unconditional"], row["gold"])[metric] for row in train)
            gain = (replace - preserve) / len(train) if train else 0.0
            policies.append({
                "fold": fold,
                "task": task,
                "train_n": len(train),
                "preserve_correct": preserve,
                "replace_correct": replace,
                "replacement_gain": gain,
                "minimum_gain": args.min_identity_gain,
                "choice": "replace" if gain >= args.min_identity_gain else "preserve",
            })
    policy = {(row["fold"], row["task"]): row["choice"] for row in policies}

    for row in records:
        task = row["task"]
        gated = copy.deepcopy(row["predictions"]["unconditional"])
        if policy[(row["fold"], task)] == "preserve":
            field = IDENTITY_FIELD[task]
            gated[field] = row["predictions"]["spatial_social"].get(field)
        row["predictions"]["oof_identity_gate"] = gated
        row["scores"] = {
            name: V2.score_prediction(task, prediction, row["gold"])
            for name, prediction in row["predictions"].items()
        }

    variants = ("direct", "spatial_social", "unconditional", "oof_identity_gate")
    result = {
        "protocol": f"{args.folds}-fold recording-level OOF identity gate; bbox always two-stage; identity policy trained only on other recording folds; replacement requires >= {args.min_identity_gain:.1%} training-fold accuracy gain",
        "samples": dict(Counter(row["task"] for row in records)),
        "policies": policies,
        "variants": {},
        "paired_gate_vs_spatial_social": {},
    }
    for variant in variants:
        result["variants"][variant] = {}
        for task in V2.TASK_FILES:
            scores = [row["scores"][variant] for row in records if row["task"] == task]
            result["variants"][variant][task] = V2.metric_summary(scores)
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        subset = [row for row in records if row["task"] == task]
        result["paired_gate_vs_spatial_social"][task] = {}
        for metric in metrics:
            result["paired_gate_vs_spatial_social"][task][metric] = V2.exact_mcnemar(
                [row["scores"]["oof_identity_gate"][metric] for row in subset],
                [row["scores"]["spatial_social"][metric] for row in subset],
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# CoMind recording-level OOF identity gate", "",
        result["protocol"], "",
        "| 字段 | Direct | 原 SSC | 无条件 two-stage | OOF identity gate | gate vs SSC (win/loss) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, metrics in V2.FIELD_TO_PREDICTION.items():
        for metric in metrics:
            values = [100 * result["variants"][variant][task]["metrics"][metric] for variant in variants]
            pair = result["paired_gate_vs_spatial_social"][task][metric]
            lines.append(f"| {task}.{metric} | {values[0]:.1f}% | {values[1]:.1f}% | {values[2]:.1f}% | {values[3]:.1f}% | {pair['candidate_only_correct']}/{pair['baseline_only_correct']} |")
    lines.extend(["", f"样本数：{result['samples']}。这是 train-only recording-fold OOF 机制实验，不是官方 hidden test。", "", "Fold policies:", "```json", json.dumps(policies, indent=2), "```"])
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"samples": result["samples"], "output": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
