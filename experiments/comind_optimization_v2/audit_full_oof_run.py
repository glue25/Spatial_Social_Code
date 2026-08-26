#!/usr/bin/env python3
"""Audit the completed CoMind grounding and recording-heldout OOF run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple[str, str, str]:
    return str(row["task"]), str(row["recording_id"]), str(row["sample_id"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_fold(recording_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(recording_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    grounding = read_jsonl(args.grounding)
    baseline = read_jsonl(args.baseline)
    oof = json.loads(args.oof.read_text())

    manifest_keys = [key(row) for row in manifest]
    grounding_keys = [key(row) for row in grounding]
    manifest_set = set(manifest_keys)
    grounding_set = set(grounding_keys)
    manifest_schema = sorted({field for row in manifest for field in row})
    forbidden_label_keys = sorted(
        set(manifest_schema)
        & {
            "answer",
            "annotation",
            "annotations",
            "gold",
            "ground_truth",
            "label",
            "labels",
            "target",
            "targets",
        }
    )

    image_paths = [Path(image["path"]) for row in manifest for image in row.get("images", [])]
    # Parallel metadata lookups keep the full (not sampled) image audit fast on
    # the shared filesystem used by the experiment host.
    with ThreadPoolExecutor(max_workers=64) as pool:
        image_exists = list(pool.map(Path.is_file, image_paths, chunksize=64))
    missing_images = [str(path) for path, exists in zip(image_paths, image_exists) if not exists]
    bad_grounding = [
        key(row)
        for row in grounding
        if row.get("error") or not isinstance(row.get("prediction"), dict)
    ]

    recording_folds: dict[str, set[int]] = defaultdict(set)
    fold_hash_mismatches = []
    for row in manifest:
        recording_id = str(row["recording_id"])
        fold = int(row["recording_fold"])
        recording_folds[recording_id].add(fold)
        if fold != expected_fold(recording_id):
            fold_hash_mismatches.append(key(row))
    split_recordings = sorted(rid for rid, folds in recording_folds.items() if len(folds) != 1)

    baseline_conditions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in baseline:
        if not row.get("error") and isinstance(row.get("prediction"), dict):
            baseline_conditions[key(row)].add(str(row.get("condition")))
    strict_common = {
        item_key
        for item_key in manifest_set & grounding_set
        if {"direct", "spatial_social"}.issubset(baseline_conditions[item_key])
    }
    strict_common_counts = dict(Counter(item_key[0] for item_key in strict_common))

    expected_train_n = {}
    fold_test_recordings = {}
    for fold in range(5):
        fold_test_recordings[str(fold)] = len(
            {row["recording_id"] for row in manifest if int(row["recording_fold"]) == fold}
        )
        for task in ("joint_attention", "scoia", "handover"):
            expected_train_n[(fold, task)] = sum(
                item_key in strict_common
                and row["task"] == task
                and int(row["recording_fold"]) != fold
                for row, item_key in zip(manifest, manifest_keys)
            )
    policy_train_n_mismatches = []
    for policy in oof["policies"]:
        identity = (int(policy["fold"]), str(policy["task"]))
        expected = expected_train_n[identity]
        if int(policy["train_n"]) != expected:
            policy_train_n_mismatches.append(
                {"fold": identity[0], "task": identity[1], "reported": policy["train_n"], "expected": expected}
            )

    oof_counts = {name: int(value) for name, value in oof["samples"].items()}
    checks = {
        "manifest_has_6877_rows": len(manifest) == 6877,
        "manifest_keys_unique": len(manifest_keys) == len(manifest_set),
        "manifest_has_no_label_fields": not forbidden_label_keys,
        "all_manifest_images_exist": not missing_images,
        "grounding_has_6877_rows": len(grounding) == 6877,
        "grounding_keys_unique": len(grounding_keys) == len(grounding_set),
        "grounding_keyset_matches_manifest": grounding_set == manifest_set,
        "all_grounding_rows_successful": not bad_grounding,
        "one_fold_per_recording": not split_recordings,
        "fold_assignment_matches_sha256": not fold_hash_mismatches,
        "strict_common_is_6673": len(strict_common) == 6673,
        "oof_sample_counts_match_strict_common": oof_counts == strict_common_counts,
        "all_policy_train_counts_exclude_heldout_fold": not policy_train_n_mismatches,
        "all_five_folds_have_heldout_recordings": len(fold_test_recordings) == 5
        and all(count > 0 for count in fold_test_recordings.values()),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "manifest_rows": len(manifest),
            "manifest_unique_keys": len(manifest_set),
            "manifest_recordings": len(recording_folds),
            "manifest_images": len(image_paths),
            "grounding_rows": len(grounding),
            "grounding_unique_keys": len(grounding_set),
            "grounding_errors": len(bad_grounding),
            "strict_common": len(strict_common),
            "strict_common_by_task": strict_common_counts,
            "oof_samples_by_task": oof_counts,
            "heldout_recordings_by_fold": fold_test_recordings,
        },
        "manifest_schema": manifest_schema,
        "forbidden_label_keys": forbidden_label_keys,
        "missing_image_count": len(missing_images),
        "split_recording_count": len(split_recordings),
        "fold_hash_mismatch_count": len(fold_hash_mismatches),
        "policy_train_n_mismatches": policy_train_n_mismatches,
        "artifacts": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "grounding": {"path": str(args.grounding), "sha256": sha256(args.grounding)},
            "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline)},
            "oof": {"path": str(args.oof), "sha256": sha256(args.oof)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# CoMind full grounding + recording-heldout OOF audit",
        "",
        f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines.extend(
        [
            "",
            f"- Grounding: {len(grounding_set):,}/{len(manifest_set):,} unique keys, errors={len(bad_grounding)}.",
            f"- Strict Direct/SSC/grounding intersection: {len(strict_common):,}; by task={strict_common_counts}.",
            f"- Recordings: {len(recording_folds)}; held-out counts by fold={fold_test_recordings}.",
            f"- Input images checked: {len(image_paths):,}; missing={len(missing_images)}.",
            "- Claim boundary: labeled training corpus with recording-level OOF policy selection; not official hidden-test performance.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": checks, "report": str(args.report)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
