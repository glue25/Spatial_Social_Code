#!/usr/bin/env python3
"""Validate two independent reviews, report agreement, and build clean states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.priority_ssc_v1.common import atomic_json, atomic_jsonl, clean_trace, digest_text, forbidden_target_keys, read_jsonl

LABELS = {"yes", "no", "uncertain", "na"}


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["claim_id"]: row for row in rows}


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right))
    ca, cb = Counter(left), Counter(right)
    expected = sum(ca[label] * cb[label] for label in labels) / (len(left) ** 2)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def parsed_correction(value: str) -> dict[str, Any] | None:
    value = value.strip()
    if not value or value.upper() == "DROP":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("kind"), str):
        raise ValueError("correction must be a trace-record JSON object or DROP")
    if forbidden_target_keys(parsed):
        raise ValueError("correction contains a target-answer field")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "workdirs/priority_ssc_v1/human_audit")
    args = parser.parse_args()
    packets = read_jsonl(args.audit_dir / "packets.jsonl")
    reviewers = [read_csv(args.audit_dir / name) for name in ("reviewer_a.csv", "reviewer_b.csv")]
    expected = {claim["claim_id"]: (packet, claim) for packet in packets for claim in packet["claims"]}
    for number, reviewer in enumerate(reviewers, start=1):
        if set(reviewer) != set(expected):
            raise ValueError(f"reviewer {number} claim IDs differ from packet")
        for claim_id, row in reviewer.items():
            layer = expected[claim_id][1]["layer"]
            state = row["state_correct"].strip().lower()
            citation = row["citation_support"].strip().lower()
            if state not in LABELS - {"na"} or citation not in LABELS - {"na"}:
                raise ValueError(f"incomplete/invalid review at {claim_id}; use yes/no/uncertain")
            if state == "no" and not row["corrected_claim_json"].strip():
                raise ValueError(f"{claim_id}: state_correct=no requires corrected JSON or DROP")
            if layer not in {"physical", "social"}:
                raise ValueError(f"unknown layer at {claim_id}")
    dimensions: dict[str, Any] = {}
    for name in ("state_correct", "citation_support"):
        left = [reviewers[0][claim_id][name].strip().lower() for claim_id in expected]
        right = [reviewers[1][claim_id][name].strip().lower() for claim_id in expected]
        dimensions[name] = {
            "n": len(left),
            "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
            "cohen_kappa": kappa(left, right),
            "reviewer_a": dict(Counter(left)),
            "reviewer_b": dict(Counter(right)),
        }
    for layer in ("physical", "social"):
        claim_ids = [claim_id for claim_id, (_, claim) in expected.items() if claim["layer"] == layer]
        dimensions[f"{layer}_state_accuracy"] = {
            f"reviewer_{letter}": sum(reviewers[i][cid]["state_correct"].strip().lower() == "yes" for cid in claim_ids) / len(claim_ids) if claim_ids else None
            for i, letter in enumerate(("a", "b"))
        }
    dimensions["physical_fact_accuracy"] = dimensions["physical_state_accuracy"]
    dimensions["access_attribution_accuracy"] = dimensions["social_state_accuracy"]
    citation_ids = list(expected)
    dimensions["citation_support_accuracy"] = {
        f"reviewer_{letter}": sum(reviewers[i][cid]["citation_support"].strip().lower() == "yes" for cid in citation_ids) / len(citation_ids) if citation_ids else None
        for i, letter in enumerate(("a", "b"))
    }
    adjudication_path = args.audit_dir / "adjudication.csv"
    adjudication = read_csv(adjudication_path) if adjudication_path.exists() else {}
    unresolved: list[str] = []
    final_claims: dict[str, dict[str, Any] | None] = {}
    for claim_id, (_, claim) in expected.items():
        a, b = reviewers[0][claim_id], reviewers[1][claim_id]
        unanimous_supported = all(row["state_correct"].strip().lower() == "yes" and row["citation_support"].strip().lower() == "yes" for row in (a, b))
        if unanimous_supported:
            final_claims[claim_id] = claim["claim"]
            continue
        decision = adjudication.get(claim_id, {}).get("final_keep", "").strip().lower()
        if decision not in {"yes", "no"}:
            unresolved.append(claim_id)
            continue
        corrected = adjudication[claim_id].get("adjudicated_claim_json", "")
        if decision == "yes":
            final_claims[claim_id] = parsed_correction(corrected) if corrected.strip() else claim["claim"]
        else:
            final_claims[claim_id] = None
    report = {
        "schema_version": 1,
        "samples": len(packets),
        "claims": len(expected),
        "agreement": dimensions,
        "unresolved_adjudications": len(unresolved),
        "status": "complete" if not unresolved else "awaiting_adjudication",
    }
    atomic_json(args.audit_dir / "audit_summary.json", report)
    if unresolved:
        atomic_json(args.audit_dir / "unresolved_adjudications.json", unresolved)
        raise SystemExit(f"awaiting adjudication for {len(unresolved)} claims")
    by_sample: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"physical": [], "social": []})
    for claim_id, (packet, claim) in expected.items():
        resolved = final_claims[claim_id]
        if resolved is not None:
            by_sample[packet["sample_id"]][claim["layer"]].append(resolved)
    verified = []
    for packet in packets:
        states = by_sample[packet["sample_id"]]
        original_physical = "\n".join(
            json.dumps(claim["claim"], ensure_ascii=False, sort_keys=True)
            for claim in packet["claims"] if claim["layer"] == "physical"
        )
        original_social = "\n".join(
            json.dumps(claim["claim"], ensure_ascii=False, sort_keys=True)
            for claim in packet["claims"] if claim["layer"] == "social"
        )
        verified_physical = "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in states["physical"])
        verified_social = "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in states["social"])
        row = {
            "schema_version": 1,
            "sample_id": packet["sample_id"],
            "dataset": packet["dataset"],
            "index": packet["index"],
            "unit_id": packet["unit_id"],
            "physical_trace": verified_physical,
            "social_trace": verified_social,
            "predicted_physical_confirmed_correct": clean_trace(original_physical) == clean_trace(verified_physical),
            "predicted_social_confirmed_correct": clean_trace(original_social, social=True) == clean_trace(verified_social, social=True),
            "predicted_physical_digest": digest_text(clean_trace(original_physical)),
            "predicted_social_digest": digest_text(clean_trace(original_social, social=True)),
        }
        forbidden = forbidden_target_keys(row)
        if forbidden:
            raise RuntimeError(f"target answer leaked into verified state: {forbidden}")
        verified.append(row)
    atomic_jsonl(args.audit_dir / "verified_states.jsonl", verified)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
