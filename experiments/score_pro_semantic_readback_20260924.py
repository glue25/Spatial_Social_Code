"""Reproduce scoring of recorded assistant semantic judgments; no model calls.

This checks evidence quotation and arithmetic, not the independent correctness
of the assistant's semantic annotations. Missing choices remain invalid.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pro_direct_semantic_readback_20260924"


def final_answer(raw):
    """Read only an explicit final nonempty line, without consulting gold."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None, "unresolved"
    line = lines[-1]
    if line.startswith("**") and line.endswith("**"):
        line = line[2:-2].strip()
    for policy, prefix in [
        ("bare_final_line", ""),
        ("answer_final_line", r"Answer\s*[:=]\s*"),
        ("explicit_statement_final_line", r"The correct statement is\s+"),
    ]:
        match = re.fullmatch(prefix + r"(?:([AB])|\*\*([AB])\*\*)[.]?", line, re.I)
        if match:
            return next(x for x in match.groups() if x).lower(), policy
    return None, "unresolved"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    pack = ROOT / "reports/flash_full_results_20260920/mmtom"
    for name, path in [("pro_results.jsonl.gz", pack / "pro_results.jsonl.gz"),
                       ("inputs.json", pack / "original_frozen_inputs/inputs.json")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["input_sha256"][name]
    packet = {r["index"]: r for r in json.loads((OUT / "review_packet_without_gold.json").read_text())}
    judgments = {r["index"]: r for r in json.loads((OUT / "semantic_decisions_without_gold.json").read_text())}
    assert set(packet) == set(judgments) and len(packet) == 69
    with gzip.open(pack / "pro_results.jsonl.gz", "rt") as f:
        originals = {r["index"]: r for r in map(json.loads, f)}
    assert sorted(originals) == list(range(600))
    for index, judgment in judgments.items():
        raw = originals[index]["direct_raw"]
        assert raw == packet[index]["raw_response"]
        assert final_answer(raw)[0] is None
        assert judgment["prediction"] in ("a", "b", None)
        assert all(quote in raw for quote in judgment["evidence_quotes"])
    rows = []
    for index, original in sorted(originals.items()):
        pred, policy = final_answer(original["direct_raw"])
        if pred is None:
            pred = judgments[index]["prediction"]
            policy = judgments[index]["category"]
        rows.append({"index": index, "type": original["question_type"], "prediction": pred,
                     "gold": original["answer"], "valid": pred is not None, "correct": pred == original["answer"],
                     "extraction": policy, "historical_direct_prediction": original["direct_prediction"],
                     "historical_direct_correct": original["direct_correct"], "ssc_correct": original["correct"]})
    assert rows == json.loads((OUT / "scores_readback_600.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert sum(r["correct"] for r in rows) == summary["correct"] == 387
    assert sum(r["valid"] for r in rows) == summary["valid"] == 537
    assert summary["unresolved"] == 63 and summary["semantic_recovered"] == 6
    assert sum(r["ssc_correct"] for r in rows) == summary["ssc_saved_correct"] == 470
    assert sum(r["correct"] for r in rows if r["extraction"] != "completed_counterfactual_exclusion") == 384
    with (OUT / "scores_readback_600.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "scope": "Historical Pro saved-response readback; retrospective, not a new benchmark run",
        "n": len(rows),
        "direct": {"correct": 387, "accuracy": 387 / 600,
                   "recoverable": 537, "unresolved_counted_incorrect": 63,
                   "status": "Re-extracted explicit answers plus recorded assistant semantic judgments"},
        "ssc": {"correct": 470, "accuracy": 470 / 600,
                "status": "Historical saved score; SSC not independently re-audited in this readback"},
        "ssc_minus_direct_correct": 83,
        "ssc_minus_direct_percentage_points": 100 * 83 / 600,
        "limitations": summary["limitations"] + [
            "Historical candidate-conditioned SSC with conditional dense refinement; not equal total compute.",
            "Difference includes answer recoverability and cannot isolate the effect of evidence links or code syntax.",
            "Cause of missing answers (generation, output cap, transport or archival loss) is not established."
        ],
        "no_new_inference_or_model_substitution": True,
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+"\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                if p.is_file() and p.name != "SHA256SUMS.json"}
    (OUT / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("PASS: 69 quotes checked; 6 semantic choices (5 correct); 63 unresolved; 387/600 under failure-inclusive readback scoring. No inference calls.")


if __name__ == "__main__":
    main()
