"""Six archived-response semantic updates; all other predictions unchanged.

Offline, standard-library-only, no inference. This intentionally scoped patch
does NOT certify the remaining historical predictions or fix all known issues.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

from score_pro_semantic_readback_20260924 import final_answer

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports/pro_direct_semantic_readback_20260924"
OUT = ROOT / "reports/pro_direct_six_response_patch_20260924"
PATCH_IDS = {2, 198, 350, 454, 541, 569}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reports/flash_full_results_20260920/mmtom/pro_results.jsonl.gz"
    protocol = json.loads((REVIEW / "protocol.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["input_sha256"][source.name]
    with gzip.open(source, "rt") as f:
        originals = [json.loads(line) for line in f]
    assert sorted(r["index"] for r in originals) == list(range(600))
    judgments = json.loads((REVIEW / "semantic_decisions_without_gold.json").read_text())
    recovered = {r["index"]: r for r in judgments if r["prediction"] is not None}
    assert set(recovered) == PATCH_IDS
    unresolved = sorted(r["index"] for r in judgments if r["prediction"] is None)
    assert len(unresolved) == 63
    rows, patches, explicit_unpatched = [], [], []
    for old in sorted(originals, key=lambda r: r["index"]):
        i, pred, gold = old["index"], old["direct_prediction"], old["answer"]
        assert old["direct_correct"] == (pred == gold)
        original_pred = pred
        if i in recovered:
            review = recovered[i]
            assert final_answer(old["direct_raw"])[0] is None
            assert all(q in old["direct_raw"] for q in review["evidence_quotes"])
            pred = review["prediction"]
            patches.append({"index": i, "historical_prediction": original_pred,
                            "recovered_prediction": pred, "gold": gold,
                            "historical_correct": old["direct_correct"], "patched_correct": pred == gold,
                            "category": review["category"], "evidence_quotes": review["evidence_quotes"],
                            "reason_zh": review["reason_zh"]})
        explicit, _ = final_answer(old["direct_raw"])
        if explicit is not None and explicit != original_pred:
            explicit_unpatched.append({"index": i, "historical_prediction_retained": original_pred,
                                       "explicit_final_answer_not_applied": explicit, "gold": gold,
                                       "historical_correct": original_pred == gold,
                                       "explicit_answer_correct": explicit == gold})
        rows.append({"index": i, "question_type": old["question_type"], "gold": gold,
                     "historical_direct_prediction": original_pred,
                     "historical_direct_correct": old["direct_correct"],
                     "direct_prediction": pred, "direct_correct": pred == gold,
                     "semantic_patch_scope": i in PATCH_IDS,
                     "prediction_changed": pred != original_pred,
                     "unresolved_readback_but_historical_prediction_retained": i in unresolved,
                     "ssc_prediction_historical": old["prediction"], "ssc_correct_historical": old["correct"]})
        if i not in PATCH_IDS:
            assert pred == original_pred
    assert [r["index"] for r in explicit_unpatched] == [72, 106, 118, 558, 575]
    assert sum(r["historical_direct_correct"] for r in rows) == 406
    assert sum(r["direct_correct"] for r in rows) == 410
    assert sum(r["ssc_correct_historical"] for r in rows) == 470
    assert sum(r["prediction_changed"] for r in rows) == 4
    write("patches_6.json", patches)
    write("scores_600.json", rows)
    with (OUT / "scores_600.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write("known_unpatched_issues.json", {
        "explicit_final_answer_discrepancies": explicit_unpatched,
        "explicit_discrepancy_net_correct_if_also_fixed": 3,
        "direct_correct_if_all_11_recoverable_updates_applied": 413,
        "unresolved_63_historical_predictions_retained": unresolved,
        "note": "Unchanged is not independently validated; no claim that historical defaults are reliable answers."
    })
    write("summary.json", {
        "policy": "Apply all six recorded semantic choices, including the wrong choice; retain other 594 historical predictions",
        "status": "User-selected partial historical correction, not a fully audited score",
        "n": 600, "patch_scope_n": 6, "predictions_changed": 4,
        "direct_historical_correct": 406, "direct_correct": 410, "direct_accuracy": 410 / 600,
        "ssc_historical_correct": 470, "ssc_historical_accuracy": 470 / 600,
        "ssc_minus_direct_correct": 60, "ssc_minus_direct_percentage_points": 10.0,
        "ssc_status": "Historical saved score; not independently re-audited here",
        "no_inference_or_model_substitution": True,
        "limitations": [
            "Five known explicit final-answer discrepancies remain unpatched by the requested scope.",
            "63 semantically unresolved responses retain historical predictions, not independently confirmed answers.",
            "Original Pro request option mapping remains unverified; semantic review was not fully blinded.",
            "Historical candidate-conditioned SSC with conditional refinement, not an option-blind equal-compute protocol.",
            "The descriptive score difference does not isolate an evidence-link or representation mechanism effect."
        ]
    })
    write("provenance.json", {
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, REVIEW / "protocol.json", REVIEW / "semantic_decisions_without_gold.json"]},
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_archives_modified": False,
        "prior_failure_inclusive_report": "../pro_direct_semantic_readback_20260924/README_zh.md"
    })
    write("SHA256SUMS.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"})
    print("PASS: six semantic choices applied, four predictions changed; other 594 unchanged. Direct410/600, historicalSSC470/600. Known remaining issues disclosed.")


if __name__ == "__main__":
    main()
