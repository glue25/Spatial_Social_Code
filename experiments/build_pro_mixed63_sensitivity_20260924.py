"""Export/verify a retrospective mixed-version sensitivity table, not a benchmark.

Default exports local saved calls; --from-export verifies public artifacts only.
No network access or inference. Historical raw predictions are never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from score_pro_semantic_readback_20260924 import final_answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pro_direct_mixed63_sensitivity_20260924"
PARTIAL = ROOT / "reports/pro_direct_six_response_patch_20260924"
REVIEW = ROOT / "reports/pro_direct_semantic_readback_20260924"
RUN = ROOT / "workdirs/pro_unresolved69_paired_20260923"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")


def selected():
    return sorted(r["index"] for r in read(REVIEW / "semantic_decisions_without_gold.json")
                  if r["prediction"] is None)


def response_text(body):
    return "\n".join(p.get("text", "") for c in body.get("candidates", [])
                     for p in c.get("content", {}).get("parts", []) if not p.get("thought"))


def export_calls():
    evidence = []
    for i in selected():
        result_path = RUN / f"results/{i:04d}/direct.json"
        ssc_path = RUN / f"results/{i:04d}/ssc.json"
        result, ssc = read(result_path), read(ssc_path)
        if i == 35:
            req_path = RUN / "reuse/direct35_request.json"
            resp_path = RUN / "reuse/direct35_response.json"
            body = read(resp_path)
            http = result["http_status"]
        else:
            assert result["logical_calls"] == 1 and not result["repair_used"]
            call = RUN / f"calls/{i:04d}/direct/01"
            accepted = read(call / "accepted.json")
            req_path = call / "request.json"
            resp_path = call / f"attempt_{accepted['attempt']:02d}.json"
            response = read(resp_path)
            body, http = response["response"], response["http_status"]
            assert accepted["modelVersion"] == body["modelVersion"]
            assert accepted["text"] == result["raw"]
        evidence.append({
            "index": i, "direct_result": result, "request_descriptor": read(req_path),
            "http_status": http, "response_envelope": body,
            "same_model_ssc_context": {k: ssc[k] for k in ["index", "prediction", "valid", "correct"]},
            "source_sha256": {str(p.relative_to(ROOT)): sha(p)
                              for p in [result_path, ssc_path, req_path, resp_path]},
        })
    write("direct63_evidence.json", evidence)
    write("source_protocol.json", read(RUN / "protocol.json"))


def calculate():
    ids = selected()
    assert len(ids) == 63
    evidence = read(OUT / "direct63_evidence.json")
    assert [r["index"] for r in evidence] == ids
    prior = read(PARTIAL / "scores_600.json")
    assert [r["index"] for r in prior] == list(range(600))
    by_id = {r["index"]: r for r in evidence}
    rows = []
    for old in prior:
        i = old["index"]
        pred = old["direct_prediction"]
        source = "historical_pro_semantic_readback" if old["semantic_patch_scope"] else "historical_pro_saved"
        if i in by_id:
            e = by_id[i]
            r, body = e["direct_result"], e["response_envelope"]
            assert e["http_status"] == 200 and body["modelVersion"] == "gemini-3.1-pro-preview"
            assert len(body["candidates"]) == 1 and body["candidates"][0]["finishReason"] == "STOP"
            assert response_text(body) == r["raw"]
            pred = final_answer(r["raw"])[0]
            assert pred in ("a", "b") and pred == r["prediction"]
            assert r["valid"] and r["correct"] == (pred == old["gold"])
            req = e["request_descriptor"]
            assert req["model"] == "Gemini 3-Pro-Preview"
            assert req["generationConfig"]["maxOutputTokens"] == 16384
            assert req["generationConfig"]["temperature"] == 0
            # Compare actual saved question/options against the frozen benchmark text.
            inputs_path = ROOT / "reports/flash_full_results_20260920/mmtom/original_frozen_inputs/inputs.json"
            if i == ids[0]:
                inputs = {x["index"]: x for x in read(inputs_path)}
            prompt = req["prompt"] if i == 35 else "\n".join(p["text"] for p in req["parts"] if "text" in p)
            assert inputs[i]["row"]["question"] in prompt
            ctx = e["same_model_ssc_context"]
            assert ctx["valid"] and ctx["correct"] == (ctx["prediction"] == old["gold"])
            source = "jdcloud_gemini_3.1_pro_preview"
        rows.append({"index": i, "question_type": old["question_type"], "gold": old["gold"],
                     "direct_before_63_replacement": old["direct_prediction"],
                     "direct_before_correct": old["direct_correct"], "direct_prediction": pred,
                     "direct_correct": pred == old["gold"], "direct_source": source,
                     "replaced_with_3_1_pro": i in by_id,
                     "ssc_prediction_historical": old["ssc_prediction_historical"],
                     "ssc_correct_historical": old["ssc_correct_historical"]})
    assert sum(r["direct_before_correct"] for r in rows) == 410
    assert sum(r["direct_before_correct"] for r in rows if r["replaced_with_3_1_pro"]) == 26
    assert sum(r["direct_correct"] for r in rows if r["replaced_with_3_1_pro"]) == 56
    assert sum(r["direct_correct"] for r in rows) == 440
    assert sum(r["ssc_correct_historical"] for r in rows) == 470
    assert sum(e["same_model_ssc_context"]["correct"] for e in evidence) == 55
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-export", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.from_export:
        export_calls()
    rows = calculate()
    summary = {
        "status": "Retrospective mixed-version sensitivity analysis; NOT a same-model benchmark",
        "n": 600, "replacement_n": 63, "remaining_historical_or_readback_n": 537,
        "direct_correct": 440, "direct_accuracy": 440 / 600,
        "ssc_historical_correct": 470, "ssc_historical_accuracy": 470 / 600,
        "ssc_minus_direct_correct": 30, "ssc_minus_direct_percentage_points": 5.0,
        "arithmetic": "410 - 26 + 56 = 440; historical SSC470 unchanged",
        "direct_source_counts": {"historical_saved": 531, "historical_semantic_readback": 6, "new_3_1_pro": 63},
        "same_model_selected63_context": {"direct_correct": 56, "ssc_correct": 55, "n": 63,
                                           "note": "SSC does not outperform Direct on the new same-model subset; selected diagnostic, not representative600"},
        "known_other5_explicit_mismatches_retained": [72, 106, 118, 558, 575],
        "direct_correct_if_those5_also_corrected": 443,
        "limitations": [
            "Direct mixes historical Pro and current3.1Pro; SSC stays historical. Asymmetric replacement is not a controlled comparison.",
            "Replacement IDs were selected retrospectively by historical response recoverability, not sampled independently.",
            "Known five explicit-answer mismatches remain uncorrected; this is not full parser repair.",
            "Historical Pro request mapping/version remains unverified; semantic readback was not fully blinded.",
            "Historical SSC saved score not independently re-audited here; candidate-conditioned and unequal-compute protocol.",
            "No claim that remaining historical and replacement model capabilities, call caps or output completeness are identical.",
            "No new calls were made for this report; existing3.1Pro predictions were reused, including previously accepted Direct35."
        ],
    }
    if args.from_export:
        assert read(OUT / "scores_600.json") == rows
        assert read(OUT / "summary.json") == summary
        for name, digest in read(OUT / "SHA256SUMS.json").items():
            assert sha(OUT / name) == digest, name
        assert read(OUT / "provenance.json")["scorer_sha256"] == sha(Path(__file__))
        provenance = read(OUT / "provenance.json")
        for path in [PARTIAL / "scores_600.json", REVIEW / "semantic_decisions_without_gold.json"]:
            assert provenance["input_sha256"][str(path.relative_to(ROOT))] == sha(path)
        with (OUT / "scores_600.csv").open() as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 600
        for saved, computed in zip(csv_rows, rows):
            assert saved == {key: str(value) for key, value in computed.items()}
    else:
        write("scores_600.json", rows)
        with (OUT / "scores_600.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        write("summary.json", summary)
        write("provenance.json", {
            "scorer_sha256": sha(Path(__file__)),
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [
                PARTIAL / "scores_600.json", REVIEW / "semantic_decisions_without_gold.json",
                RUN / "protocol.json", RUN / "runner.py", RUN / "answer_parser.py"]},
            "scope": "63 Direct request descriptors, full saved response envelopes and result records; context-only SSC predictions",
            "original_archives_modified": False,
        })
        write("SHA256SUMS.json", {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"})
    print("PASS: 63 Direct responses/version/STOP/choices verified; mixedDirect440/600, historicalSSC470/600. New same-model63: Direct56, SSC55.")


if __name__ == "__main__":
    main()
