"""Standard-library public-archive verification; no API, no raw dataset required."""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from build_figure2_pro_replacement_20260923 import final_answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figure2_pro_replacement_20260923"


def main():
    manifest = json.loads((OUT / "SHA256SUMS.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
    case = json.loads((OUT / "mmtom_pro_0302/case.json").read_text())
    with gzip.open(ROOT / "reports/flash_full_results_20260920/mmtom/pro_results.jsonl.gz", "rt") as f:
        originals = {r["index"]: r for r in map(json.loads, f)}
    assert case["saved_pro_record"] == originals[302]
    assert originals[302]["direct_raw"] == "B"
    assert final_answer(originals[302]["direct_raw"])[0] == originals[302]["direct_prediction"] == "b"
    assert originals[302]["answer"] == originals[302]["prediction"] == "a"
    assert final_answer(originals[0]["direct_raw"])[0] is None
    assert case["checks"]["actual_pro_wire_request_available"] is False
    assert case["checks"]["question_premise_is_not_visual_inference"] is True
    for frame in case["frames"]:
        assert hashlib.sha256((OUT / frame["file"]).read_bytes()).hexdigest() == frame["sha256"]
    for name in ("spatial_code", "social_code"):
        assert (OUT / "mmtom_pro_0302" / (name + ".py")).read_text() == originals[302][name]+"\n"
    ledger = json.loads((OUT / "selection_ledger_600.json").read_text())
    assert [r["index"] for r in ledger] == list(range(600))
    assert min(r["index"] for r in ledger if r["candidate"] and not r["dense"]) == 302
    text = " ".join(ET.parse(OUT / "mmtom_pro_0302/panel_a_input_state_readout.svg").getroot().itertext())
    assert not any(c in text for c in "✓✗✔✘")
    assert "query premise" in text and "Candidate-conditioned" in text
    assert "Archived Direct response:  B" in text
    print(f"PASS: {len(manifest)} hashes; Pro302 raw/code/frame correspondence; index0 withheld; no win/loss badge. No semantic or wire-request certification.")


if __name__ == "__main__":
    main()
