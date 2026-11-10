"""Rescore saved pre-verifier answers as a standalone Spatial-Social method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent.parent
for import_path in (ROOT, PROJECT_ROOT / "spatial_social_code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from egomas.src.run_video_grounded_v2_jdcloud import _answer_index
from egomas.src.run_video_provenance_v3_jdcloud import (
    METHOD,
    SCHEMA_VERSION,
    _validated_citations,
    load_jsonl,
    write_outputs,
)


DEFAULT_INPUT = ROOT / "data" / "baselines" / "video_provenance_v3_gemini3pro.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "data" / "baselines" / "video_provenance_v3_standalone_gemini3pro.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    for row in records.values():
        prediction, data = _answer_index(str(row.get("structured_raw", "")))
        valid_ids = set(row.get("valid_evidence_ids", []))
        data["evidence_ids"] = _validated_citations(data, valid_ids)
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "structured_data": data,
                "structured_prediction_index": prediction,
                "structured_correct": prediction == row["answer_index"],
                "structured_citation_valid": bool(data["evidence_ids"]),
                "prediction_index": prediction,
                "correct": prediction == row["answer_index"],
            }
        )
        for key in (
            "verifier_triggered",
            "verifier_clip_ids",
            "verifier_raw",
            "verifier_data",
            "verifier_switch_accepted",
        ):
            row.pop(key, None)

    selected = sorted(records)
    model = str(next(iter(records.values())).get("model", "")) if records else ""
    write_outputs(args.output, records, selected, model)


if __name__ == "__main__":
    main()
