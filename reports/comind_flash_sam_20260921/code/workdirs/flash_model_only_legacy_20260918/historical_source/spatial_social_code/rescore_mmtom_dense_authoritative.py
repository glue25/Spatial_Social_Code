"""Use the denser independent pass as authoritative on uncertain MMToM items."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_mmtom_jdcloud import load_jsonl, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("spatial_social_code/mmtom_uncertainty_role_rerun_gemini3pro.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl"))
    args = parser.parse_args()
    records = load_jsonl(args.input)
    for row in records.values():
        row["method"] = "Dense-authoritative uncertainty-refined Spatial-Social Code"
        if row.get("uncertainty_triggered"):
            a = float(row["second_prob_a"])
            b = float(row["second_prob_b"])
            prediction = "a" if a >= b else "b"
            row.update({"prediction": prediction, "correct": prediction == str(row["answer"]).lower(), "prob_a": a, "prob_b": b, "fusion_policy": "dense_authoritative"})
    model = str(next(iter(records.values())).get("model", "")) if records else ""
    write_outputs(args.output, records, sorted(records), model)


if __name__ == "__main__":
    main()
