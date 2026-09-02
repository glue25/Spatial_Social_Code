"""Offline consistency checks for the public aggregate result artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{left!r} != {right!r}")


def main() -> None:
    summary = load_json("reports/submission_results_summary.json")
    paired = {row["benchmark"]: row for row in summary["paired_results"]}

    for row in paired.values():
        close(row["method_accuracy"], row["method_correct"] / row["total"])
        close(row["direct_accuracy"], row["direct_correct"] / row["total"])
        close(
            row["delta_pp"],
            100.0 * (row["method_accuracy"] - row["direct_accuracy"]),
        )

    mmtom_source = load_json("reports/mmtom_iclr_ablation_summary.json")
    mmtom = paired["MMToM"]
    source_direct = mmtom_source["main_component_ablation"]["direct"]
    source_method = mmtom_source["main_component_ablation"][
        "full_hybrid_dense_authoritative"
    ]
    assert (source_direct["correct"], source_direct["total"]) == (
        mmtom["direct_correct"],
        mmtom["total"],
    )
    assert (source_method["correct"], source_method["total"]) == (
        mmtom["method_correct"],
        mmtom["total"],
    )

    habitat = summary["habitat"]
    runs = habitat["runs_per_condition"]
    for condition in ("direct", "spatial_only", "social_only", "spatial_social"):
        values = habitat[condition]
        close(values["success_rate"], values["successes"] / runs)
    close(
        habitat["spatial_social_minus_direct_completion_pp"],
        100.0
        * (
            habitat["spatial_social"]["mean_goal_completion"]
            - habitat["direct"]["mean_goal_completion"]
        ),
    )
    assert habitat["strongest_condition"] == "social_only"

    assert summary["external_references"]
    assert all(not row["strictly_comparable"] for row in summary["external_references"])
    assert load_json("reports/comind_grounding_oof_summary.json")["official_hidden_test"] is False

    required = [
        "README.md",
        "docs/benchmark_sota_zh.md",
        "reports/submission_results_and_baselines_zh.md",
        "reports/habitat_spatial_social_status_zh.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"missing public artifact files: {missing}")
    print("reported-result checks passed")


if __name__ == "__main__":
    main()
