#!/usr/bin/env python3
"""Analyze executable SSC environment cases without treating four cases as a benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    tasks_dir = args.tasks.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    task_data = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in tasks_dir.glob("*.json") if path.name != "manifest.json"
    }
    summaries = sorted(run_root.glob("results/*/*/results/*/closed_loop_artifacts/summary.json"))
    if len(summaries) != 8:
        raise SystemExit(f"expected 8 completed artifacts, found {len(summaries)}")

    rows = []
    for summary_path in summaries:
        artifact = summary_path.parent
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        case_id = summary_path.parents[3].name
        condition = summary_path.parents[4].name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks = jsonl(artifact / "physical_checks.jsonl")
        final = checks[-1]
        events = jsonl(artifact / "events.jsonl")
        messages = [
            {
                "turn": event.get("turn"), "sender": event.get("sender"),
                "receivers": event.get("receivers"), "message": event.get("message"),
            }
            for event in events if event.get("event_type") == "message_delivered"
        ]
        relevant_messages = [
            message for message in messages
            if "spray_bottle_3" in str(message.get("message")) or "table_12" in str(message.get("message"))
        ]
        usage_files = list(summary_path.parents[1].glob("**/api_usage.jsonl"))
        usage = [item for path in usage_files for item in jsonl(path)]
        task = task_data[case_id]
        generated = task["ssc_generated_case"]
        rows.append({
            "case_id": case_id,
            "condition": condition,
            "task_id": manifest["task_id"],
            "seed": int(manifest["seed"]),
            "key_fact_owner": generated["key_fact_owner"],
            "intervention": generated["intervention"],
            "strict_success": bool(final["strict_success"]),
            "completed_goals": int(final["completed_goals"]),
            "total_goals": int(final["total_goals"]),
            "turns": int(final["turn"]),
            "api_calls": sum(int(item.get("api_calls", 0) or 0) for item in usage),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in usage),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in usage),
            "delivered_messages": messages,
            "target_relevant_messages": relevant_messages,
            "artifact_dir": str(artifact.relative_to(ROOT)),
        })

    pairs = []
    for case_id in sorted(task_data):
        pair = {row["condition"]: row for row in rows if row["case_id"] == case_id}
        if set(pair) != {"direct", "dynamic_ssc"}:
            raise SystemExit(f"incomplete pair for {case_id}: {sorted(pair)}")
        pairs.append({
            "case_id": case_id,
            "key_fact_owner": pair["direct"]["key_fact_owner"],
            "intervention": pair["direct"]["intervention"],
            "direct_goals": f"{pair['direct']['completed_goals']}/{pair['direct']['total_goals']}",
            "dynamic_ssc_goals": f"{pair['dynamic_ssc']['completed_goals']}/{pair['dynamic_ssc']['total_goals']}",
            "direct_success": pair["direct"]["strict_success"],
            "dynamic_ssc_success": pair["dynamic_ssc"]["strict_success"],
            "direct_target_relevant_messages": pair["direct"]["target_relevant_messages"],
            "dynamic_target_relevant_messages": pair["dynamic_ssc"]["target_relevant_messages"],
        })

    manifest = json.loads((tasks_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "ssc_environment_case_results_v1",
        "status": "illustrative_fixed_scene_cases_not_population_benchmark",
        "model": "qwen3-vl-8b-instruct",
        "same_physical_state_and_goal": manifest["same_physical_state_and_goal"],
        "answer_bearing_fields_removed": not any(
            manifest["leakage_audit"][key]
            for key in ("golden_trajectory_present", "literal_tom_expected_response_present", "target_answer_present")
        ),
        "pairs": pairs,
        "runs": rows,
        "interpretation_boundary": (
            "Four interventions in one episode illustrate executable information attribution; "
            "they do not estimate population-level accuracy or significance."
        ),
    }
    json_path = output / "habitat_ssc_environment_cases_results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# Executable SSC-generated Habitat cases", "",
        "All cases use the same HSSD stage, episode, spawns, physical initial state, goals and seed. ",
        "Only the owner of one private location fact changes; the fourth case adds an irrelevant fact.", "",
        "| Case | Key fact owner | Direct | Dynamic SSC |", "|---|---|---:|---:|",
    ]
    for row in pairs:
        md.append(
            f"| {row['case_id']} | {row['key_fact_owner'] or 'withheld'} | "
            f"{row['direct_goals']} | {row['dynamic_ssc_goals']} |"
        )
    md.extend([
        "", "These are mechanism cases, not an accuracy benchmark. Golden trajectories, target answers, "
        "and persisted literal-ToM expected responses were removed before execution.", "",
    ])
    (output / "habitat_ssc_environment_cases_results.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
