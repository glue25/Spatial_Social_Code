#!/usr/bin/env python3
"""Build executable SSC information-attribution cases in one fixed Habitat scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workdirs/habitat_enacttom/frozen12/tasks/standard_108294897_176710602_1566.json"
OUTPUT = ROOT / "reports/habitat_generated_environment_cases/tasks"
KEY_FACT = "spray_bottle_3 starts on table_12 in living_room_1."
IRRELEVANT_FACT = "chair_16 is in dining_room_1."

PHYSICAL_FIELDS = (
    "scene_id", "episode_id", "agent_spawns", "initial_states", "problem_pddl",
    "functional_goal_pddl", "agent_actions",
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def clean_source(source: dict) -> dict:
    task = copy.deepcopy(source)
    for field in (
        "golden_trajectory", "golden_trajectory_metadata", "literal_tom_probes",
        "calibration", "oracle_schedule", "coordination_cards",
        "coordination_replay_responses", "replay_audit_specs",
    ):
        task.pop(field, None)
    task["tom_level"] = 0
    task["tom_reasoning"] = "Not evaluated in the generated physical-planning cases."
    task["runtime_semantics_version"] = "functional_only_ssc_generated_v1"
    return task


def without_key_fact(secrets: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        agent: [fact for fact in facts if fact != KEY_FACT]
        for agent, facts in secrets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    base = clean_source(source)
    baseline_secrets = copy.deepcopy(base["agent_secrets"])
    stripped = without_key_fact(baseline_secrets)
    assert KEY_FACT in baseline_secrets["agent_1"]

    cases = {
        "fact_owner_agent1": {
            "secrets": baseline_secrets,
            "intervention": "none; original private information owner",
            "key_fact_owner": "agent_1",
        },
        "fact_owner_agent0": {
            "secrets": {**copy.deepcopy(stripped), "agent_0": stripped["agent_0"] + [KEY_FACT]},
            "intervention": "move exact spray-bottle location from agent_1 to agent_0",
            "key_fact_owner": "agent_0",
        },
        "fact_withheld": {
            "secrets": copy.deepcopy(stripped),
            "intervention": "remove the exact spray-bottle location from both agents",
            "key_fact_owner": None,
        },
        "irrelevant_fact_control": {
            "secrets": {**copy.deepcopy(baseline_secrets), "agent_0": baseline_secrets["agent_0"] + [IRRELEVANT_FACT]},
            "intervention": "add one task-irrelevant furniture-room fact to agent_0",
            "key_fact_owner": "agent_1",
        },
    }

    physical_signature = {field: base.get(field) for field in PHYSICAL_FIELDS}
    physical_hash = canonical_hash(physical_signature)
    rows = []
    for case_id, spec in cases.items():
        task = copy.deepcopy(base)
        task["task_id"] = f"sscgen-1566-{case_id}"
        task["title"] = f"SSC-generated 1566 case: {case_id}"
        task["agent_secrets"] = spec["secrets"]
        task["ssc_generated_case"] = {
            "case_id": case_id,
            "intervention": spec["intervention"],
            "key_fact": KEY_FACT,
            "key_fact_owner": spec["key_fact_owner"],
            "physical_signature_sha256": physical_hash,
            "answer_bearing_fields_removed": True,
        }
        path = output / f"{case_id}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append({
            "case_id": case_id,
            "task_path": str(path.relative_to(ROOT)),
            "task_sha256": digest_bytes(path.read_bytes()),
            "intervention": spec["intervention"],
            "key_fact_owner": spec["key_fact_owner"],
            "physical_signature_sha256": physical_hash,
        })

    manifest = {
        "schema_version": "ssc_generated_environment_cases_v1",
        "source_task": str(source_path.relative_to(ROOT)),
        "source_task_sha256": digest_bytes(source_path.read_bytes()),
        "scene_id": base["scene_id"],
        "episode_id": base["episode_id"],
        "physical_signature_sha256": physical_hash,
        "same_physical_state_and_goal": len({row["physical_signature_sha256"] for row in rows}) == 1,
        "cases": rows,
        "leakage_audit": {
            "golden_trajectory_present": False,
            "literal_tom_expected_response_present": False,
            "target_answer_present": False,
            "functional_goal_retained_for_environment_execution": True,
        },
        "scope": "Existing HSSD assets instantiated with SSC-derived social-state interventions; no new mesh synthesis.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
