#!/usr/bin/env python3
"""Create three single-factor intervention scenes from one normalized world."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.astra_world_demo_v1.world_dsl import execute_world


def q(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def to_code(statements: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in statements:
        kind, values = row["kind"], row["values"]
        source = row["source"]
        evidence_id = row.get("evidence_id", "")
        confidence = row.get("confidence", 1.0)
        basis = row.get("basis", "")
        provenance = f"source={q(source)}, evidence_id={q(evidence_id)}, confidence={confidence!r}, basis={q(basis)}"
        if kind == "entity":
            line = f"W.entity({q(values['entity_id'])}, {q(values['entity_type'])}, {provenance})"
        elif kind == "relation":
            line = f"W.relation({q(values['relation'])}, {q(values['subject'])}, {q(values['object'])}, {provenance})"
        elif kind == "state":
            line = f"W.state({q(values['entity_id'])}, {q(values['state'])}, {bool(values['value'])!r}, {provenance})"
        elif kind == "pose":
            line = f"W.pose({q(values['pose'])}, {q(values['agent'])}, {q(values['target'])}, {provenance})"
        elif kind == "utterance":
            line = f"W.utterance({q(values['speaker'])}, {q(values['text'])}, {provenance})"
        elif kind == "social":
            line = f"W.social({q(values['social_type'])}, {q(values['agent'])}, {q(values['text'])}, {provenance})"
        elif kind == "unknown":
            line = f"W.unknown({q(values['variable'])}, evidence_id={q(evidence_id)}, basis={q(basis)})"
        else:
            raise ValueError(f"unsupported normalized statement kind {kind!r}")
        lines.append(line)
    lines.append("W.build()")
    return "\n".join(lines) + "\n"


def entity_types(statements: list[dict[str, Any]]) -> dict[str, str]:
    return {
        row["values"]["entity_id"]: row["values"]["entity_type"]
        for row in statements
        if row["kind"] == "entity"
    }


def references(row: dict[str, Any], entity_id: str) -> bool:
    values = row.get("values", {})
    return entity_id in {
        values.get("entity_id"), values.get("subject"), values.get("object"),
        values.get("agent"), values.get("target"), values.get("speaker"),
    }


def save_case(root: Path, case_id: str, statements: list[dict[str, Any]], intervention: dict[str, Any]) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    code = to_code(statements)
    normalized = execute_world(code)
    normalized["case_id"] = case_id
    normalized["intervention"] = intervention
    (case_dir / "world_code.py").write_text(code, encoding="utf-8")
    (case_dir / "normalized_scene.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "intervention.json").write_text(
        json.dumps(intervention, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--base-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_scene.read_text(encoding="utf-8"))
    statements = copy.deepcopy(base["statements"])
    save_case(root, "base", statements, {"type": "none", "grounding": "original_case"})
    types = entity_types(statements)
    counters = sorted(key for key, value in types.items() if value in {"counter", "surface"})
    if not counters:
        raise ValueError("intervention requires an SSC-declared counter or surface")
    nested_bowls = [
        row for row in statements
        if row["kind"] == "relation"
        and row["values"]["relation"] == "inside"
        and types.get(row["values"]["subject"]) == "bowl"
        and types.get(row["values"]["object"]) in {"bowl", "drawer"}
    ]
    if not nested_bowls:
        raise ValueError("intervention requires an SSC-declared bowl inside a bowl or drawer")
    nested_bowls.sort(key=lambda row: types.get(row["values"]["object"]) != "bowl")
    moved = nested_bowls[0]["values"]["subject"]
    old_target = nested_bowls[0]["values"]["object"]

    moved_statements = copy.deepcopy(statements)
    changed = next(
        row for row in moved_statements
        if row["kind"] == "relation"
        and row["values"] == nested_bowls[0]["values"]
    )
    changed["values"] = {"relation": "on_top_of", "subject": moved, "object": counters[0]}
    changed.update({
        "source": "intervention",
        "evidence_id": "I1_MOVE_BOWL",
        "confidence": 1.0,
        "basis": f"single-factor intervention: replace inside({moved},{old_target}) with on_top_of({moved},{counters[0]})",
    })
    save_case(root, "intervention_move_bowl", moved_statements, {
        "type": "relation_replacement",
        "factor": "bowl_support",
        "subject": moved,
        "before": ["inside", old_target],
        "after": ["on_top_of", counters[0]],
        "is_original_video_fact": False,
    })

    closed_statements = copy.deepcopy(statements)
    drawer_state = next(
        row for row in closed_statements
        if row["kind"] == "state"
        and row["values"]["state"] == "open"
        and types.get(row["values"]["entity_id"]) == "drawer"
    )
    drawer_state["values"]["value"] = not bool(drawer_state["values"]["value"])
    drawer_state.update({
        "source": "intervention",
        "evidence_id": "I2_TOGGLE_DRAWER",
        "confidence": 1.0,
        "basis": "single-factor intervention: toggle drawer open state",
    })
    save_case(root, "intervention_toggle_drawer", closed_statements, {
        "type": "state_toggle",
        "factor": "drawer_open",
        "entity_id": drawer_state["values"]["entity_id"],
        "after": drawer_state["values"]["value"],
        "is_original_video_fact": False,
    })

    removed_statements = [row for row in copy.deepcopy(statements) if not references(row, moved)]
    save_case(root, "intervention_remove_bowl", removed_statements, {
        "type": "entity_removal",
        "factor": "task_related_object_presence",
        "removed_entity": moved,
        "causal_relations_removed": [row["values"] for row in statements if row["kind"] != "entity" and references(row, moved)],
        "is_original_video_fact": False,
    })
    print(json.dumps({"output": str(root), "cases": sorted(path.name for path in root.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
