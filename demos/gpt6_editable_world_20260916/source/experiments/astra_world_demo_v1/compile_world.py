#!/usr/bin/env python3
"""Validate Spatial/Social Code and deterministically compile it to World DSL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_social_code.runtime import execute_social, execute_spatial

from experiments.astra_world_demo_v1.world_dsl import (
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_POSES,
    SUPPORTED_RELATIONS,
    WorldProgramError,
    execute_world,
)


def read_jsonl_text(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def q(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def joined_code(payload: dict[str, Any], key: str) -> str:
    values = payload.get(key)
    if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{key} must be a nonempty array of code strings")
    return "\n".join(item.strip() for item in values if item.strip())


def source_for(record: dict[str, Any], social: bool = False) -> str:
    if social and record["kind"] in {"belief", "goal", "intent", "social_goal"}:
        return "inferred"
    raw = str(record.get("values", {}).get("source", "")).lower()
    if raw in {"video", "transcript", "spatial_trace", "audio"}:
        return "observed"
    return "inferred" if social else "observed"


def evidence_for(record: dict[str, Any]) -> str:
    values = record.get("values", {})
    direct = str(values.get("evidence_id", "")).strip()
    refs = values.get("evidence_ids") or []
    if direct:
        return direct
    if refs:
        return ",".join(str(item) for item in refs)
    return ""


def compile_payloads(spatial_payload: dict[str, Any], social_payload: dict[str, Any]) -> dict[str, Any]:
    spatial_code = joined_code(spatial_payload, "spatial_code")
    spatial_trace_text = execute_spatial(spatial_code)
    spatial_trace = read_jsonl_text(spatial_trace_text)
    evidence_ids = {
        str(row.get("values", {}).get("evidence_id"))
        for row in spatial_trace
        if row.get("values", {}).get("evidence_id")
    }
    social_code = joined_code(social_payload, "social_code")
    social_trace_text = execute_social(
        social_code,
        valid_evidence_ids=evidence_ids,
        require_evidence=True,
    )
    social_trace = read_jsonl_text(social_trace_text)

    entities: dict[str, tuple[str, dict[str, Any]]] = {}
    constraints: list[dict[str, Any]] = []
    utterances: list[dict[str, Any]] = []
    unknowns: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for index, row in enumerate(spatial_trace):
        kind = row.get("kind")
        values = row.get("values", {})
        relation = str(values.get("relation", "")).lower()
        source_record = {"trace": "spatial", "index": index, "record": row}
        if kind == "spatial_fact" and relation == "entity_type":
            entity_id = str(values.get("subject", "")).lower()
            entity_type = str(values.get("object", "")).lower()
            if entity_type not in SUPPORTED_ENTITY_TYPES:
                unsupported.append({**source_record, "reason": f"unsupported entity_type {entity_type!r}"})
            elif entity_id in entities:
                unsupported.append({**source_record, "reason": f"duplicate entity declaration {entity_id!r}"})
            else:
                entities[entity_id] = (entity_type, row)
        elif kind == "spatial_fact" and relation in SUPPORTED_RELATIONS:
            constraints.append({"kind": "relation", "relation": relation, "subject": str(values.get("subject", "")).lower(), "object": str(values.get("object", "")).lower(), "record": row})
        elif kind == "spatial_fact" and relation == "state":
            raw_value = str(values.get("object", "")).strip().lower()
            if raw_value not in {"open", "closed", "true", "false"}:
                unsupported.append({**source_record, "reason": f"unsupported state value {raw_value!r}"})
            else:
                constraints.append({"kind": "state", "entity_id": str(values.get("subject", "")).lower(), "state": "open", "value": raw_value in {"open", "true"}, "record": row})
        elif kind == "spatial_fact" and relation in SUPPORTED_POSES:
            constraints.append({"kind": "pose", "pose": relation, "agent": str(values.get("subject", "")).lower(), "target": str(values.get("object", "")).lower(), "record": row})
        elif kind == "event" and str(values.get("action", "")).lower() in {"utter", "say", "request"}:
            quote = str(values.get("quote", "") or values.get("target", "")).strip()
            utterances.append({"speaker": str(values.get("actor", "unknown_speaker")).lower(), "text": quote, "record": row})
        elif kind == "event" and str(values.get("action", "")).lower() in SUPPORTED_POSES:
            constraints.append({"kind": "pose", "pose": str(values["action"]).lower(), "agent": str(values.get("actor", "")).lower(), "target": str(values.get("target", "")).lower(), "record": row})
        elif kind == "spatial_unknown":
            unknowns.append({"variable": f"{values.get('relation')}:{values.get('subject')}:{values.get('object')}", "record": row})
        else:
            unsupported.append({**source_record, "reason": "record has no World DSL mapping"})

    world_lines: list[str] = []
    mapping: list[dict[str, Any]] = []

    def append(line: str, record: dict[str, Any], mapping_kind: str) -> None:
        world_lines.append(line)
        mapping.append({
            "world_line": len(world_lines),
            "mapping_kind": mapping_kind,
            "source_record": record,
        })

    for entity_id, (entity_type, row) in entities.items():
        values = row["values"]
        append(
            f"W.entity({q(entity_id)}, {q(entity_type)}, source={q(source_for(row))}, evidence_id={q(evidence_for(row))}, confidence={float(values.get('confidence', 1.0))!r}, basis={q(values.get('evidence_span', 'entity visible in cited evidence'))})",
            row,
            "entity_type",
        )

    referenced: set[str] = set()
    for item in constraints:
        row = item["record"]
        values = row["values"]
        provenance = (
            f"source={q(source_for(row))}, evidence_id={q(evidence_for(row))}, "
            f"confidence={float(values.get('confidence', 1.0))!r}, "
            f"basis={q(values.get('evidence_span', values.get('quote', 'cited SSC evidence')))}"
        )
        if item["kind"] == "relation":
            referenced.update((item["subject"], item["object"]))
            line = f"W.relation({q(item['relation'])}, {q(item['subject'])}, {q(item['object'])}, {provenance})"
        elif item["kind"] == "state":
            referenced.add(item["entity_id"])
            line = f"W.state({q(item['entity_id'])}, {q(item['state'])}, {item['value']!r}, {provenance})"
        else:
            referenced.update((item["agent"], item["target"]))
            line = f"W.pose({q(item['pose'])}, {q(item['agent'])}, {q(item['target'])}, {provenance})"
        append(line, row, item["kind"])

    for item in utterances:
        row = item["record"]
        values = row["values"]
        if item["speaker"] != "unknown_speaker":
            referenced.add(item["speaker"])
        append(
            f"W.utterance({q(item['speaker'])}, {q(item['text'])}, source='observed', evidence_id={q(evidence_for(row))}, confidence={float(values.get('confidence', 1.0))!r}, basis={q(values.get('quote', 'verbatim transcript'))})",
            row,
            "utterance",
        )

    included_social = 0
    for index, row in enumerate(social_trace):
        kind = row.get("kind")
        values = row.get("values", {})
        if kind == "observation":
            decisions.append({"trace": "social", "index": index, "decision": "retained_in_social_trace_not_duplicated_in_world_code", "record": row})
            continue
        if kind == "social_unknown":
            unknowns.append({"variable": f"{values.get('agent')}:{values.get('variable')}", "record": row})
            continue
        if kind not in {"belief", "goal", "intent", "social_goal"}:
            unsupported.append({"trace": "social", "index": index, "record": row, "reason": "record has no World DSL mapping"})
            continue
        agent = str(values.get("agent", "unknown_agent")).lower()
        text = str(values.get("fact") or values.get("goal") or values.get("intent") or "").strip()
        if agent not in {"unknown_speaker", "unknown_agent"}:
            referenced.add(agent)
        evidence_id = evidence_for(row)
        append(
            f"W.social({q(kind)}, {q(agent)}, {q(text)}, source='inferred', evidence_id={q(evidence_id)}, confidence={float(values.get('confidence', 0.5))!r}, basis={q(values.get('evidence', 'model inference from cited evidence'))})",
            row,
            "social_state",
        )
        included_social += 1

    for item in unknowns:
        row = item["record"]
        append(
            f"W.unknown({q(item['variable'])}, evidence_id={q(evidence_for(row))}, basis={q(row.get('values', {}).get('evidence_span', 'evidence does not resolve this variable'))})",
            row,
            "unknown",
        )

    missing = sorted(referenced - set(entities))
    if missing:
        unsupported.append({"reason": "referenced entities lack entity_type declarations", "entities": missing})
    if unsupported:
        raise WorldProgramError(json.dumps({"unsupported": unsupported, "compiler_decisions": decisions}, ensure_ascii=False))
    world_lines.append("W.build()")
    world_code = "\n".join(world_lines) + "\n"
    normalized = execute_world(world_code)
    normalized.update({
        "ssc_mapping": mapping,
        "compiler_decisions": decisions,
        "validated_spatial_evidence_ids": sorted(evidence_ids),
        "spatial_trace": spatial_trace,
        "social_trace": social_trace,
    })
    return {
        "world_code": world_code,
        "normalized": normalized,
        "spatial_code": spatial_code,
        "social_code": social_code,
        "spatial_trace_text": spatial_trace_text,
        "social_trace_text": social_trace_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--social", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation: dict[str, Any] = {"accepted": False}
    try:
        result = compile_payloads(
            json.loads(args.spatial.read_text(encoding="utf-8")),
            json.loads(args.social.read_text(encoding="utf-8")),
        )
        (args.output / "world_code.py").write_text(result["world_code"], encoding="utf-8")
        (args.output / "normalized_scene.json").write_text(
            json.dumps(result["normalized"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output / "spatial_trace.jsonl").write_text(result["spatial_trace_text"] + "\n", encoding="utf-8")
        (args.output / "social_trace.jsonl").write_text(result["social_trace_text"] + "\n", encoding="utf-8")
        validation = {
            "accepted": True,
            "spatial_code_executed": True,
            "social_code_executed": True,
            "world_code_executed": True,
            "statement_count": result["normalized"]["statement_count"],
            "unsupported_records": 0,
        }
    except Exception as error:
        validation = {"accepted": False, "error_type": type(error).__name__, "error": str(error)}
    (args.output / "compile_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if not validation["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
