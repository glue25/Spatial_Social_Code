"""Task-profile validation for Spatial Evidence v2 traces.

The contract deliberately keeps metric geometry optional.  A task profile
states which capabilities its downstream consumer needs; every numeric
geometry record is nevertheless required to carry time, unit, coordinate
frame, and provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


LOCATION_RELATIONS = {
    "at", "in", "inside", "on", "on_top_of", "near", "left_of", "right_of",
    "front_of", "behind",
}
CONTAINER_RELATIONS = {"in", "inside", "on", "on_top_of"}
SPEECH_ACTIONS = {"say", "says", "said", "speak", "speaks", "utter", "utters"}


@dataclass(frozen=True)
class TaskProfile:
    name: str
    required_capabilities: frozenset[str]
    geometry_required: bool = False


TASK_PROFILES = {
    "mmtom_belief": TaskProfile(
        "mmtom_belief",
        frozenset({"location_relation", "temporal_event", "observation_access"}),
    ),
    "ma_egoqa_social": TaskProfile(
        "ma_egoqa_social",
        frozenset({"entity", "interaction", "speaker", "time_span"}),
    ),
    "comind_localization": TaskProfile(
        "comind_localization",
        frozenset({"entity", "bbox_2d", "interaction"}),
    ),
    "habitat_execution": TaskProfile(
        "habitat_execution",
        frozenset({"container_relation", "grasp_state", "joint_state"}),
    ),
    "environment_instantiation": TaskProfile(
        "environment_instantiation",
        frozenset({"asset", "pose_3d"}),
        geometry_required=True,
    ),
}


class SpatialContractError(ValueError):
    pass


def read_trace(trace: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(trace).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialContractError(f"invalid JSON trace line {line_number}: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("values"), dict):
            raise SpatialContractError(f"trace line {line_number} is not a runtime record")
        records.append(row)
    return records


def capabilities(records: Iterable[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for row in records:
        kind = str(row.get("kind", ""))
        values = row.get("values") or {}
        if kind == "entity":
            found.add("entity")
        elif kind == "bbox_2d":
            found.add("bbox_2d")
        elif kind == "pose_3d":
            found.add("pose_3d")
        elif kind == "asset":
            found.add("asset")
        elif kind == "visibility":
            found.add("observation_access")
        elif kind == "event":
            found.update({"temporal_event", "interaction"})
            if str(values.get("action", "")).lower() in SPEECH_ACTIONS:
                found.add("speaker")
            if values.get("evidence_span"):
                found.add("time_span")
        elif kind == "spatial_fact":
            relation = str(values.get("relation", "")).lower()
            if relation in LOCATION_RELATIONS:
                found.add("location_relation")
            if relation in CONTAINER_RELATIONS:
                found.add("container_relation")
            if relation in {"holding", "held_by", "grasped_by"}:
                found.add("grasp_state")
            if relation in {"joint_state", "open", "closed"}:
                found.add("joint_state")
        elif kind == "spatial_state":
            attribute = str(values.get("attribute", "")).lower()
            if attribute in {"holding", "held", "held_by", "grasped_by"}:
                found.add("grasp_state")
            if attribute in {"joint_state", "open", "closed"}:
                found.add("joint_state")
    return found


def validate_spatial_trace(trace: str, profile: str) -> dict[str, Any]:
    if profile not in TASK_PROFILES:
        raise SpatialContractError(f"unknown task profile: {profile!r}")
    records = read_trace(trace)
    found = capabilities(records)
    required = TASK_PROFILES[profile].required_capabilities
    missing = sorted(required - found)
    report = {
        "schema_version": "spatial_evidence_v2",
        "profile": profile,
        "record_count": len(records),
        "geometry_required": TASK_PROFILES[profile].geometry_required,
        "required_capabilities": sorted(required),
        "found_capabilities": sorted(found),
        "missing_capabilities": missing,
        "accepted": not missing,
    }
    if missing:
        raise SpatialContractError(
            f"profile {profile} missing required capabilities: {', '.join(missing)}"
        )
    return report

