"""Restricted, provenance-preserving DSL for task-local 3D worlds.

This language is deliberately smaller than Blender.  It records only semantic
constraints that must survive generation.  Mesh sizes, materials, lighting,
and camera placement are generator defaults and never masquerade as observed
facts.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from typing import Any


SUPPORTED_ENTITY_TYPES = {"drawer", "bowl", "agent", "vegetable_scrap", "counter", "surface"}
SUPPORTED_RELATIONS = {"inside", "on_top_of", "near"}
SUPPORTED_STATES = {"open"}
SUPPORTED_POSES = {"holding", "reaching_toward"}
SOURCE_CLASSES = {"observed", "inferred", "default", "unknown", "intervention"}
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
EVIDENCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}(?:,[A-Za-z][A-Za-z0-9_.:-]{0,63})*$")


class WorldProgramError(ValueError):
    """Raised when world code is unsafe, incomplete, or semantically invalid."""


@dataclass(frozen=True)
class Statement:
    statement_id: str
    kind: str
    values: dict[str, Any]
    source: str
    evidence_id: str
    confidence: float
    basis: str


class World:
    """In-memory target for restricted ``W.method(...)`` programs."""

    def __init__(self) -> None:
        self.statements: list[Statement] = []
        self.entities: dict[str, str] = {}
        self.built = False

    def _provenance(self, source: str, evidence_id: str, confidence: float, basis: str) -> tuple[str, str, float, str]:
        source = str(source).strip().lower()
        if source not in SOURCE_CLASSES:
            raise WorldProgramError(f"invalid source class {source!r}")
        evidence_id = str(evidence_id).strip()
        if evidence_id and not EVIDENCE_RE.fullmatch(evidence_id):
            raise WorldProgramError(f"invalid evidence_id {evidence_id!r}")
        if source in {"observed", "inferred", "intervention"} and not evidence_id:
            raise WorldProgramError(f"{source} statement requires evidence_id")
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise WorldProgramError("confidence must be in [0, 1]")
        return source, evidence_id, value, str(basis).strip()

    def _add(
        self,
        kind: str,
        values: dict[str, Any],
        source: str,
        evidence_id: str,
        confidence: float,
        basis: str,
    ) -> None:
        if self.built:
            raise WorldProgramError("no statements are allowed after W.build()")
        source, evidence_id, confidence, basis = self._provenance(source, evidence_id, confidence, basis)
        self.statements.append(
            Statement(
                statement_id=f"W{len(self.statements) + 1:02d}",
                kind=kind,
                values=values,
                source=source,
                evidence_id=evidence_id,
                confidence=confidence,
                basis=basis,
            )
        )

    def entity(
        self,
        entity_id: str,
        entity_type: str,
        *,
        source: str,
        evidence_id: str,
        confidence: float = 1.0,
        basis: str = "",
    ) -> None:
        entity_id = str(entity_id).strip().lower()
        entity_type = str(entity_type).strip().lower()
        if not ID_RE.fullmatch(entity_id):
            raise WorldProgramError(f"invalid entity id {entity_id!r}")
        if entity_type not in SUPPORTED_ENTITY_TYPES:
            raise WorldProgramError(
                f"unsupported entity type {entity_type!r}; supported={sorted(SUPPORTED_ENTITY_TYPES)}"
            )
        if entity_id in self.entities:
            raise WorldProgramError(f"duplicate entity {entity_id!r}")
        self.entities[entity_id] = entity_type
        self._add("entity", {"entity_id": entity_id, "entity_type": entity_type}, source, evidence_id, confidence, basis)

    def relation(
        self,
        relation: str,
        subject: str,
        object: str,
        *,
        source: str,
        evidence_id: str,
        confidence: float = 1.0,
        basis: str = "",
    ) -> None:
        relation = str(relation).strip().lower()
        if relation not in SUPPORTED_RELATIONS:
            raise WorldProgramError(
                f"unsupported relation {relation!r}; supported={sorted(SUPPORTED_RELATIONS)}"
            )
        self._require_entities(subject, object)
        self._add("relation", {"relation": relation, "subject": subject, "object": object}, source, evidence_id, confidence, basis)

    def state(
        self,
        entity_id: str,
        state: str,
        value: bool,
        *,
        source: str,
        evidence_id: str,
        confidence: float = 1.0,
        basis: str = "",
    ) -> None:
        state = str(state).strip().lower()
        self._require_entities(entity_id)
        if state not in SUPPORTED_STATES:
            raise WorldProgramError(f"unsupported state {state!r}; supported={sorted(SUPPORTED_STATES)}")
        if type(value) is not bool:
            raise WorldProgramError("state value must be a bool")
        self._add("state", {"entity_id": entity_id, "state": state, "value": value}, source, evidence_id, confidence, basis)

    def pose(
        self,
        pose: str,
        agent: str,
        target: str,
        *,
        source: str,
        evidence_id: str,
        confidence: float = 1.0,
        basis: str = "",
    ) -> None:
        pose = str(pose).strip().lower()
        if pose not in SUPPORTED_POSES:
            raise WorldProgramError(f"unsupported pose {pose!r}; supported={sorted(SUPPORTED_POSES)}")
        self._require_entities(agent, target)
        if self.entities[agent] != "agent":
            raise WorldProgramError(f"pose subject {agent!r} is not an agent")
        self._add("pose", {"pose": pose, "agent": agent, "target": target}, source, evidence_id, confidence, basis)

    def utterance(
        self,
        speaker: str,
        text: str,
        *,
        source: str,
        evidence_id: str,
        confidence: float = 1.0,
        basis: str = "",
    ) -> None:
        speaker = str(speaker).strip().lower()
        if speaker != "unknown_speaker":
            self._require_entities(speaker)
        if not str(text).strip():
            raise WorldProgramError("utterance text must not be empty")
        self._add("utterance", {"speaker": speaker, "text": str(text).strip()}, source, evidence_id, confidence, basis)

    def social(
        self,
        social_type: str,
        agent: str,
        text: str,
        *,
        source: str,
        evidence_id: str,
        confidence: float,
        basis: str,
    ) -> None:
        social_type = str(social_type).strip().lower()
        if social_type not in {"observation", "belief", "goal", "intent", "social_goal"}:
            raise WorldProgramError(f"unsupported social type {social_type!r}")
        if agent not in {"unknown_speaker", "unknown_agent"}:
            self._require_entities(agent)
        if source != "inferred" and social_type in {"belief", "goal", "intent", "social_goal"}:
            raise WorldProgramError(f"{social_type} must be marked inferred")
        self._add(
            "social",
            {"social_type": social_type, "agent": agent, "text": str(text).strip()},
            source,
            evidence_id,
            confidence,
            basis,
        )

    def unknown(
        self,
        variable: str,
        *,
        evidence_id: str = "",
        basis: str = "",
    ) -> None:
        self._add("unknown", {"variable": str(variable).strip()}, "unknown", evidence_id, 0.0, basis)

    def build(self) -> None:
        if self.built:
            raise WorldProgramError("W.build() may appear only once")
        self.built = True

    def _require_entities(self, *entity_ids: str) -> None:
        missing = [value for value in entity_ids if value not in self.entities]
        if missing:
            raise WorldProgramError(f"referenced entities must be declared first: {missing}")

    def normalized(self, code: str) -> dict[str, Any]:
        if not self.built:
            raise WorldProgramError("program must end with W.build()")
        if not 8 <= len(self.statements) <= 20:
            raise WorldProgramError(
                f"world code must contain 8-20 semantic statements, got {len(self.statements)}"
            )
        self._validate_constraints()
        return {
            "schema_version": "task_local_world_v1",
            "supported_subset": {
                "entity_types": sorted(SUPPORTED_ENTITY_TYPES),
                "relations": sorted(SUPPORTED_RELATIONS),
                "states": sorted(SUPPORTED_STATES),
                "poses": sorted(SUPPORTED_POSES),
            },
            "statement_count": len(self.statements),
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "statements": [asdict(row) for row in self.statements],
            "generator_defaults_boundary": [
                "metric dimensions and depth",
                "materials and mesh tessellation",
                "camera, lighting, and background floor",
                "unconstrained object spacing",
            ],
        }

    def _validate_constraints(self) -> None:
        placements: dict[str, dict[str, Any]] = {}
        held: set[str] = set()
        states: set[tuple[str, str]] = set()
        for row in self.statements:
            values = row.values
            if row.kind == "relation" and values["relation"] in {"inside", "on_top_of"}:
                subject = values["subject"]
                if subject in placements:
                    raise WorldProgramError(
                        f"conflicting placement constraints for {subject!r}: "
                        f"{placements[subject]} and {values}"
                    )
                placements[subject] = values
                if values["relation"] == "inside" and self.entities[values["object"]] not in {"drawer", "bowl"}:
                    raise WorldProgramError("inside target must be a drawer or bowl in the implemented geometry subset")
            elif row.kind == "pose" and values["pose"] == "holding":
                held.add(values["target"])
            elif row.kind == "state":
                key = (values["entity_id"], values["state"])
                if key in states:
                    raise WorldProgramError(f"duplicate state declaration for {key!r}")
                states.add(key)
                if values["state"] == "open" and self.entities[values["entity_id"]] != "drawer":
                    raise WorldProgramError("open state is implemented only for drawer entities")
        conflict = sorted(held & set(placements))
        if conflict:
            raise WorldProgramError(f"objects cannot be both held and placed: {conflict}")


ALLOWED_METHODS = {"entity", "relation", "state", "pose", "utterance", "social", "unknown", "build"}


def execute_world(code: str) -> dict[str, Any]:
    """Validate and execute literal-only world code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise WorldProgramError(f"syntax error: {error.msg}") from error
    allowed_nodes = (
        ast.Module,
        ast.Expr,
        ast.Call,
        ast.Attribute,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.keyword,
        ast.UnaryOp,
        ast.USub,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise WorldProgramError(f"unsupported syntax {type(node).__name__}")
    world = World()
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            raise WorldProgramError("every line must be one W.method(...) call")
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "W":
            raise WorldProgramError("calls must target W")
        if call.func.attr not in ALLOWED_METHODS:
            raise WorldProgramError(f"unsupported W method {call.func.attr!r}")
        if any(keyword.arg is None for keyword in call.keywords):
            raise WorldProgramError("expanded keyword arguments are forbidden")
        method = getattr(world, call.func.attr)
        try:
            inspect.signature(method).bind(
                *([None] * len(call.args)),
                **{keyword.arg: None for keyword in call.keywords},
            )
        except TypeError as error:
            raise WorldProgramError(f"invalid W.{call.func.attr} call: {error}") from error
    try:
        exec(compile(tree, "<world-code>", "exec"), {"__builtins__": {}}, {"W": world})
    except (TypeError, ValueError, WorldProgramError) as error:
        if isinstance(error, WorldProgramError):
            raise
        raise WorldProgramError(f"invalid runtime value: {error}") from error
    return world.normalized(code)


def dumps_normalized(code: str) -> str:
    return json.dumps(execute_world(code), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
