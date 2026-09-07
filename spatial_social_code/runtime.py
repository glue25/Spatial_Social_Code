"""Small, safe runtimes for generated Spatial-Social Code programs."""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Record:
    kind: str
    values: dict[str, Any]


class _Memory:
    def __init__(self) -> None:
        self.records: list[Record] = []

    def _add(self, kind: str, **values: Any) -> None:
        self.records.append(Record(kind=kind, values=values))

    def trace(self) -> str:
        return "\n".join(
            json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
            for record in self.records
        )


def _timestamp(value: Any) -> int:
    if type(value) is not int:
        raise ValueError(f"timestamp must be an integer, got {value!r}")
    return value


_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_CLAIM_ID_RE = re.compile(r"^C[1-9][0-9]{0,2}$")
_EVENT_ROLES = {
    "initiator",
    "executor",
    "speaker",
    "reactor",
    "observer",
    "participant",
    "unknown",
}
_EVENT_PHASES = {"before", "during", "after", "unknown"}
_QUERY_VARIABLE_TYPES = {
    "world_state",
    "observation",
    "belief",
    "intent",
    "responsibility/causality",
    "temporal",
    "unknown",
}
_HYPOTHESIS_CONSTRAINTS = {"all", "any", "order"}
_CLAUSE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


def _enum_value(value: Any, allowed: set[str], field: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"invalid {field}: {value!r}; expected one of {sorted(allowed)!r}")
    return normalized


def _optional_provenance(
    evidence_id: str = "",
    clip_id: str = "",
    observer: str = "",
    evidence_span: str = "",
    quote: str = "",
    claim_ids: str = "",
) -> dict[str, str]:
    claims = list(
        dict.fromkeys(value.strip() for value in str(claim_ids).split(",") if value.strip())
    )
    invalid_claims = [value for value in claims if not _CLAIM_ID_RE.fullmatch(value)]
    if invalid_claims:
        raise ValueError(f"invalid claim_ids: {invalid_claims!r}")
    values = {
        "evidence_id": str(evidence_id).strip(),
        "clip_id": str(clip_id).strip(),
        "observer": str(observer).strip(),
        "evidence_span": str(evidence_span).strip(),
        "quote": str(quote).strip(),
        "claim_ids": ",".join(claims),
    }
    if values["evidence_id"] and not _EVIDENCE_ID_RE.fullmatch(values["evidence_id"]):
        raise ValueError(f"invalid evidence_id: {values['evidence_id']!r}")
    return {key: value for key, value in values.items() if value}


class SpatialMemory(_Memory):
    def __init__(self) -> None:
        super().__init__()
        self.evidence_ids: set[str] = set()

    def _provenance(self, **values: str) -> dict[str, str]:
        provenance = _optional_provenance(**values)
        evidence_id = provenance.get("evidence_id", "")
        if evidence_id in self.evidence_ids:
            raise ValueError(f"duplicate spatial evidence_id: {evidence_id!r}")
        if evidence_id:
            self.evidence_ids.add(evidence_id)
        return provenance

    def fact(
        self,
        t: int,
        relation: str,
        subject: str,
        object: str,
        source: str = "text",
        confidence: float = 1.0,
        evidence_id: str = "",
        clip_id: str = "",
        observer: str = "",
        evidence_span: str = "",
        quote: str = "",
        claim_ids: str = "",
    ) -> None:
        self._add(
            "spatial_fact",
            t=_timestamp(t),
            relation=relation,
            subject=subject,
            object=object,
            source=source,
            confidence=float(confidence),
            **self._provenance(
                evidence_id=evidence_id,
                clip_id=clip_id,
                observer=observer,
                evidence_span=evidence_span,
                quote=quote,
                claim_ids=claim_ids,
            ),
        )

    def event(
        self,
        t: int,
        actor: str,
        action: str,
        target: str,
        source: str = "text",
        confidence: float = 1.0,
        evidence_id: str = "",
        clip_id: str = "",
        observer: str = "",
        evidence_span: str = "",
        quote: str = "",
        claim_ids: str = "",
        event_role: str = "",
        phase: str = "",
    ) -> None:
        values: dict[str, Any] = dict(
            t=_timestamp(t),
            actor=actor,
            action=action,
            target=target,
            source=source,
            confidence=float(confidence),
            **self._provenance(
                evidence_id=evidence_id,
                clip_id=clip_id,
                observer=observer,
                evidence_span=evidence_span,
                quote=quote,
                claim_ids=claim_ids,
            ),
        )
        if str(event_role).strip():
            values["event_role"] = _enum_value(event_role, _EVENT_ROLES, "event_role")
        if str(phase).strip():
            values["phase"] = _enum_value(phase, _EVENT_PHASES, "phase")
        self._add("event", **values)

    def visible(
        self,
        t: int,
        observer: str,
        object: str,
        value: bool,
        source: str = "video",
        confidence: float = 1.0,
        evidence_id: str = "",
        clip_id: str = "",
        evidence_span: str = "",
        quote: str = "",
        claim_ids: str = "",
    ) -> None:
        self._add(
            "visibility",
            t=_timestamp(t),
            observer=observer,
            object=object,
            value=bool(value),
            source=source,
            confidence=float(confidence),
            **self._provenance(
                evidence_id=evidence_id,
                clip_id=clip_id,
                evidence_span=evidence_span,
                quote=quote,
                claim_ids=claim_ids,
            ),
        )

    def unknown(
        self,
        t: int,
        relation: str,
        subject: str,
        object: str,
        source: str = "unknown",
        confidence: float = 0.0,
        evidence_id: str = "",
        clip_id: str = "",
        observer: str = "",
        evidence_span: str = "",
        quote: str = "",
        claim_ids: str = "",
    ) -> None:
        self._add(
            "spatial_unknown",
            t=_timestamp(t),
            relation=relation,
            subject=subject,
            object=object,
            source=source,
            confidence=float(confidence),
            **self._provenance(
                evidence_id=evidence_id,
                clip_id=clip_id,
                observer=observer,
                evidence_span=evidence_span,
                quote=quote,
                claim_ids=claim_ids,
            ),
        )


class SocialMemory(_Memory):
    def __init__(
        self,
        valid_evidence_ids: set[str] | None = None,
        require_evidence: bool = False,
    ) -> None:
        super().__init__()
        self.valid_evidence_ids = valid_evidence_ids
        self.require_evidence = require_evidence

    def _evidence_references(self, evidence_ids: str) -> dict[str, list[str]]:
        references = list(
            dict.fromkeys(
                value.strip()
                for value in str(evidence_ids).split(",")
                if value.strip()
            )
        )
        if self.require_evidence and not references:
            raise ValueError("social record requires at least one spatial evidence_id")
        invalid = [
            value
            for value in references
            if not _EVIDENCE_ID_RE.fullmatch(value)
            or (
                self.valid_evidence_ids is not None
                and value not in self.valid_evidence_ids
            )
        ]
        if invalid:
            raise ValueError(f"invalid or unknown social evidence_ids: {invalid!r}")
        return {"evidence_ids": references} if references else {}

    def candidate(
        self,
        agent: str,
        variable: str,
        label: str,
        hypothesis: str,
        prior: float = 0.5,
        source: str = "answer_option",
        variable_type: str = "unknown",
        constraint: str = "ALL",
    ) -> None:
        """Register an answer-aligned hypothesis after evidence extraction."""
        normalized_type = _enum_value(
            variable_type, _QUERY_VARIABLE_TYPES, "query variable type"
        )
        normalized_constraint = _enum_value(
            constraint, _HYPOTHESIS_CONSTRAINTS, "hypothesis constraint"
        ).upper()
        self._add(
            "query_candidate",
            agent=agent,
            variable=variable,
            label=label,
            hypothesis=hypothesis,
            prior=float(prior),
            source=source,
            variable_type=normalized_type,
            constraint=normalized_constraint,
        )

    def clause(
        self,
        label: str,
        clause_id: str,
        text: str,
        subject: str = "",
        predicate: str = "",
        required: bool = True,
        endpoint: str = "",
    ) -> None:
        """Register one atomic, answer-option clause."""
        normalized_id = str(clause_id).strip().upper()
        if not _CLAUSE_ID_RE.fullmatch(normalized_id):
            raise ValueError(f"invalid clause_id: {clause_id!r}")
        self._add(
            "hypothesis_clause",
            label=str(label).strip().upper(),
            clause_id=normalized_id,
            text=str(text).strip(),
            subject=str(subject).strip(),
            predicate=str(predicate).strip(),
            required=bool(required),
            endpoint=str(endpoint).strip(),
        )

    def constraint(self, label: str, kind: str, clause_ids: str) -> None:
        """Register an ALL, ANY, or ORDER relation over atomic clauses."""
        normalized_kind = _enum_value(
            kind, _HYPOTHESIS_CONSTRAINTS, "hypothesis constraint"
        ).upper()
        members = [
            value.strip().upper()
            for value in str(clause_ids).split(",")
            if value.strip()
        ]
        if not members or any(not _CLAUSE_ID_RE.fullmatch(value) for value in members):
            raise ValueError(f"invalid constraint clause_ids: {clause_ids!r}")
        if len(members) != len(set(members)):
            raise ValueError("constraint clause_ids must be unique")
        self._add(
            "hypothesis_constraint",
            label=str(label).strip().upper(),
            constraint=normalized_kind,
            clause_ids=members,
        )

    def _clause_verdict(
        self,
        verdict: str,
        t: int,
        label: str,
        clause_id: str,
        evidence_ids: str,
        evidence: str,
    ) -> None:
        normalized_id = str(clause_id).strip().upper()
        if not _CLAUSE_ID_RE.fullmatch(normalized_id):
            raise ValueError(f"invalid clause_id: {clause_id!r}")
        self._add(
            "hypothesis_evidence",
            t=_timestamp(t),
            label=str(label).strip().upper(),
            clause_id=normalized_id,
            verdict=verdict,
            evidence=str(evidence).strip(),
            **self._evidence_references(evidence_ids),
        )

    def support(
        self,
        t: int,
        label: str,
        clause_id: str,
        evidence_ids: str,
        evidence: str = "",
    ) -> None:
        self._clause_verdict(
            "support", t, label, clause_id, evidence_ids, evidence
        )

    def contradict(
        self,
        t: int,
        label: str,
        clause_id: str,
        evidence_ids: str,
        evidence: str = "",
    ) -> None:
        self._clause_verdict(
            "contradict", t, label, clause_id, evidence_ids, evidence
        )

    def clause_unknown(
        self,
        t: int,
        label: str,
        clause_id: str,
        evidence: str = "",
    ) -> None:
        """Record missing evidence without treating it as contradiction."""
        normalized_id = str(clause_id).strip().upper()
        if not _CLAUSE_ID_RE.fullmatch(normalized_id):
            raise ValueError(f"invalid clause_id: {clause_id!r}")
        self._add(
            "hypothesis_evidence",
            t=_timestamp(t),
            label=str(label).strip().upper(),
            clause_id=normalized_id,
            verdict="unknown",
            evidence=str(evidence).strip(),
            evidence_ids=[],
        )

    def query_evidence(
        self,
        t: int,
        agent: str,
        variable: str,
        hypothesis: str,
        likelihood: float,
        confidence: float,
        evidence: str,
    ) -> None:
        """Record calibrated evidence for one answer-aligned hypothesis."""
        self._add(
            "query_evidence",
            t=_timestamp(t),
            agent=agent,
            variable=variable,
            hypothesis=hypothesis,
            likelihood=float(likelihood),
            confidence=float(confidence),
            evidence=evidence,
        )

    def observe(
        self,
        t: int,
        agent: str,
        fact: str,
        source: str = "spatial_trace",
        confidence: float = 1.0,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "observation",
            t=_timestamp(t),
            agent=agent,
            fact=fact,
            source=source,
            confidence=float(confidence),
            **self._evidence_references(evidence_ids),
        )

    def belief(
        self,
        t: int,
        agent: str,
        fact: str,
        confidence: float,
        evidence: str,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "belief",
            t=_timestamp(t),
            agent=agent,
            fact=fact,
            confidence=float(confidence),
            evidence=evidence,
            **self._evidence_references(evidence_ids),
        )

    def goal(
        self,
        agent: str,
        goal: str,
        confidence: float,
        evidence: str,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "goal",
            agent=agent,
            goal=goal,
            confidence=float(confidence),
            evidence=evidence,
            **self._evidence_references(evidence_ids),
        )

    def intent(
        self,
        t: int,
        agent: str,
        intent: str,
        confidence: float,
        evidence: str,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "intent",
            t=_timestamp(t),
            agent=agent,
            intent=intent,
            confidence=float(confidence),
            evidence=evidence,
            **self._evidence_references(evidence_ids),
        )

    def action_evidence(
        self,
        t: int,
        agent: str,
        action: str,
        hypothesis: str,
        confidence: float,
        evidence: str,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "action_evidence",
            t=_timestamp(t),
            agent=agent,
            action=action,
            hypothesis=hypothesis,
            confidence=float(confidence),
            evidence=evidence,
            **self._evidence_references(evidence_ids),
        )

    def contradiction(
        self,
        agent: str,
        claim: str,
        source_a: str,
        source_b: str,
    ) -> None:
        self._add(
            "contradiction",
            agent=agent,
            claim=claim,
            source_a=source_a,
            source_b=source_b,
        )

    def social_goal(
        self,
        agent: str,
        target_agent: str,
        goal: str,
        confidence: float,
        evidence: str,
        evidence_ids: str = "",
    ) -> None:
        self._add(
            "social_goal",
            agent=agent,
            target_agent=target_agent,
            goal=goal,
            confidence=float(confidence),
            evidence=evidence,
            **self._evidence_references(evidence_ids),
        )

    def likelihood(
        self,
        agent: str,
        action: str,
        hypothesis: str,
        value: float,
    ) -> None:
        self._add(
            "action_likelihood",
            agent=agent,
            action=action,
            hypothesis=hypothesis,
            value=float(value),
        )

    def posterior(
        self,
        t: int,
        agent: str,
        variable: str,
        hypothesis: str,
        value: float,
        evidence: str,
        label: str = "",
    ) -> None:
        """Record a posterior produced by the native Social Code reasoner.

        ``likelihood`` is retained for external Bayesian adapters.  A posterior
        is a different object: it is normalized within a set of competing
        hypotheses after the Social Code reasoner has combined its priors and
        action/observation evidence.
        """
        self._add(
            "posterior",
            t=_timestamp(t),
            agent=agent,
            variable=variable,
            hypothesis=hypothesis,
            value=float(value),
            evidence=evidence,
            label=label,
        )

    def unknown(self, agent: str, variable: str) -> None:
        self._add("social_unknown", agent=agent, variable=variable)


class UnsafeProgram(ValueError):
    pass


def execute_program(
    code: str,
    variable: str,
    memory: _Memory,
    allowed_methods: set[str],
) -> str:
    """Execute only literal calls on the supplied S or M runtime object."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise UnsafeProgram(f"syntax error: {error.msg}") from error

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.Expr, ast.Call, ast.Attribute, ast.Name)):
            continue
        if isinstance(node, (ast.Load, ast.Constant, ast.keyword, ast.UnaryOp, ast.USub)):
            continue
        raise UnsafeProgram(f"unsupported syntax: {type(node).__name__}")
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise UnsafeProgram("every line must be one runtime method call")
        call = statement.value
        if not isinstance(call.func, ast.Attribute):
            raise UnsafeProgram("calls must target the runtime object")
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != variable:
            raise UnsafeProgram(f"only {variable}.method(...) calls are allowed")
        if call.func.attr not in allowed_methods:
            raise UnsafeProgram(f"unsupported {variable} method: {call.func.attr}")
        keywords = [keyword.arg for keyword in call.keywords]
        if any(keyword is None for keyword in keywords):
            raise UnsafeProgram("expanded keyword arguments are not allowed")
        if len(keywords) != len(set(keywords)):
            raise UnsafeProgram("duplicate keyword arguments are not allowed")
        method = getattr(memory, call.func.attr)
        try:
            inspect.signature(method).bind(
                *([None] * len(call.args)),
                **{keyword: None for keyword in keywords},
            )
        except TypeError as error:
            raise UnsafeProgram(
                f"invalid {variable}.{call.func.attr} call: {error}"
            ) from error

    # VLMs occasionally emit identifiers such as ``inside`` or ``Mark``
    # without quotes. Since calls and control flow are already restricted,
    # interpreting every argument identifier as its literal spelling is safe.
    bare_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id != variable
    }
    literal_names = {name: name for name in bare_names}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {variable, *literal_names}:
            raise UnsafeProgram(f"unknown bare name: {node.id}")
    globals_dict = {"__builtins__": {}, **literal_names}
    try:
        exec(compile(tree, "<generated-code>", "exec"), globals_dict, {variable: memory})
    except (TypeError, ValueError) as error:
        raise UnsafeProgram(f"invalid runtime argument: {error}") from error
    return memory.trace()


def execute_spatial(code: str) -> str:
    return execute_program(
        code,
        "S",
        SpatialMemory(),
        {"fact", "event", "visible", "unknown"},
    )


def execute_social(
    code: str,
    valid_evidence_ids: set[str] | None = None,
    require_evidence: bool = False,
) -> str:
    """Execute the public Social Code state language.

    ``query_evidence`` is intentionally absent: it was a legacy, answer-option
    scoring primitive rather than an agent state.  Historical artifacts can be
    replayed only through :func:`execute_social_legacy`.
    """
    return execute_program(
        code,
        "M",
        SocialMemory(valid_evidence_ids, require_evidence),
        {
            "candidate",
            "clause",
            "constraint",
            "support",
            "contradict",
            "clause_unknown",
            "observe",
            "belief",
            "goal",
            "intent",
            "action_evidence",
            "contradiction",
            "social_goal",
            "likelihood",
            "posterior",
            "unknown",
        },
    )


def execute_social_legacy(
    code: str,
    valid_evidence_ids: set[str] | None = None,
    require_evidence: bool = False,
) -> str:
    """Replay pre-cleanup traces that used answer-directed query evidence.

    This entry point is deliberately named and separated so new method code
    cannot accidentally admit ``M.query_evidence``.
    """
    return execute_program(
        code,
        "M",
        SocialMemory(valid_evidence_ids, require_evidence),
        {
            "candidate",
            "clause",
            "constraint",
            "support",
            "contradict",
            "clause_unknown",
            "query_evidence",
            "observe",
            "belief",
            "goal",
            "intent",
            "action_evidence",
            "contradiction",
            "social_goal",
            "likelihood",
            "posterior",
            "unknown",
        },
    )
