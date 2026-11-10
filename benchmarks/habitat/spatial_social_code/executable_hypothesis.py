"""Typed executable hypotheses and deterministic evidence-grounded scoring.

The compiler-facing schema is deliberately narrower than a free-form answer
prompt.  A model may atomize answer options and attach Spatial evidence IDs;
only this module validates those links, evaluates constraints, and chooses an
answer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from runtime import execute_social


VARIABLE_TYPES = {
    "world_state",
    "observation",
    "belief",
    "intent",
    "responsibility/causality",
    "temporal",
    "unknown",
}
CONSTRAINTS = {"ALL", "ANY", "ORDER"}
VERDICTS = {"support", "contradict", "unknown"}
_CLAUSE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


@dataclass(frozen=True)
class ExecutableDecision:
    choice: str
    posteriors: dict[str, float]
    scores: dict[str, float]
    trace: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def trace_records(trace: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in str(trace).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("values"), dict):
            records.append(record)
    return records


def ensure_spatial_evidence_ids(
    spatial_trace: str, prefix: str = "E"
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Assign stable IDs to existing records without changing their evidence."""
    records = trace_records(spatial_trace)
    used = {
        str(record["values"].get("evidence_id", "")).strip()
        for record in records
        if str(record["values"].get("evidence_id", "")).strip()
    }
    next_id = 1
    evidence: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    for record in records:
        values = dict(record["values"])
        evidence_id = str(values.get("evidence_id", "")).strip()
        if not evidence_id:
            while f"{prefix}{next_id}" in used:
                next_id += 1
            evidence_id = f"{prefix}{next_id}"
            next_id += 1
            values["evidence_id"] = evidence_id
            record = {**record, "values": values}
        if evidence_id in evidence:
            raise ValueError(f"duplicate Spatial evidence ID: {evidence_id!r}")
        used.add(evidence_id)
        evidence[evidence_id] = record
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines), evidence


def infer_query_variable_type(question: str, options: Sequence[str] = ()) -> str:
    """Infer a general semantic variable type without benchmark labels."""
    query_text = str(question).lower()
    text = " ".join([str(question), *map(str, options)]).lower()
    if re.search(
        r"\b(?:before|after|first|last|earlier|later|sequence|order|prior to|subsequent)\b",
        query_text,
    ):
        return "temporal"
    if re.search(
        r"\b(?:responsib|caus(?:e|ed|ing)|initiated?|prompted?|made .+ (?:do|happen)|led to)\b",
        query_text,
    ):
        return "responsibility/causality"
    rules = (
        ("belief", r"\b(?:believ|think|thought|knew|know|aware|expect|assum)"),
        ("intent", r"\b(?:intend|intent|want|goal|plan|purpose|trying to|try to|aim)"),
        (
            "observation",
            r"\b(?:saw|see|seen|heard|hear|notice|observe|watch|visible|viewpoint|perspective)\b",
        ),
        (
            "world_state",
            r"\b(?:who|what|where|which|how many|doing|did|happened|located|holding|wearing|said|spoke|suggest)\b",
        ),
    )
    for variable_type, pattern in rules:
        if re.search(pattern, text):
            return variable_type
    return "unknown"


def _literal(value: Any) -> str:
    return repr(value)


def _members(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    return [str(item).strip().upper() for item in raw if str(item).strip()]


def _payload_options(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("options", [])
    return [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def compile_payload(
    payload: Mapping[str, Any],
    options: Sequence[tuple[str, str]],
    spatial_trace: str,
    question: str,
    evidence_prefix: str = "E",
) -> tuple[str, str, ExecutableDecision]:
    """Compile model JSON into runtime calls, then execute and score it."""
    indexed_trace, spatial_by_id = ensure_spatial_evidence_ids(
        spatial_trace, prefix=evidence_prefix
    )
    expected = {str(label).upper(): str(text) for label, text in options}
    payload_by_label = {
        str(item.get("label", "")).strip().upper(): item
        for item in _payload_options(payload)
        if str(item.get("label", "")).strip().upper() in expected
    }
    inferred_type = infer_query_variable_type(question, list(expected.values()))
    declared_type = str(payload.get("variable_type", "unknown")).strip().lower()
    variable_type = inferred_type if inferred_type != "unknown" else declared_type
    if variable_type not in VARIABLE_TYPES:
        variable_type = "unknown"
    agent = " ".join(str(payload.get("agent", "agent")).split()) or "agent"
    calls: list[str] = []

    for label, option_text in expected.items():
        item = payload_by_label.get(label, {})
        raw_clauses = item.get("clauses", []) if isinstance(item, Mapping) else []
        clauses = [value for value in raw_clauses if isinstance(value, Mapping)]
        clauses = clauses[:8]
        known_ids: list[str] = []
        for offset, clause in enumerate(clauses, start=1):
            clause_id = str(clause.get("id", f"{label}{offset}")).strip().upper()
            if not _CLAUSE_ID_RE.fullmatch(clause_id) or clause_id in known_ids:
                clause_id = f"{label}{offset}"
            known_ids.append(clause_id)
            calls.append(
                "M.clause("
                f"{label!r}, {clause_id!r}, {_literal(' '.join(str(clause.get('text', option_text)).split()))}, "
                f"subject={_literal(' '.join(str(clause.get('subject', '')).split()))}, "
                f"predicate={_literal(' '.join(str(clause.get('predicate', '')).split()))}, "
                f"required={bool(clause.get('required', True))!r}, "
                f"endpoint={_literal(' '.join(str(clause.get('endpoint', '')).split()))})"
            )
        if not known_ids:
            known_ids = [f"{label}1"]
            calls.append(
                f"M.clause({label!r}, {known_ids[0]!r}, {option_text!r}, required=True)"
            )

        raw_constraints = item.get("constraints", []) if isinstance(item, Mapping) else []
        constraints = [value for value in raw_constraints if isinstance(value, Mapping)]
        valid_constraints: list[tuple[str, list[str]]] = []
        for constraint in constraints[:4]:
            kind = str(constraint.get("kind", "ALL")).strip().upper()
            members = [value for value in _members(constraint.get("members", [])) if value in known_ids]
            if kind in CONSTRAINTS and members:
                valid_constraints.append((kind, members))
        if not valid_constraints:
            default_kind = "ORDER" if variable_type == "temporal" and len(known_ids) > 1 else "ALL"
            valid_constraints = [(default_kind, known_ids)]
        root_constraint = valid_constraints[0][0]
        calls.insert(
            len(calls) - len(known_ids),
            "M.candidate("
            f"{agent!r}, 'answer', {label!r}, {option_text!r}, 0.5, "
            f"'answer_option', {variable_type!r}, {root_constraint!r})",
        )
        for kind, members in valid_constraints:
            calls.append(
                f"M.constraint({label!r}, {kind!r}, {','.join(members)!r})"
            )

        raw_evidence = item.get("evidence", []) if isinstance(item, Mapping) else []
        evidence_items = [value for value in raw_evidence if isinstance(value, Mapping)]
        covered: set[str] = set()
        for evidence in evidence_items[:24]:
            clause_id = str(evidence.get("clause_id", "")).strip().upper()
            verdict = str(evidence.get("verdict", "unknown")).strip().lower()
            if clause_id not in known_ids or verdict not in VERDICTS:
                continue
            reason = " ".join(str(evidence.get("reason", "")).split())
            evidence_ids = [
                value
                for value in _members(evidence.get("evidence_ids", []))
                if value in spatial_by_id
            ]
            if verdict == "unknown" or not evidence_ids:
                calls.append(
                    f"M.clause_unknown(0, {label!r}, {clause_id!r}, {reason!r})"
                )
                covered.add(clause_id)
                continue
            evidence_t = min(
                int(spatial_by_id[value]["values"].get("t", 0))
                for value in evidence_ids
            )
            method = "support" if verdict == "support" else "contradict"
            calls.append(
                f"M.{method}({evidence_t!r}, {label!r}, {clause_id!r}, "
                f"{','.join(evidence_ids)!r}, {reason!r})"
            )
            covered.add(clause_id)
        for clause_id in known_ids:
            if clause_id not in covered:
                calls.append(
                    f"M.clause_unknown(0, {label!r}, {clause_id!r}, 'no cited evidence')"
                )

    code = "\n".join(calls)
    trace = execute_social(
        code,
        valid_evidence_ids=set(spatial_by_id),
        require_evidence=True,
    )
    decision = score_hypotheses(trace, indexed_trace)
    return code, indexed_trace, decision


def _matches(left: str, right: str) -> bool:
    left = re.sub(r"[^a-z0-9]+", " ", str(left).lower()).strip()
    right = re.sub(r"[^a-z0-9]+", " ", str(right).lower()).strip()
    return bool(left and right and (left == right or left in right or right in left))


_GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "did", "does", "for",
    "from", "in", "is", "it", "of", "on", "or", "the", "then", "to",
    "during", "group", "people", "person", "someone", "was", "were", "while",
    "whoever", "with",
}


def _grounding_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value).lower()):
        if token in _GROUNDING_STOPWORDS or len(token) <= 1:
            continue
        for suffix in ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
        tokens.add(token[:8])
    return tokens


def _semantically_grounded(
    clause: Mapping[str, Any], cited: Sequence[Mapping[str, Any]]
) -> bool:
    subject = str(clause.get("subject", "")).strip()
    predicate = str(clause.get("predicate", "")).strip()
    clause_tokens = _grounding_tokens(predicate or clause.get("text", ""))
    for record in cited:
        values = record["values"]
        entity_values = [
            values.get(key, "")
            for key in ("actor", "subject", "observer", "target", "object", "speaker")
        ]
        if subject and not any(_matches(subject, value) for value in entity_values):
            continue
        evidence_text = " ".join(
            str(values.get(key, ""))
            for key in (
                "relation", "subject", "object", "actor", "action", "target",
                "observer", "quote",
            )
        )
        evidence_tokens = _grounding_tokens(evidence_text)
        subject_tokens = _grounding_tokens(subject)
        informative_clause_tokens = clause_tokens - subject_tokens
        if not informative_clause_tokens or informative_clause_tokens & evidence_tokens:
            return True
    return False


def _explicitly_negative(cited: Sequence[Mapping[str, Any]]) -> bool:
    for record in cited:
        values = record["values"]
        if record.get("kind") == "visibility" and values.get("value") is False:
            return True
        text = " ".join(
            str(values.get(key, ""))
            for key in ("relation", "action", "object", "target", "quote")
        ).lower()
        if re.search(r"\b(?:not|no|never|without|absent|false|prevent|refus)\w*\b", text):
            return True
    return False


def _valid_chain(
    variable_type: str,
    candidate: Mapping[str, Any],
    clause: Mapping[str, Any],
    cited: Sequence[Mapping[str, Any]],
    verdict: str,
) -> bool:
    direct = [record for record in cited if record.get("kind") != "spatial_unknown"]
    if not direct or not _semantically_grounded(clause, direct):
        return False
    if verdict == "contradict" and not _explicitly_negative(direct):
        return False
    if variable_type in {"world_state", "temporal", "unknown"}:
        return True
    subject = str(clause.get("subject", "")).strip() or str(candidate.get("agent", "")).strip()
    if variable_type == "observation":
        return any(
            _matches(subject, record["values"].get("observer", ""))
            for record in direct
        )
    if variable_type == "belief":
        for record in direct:
            values = record["values"]
            if _matches(subject, values.get("observer", "")):
                return True
            if values.get("quote") and _matches(subject, values.get("target", "")):
                return True
        return False
    if variable_type == "intent":
        return any(
            _matches(subject, record["values"].get("actor", ""))
            or _matches(subject, record["values"].get("observer", ""))
            for record in direct
        )
    if variable_type == "responsibility/causality":
        for record in direct:
            values = record["values"]
            role = str(values.get("event_role", "")).lower()
            relation = str(values.get("relation", "")).lower()
            action = str(values.get("action", "")).lower()
            actor_matches = not subject or _matches(subject, values.get("actor", ""))
            if actor_matches and role == "initiator":
                return True
            if actor_matches and re.search(r"\b(?:initiat|caus|responsib)\w*\b", action):
                return True
            if re.search(r"\b(?:cause|responsib|initiator)\w*\b", relation):
                return True
        return False
    return False


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {label: math.exp(3.0 * (score - maximum)) for label, score in scores.items()}
    denominator = sum(weights.values())
    return {label: value / denominator for label, value in weights.items()}


def score_hypotheses(hypothesis_trace: str, spatial_trace: str) -> ExecutableDecision:
    """Validate evidence links, execute constraints, and choose a candidate."""
    spatial_records = trace_records(spatial_trace)
    spatial_by_id = {
        str(record["values"].get("evidence_id", "")): record
        for record in spatial_records
        if str(record["values"].get("evidence_id", ""))
    }
    records = trace_records(hypothesis_trace)
    candidates: dict[str, dict[str, Any]] = {}
    candidate_order: list[str] = []
    clauses: dict[str, dict[str, dict[str, Any]]] = {}
    constraints: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        kind, values = record["kind"], record["values"]
        label = str(values.get("label", "")).upper()
        if kind == "query_candidate" and label and label not in candidates:
            candidates[label] = values
            candidate_order.append(label)
        elif kind == "hypothesis_clause" and label:
            clauses.setdefault(label, {})[str(values.get("clause_id", ""))] = values
        elif kind == "hypothesis_constraint" and label:
            constraints.setdefault(label, []).append(values)
        elif kind == "hypothesis_evidence" and label:
            key = (label, str(values.get("clause_id", "")))
            evidence.setdefault(key, []).append(values)
    if not candidates:
        raise ValueError("no executable answer candidates")

    invalid_links = 0
    clause_states: dict[str, dict[str, str]] = {}
    clause_times: dict[tuple[str, str], list[int]] = {}
    for label, candidate in candidates.items():
        variable_type = str(candidate.get("variable_type", "unknown"))
        clause_states[label] = {}
        for clause_id, clause in clauses.get(label, {}).items():
            valid_support = False
            valid_contradict = False
            times: list[int] = []
            for item in evidence.get((label, clause_id), []):
                verdict = str(item.get("verdict", "unknown"))
                ids = [str(value) for value in item.get("evidence_ids", [])]
                cited = [spatial_by_id[value] for value in ids if value in spatial_by_id]
                valid = bool(ids) and len(cited) == len(ids) and _valid_chain(
                    variable_type, candidate, clause, cited, verdict
                )
                if verdict in {"support", "contradict"} and not valid:
                    invalid_links += 1
                    continue
                if verdict == "support":
                    valid_support = True
                    times.extend(int(record["values"].get("t", 0)) for record in cited)
                elif verdict == "contradict":
                    valid_contradict = True
            if valid_support and not valid_contradict:
                state = "support"
            elif valid_contradict and not valid_support:
                state = "contradict"
            elif valid_support and valid_contradict:
                state = "conflict"
            else:
                state = "unknown"
            clause_states[label][clause_id] = state
            clause_times[(label, clause_id)] = sorted(set(times))

    scores: dict[str, float] = {}
    constraint_results: dict[str, list[dict[str, Any]]] = {}
    for label, candidate in candidates.items():
        option_clauses = clauses.get(label, {})
        option_constraints = constraints.get(label, [])
        if not option_constraints and option_clauses:
            option_constraints = [{
                "constraint": candidate.get("constraint", "ALL"),
                "clause_ids": list(option_clauses),
            }]
        results: list[dict[str, Any]] = []
        for constraint in option_constraints:
            kind = str(constraint.get("constraint", "ALL")).upper()
            members = [
                str(value) for value in constraint.get("clause_ids", []) if str(value) in option_clauses
            ]
            states = [clause_states[label].get(value, "unknown") for value in members]
            support_count = sum(value == "support" for value in states)
            if not members:
                value = -1.0
            elif kind == "ALL":
                if any(state == "contradict" for state in states):
                    value = -1.0
                elif all(state == "support" for state in states):
                    value = 1.0
                else:
                    value = 0.49 * support_count / len(members)
            elif kind == "ANY":
                if any(state == "support" for state in states):
                    value = 1.0
                elif all(state == "contradict" for state in states):
                    value = -1.0
                else:
                    value = 0.0
            elif kind == "ORDER":
                if any(state == "contradict" for state in states):
                    value = -1.0
                elif not all(state == "support" for state in states):
                    value = 0.49 * support_count / len(members)
                else:
                    endpoints = [clause_times[(label, member)] for member in members]
                    if all(endpoints):
                        ordered = all(
                            max(endpoints[offset]) < min(endpoints[offset + 1])
                            for offset in range(len(endpoints) - 1)
                        )
                        value = 1.0 if ordered else -1.0
                    else:
                        value = 0.0
            else:
                value = -1.0
            results.append({"constraint": kind, "members": members, "value": value})
        constraint_results[label] = results
        scores[label] = sum(result["value"] for result in results) / len(results) if results else -1.0

    posteriors = _softmax(scores)
    rank = {label: offset for offset, label in enumerate(candidate_order)}
    choice = max(candidate_order, key=lambda label: (scores[label], -rank[label]))
    posterior_calls = "\n".join(
        "M.posterior(0, "
        f"{str(candidates[label].get('agent', 'agent'))!r}, 'executable_answer', "
        f"{str(candidates[label].get('hypothesis', ''))!r}, {posteriors[label]!r}, "
        f"'deterministic executable constraint score={scores[label]:.6f}', label={label!r})"
        for label in candidate_order
    )
    posterior_trace = execute_social(posterior_calls)
    full_trace = hypothesis_trace + ("\n" if hypothesis_trace else "") + posterior_trace
    return ExecutableDecision(
        choice=choice,
        posteriors=posteriors,
        scores=scores,
        trace=full_trace,
        diagnostics={
            "variable_types": {
                label: candidates[label].get("variable_type", "unknown") for label in candidate_order
            },
            "clause_states": clause_states,
            "constraints": constraint_results,
            "invalid_evidence_links": invalid_links,
            "spatial_evidence_count": len(spatial_by_id),
            "tie_break": "stable answer-option order",
        },
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a possibly fenced model response."""
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def deterministic_payload(
    question: str,
    options: Sequence[tuple[str, str]],
    spatial_trace: str,
    evidence_prefix: str = "E",
) -> tuple[dict[str, Any], str]:
    """Build a conservative no-model payload for reproducible ablations.

    This compiler uses only query text and already-executed Spatial records.
    It never consumes gold labels, baseline outputs, or a prior method answer.
    """
    indexed_trace, evidence_by_id = ensure_spatial_evidence_ids(
        spatial_trace, prefix=evidence_prefix
    )
    variable_type = infer_query_variable_type(question, [text for _, text in options])
    evidence_entities: list[str] = []
    for record in evidence_by_id.values():
        values = record["values"]
        for key in ("actor", "subject", "observer", "target", "object"):
            value = " ".join(str(values.get(key, "")).split())
            if value and value.lower() not in {item.lower() for item in evidence_entities}:
                evidence_entities.append(value)

    def subject_in(text: str) -> str:
        matches = [value for value in evidence_entities if re.search(rf"\b{re.escape(value)}\b", text, re.I)]
        return max(matches, key=len) if matches else ""

    def atomic_clauses(label: str, text: str) -> tuple[list[dict[str, Any]], str]:
        clean = " ".join(text.split()).strip(" .")
        constraint = "ALL"
        parts: list[str]
        before = re.split(r"\s+before\s+", clean, maxsplit=1, flags=re.I)
        after = re.split(r"\s+after\s+", clean, maxsplit=1, flags=re.I)
        if variable_type == "temporal" and len(before) == 2:
            parts, constraint = before, "ORDER"
        elif variable_type == "temporal" and len(after) == 2:
            parts, constraint = [after[1], after[0]], "ORDER"
        else:
            comma = [part.strip() for part in clean.split(",", 1)]
            names = re.split(r"\s+and\s+", comma[0], flags=re.I)
            if len(comma) == 2 and len(names) > 1 and all(subject_in(name) for name in names):
                parts = [f"{name} {comma[1]}" for name in names]
            else:
                parts = [part.strip() for part in re.split(r"\s+and\s+|;", clean, flags=re.I) if part.strip()]
        parts = parts[:6] or [clean]
        clauses = []
        for offset, part in enumerate(parts, start=1):
            subject = subject_in(part)
            predicate = part
            if subject:
                predicate = re.sub(rf"\b{re.escape(subject)}\b", " ", predicate, flags=re.I)
                predicate = " ".join(predicate.split())
            clauses.append({
                "id": f"{label}{offset}",
                "text": part,
                "subject": subject,
                "predicate": predicate,
                "required": True,
                "endpoint": str(offset) if constraint == "ORDER" else "",
            })
        return clauses, constraint

    payload_options: list[dict[str, Any]] = []
    for label, option_text in options:
        clauses, constraint = atomic_clauses(str(label).upper(), str(option_text))
        links: list[dict[str, Any]] = []
        for clause in clauses:
            clause_tokens = _grounding_tokens(clause["predicate"] or clause["text"])
            subject = clause["subject"]
            ranked: list[tuple[float, str, Mapping[str, Any]]] = []
            for evidence_id, record in evidence_by_id.items():
                if record.get("kind") == "spatial_unknown":
                    continue
                values = record["values"]
                if subject:
                    entities = [values.get(key, "") for key in ("actor", "subject", "observer", "target", "object")]
                    if not any(_matches(subject, value) for value in entities):
                        continue
                evidence_text = " ".join(
                    str(values.get(key, ""))
                    for key in ("relation", "subject", "object", "actor", "action", "target", "observer", "quote")
                )
                evidence_tokens = _grounding_tokens(evidence_text)
                informative = clause_tokens - _grounding_tokens(subject)
                overlap = len(informative & evidence_tokens)
                required_overlap = 1 if subject or len(informative) <= 2 else 2
                if overlap < required_overlap:
                    continue
                score = overlap / math.sqrt(max(1, len(informative) * len(evidence_tokens)))
                ranked.append((score, evidence_id, record))
            if ranked:
                _, evidence_id, record = max(ranked, key=lambda item: (item[0], item[1]))
                values = record["values"]
                negative = (
                    record.get("kind") == "visibility" and values.get("value") is False
                ) or bool(re.search(
                    r"\b(?:not|no|never|without|absent|false|prevent|refus)\w*\b",
                    " ".join(str(values.get(key, "")) for key in ("relation", "action", "object", "target")),
                    flags=re.I,
                ))
                links.append({
                    "clause_id": clause["id"],
                    "verdict": "contradict" if negative else "support",
                    "evidence_ids": [evidence_id],
                    "reason": "deterministic lexical/role link to executed Spatial record",
                    "_match_score": score,
                })
            else:
                links.append({
                    "clause_id": clause["id"],
                    "verdict": "unknown",
                    "evidence_ids": [],
                    "reason": "no sufficiently matching executed Spatial record",
                })
        payload_options.append({
            "label": str(label).upper(),
            "clauses": clauses,
            "constraints": [{"kind": constraint, "members": [clause["id"] for clause in clauses]}],
            "evidence": links,
        })
    best_by_evidence: dict[str, float] = {}
    for option in payload_options:
        for link in option["evidence"]:
            if link.get("verdict") != "support" or not link.get("evidence_ids"):
                continue
            evidence_id = str(link["evidence_ids"][0])
            best_by_evidence[evidence_id] = max(
                best_by_evidence.get(evidence_id, -1.0), float(link.get("_match_score", 0.0))
            )
    for option in payload_options:
        for link in option["evidence"]:
            score = float(link.pop("_match_score", 0.0))
            if link.get("verdict") != "support" or not link.get("evidence_ids"):
                continue
            evidence_id = str(link["evidence_ids"][0])
            if score + 1e-12 < best_by_evidence[evidence_id]:
                link.update({
                    "verdict": "unknown",
                    "evidence_ids": [],
                    "reason": "a competing clause is a more specific match to this Spatial record",
                })
    return {
        "variable_type": variable_type,
        "agent": subject_in(question) or "agent",
        "options": payload_options,
    }, indexed_trace
