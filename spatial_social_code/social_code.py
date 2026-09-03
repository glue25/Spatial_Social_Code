"""Compilers and inference for the Social Code intermediate representation.

Social Code is an independent social-state model.  Neural extraction,
deterministic action rules, and Bayesian inference are evidence components;
none of them defines the method by itself.  The default ``hybrid`` path uses
all three native components and does not depend on AutoToM.  An AutoToM
posterior can be admitted explicitly as one additional external evidence
source.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from runtime import UnsafeProgram, execute_social


SOCIAL_SOURCES = ("neural", "rules", "bayesian", "neural_rules", "hybrid")


NEURAL_SOCIAL_PROMPT = """You are the Social Code evidence compiler in a video understanding system.
Infer only socially relevant mental-state evidence from the action history and ordered RGB frames. Use both answer choices only to identify the social distinctions that need evidence; never choose, rank, copy, or encode an answer. Separate observation, belief, goal, and immediate intent. A world fact is not automatically an agent belief: only attribute a belief when the agent could observe supporting evidence. Treat the question and choices as query context, not as observed facts. Missing evidence means unknown, not false.

The runtime object M already exists. Use ONLY literal calls in these forms:
M.observe(t, agent, fact, source="video_or_action", confidence=0.8)
M.belief(t, agent, fact, confidence=0.7, evidence="...")
M.goal(agent, goal, confidence=0.7, evidence="...")
M.intent(t, agent, intent, confidence=0.7, evidence="...")
M.action_evidence(t, agent, action, hypothesis, confidence=0.7, evidence="...")
M.unknown(agent, variable)

Use timestamps in action order. Emit 3-12 complete calls. No assignments, variables, imports, loops, markdown, comments, answer labels, or repeated calls.

Evidence and complete query context:
{evidence}
"""


@dataclass(frozen=True)
class SocialCodeResult:
    source: str
    code: str
    trace: str
    components: tuple[str, ...]
    raw_code: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def normalize_social_code(text: str, max_lines: int = 16) -> str:
    """Keep unique, syntactically valid calls to the Social Code runtime."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    allowed = {
        "candidate",
        "query_evidence",
        "observe",
        "belief",
        "goal",
        "intent",
        "action_evidence",
        "contradiction",
        "unknown",
        "likelihood",  # External Bayesian evidence compatibility.
        "posterior",  # Native Social Code Bayesian inference.
    }
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        try:
            parsed = ast.parse(line)
        except SyntaxError:
            continue
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
            continue
        call = parsed.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "M":
            continue
        if call.func.attr not in allowed:
            continue
        try:
            execute_social(line)
        except UnsafeProgram:
            continue
        seen.add(line)
        kept.append(line)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def question_without_options(question: str) -> str:
    """Retain evidence and question premise while hiding candidate answers."""
    match = re.search(r"\s+\(a\)\s+", question, flags=re.IGNORECASE)
    return question[: match.start()].strip() if match else question.strip()


def answer_options(question: str) -> list[tuple[str, str]]:
    """Parse the two MMToM options without exposing them to evidence extraction."""
    match = re.search(
        r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)(?:\s*Please respond\b|$)",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return []
    return [
        ("A", " ".join(match.group(1).split()).rstrip(". ")),
        ("B", " ".join(match.group(2).split()).rstrip(". ")),
    ]


_ORDINALS = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "one": "first",
    "two": "second",
    "three": "third",
    "four": "fourth",
}


def _canonical_text(text: str) -> str:
    text = text.lower()
    for contraction, expanded in {
        "isn't": "is not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
    }.items():
        text = text.replace(contraction, expanded)
    text = text.replace("refrigerator", "fridge").replace("dishbowl", "dish bowl")
    for source, target in _ORDINALS.items():
        text = re.sub(rf"\b{source}\b", target, text)
    text = re.sub(r"\b(?:from the left|from left to right|kitchen)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _canonical_text(text).split()
        if token not in _STOPWORDS and token not in {"agent", "thinks", "think"}
    }


def _entity_match(left: str, right: str) -> bool:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    return bool(left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens))


def _belief_claim(text: str) -> tuple[str, str, bool] | None:
    canonical = _canonical_text(text)
    direct = re.search(
        r"^(?:(?:.+? )?thinks? that )?(?:the )?(.+?) is (not )?(?:inside|in) (?:the )?(.+)$",
        canonical,
    )
    if direct:
        return direct.group(1).strip(), direct.group(3).strip(), bool(direct.group(2))
    existential = re.search(
        r"^(?:(?:.+? )?(?:does not think|thinks? that|think) )?(?:there )?is (not )?(?:a |an |any )?(.+?) (?:inside|in) (?:the )?(.+)$",
        canonical,
    )
    if existential:
        negated = bool(existential.group(1)) or "does not think" in canonical
        return existential.group(2).strip(), existential.group(3).strip(), negated
    return None


def _goal_object(text: str) -> str | None:
    canonical = _canonical_text(text)
    match = re.search(r"(?:trying to get|goal (?:is )?to get|obtain|get) (?:a |an |the )?(.+)$", canonical)
    return match.group(1).strip() if match else None


def _query_variable(options: list[tuple[str, str]]) -> str:
    joined = " ".join(option for _, option in options).lower()
    if "has been trying" in joined or "trying to get" in joined:
        return "goal"
    return "belief"


def _query_context(question: str, agent: str) -> tuple[str | None, str | None]:
    """Return a known goal and belief explicitly supplied in the question premise."""
    prefix = question_without_options(question)
    goal_match = re.search(
        r"If\s+.+?\s+has been trying to get\s+(?:a |an |the )?(.+?),\s*which",
        prefix,
        flags=re.IGNORECASE | re.DOTALL,
    )
    belief_match = re.search(
        r"If\s+.+?\s+think(?:s)?\s+(.+?),\s*which",
        prefix,
        flags=re.IGNORECASE | re.DOTALL,
    )
    goal = f"get {' '.join(goal_match.group(1).split())}" if goal_match else None
    belief = " ".join(belief_match.group(1).split()) if belief_match else None
    return goal, belief


def _action_sentences(question: str) -> list[str]:
    _, history = action_history(question)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", history) if part.strip()]


def _action_target(sentence: str) -> str | None:
    steps = _sentence_location_steps(sentence)
    return steps[-1][0] if steps else None


def _sentence_location_steps(sentence: str) -> list[tuple[str, bool]]:
    """Split a sentence containing several locations into ordered search steps."""
    patterns = (
        (
            r"(?:towards?|toward|approaches?|approach|heads? (?:towards?|to)|moves? (?:towards?|to)|advances? (?:towards?|to)|walks? (?:towards?|to)|strides? (?:towards?|to)|proceeds? (?:towards?|to))\s+(?:the\s+)?([^,.]+)",
            False,
        ),
        (
            r"\b(?:opens?|checks?|inspects?|searches?)\b\s+(?:the\s+)?((?!(?:it|them|and)\b)[^,.]+)",
            True,
        ),
    )
    matches: list[dict[str, Any]] = []
    for pattern, direct_inspection in patterns:
        for found in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            target = re.split(
                r"\b(?:which|and then|and is|and (?:opens?|closes?|checks?|inspects?|searches?)|before|then)\b",
                found.group(1),
                maxsplit=1,
                flags=re.I,
            )[0].strip().rstrip(".")
            if not _canonical_text(target) or _canonical_text(target) in {"it", "them", "him", "her"}:
                continue
            matches.append(
                {
                    "start": found.start(),
                    "end": found.end(),
                    "target": target,
                    "direct": direct_inspection,
                }
            )
    matches.sort(key=lambda item: (item["start"], not item["direct"]))
    merged: list[dict[str, Any]] = []
    for item in matches:
        if merged and _canonical_text(merged[-1]["target"]) == _canonical_text(item["target"]):
            merged[-1]["direct"] = merged[-1]["direct"] or item["direct"]
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        else:
            merged.append(item)
    steps: list[tuple[str, bool]] = []
    for offset, item in enumerate(merged):
        end = merged[offset + 1]["start"] if offset + 1 < len(merged) else len(sentence)
        segment = sentence[item["start"] : end]
        steps.append((item["target"], bool(item["direct"] or _completed_inspection(segment))))
    return steps


def _action_location_steps(question: str) -> list[tuple[int, str, bool, str]]:
    steps: list[tuple[int, str, bool, str]] = []
    for sentence in _action_sentences(question):
        for target, inspected in _sentence_location_steps(sentence):
            steps.append((len(steps), target, inspected, sentence))
    return steps


def _completed_inspection(sentence: str) -> bool:
    lower = sentence.lower()
    return bool(re.search(r"\b(?:open(?:s|ed)?|check(?:s|ed)?|inspect(?:s|ed)?|search(?:es|ed)?)\b", lower)) and not bool(
        re.search(r"\b(?:about to|preparing to|on the verge of)\s+open", lower)
    )


def compile_query_candidates(
    question: str, *, include_heuristic_evidence: bool = True
) -> SocialCodeResult:
    """Register A/B hypotheses, optionally adding question-derived heuristics.

    Component ablations set ``include_heuristic_evidence=False`` so the shared
    query adapter cannot itself answer the question and mask the contribution
    of Physical or Social evidence.
    """
    options = answer_options(question)
    agent, _ = action_history(question)
    if len(options) != 2:
        code = f"M.unknown({_literal(agent)}, 'answer_candidates')"
        return SocialCodeResult("query", code, execute_social(code), ("query",))

    variable = _query_variable(options)
    answer_variable = f"answer_{variable}"
    known_goal, known_belief = _query_context(question, agent)
    calls: list[str] = []
    if known_goal:
        calls.append(f"M.goal({_literal(agent)}, {_literal(known_goal)}, 1.0, 'given query premise')")
    if known_belief:
        calls.append(f"M.belief(0, {_literal(agent)}, {_literal(known_belief)}, 1.0, 'given query premise')")
    for label, hypothesis in options:
        calls.append(
            f"M.candidate({_literal(agent)}, {_literal(answer_variable)}, {label!r}, "
            f"{_literal(hypothesis)}, 0.5, 'MMToM answer option')"
        )

    if not include_heuristic_evidence:
        code = "\n".join(calls)
        return SocialCodeResult(
            "query_candidates",
            code,
            execute_social(code),
            ("query_candidates",),
            diagnostics={"variable": answer_variable, "candidates": len(options)},
        )

    sentences = _action_sentences(question)
    location_steps = _action_location_steps(question)
    if variable == "belief" and known_goal:
        goal_object = _goal_object(known_goal)
        final_target = location_steps[-1][1] if location_steps else None
        for _, hypothesis in options:
            claim = _belief_claim(hypothesis)
            if claim is None or goal_object is None or not _entity_match(claim[0], goal_object):
                continue
            relevant = [
                step
                for step in location_steps
                if _entity_match(step[1], claim[1])
            ]
            if relevant:
                t, _, searched, sentence = relevant[-1]
                likelihood = 0.82 if claim[2] == searched else 0.18
                confidence = 0.85
                reason = "completed search supports absence belief" if searched else "goal-directed approach supports presence belief"
            elif final_target and not _entity_match(final_target, claim[1]):
                t = location_steps[-1][0]
                sentence = location_steps[-1][3]
                likelihood = 0.76 if claim[2] else 0.24
                confidence = 0.65
                reason = "long-horizon trajectory ends away from queried location"
            else:
                continue
            calls.append(
                f"M.query_evidence({t}, {_literal(agent)}, {_literal(answer_variable)}, "
                f"{_literal(hypothesis)}, {likelihood!r}, {confidence!r}, {_literal(reason + ': ' + sentence)})"
            )

    if variable == "goal":
        premise_claim = _belief_claim(known_belief) if known_belief else None
        premise_object = premise_claim[0] if premise_claim else None
        premise_location_visited = bool(
            premise_claim
            and any(_entity_match(step[1], premise_claim[1]) for step in location_steps)
        )
        for _, hypothesis in options:
            candidate_object = _goal_object(hypothesis)
            if candidate_object and premise_object and premise_location_visited:
                matches_premise_object = _entity_match(candidate_object, premise_object)
                if premise_claim and premise_claim[2]:
                    likelihood = 0.18 if matches_premise_object else 0.72
                    reason = "visited location is believed not to contain this goal object"
                else:
                    likelihood = 0.82 if matches_premise_object else 0.38
                    reason = "visited location is believed to contain this goal object"
                calls.append(
                    f"M.query_evidence(0, {_literal(agent)}, {_literal(answer_variable)}, "
                    f"{_literal(hypothesis)}, {likelihood!r}, 0.8, {_literal(reason)})"
                )
            direct_steps = [
                (t, sentence)
                for t, sentence in enumerate(sentences)
                if candidate_object and _entity_match(candidate_object, sentence)
            ]
            if direct_steps:
                t, sentence = direct_steps[-1]
                calls.append(
                    f"M.query_evidence({t}, {_literal(agent)}, {_literal(answer_variable)}, "
                    f"{_literal(hypothesis)}, 0.88, 0.8, {_literal('action directly targets candidate object: ' + sentence)})"
                )

    code = "\n".join(calls)
    return SocialCodeResult(
        "query",
        code,
        execute_social(code),
        ("query",),
        diagnostics={"variable": answer_variable, "candidates": len(options)},
    )


def action_history(question: str) -> tuple[str, str]:
    match = re.search(
        r"Actions taken by\s+([^:]+):\s*(.*?)(?:\s+Question:|$)",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return "agent", ""
    return match.group(1).strip(), " ".join(match.group(2).split())


def _literal(value: object) -> str:
    return repr(value)


def compile_rules(question: str) -> SocialCodeResult:
    """Compile conservative action-to-intent evidence without another model."""
    agent, history = action_history(question)
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", history)
        if part.strip()
    ]
    calls: list[str] = []
    target_patterns = (
        r"(?:towards?|toward|approaches?|approach|heads? (?:towards?|to))\s+(?:the\s+)?([^,.]+)",
        r"(?:opens?|closes?|checks?|inspects?|searches?)\s+(?:the\s+)?([^,.]+)",
        r"(?:picks? up|takes?|grabs?|gets?)\s+(?:the\s+)?([^,.]+)",
    )
    for t, sentence in enumerate(sentences):
        normalized = re.sub(r"^(?:she|he|they)\s+", f"{agent} ", sentence, flags=re.I)
        target = None
        for pattern in target_patterns:
            found = re.search(pattern, normalized, flags=re.I)
            if found:
                target = found.group(1).strip().rstrip(".")
                break
        if target is None:
            continue

        lower = normalized.lower()
        if re.search(r"\b(?:pick|take|grab|get)", lower):
            hypothesis = f"obtain {target}"
            confidence = 0.9
            calls.append(
                f"M.goal({_literal(agent)}, {_literal(hypothesis)}, {confidence!r}, "
                f"{_literal('explicit acquisition action')})"
            )
        elif re.search(r"\b(?:open|check|inspect|search)", lower):
            hypothesis = f"inspect {target}"
            confidence = 0.75
        else:
            hypothesis = f"interact with {target}"
            confidence = 0.6

        calls.append(
            f"M.intent({t}, {_literal(agent)}, {_literal(hypothesis)}, {confidence!r}, "
            f"{_literal(normalized)})"
        )
        calls.append(
            f"M.action_evidence({t}, {_literal(agent)}, {_literal(normalized)}, "
            f"{_literal(hypothesis)}, {confidence!r}, "
            f"{_literal('deterministic action rule')})"
        )

    if not calls:
        calls.append(f"M.unknown({_literal(agent)}, 'goal_or_intent')")
    code = "\n".join(calls[:16])
    return SocialCodeResult("rules", code, execute_social(code), ("rules",))


def compile_neural(
    question: str,
    generate: Callable[..., str],
) -> SocialCodeResult:
    prompt = NEURAL_SOCIAL_PROMPT.format(evidence=question)
    code = normalize_social_code(generate(prompt, max_new_tokens=384))
    if not code:
        agent, _ = action_history(question)
        code = f"M.unknown({_literal(agent)}, 'social_state')"
    return SocialCodeResult("neural", code, execute_social(code), ("neural",))


def compile_spatial_bridge(question: str, spatial_trace: str) -> SocialCodeResult:
    """Turn physically observable Spatial records into Social observations.

    Static world facts are never copied directly into an agent belief. Container
    contents become observations only after an explicit open/check/search action.
    """
    agent, _ = action_history(question)
    options = answer_options(question)
    variable = _query_variable(options) if options else "belief"
    answer_variable = f"answer_{variable}"
    records = _trace_records(spatial_trace) if spatial_trace else []
    contains: list[tuple[str, str, float]] = []
    calls: list[str] = []
    for record in records:
        values = record["values"]
        if record["kind"] == "spatial_fact" and str(values.get("relation", "")).lower() in {
            "contains",
            "inside",
            "in",
        }:
            contains.append(
                (
                    str(values.get("subject", "")),
                    str(values.get("object", "")),
                    float(values.get("confidence", 1.0)),
                )
            )
        elif record["kind"] == "visibility" and bool(values.get("value")):
            observer = str(values.get("observer", agent))
            calls.append(
                f"M.observe({int(values.get('t', 0))}, {_literal(observer)}, "
                f"{_literal(str(values.get('object', 'object')) + ' is visible')}, "
                f"'spatial visibility', {float(values.get('confidence', 0.8))!r})"
            )

    inspected: list[tuple[int, str, str]] = []
    approached: list[tuple[int, str, str]] = []
    for t, target, was_inspected, sentence in _action_location_steps(question):
        if was_inspected:
            inspected.append((t, target, sentence))
        else:
            approached.append((t, target, sentence))

    observed_pairs: set[tuple[int, str, str]] = set()
    for t, container, sentence in inspected:
        for fact_container, item, confidence in contains:
            if not _entity_match(container, fact_container):
                continue
            key = (t, _canonical_text(item), _canonical_text(fact_container))
            if key in observed_pairs:
                continue
            observed_pairs.add(key)
            fact = f"{item} is inside {fact_container}"
            calls.append(
                f"M.observe({t}, {_literal(agent)}, {_literal(fact)}, "
                f"'opened-container spatial bridge', {min(0.95, confidence * 0.9)!r})"
            )

    if variable == "belief":
        for _, hypothesis in options:
            claim = _belief_claim(hypothesis)
            if claim is None:
                continue
            item, container, negated = claim
            relevant_inspections = [entry for entry in inspected if _entity_match(entry[1], container)]
            if relevant_inspections:
                t, _, sentence = relevant_inspections[-1]
                present = any(
                    _entity_match(fact_container, container) and _entity_match(fact_item, item)
                    for fact_container, fact_item, _ in contains
                )
                # Spatial Code is open-world: an omitted containment fact is
                # unknown, not evidence that the item is absent.  Only emit
                # answer evidence when a matching positive fact was actually
                # compiled.  Negative evidence requires an explicit negative
                # or complete-inventory record, neither of which this schema
                # currently provides.
                if not present:
                    continue
                likelihood = 0.92 if negated != present else 0.08
                calls.append(
                    f"M.query_evidence({t}, {_literal(agent)}, {_literal(answer_variable)}, "
                    f"{_literal(hypothesis)}, {likelihood!r}, 0.9, "
                    f"{_literal('opened-container observation grounded in Spatial Code: ' + sentence)})"
                )

    if variable == "goal" and contains:
        for _, hypothesis in options:
            item = _goal_object(hypothesis)
            if item is None:
                continue
            known_in_world = any(_entity_match(fact_item, item) for _, fact_item, _ in contains)
            if not known_in_world or not inspected:
                continue
            found_during_search = any(
                _entity_match(fact_container, searched_container) and _entity_match(fact_item, item)
                for _, searched_container, _ in inspected
                for fact_container, fact_item, _ in contains
            )
            likelihood = 0.16 if found_during_search else 0.70
            calls.append(
                f"M.query_evidence({inspected[-1][0]}, {_literal(agent)}, {_literal(answer_variable)}, "
                f"{_literal(hypothesis)}, {likelihood!r}, 0.75, "
                f"'inverse planning over completed searches: an unretrieved visible target is unlikely to be the goal')"
            )

        final_approach = approached[-1] if approached else None
        if final_approach:
            t, container, sentence = final_approach
            for _, hypothesis in options:
                item = _goal_object(hypothesis)
                if item is None:
                    continue
                present = any(
                    _entity_match(fact_container, container) and _entity_match(fact_item, item)
                    for fact_container, fact_item, _ in contains
                )
                likelihood = 0.76 if present else 0.34
                calls.append(
                    f"M.query_evidence({t}, {_literal(agent)}, {_literal(answer_variable)}, "
                    f"{_literal(hypothesis)}, {likelihood!r}, 0.55, "
                    f"{_literal('candidate compatibility with next approached location: ' + sentence)})"
                )

    if not calls:
        calls.append(f"M.unknown({_literal(agent)}, 'spatial_social_bridge')")
    code = "\n".join(calls)
    return SocialCodeResult(
        "spatial_bridge",
        code,
        execute_social(code),
        ("spatial_bridge",),
        diagnostics={
            "spatial_records": len(records),
            "inspections": len(inspected),
            "approaches": len(approached),
            "bridged_observations": len(observed_pairs),
        },
    )


def _trace_records(trace: str) -> list[dict[str, Any]]:
    records = []
    for line in trace.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and isinstance(record.get("values"), dict):
            records.append(record)
    return records


def answer_posterior_probs(code_context: str) -> dict[str, float] | None:
    """Extract and normalize an explicit native answer posterior from a trace."""
    probabilities: dict[str, float] = {}
    for line in code_context.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = record.get("values", {})
        if record.get("kind") != "posterior":
            continue
        if not str(values.get("variable", "")).startswith("answer_"):
            continue
        label = str(values.get("label", "")).upper()
        if label in {"A", "B"}:
            probabilities[label] = float(values.get("value", 0.0))
    if set(probabilities) != {"A", "B"}:
        return None
    denominator = probabilities["A"] + probabilities["B"]
    if denominator <= 0:
        return None
    return {
        "A": probabilities["A"] / denominator,
        "B": probabilities["B"] / denominator,
    }


_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "toward",
    "towards",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    }


def _compatibility(evidence: str, hypothesis: str) -> float:
    """Estimate a calibrated semantic likelihood for non-query evidence."""
    evidence_claim = _belief_claim(evidence)
    hypothesis_claim = _belief_claim(hypothesis)
    if evidence_claim and hypothesis_claim:
        same_relation = _entity_match(evidence_claim[0], hypothesis_claim[0]) and _entity_match(
            evidence_claim[1], hypothesis_claim[1]
        )
        if same_relation:
            return 0.9 if evidence_claim[2] == hypothesis_claim[2] else 0.1

    evidence_tokens = _tokens(evidence)
    hypothesis_tokens = _tokens(hypothesis)
    union = evidence_tokens | hypothesis_tokens
    overlap = len(evidence_tokens & hypothesis_tokens) / len(union) if union else 0.0
    lower_evidence = evidence.lower()
    lower_hypothesis = hypothesis.lower()
    verb_match = 0.0
    verb_families = (
        (("pick", "take", "grab", "get"), ("obtain", "acquire", "get", "take")),
        (("open", "check", "inspect", "search"), ("inspect", "search", "find")),
        (("walk", "approach", "head"), ("interact", "inspect", "obtain", "reach")),
    )
    for evidence_verbs, hypothesis_verbs in verb_families:
        if any(v in lower_evidence for v in evidence_verbs) and any(
            v in lower_hypothesis for v in hypothesis_verbs
        ):
            verb_match = 0.2
            break
    return min(0.92, max(0.08, 0.30 + 0.50 * overlap + verb_match))


def _explicit_belief_likelihood(evidence: str, hypothesis: str) -> float | None:
    """Match an explicit belief to an answer belief without lexical guessing."""
    evidence_claim = _belief_claim(evidence)
    hypothesis_claim = _belief_claim(hypothesis)
    if evidence_claim is None or hypothesis_claim is None:
        return None
    same_relation = _entity_match(evidence_claim[0], hypothesis_claim[0]) and _entity_match(
        evidence_claim[1], hypothesis_claim[1]
    )
    if not same_relation:
        return None
    return 0.9 if evidence_claim[2] == hypothesis_claim[2] else 0.1


def _noisy_or(left: float, right: float) -> float:
    return 1.0 - (1.0 - min(max(left, 0.0), 1.0)) * (
        1.0 - min(max(right, 0.0), 1.0)
    )


def compile_native_bayesian(
    question: str,
    evidence_results: Iterable[SocialCodeResult],
) -> SocialCodeResult:
    """Score Social Code hypotheses with a small native Bayesian reasoner.

    Free-form hypotheses come from candidate-hidden evidence. Answer-aligned
    hypotheses are registered only after that extraction step. Priors and
    calibrated action/observation/query evidence are normalized within each
    agent-variable-timestep hypothesis set, using no future evidence.
    """
    evidence_parts = list(evidence_results)
    records = [
        record
        for part in evidence_parts
        for record in _trace_records(part.trace)
    ]
    candidates: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    beliefs: list[dict[str, Any]] = []
    query_evidence: list[dict[str, Any]] = []

    for record in records:
        kind = record["kind"]
        values = record["values"]
        agent = str(values.get("agent", "agent"))
        variable = None
        hypothesis = None
        timestep = int(values.get("t", -1))
        prior = float(values.get("confidence", values.get("prior", 0.5)))
        if kind == "belief":
            variable, hypothesis = "belief", str(values["fact"])
            beliefs.append(values)
        elif kind == "goal":
            variable, hypothesis = "goal", str(values["goal"])
        elif kind == "intent":
            variable, hypothesis = "intent", str(values["intent"])
        elif kind == "action_evidence":
            variable, hypothesis = "intent", str(values["hypothesis"])
            actions.append(values)
        elif kind == "observation":
            variable, hypothesis = "belief", str(values["fact"])
            prior *= 0.9
            observations.append(values)
        elif kind == "query_candidate":
            variable = str(values["variable"])
            hypothesis = str(values["hypothesis"])
        elif kind == "query_evidence":
            query_evidence.append(values)
        if variable is None or hypothesis is None:
            continue
        key = (
            agent,
            variable,
            timestep,
            re.sub(r"\W+", " ", hypothesis.lower()).strip(),
        )
        if key in candidates:
            candidates[key]["prior"] = _noisy_or(candidates[key]["prior"], prior)
        else:
            candidates[key] = {
                "agent": agent,
                "variable": variable,
                "t": timestep,
                "hypothesis": hypothesis,
                "prior": min(max(prior, 0.01), 0.99),
                "label": str(values.get("label", "")),
            }

    if not candidates:
        agent, _ = action_history(question)
        code = f"M.unknown({_literal(agent)}, 'bayesian_hypotheses')"
        return SocialCodeResult(
            "bayesian",
            code,
            execute_social(code),
            ("bayesian",),
            diagnostics={"hypothesis_sets": 0, "hypotheses": 0},
        )

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates.values():
        prior = min(max(float(candidate["prior"]), 0.01), 0.99)
        candidate["prior"] = prior
        score = math.log(prior / (1.0 - prior))
        used = 0
        base_variable = str(candidate["variable"]).removeprefix("answer_")
        if base_variable in {"goal", "intent"}:
            for action in actions:
                if str(action.get("agent")) != candidate["agent"]:
                    continue
                action_t = int(action.get("t", -1))
                if base_variable == "intent" and action_t > candidate["t"]:
                    continue
                likelihood = _compatibility(
                    str(action.get("action", "")), candidate["hypothesis"]
                )
                weight = float(action.get("confidence", 0.5))
                if base_variable == "intent":
                    weight *= 0.7 ** max(0, candidate["t"] - action_t)
                score += weight * math.log(likelihood / (1.0 - likelihood))
                used += 1
        if base_variable in {"goal", "belief"} and candidate["variable"].startswith("answer_"):
            for evidence in query_evidence:
                if str(evidence.get("agent")) != candidate["agent"]:
                    continue
                if str(evidence.get("variable")) != candidate["variable"]:
                    continue
                if _canonical_text(str(evidence.get("hypothesis", ""))) != _canonical_text(
                    candidate["hypothesis"]
                ):
                    continue
                likelihood = min(max(float(evidence.get("likelihood", 0.5)), 0.02), 0.98)
                weight = float(evidence.get("confidence", 0.5))
                score += weight * math.log(likelihood / (1.0 - likelihood))
                used += 1
        if base_variable == "belief" and candidate["variable"].startswith("answer_"):
            # Answer candidates use t=-1 as a sentinel, so the ordinary
            # temporal filter below would reject every explicit belief.  Only
            # semantically matching belief records are admitted here; generic
            # observations remain excluded to avoid treating visibility or
            # motion as a belief about an answer choice.
            for belief in beliefs:
                if str(belief.get("agent")) != candidate["agent"]:
                    continue
                likelihood = _explicit_belief_likelihood(
                    str(belief.get("fact", "")), candidate["hypothesis"]
                )
                if likelihood is None:
                    continue
                weight = float(belief.get("confidence", 0.5))
                score += weight * math.log(likelihood / (1.0 - likelihood))
                used += 1
        elif base_variable == "belief":
            for observation in observations:
                if str(observation.get("agent")) != candidate["agent"]:
                    continue
                if int(observation.get("t", -1)) > candidate["t"]:
                    continue
                likelihood = _compatibility(
                    str(observation.get("fact", "")), candidate["hypothesis"]
                )
                weight = float(observation.get("confidence", 0.5))
                score += weight * math.log(likelihood / (1.0 - likelihood))
                used += 1
        candidate["score"] = max(-20.0, min(20.0, score))
        candidate["evidence_count"] = used
        grouped[(candidate["agent"], candidate["variable"], candidate["t"])].append(
            candidate
        )

    calls: list[str] = []
    for group_key in sorted(grouped):
        group = grouped[group_key]
        if len(group) == 1:
            probabilities = [1.0 / (1.0 + math.exp(-group[0]["score"]))]
        else:
            maximum = max(item["score"] for item in group)
            weights = [math.exp(item["score"] - maximum) for item in group]
            denominator = sum(weights)
            probabilities = [weight / denominator for weight in weights]
        for candidate, probability in sorted(
            zip(group, probabilities, strict=True), key=lambda pair: pair[0]["hypothesis"]
        ):
            evidence = (
                "native Social Code Bayesian update: compiler prior + "
                f"{candidate['evidence_count']} action/observation evidence records"
            )
            calls.append(
                "M.posterior("
                f"{candidate['t']!r}, {candidate['agent']!r}, "
                f"{candidate['variable']!r}, "
                f"{candidate['hypothesis']!r}, {probability!r}, {evidence!r}, "
                f"label={candidate.get('label', '')!r})"
            )

    code = "\n".join(calls)
    return SocialCodeResult(
        "bayesian",
        code,
        execute_social(code),
        ("bayesian",),
        diagnostics={
            "hypothesis_sets": len(grouped),
            "hypotheses": len(candidates),
            "evidence_records": len(actions) + len(observations) + len(query_evidence),
            "query_evidence_records": len(query_evidence),
        },
    )


def _claim_signature(text: str) -> tuple[str, bool]:
    lower = text.lower()
    negated = bool(
        re.search(r"\b(?:not|no|never|cannot|can't|isn't|wasn't|doesn't|didn't)\b", lower)
    )
    base = re.sub(
        r"\b(?:not|no|never|cannot|can't|isn't|wasn't|doesn't|didn't)\b",
        " ",
        lower,
    )
    return re.sub(r"[^a-z0-9]+", " ", base).strip(), negated


def _semantic_key(record: dict[str, Any]) -> tuple[Any, ...]:
    kind = record["kind"]
    values = record["values"]
    agent = str(values.get("agent", ""))
    content_fields = {
        "query_candidate": ("variable", "label", "hypothesis"),
        "query_evidence": ("variable", "hypothesis", "evidence"),
        "observation": ("fact",),
        "belief": ("fact",),
        "goal": ("goal",),
        "intent": ("intent",),
        "action_evidence": ("action", "hypothesis"),
        "posterior": ("variable", "hypothesis"),
        "action_likelihood": ("action", "hypothesis"),
        "social_unknown": ("variable",),
        "contradiction": ("claim", "source_a", "source_b"),
    }
    fields = content_fields.get(kind, tuple(sorted(values)))
    content = tuple(
        re.sub(r"\W+", " ", str(values.get(name, "")).lower()).strip()
        for name in fields
    )
    # Beliefs and goals accumulate through time.  Events remain time-specific.
    time = (
        values.get("t")
        if kind in {"query_evidence", "observation", "intent", "action_evidence", "posterior"}
        else None
    )
    return (kind, agent, time, *content)


def _merge_text(left: str, right: str) -> str:
    pieces = []
    for piece in [*left.split(" | "), *right.split(" | ")]:
        if piece and piece not in pieces:
            pieces.append(piece)
    return " | ".join(pieces)


def _aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    for record in records:
        record = {"kind": record["kind"], "values": dict(record["values"])}
        key = _semantic_key(record)
        if key not in aggregated:
            aggregated[key] = record
            continue
        current = aggregated[key]["values"]
        incoming = record["values"]
        if "confidence" in current and "confidence" in incoming:
            current["confidence"] = _noisy_or(
                float(current["confidence"]), float(incoming["confidence"])
            )
        if "t" in incoming:
            current["t"] = max(int(current.get("t", -1)), int(incoming["t"]))
        for field_name in ("evidence", "source"):
            if field_name in incoming:
                current[field_name] = _merge_text(
                    str(current.get(field_name, "")), str(incoming[field_name])
                )
    return list(aggregated.values())


def _resolve_contradictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    content_field = {"observation": "fact", "belief": "fact"}
    for record in records:
        field_name = content_field.get(record["kind"])
        if field_name is None:
            continue
        values = record["values"]
        base, negated = _claim_signature(str(values[field_name]))
        indexed[(record["kind"], str(values.get("agent", "")), base)].append(
            (record, negated)
        )

    contradictions: list[dict[str, Any]] = []
    for (_, agent, base), group in indexed.items():
        positives = [item for item in group if not item[1]]
        negatives = [item for item in group if item[1]]
        if not positives or not negatives:
            continue
        positive = max(
            (item[0] for item in positives),
            key=lambda item: (
                int(item["values"].get("t", -1)),
                float(item["values"].get("confidence", 0.0)),
            ),
        )
        negative = max(
            (item[0] for item in negatives),
            key=lambda item: (
                int(item["values"].get("t", -1)),
                float(item["values"].get("confidence", 0.0)),
            ),
        )
        ranked = sorted(
            [positive, negative],
            key=lambda item: (
                int(item["values"].get("t", -1)),
                float(item["values"].get("confidence", 0.0)),
            ),
            reverse=True,
        )
        loser = ranked[1]["values"]
        if "confidence" in loser:
            loser["confidence"] = float(loser["confidence"]) * 0.35
            loser["evidence"] = _merge_text(
                str(loser.get("evidence", "")),
                "discounted by conflicting newer or stronger evidence",
            )
        contradictions.append(
            {
                "kind": "contradiction",
                "values": {
                    "agent": agent,
                    "claim": base,
                    "source_a": str(positive["values"].get("evidence", "positive evidence")),
                    "source_b": str(negative["values"].get("evidence", "negative evidence")),
                },
            }
        )
    return records + contradictions


def _record_to_call(record: dict[str, Any]) -> str | None:
    kind = record["kind"]
    v = record["values"]
    if kind == "query_candidate":
        return (
            f"M.candidate({v['agent']!r}, {v['variable']!r}, {v['label']!r}, "
            f"{v['hypothesis']!r}, {v.get('prior', 0.5)!r}, {v.get('source', 'answer_option')!r})"
        )
    if kind == "query_evidence":
        return (
            f"M.query_evidence({v['t']!r}, {v['agent']!r}, {v['variable']!r}, "
            f"{v['hypothesis']!r}, {v['likelihood']!r}, {v.get('confidence', 0.5)!r}, "
            f"{v.get('evidence', '')!r})"
        )
    if kind == "observation":
        return (
            f"M.observe({v['t']!r}, {v['agent']!r}, {v['fact']!r}, "
            f"source={v.get('source', 'unknown')!r}, confidence={v.get('confidence', 1.0)!r})"
        )
    if kind == "belief":
        return (
            f"M.belief({v['t']!r}, {v['agent']!r}, {v['fact']!r}, "
            f"{v.get('confidence', 0.5)!r}, {v.get('evidence', '')!r})"
        )
    if kind == "goal":
        return (
            f"M.goal({v['agent']!r}, {v['goal']!r}, {v.get('confidence', 0.5)!r}, "
            f"{v.get('evidence', '')!r})"
        )
    if kind == "intent":
        return (
            f"M.intent({v['t']!r}, {v['agent']!r}, {v['intent']!r}, "
            f"{v.get('confidence', 0.5)!r}, {v.get('evidence', '')!r})"
        )
    if kind == "action_evidence":
        return (
            f"M.action_evidence({v['t']!r}, {v['agent']!r}, {v['action']!r}, "
            f"{v['hypothesis']!r}, {v.get('confidence', 0.5)!r}, "
            f"{v.get('evidence', '')!r})"
        )
    if kind == "posterior":
        return (
            f"M.posterior({v['t']!r}, {v['agent']!r}, {v['variable']!r}, "
            f"{v['hypothesis']!r}, "
            f"{v['value']!r}, {v.get('evidence', '')!r}, label={v.get('label', '')!r})"
        )
    if kind == "action_likelihood":
        return (
            f"M.likelihood({v['agent']!r}, {v['action']!r}, "
            f"{v['hypothesis']!r}, {v['value']!r})"
        )
    if kind == "contradiction":
        return (
            f"M.contradiction({v['agent']!r}, {v['claim']!r}, "
            f"{v['source_a']!r}, {v['source_b']!r})"
        )
    if kind == "social_unknown":
        return f"M.unknown({v['agent']!r}, {v['variable']!r})"
    return None


def fuse_social_results(
    source: str,
    results: Iterable[SocialCodeResult],
) -> SocialCodeResult:
    """Build a temporally updated and conflict-aware executable social state."""
    parts = list(results)
    raw_records = [
        record
        for part in parts
        for record in _trace_records(part.trace)
    ]
    timeline_end = max(
        (
            int(record["values"].get("t", -1))
            for record in raw_records
            if record["kind"] in {"observation", "intent", "action_evidence", "query_evidence"}
        ),
        default=-1,
    )
    latest_observations: dict[tuple[str, str], dict[str, Any]] = {}
    for record in raw_records:
        if record["kind"] != "observation":
            continue
        values = record["values"]
        base, _ = _claim_signature(str(values["fact"]))
        key = (str(values["agent"]), base)
        current = latest_observations.get(key)
        rank = (int(values["t"]), float(values.get("confidence", 0.0)))
        if current is None or rank > (
            int(current["t"]),
            float(current.get("confidence", 0.0)),
        ):
            latest_observations[key] = values

    derived_records = []
    carried_beliefs = 0
    for values in latest_observations.values():
        observed_t = int(values["t"])
        carried = timeline_end > observed_t
        carried_beliefs += int(carried)
        evidence = f"belief update from observation at t={observed_t}"
        if carried:
            evidence += f"; carried to t={timeline_end} because no contradictory observation occurred"
        derived_records.append(
            {
                "kind": "belief",
                "values": {
                    "t": timeline_end if carried else observed_t,
                    "agent": values["agent"],
                    "fact": values["fact"],
                    "confidence": float(values.get("confidence", 1.0)) * (0.88 if carried else 0.9),
                    "evidence": evidence,
                },
            }
        )

    fused_records = _aggregate_records([*raw_records, *derived_records])
    fused_records = _resolve_contradictions(fused_records)
    calls = [call for record in fused_records if (call := _record_to_call(record))]
    code = "\n".join(calls)
    if not code:
        code = "M.unknown('agent', 'social_state')"
    components = tuple(component for part in parts for component in part.components)
    raw_code = "\n".join(part.code for part in parts if part.code)
    return SocialCodeResult(
        source,
        code,
        execute_social(code),
        components,
        raw_code=raw_code,
        diagnostics={
            "raw_records": len(raw_records),
            "derived_beliefs": len(derived_records),
            "carried_beliefs": carried_beliefs,
            "timeline_end": timeline_end,
            "fused_records": len(fused_records),
            "contradictions": sum(
                record["kind"] == "contradiction" for record in fused_records
            ),
            "component_diagnostics": {
                part.source: part.diagnostics for part in parts if part.diagnostics
            },
        },
    )


def combine_results(source: str, results: Iterable[SocialCodeResult]) -> SocialCodeResult:
    """Compatibility name for the stateful fusion operation."""
    return fuse_social_results(source, results)


def compile_social_code(
    source: str,
    question: str,
    generate: Callable[..., str] | None = None,
    spatial_trace: str = "",
    bayesian_loader: Callable[[], tuple[str, str]] | None = None,
    hybrid_include_bayesian: bool = False,
    include_external_bayesian: bool = False,
) -> SocialCodeResult:
    """Compile one Social Code state from the selected native components.

    ``bayesian_loader`` is exclusively an adapter for optional external
    evidence.  Native Bayesian inference is always performed locally for the
    ``bayesian`` and ``hybrid`` configurations.
    """
    if source not in SOCIAL_SOURCES:
        raise ValueError(f"unknown Social Code source: {source}")

    external_requested = include_external_bayesian or hybrid_include_bayesian
    parts: list[SocialCodeResult] = []
    if source in {"neural", "neural_rules", "hybrid"}:
        if generate is None:
            raise ValueError(f"{source} Social Code requires a generation function")
        parts.append(compile_neural(question, generate))
    if source in {"rules", "bayesian", "neural_rules", "hybrid"}:
        parts.append(compile_rules(question))
    if source in {"bayesian", "hybrid"}:
        parts.append(compile_query_candidates(question))
        if spatial_trace:
            parts.append(compile_spatial_bridge(question, spatial_trace))
        parts.append(compile_native_bayesian(question, parts))

    if external_requested:
        if bayesian_loader is None:
            raise ValueError("external Bayesian evidence requires a loader")
        code, _ = bayesian_loader()
        code = normalize_social_code(code)
        if code:
            parts.append(
                SocialCodeResult(
                    "external_bayesian",
                    code,
                    execute_social(code),
                    ("external_bayesian",),
                )
            )

    return fuse_social_results(source, parts)
