"""Answer inference from clean Physical and Social state.

This module replaces the removed ``query_evidence`` mechanism.  It never
creates evidence attached to an answer label.  Instead, it evaluates every
option with the same fixed state-transition and inverse-planning likelihoods.
Gold labels and MMToM question-type identifiers are not inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from runtime import execute_social
from social_code import (
    _action_location_steps,
    _belief_claim,
    _entity_match,
    _explicit_belief_likelihood,
    _goal_object,
    _location_match,
    _query_context,
    _query_variable,
    action_history,
    answer_options,
)


REMOVED_ANSWER_DIRECTED_KINDS = {
    "query_candidate",
    "query_evidence",
    "action_evidence",
    "posterior",
}


@dataclass(frozen=True)
class CleanBayesianDecision:
    choice: str
    probabilities: dict[str, float]
    scores: dict[str, float]
    terms: dict[str, dict[str, float]]
    posterior_code: str
    posterior_trace: str


def _records(trace: str, *, social: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in str(trace or "").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("values"), dict):
            continue
        if social and record.get("kind") in REMOVED_ANSWER_DIRECTED_KINDS:
            continue
        records.append(record)
    return records


def _containment_facts(spatial_trace: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    for record in _records(spatial_trace):
        values = record["values"]
        if record.get("kind") != "spatial_fact":
            continue
        relation = str(values.get("relation", "")).lower()
        if relation not in {"contains", "inside", "in"}:
            continue
        subject = str(values.get("subject", ""))
        object_ = str(values.get("object", ""))
        facts.append((object_, subject) if relation == "contains" else (subject, object_))
    return facts


def _logit(probability: float) -> float:
    value = min(max(float(probability), 0.02), 0.98)
    return math.log(value / (1.0 - value))


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {label: math.exp(value - maximum) for label, value in scores.items()}
    denominator = sum(weights.values())
    return {label: value / denominator for label, value in weights.items()}


def _present(
    item: str,
    location: str,
    containment: list[tuple[str, str]],
) -> bool:
    return any(
        _entity_match(item, fact_item) and _location_match(location, fact_location)
        for fact_item, fact_location in containment
    )


def score_clean_answers(
    question: str,
    spatial_trace: str,
    social_trace: str,
) -> CleanBayesianDecision:
    """Apply one fixed state-to-answer likelihood model to both options.

    Belief options are matched against explicit agent beliefs and receive a
    small inverse-planning term from the final search step.  Goal options use
    three general action semantics: a sought object is unlikely to have been
    visibly left behind in an inspected container; a pending target containing
    it is supportive; and an explicitly believed-absent object is unlikely to
    explain movement toward that location.  Missing facts remain neutral.
    """
    options = answer_options(question)
    if len(options) != 2:
        raise ValueError("clean Bayesian answer inference requires two options")

    variable = _query_variable(options)
    social_records = _records(social_trace, social=True)
    containment = _containment_facts(spatial_trace)
    steps = _action_location_steps(question)
    agent, _ = action_history(question)
    _, premise_belief = _query_context(question, agent)
    premise_claim = _belief_claim(premise_belief or "")

    scores: dict[str, float] = {}
    terms: dict[str, dict[str, float]] = {}
    for label, hypothesis in options:
        option_terms: dict[str, float] = {}
        if variable == "belief":
            explicit = 0.0
            for record in social_records:
                if record.get("kind") != "belief":
                    continue
                values = record["values"]
                likelihood = _explicit_belief_likelihood(
                    str(values.get("fact", "")), hypothesis
                )
                if likelihood is None:
                    continue
                explicit += float(values.get("confidence", 0.5)) * _logit(likelihood)
            option_terms["explicit_belief"] = explicit

            plan = 0.0
            claim = _belief_claim(hypothesis)
            if claim is not None and steps:
                _, location, negated = claim
                _, final_location, inspected, _ = steps[-1]
                predicts_presence = _location_match(location, final_location) and not inspected
                plan = 0.25 if negated != predicts_presence else -0.25
            option_terms["inverse_plan"] = plan
        else:
            candidate_object = _goal_object(hypothesis) or hypothesis
            explicit_goal = 0.0
            for record in social_records:
                if record.get("kind") != "goal":
                    continue
                values = record["values"]
                state_goal = _goal_object(str(values.get("goal", ""))) or str(
                    values.get("goal", "")
                )
                if _entity_match(candidate_object, state_goal):
                    explicit_goal = max(
                        explicit_goal, float(values.get("confidence", 0.5))
                    )
            option_terms["explicit_goal"] = explicit_goal

            visibly_left_behind = sum(
                _present(candidate_object, location, containment)
                for _, location, inspected, _ in steps
                if inspected
            )
            option_terms["left_behind_after_inspection"] = -2.0 * visibly_left_behind

            pending_target = 0.0
            if steps and not steps[-1][2] and _present(
                candidate_object, steps[-1][1], containment
            ):
                pending_target = 1.0
            option_terms["pending_target"] = pending_target

            contradicted_plan = 0.0
            if (
                premise_claim is not None
                and premise_claim[2]
                and _entity_match(candidate_object, premise_claim[0])
                and any(_location_match(premise_claim[1], step[1]) for step in steps)
            ):
                contradicted_plan = -3.0
            option_terms["belief_plan_contradiction"] = contradicted_plan

        terms[label] = option_terms
        scores[label] = sum(option_terms.values())

    probabilities = _softmax(scores)
    choice = max((label for label, _ in options), key=lambda label: (scores[label], -ord(label[0])))
    code = "\n".join(
        "M.posterior(0, "
        f"{agent!r}, 'clean_answer_{variable}', {hypothesis!r}, "
        f"{probabilities[label]!r}, "
        f"'fixed clean state likelihood; score={scores[label]:.6f}', label={label!r})"
        for label, hypothesis in options
    )
    return CleanBayesianDecision(
        choice=choice,
        probabilities=probabilities,
        scores=scores,
        terms=terms,
        posterior_code=code,
        posterior_trace=execute_social(code),
    )
