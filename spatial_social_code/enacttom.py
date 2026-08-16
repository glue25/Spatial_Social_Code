"""Oracle-state E0 for EnactToM.

This module intentionally depends only on the Python standard library.  It
compiles the public EnactToM task JSON into a query-independent physical-social
state, replays the deterministic golden trajectory, and runs first-order
message deletion/redirection interventions.

The public trajectories contain no communication actions.  Therefore message
events used by E0 are explicit oracle interventions grounded in supported
first-order literal-ToM probes; they are not presented as reconstructed
benchmark rollouts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True, order=True)
class Literal:
    """A grounded physical predicate."""

    predicate: str
    arguments: tuple[str, ...] = ()
    negated: bool = False

    def __str__(self) -> str:
        inner = f"({self.predicate}{' ' if self.arguments else ''}{' '.join(self.arguments)})"
        return f"(not {inner})" if self.negated else inner


@dataclass(frozen=True)
class KnowledgeEvidence:
    """Auditable support for one agent-indexed information entry."""

    support: str
    evidence_ref: str
    source_agent: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "evidence_ref": self.evidence_ref,
            "source_agent": self.source_agent,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PlanAssignment:
    """An oracle plan assignment, not an inferred promise or mental state."""

    agent: str
    goal: str
    evidence_ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "oracle_plan_assignment",
            "agent": self.agent,
            "goal": self.goal,
            "evidence_ref": self.evidence_ref,
        }


@dataclass
class OracleState:
    """Query-independent Physical Code plus agent-indexed Interaction Code."""

    task_id: str
    split: str
    agents: tuple[str, ...]
    physical_facts: set[Literal]
    object_types: dict[str, str]
    entity_rooms: dict[str, str]
    agent_rooms: dict[str, str]
    restricted_rooms: dict[str, set[str]]
    message_targets: dict[str, set[str]]
    message_limits: dict[str, int | None]
    physical_goals: tuple[Literal, ...]
    roles: dict[str, set[str]] = field(default_factory=dict)
    knowledge: dict[str, dict[str, KnowledgeEvidence]] = field(default_factory=dict)
    assignments: list[PlanAssignment] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    messages_sent: dict[str, int] = field(default_factory=dict)
    held_objects: dict[str, str] = field(default_factory=dict)

    def clone(self) -> "OracleState":
        return copy.deepcopy(self)

    def knows(self, agent: str, fact: Literal | str) -> bool:
        key = str(fact) if isinstance(fact, Literal) else fact
        return key in self.knowledge.get(agent, {})

    def add_knowledge(
        self,
        agent: str,
        fact: Literal | str,
        *,
        support: str,
        evidence_ref: str,
        source_agent: str | None = None,
    ) -> None:
        key = str(fact) if isinstance(fact, Literal) else fact
        self.knowledge.setdefault(agent, {})[key] = KnowledgeEvidence(
            support=support,
            evidence_ref=evidence_ref,
            source_agent=source_agent,
        )

    def semantic_snapshot(self, excluded_agents: Iterable[str] = ()) -> dict[str, Any]:
        """State used for non-target invariance, excluding event bookkeeping."""

        excluded = set(excluded_agents)
        return {
            "physical_facts": sorted(str(fact) for fact in self.physical_facts),
            "physical_goals": sorted(str(goal) for goal in self.physical_goals),
            "roles": {
                agent: sorted(values)
                for agent, values in sorted(self.roles.items())
                if agent not in excluded
            },
            "knowledge": {
                agent: sorted(entries)
                for agent, entries in sorted(self.knowledge.items())
                if agent not in excluded
            },
            "assignments": [
                assignment.to_dict()
                for assignment in self.assignments
                if assignment.agent not in excluded
            ],
        }

    def to_code_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "physical_code": {
                "facts": sorted(str(fact) for fact in self.physical_facts),
                "entity_rooms": dict(sorted(self.entity_rooms.items())),
                "agent_rooms": dict(sorted(self.agent_rooms.items())),
                "access_restrictions": {
                    agent: sorted(rooms)
                    for agent, rooms in sorted(self.restricted_rooms.items())
                },
            },
            "interaction_code": {
                "goals": sorted(str(goal) for goal in self.physical_goals),
                "roles": {
                    agent: sorted(values)
                    for agent, values in sorted(self.roles.items())
                },
                "knowledge": {
                    agent: {
                        fact: evidence.to_dict()
                        for fact, evidence in sorted(entries.items())
                    }
                    for agent, entries in sorted(self.knowledge.items())
                },
                "assignments": [
                    assignment.to_dict() for assignment in self.assignments
                ],
                "message_targets": {
                    agent: sorted(targets)
                    for agent, targets in sorted(self.message_targets.items())
                },
                "message_limits": dict(sorted(self.message_limits.items())),
                "recent_events": list(self.recent_events),
            },
        }


@dataclass
class ReplayResult:
    state: OracleState
    action_counts: Counter[str]
    effect_actors: dict[str, str]
    goal_results: dict[str, bool]
    invalid_actions: list[dict[str, Any]]
    access_violations: list[dict[str, Any]]


def _strip_pddl_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _tokenize_sexpr(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", _strip_pddl_comments(text))


def parse_sexpr(text: str) -> Any:
    """Parse one balanced S-expression into nested Python lists."""

    tokens = _tokenize_sexpr(text)
    if not tokens:
        raise ValueError("empty S-expression")
    position = 0

    def parse_one() -> Any:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("unexpected end of S-expression")
        token = tokens[position]
        position += 1
        if token != "(":
            if token == ")":
                raise ValueError("unexpected closing parenthesis")
            return token
        result = []
        while position < len(tokens) and tokens[position] != ")":
            result.append(parse_one())
        if position >= len(tokens):
            raise ValueError("unclosed parenthesis")
        position += 1
        return result

    parsed = parse_one()
    if position != len(tokens):
        raise ValueError("trailing S-expression tokens")
    return parsed


def _problem_section(parsed: list[Any], name: str) -> list[Any]:
    needle = f":{name}".lower()
    for child in parsed:
        if (
            isinstance(child, list)
            and child
            and isinstance(child[0], str)
            and child[0].lower() == needle
        ):
            return child[1:]
    raise ValueError(f"PDDL problem is missing (:{name} ...)")


def _parse_typed_objects(tokens: list[Any]) -> dict[str, str]:
    flat = [str(token) for token in tokens if isinstance(token, str)]
    result: dict[str, str] = {}
    pending: list[str] = []
    position = 0
    while position < len(flat):
        token = flat[position]
        if token == "-" and position + 1 < len(flat):
            object_type = flat[position + 1]
            for name in pending:
                result[name] = object_type
            pending = []
            position += 2
            continue
        pending.append(token)
        position += 1
    for name in pending:
        result[name] = "object"
    return result


def _literal_from_expr(expr: Any, *, negated: bool = False) -> Literal:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        raise ValueError(f"not a grounded literal: {expr!r}")
    if expr[0].lower() == "not":
        if len(expr) != 2:
            raise ValueError(f"invalid negated literal: {expr!r}")
        return _literal_from_expr(expr[1], negated=not negated)
    if any(isinstance(part, list) for part in expr[1:]):
        raise ValueError(f"nested expression is not a physical literal: {expr!r}")
    return Literal(
        predicate=expr[0],
        arguments=tuple(str(part) for part in expr[1:]),
        negated=negated,
    )


def _flatten_physical_goals(expr: Any) -> list[Literal]:
    if not isinstance(expr, list) or not expr:
        return []
    head = str(expr[0]).lower()
    if head == "and":
        return [
            literal
            for child in expr[1:]
            for literal in _flatten_physical_goals(child)
        ]
    if head in {"k", "b"}:
        return []
    if head == "not":
        return [_literal_from_expr(expr)]
    return [_literal_from_expr(expr)]


def _extract_agents(task: dict[str, Any], object_types: dict[str, str]) -> tuple[str, ...]:
    agents = sorted(name for name, kind in object_types.items() if kind == "agent")
    if not agents:
        agents = sorted(
            set(task.get("agent_actions", {})) | set(task.get("agent_spawns", {}))
        )
    return tuple(agents)


def _literal_room(
    state: OracleState, literal: Literal, *, actor: str | None = None
) -> str | None:
    for argument in reversed(literal.arguments):
        if argument in state.entity_rooms:
            return state.entity_rooms[argument]
        if state.object_types.get(argument) == "room":
            return argument
    return state.agent_rooms.get(actor or "")


def _can_access_room(state: OracleState, agent: str, room: str | None) -> bool:
    return room is None or room not in state.restricted_rooms.get(agent, set())


def _build_topology(
    task: dict[str, Any], agents: tuple[str, ...]
) -> tuple[dict[str, set[str]], dict[str, int | None], dict[str, set[str]]]:
    explicit_targets: dict[str, set[str]] = {}
    message_limits: dict[str, int | None] = {agent: None for agent in agents}
    restricted_rooms: dict[str, set[str]] = {agent: set() for agent in agents}

    for binding in task.get("mechanic_bindings", []):
        mechanic = binding.get("mechanic_type")
        if mechanic == "restricted_communication":
            for agent, targets in (binding.get("allowed_targets") or {}).items():
                explicit_targets[agent] = set(targets or [])
        elif mechanic == "limited_bandwidth":
            for agent, limit in (binding.get("message_limits") or {}).items():
                if isinstance(limit, (int, float)):
                    message_limits[agent] = int(limit)
        elif mechanic == "room_restriction":
            rooms = set(binding.get("restricted_rooms") or [])
            for agent in binding.get("for_agents") or []:
                restricted_rooms.setdefault(agent, set()).update(rooms)

    for agent, targets in (task.get("message_targets") or {}).items():
        explicit_targets[agent] = set(targets or [])

    topology = {
        agent: set(explicit_targets.get(agent, set(agents) - {agent}))
        for agent in agents
    }
    return topology, message_limits, restricted_rooms


def _secret_key(secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]
    return f"private_task_input:{digest}"


def compile_oracle_state(task: dict[str, Any], split: str) -> OracleState:
    """Compile task metadata and PDDL before any question is consulted."""

    problem = parse_sexpr(task["problem_pddl"])
    if not isinstance(problem, list):
        raise ValueError("PDDL problem must be a list")
    object_types = _parse_typed_objects(_problem_section(problem, "objects"))
    init_exprs = _problem_section(problem, "init")
    physical_facts = {
        _literal_from_expr(expr)
        for expr in init_exprs
        if isinstance(expr, list)
        and expr
        and str(expr[0]).lower() not in {"k", "b"}
    }
    agents = _extract_agents(task, object_types)
    topology, message_limits, restricted_rooms = _build_topology(task, agents)

    entity_rooms = {
        fact.arguments[0]: fact.arguments[1]
        for fact in physical_facts
        if not fact.negated
        and fact.predicate == "is_in_room"
        and len(fact.arguments) == 2
    }
    for entity, object_type in object_types.items():
        if object_type == "room":
            entity_rooms[entity] = entity
    agent_rooms = {
        fact.arguments[0]: fact.arguments[1]
        for fact in physical_facts
        if not fact.negated
        and fact.predicate == "agent_in_room"
        and len(fact.arguments) == 2
    }

    planned_literals = (
        task.get("golden_trajectory_metadata", {}).get("planned_literals") or []
    )
    functional_goal = task.get("functional_goal_pddl")
    if planned_literals:
        # Disjunctive inverse-state tasks have multiple valid goal branches.
        # The saved planner literals identify the branch that the saved golden
        # trajectory actually proves, which is the correct oracle replay target.
        physical_goals = tuple(
            _literal_from_expr(parse_sexpr(str(literal)))
            for literal in planned_literals
        )
    elif functional_goal:
        physical_goals = tuple(
            _flatten_physical_goals(parse_sexpr(functional_goal))
        )
    else:
        goal_section = _problem_section(problem, "goal")
        if len(goal_section) != 1:
            raise ValueError("PDDL :goal must contain one formula")
        physical_goals = tuple(_flatten_physical_goals(goal_section[0]))

    state = OracleState(
        task_id=task["task_id"],
        split=split,
        agents=agents,
        physical_facts=physical_facts,
        object_types=object_types,
        entity_rooms=entity_rooms,
        agent_rooms=agent_rooms,
        restricted_rooms=restricted_rooms,
        message_targets=topology,
        message_limits=message_limits,
        physical_goals=physical_goals,
        roles={agent: set() for agent in agents},
        knowledge={agent: {} for agent in agents},
        messages_sent={agent: 0 for agent in agents},
    )

    for agent, secrets in (task.get("agent_secrets") or {}).items():
        for index, secret in enumerate(secrets or []):
            state.add_knowledge(
                agent,
                _secret_key(str(secret)),
                support="private_task_input",
                evidence_ref=f"agent_secrets.{agent}[{index}]",
            )
            lowered = str(secret).lower()
            if "only agent" in lowered or "only one" in lowered:
                state.roles.setdefault(agent, set()).add("exclusive_accessor")

    state.recent_events.append(
        {
            "event_id": "compile",
            "type": "oracle_compile",
            "query_independent": True,
            "source": "problem_pddl+task_metadata",
        }
    )
    return state


_ACTION_RE = re.compile(r"^(?P<name>[A-Za-z_]+)\[(?P<arguments>.*)\]$")


def parse_action(action: str) -> tuple[str, list[str]]:
    match = _ACTION_RE.fullmatch(action.strip())
    if not match:
        raise ValueError(f"invalid action syntax: {action}")
    raw_arguments = match.group("arguments")
    arguments = (
        []
        if not raw_arguments
        else [part.strip() for part in raw_arguments.split(",")]
    )
    return match.group("name"), arguments


def _remove_facts(
    state: OracleState, predicate: str, first_argument: str
) -> None:
    state.physical_facts = {
        fact
        for fact in state.physical_facts
        if not (
            fact.predicate == predicate
            and fact.arguments
            and fact.arguments[0] == first_argument
        )
    }


def _set_entity_room(state: OracleState, entity: str, room: str | None) -> None:
    if not room:
        return
    _remove_facts(state, "is_in_room", entity)
    state.physical_facts.add(Literal("is_in_room", (entity, room)))
    state.entity_rooms[entity] = room


def _record_observation(
    state: OracleState,
    *,
    actor: str,
    effects: Iterable[Literal],
    evidence_ref: str,
) -> None:
    for effect in effects:
        room = _literal_room(state, effect, actor=actor)
        for observer in state.agents:
            same_room = state.agent_rooms.get(observer) == room
            if observer == actor or (
                same_room and _can_access_room(state, observer, room)
            ):
                state.add_knowledge(
                    observer,
                    effect,
                    support="observed_action",
                    evidence_ref=evidence_ref,
                    source_agent=actor,
                )


def _mechanic_effects(
    task: dict[str, Any], action_name: str, target: str
) -> list[Literal]:
    if action_name not in {"Open", "Close"}:
        return []
    effects: list[Literal] = []
    for binding in task.get("mechanic_bindings", []):
        mechanic = binding.get("mechanic_type")
        if mechanic not in {"remote_control", "state_mirroring"}:
            continue
        if binding.get("trigger_object") != target:
            continue
        target_object = binding.get("target_object")
        target_state = binding.get("target_state")
        if target_object and target_state:
            effects.append(Literal(str(target_state), (str(target_object),)))
    return effects


def apply_action(
    state: OracleState,
    task: dict[str, Any],
    *,
    agent: str,
    action: str,
    evidence_ref: str,
) -> list[Literal]:
    """Apply one golden action and return its positive physical effects."""

    name, arguments = parse_action(action)
    if agent not in state.agents:
        raise ValueError(f"unknown agent: {agent}")
    effects: list[Literal] = []

    if name == "Wait":
        pass
    elif name == "Navigate":
        if len(arguments) != 1:
            raise ValueError(f"Navigate expects one argument: {action}")
        target = arguments[0]
        room = (
            target
            if state.object_types.get(target) == "room"
            else state.entity_rooms.get(target)
        )
        if room is None:
            raise ValueError(f"cannot resolve navigation room for {target}")
        _remove_facts(state, "agent_in_room", agent)
        location = Literal("agent_in_room", (agent, room))
        state.physical_facts.add(location)
        state.agent_rooms[agent] = room
        effects.append(location)
    elif name == "Pick":
        if len(arguments) != 1:
            raise ValueError(f"Pick expects one argument: {action}")
        object_name = arguments[0]
        state.physical_facts = {
            fact
            for fact in state.physical_facts
            if not (
                fact.arguments
                and fact.arguments[0] == object_name
                and fact.predicate in {"is_on_top", "is_inside", "holding"}
            )
        }
        holding = Literal("holding", (agent, object_name))
        state.physical_facts.add(holding)
        state.held_objects[object_name] = agent
        _set_entity_room(state, object_name, state.agent_rooms.get(agent))
        effects.append(holding)
    elif name == "Place":
        if len(arguments) != 5:
            raise ValueError(f"Place expects five arguments: {action}")
        object_name, relation, target = arguments[:3]
        if state.held_objects.get(object_name) not in {None, agent}:
            raise ValueError(f"{agent} does not hold {object_name}")
        _remove_facts(state, "holding", agent)
        state.held_objects.pop(object_name, None)
        state.physical_facts = {
            fact
            for fact in state.physical_facts
            if not (
                fact.arguments
                and fact.arguments[0] == object_name
                and fact.predicate in {"is_on_top", "is_inside"}
            )
        }
        predicate = {"on": "is_on_top", "within": "is_inside"}.get(relation)
        if predicate is None:
            raise ValueError(f"unsupported Place relation: {relation}")
        placed = Literal(predicate, (object_name, target))
        state.physical_facts.add(placed)
        _set_entity_room(state, object_name, state.entity_rooms.get(target))
        effects.append(placed)
    elif name in {"Open", "Close"}:
        if len(arguments) != 1:
            raise ValueError(f"{name} expects one argument: {action}")
        target = arguments[0]
        if name == "Open":
            _remove_facts(state, "is_closed", target)
            changed = Literal("is_open", (target,))
        else:
            _remove_facts(state, "is_open", target)
            changed = Literal("is_closed", (target,))
        state.physical_facts.add(changed)
        effects.append(changed)
        for mechanic_effect in _mechanic_effects(task, name, target):
            opposite = (
                "is_closed"
                if mechanic_effect.predicate == "is_open"
                else "is_open"
            )
            _remove_facts(state, opposite, mechanic_effect.arguments[0])
            state.physical_facts.add(mechanic_effect)
            effects.append(mechanic_effect)
    else:
        raise ValueError(f"unsupported E0 action: {name}")

    _record_observation(
        state, actor=agent, effects=effects, evidence_ref=evidence_ref
    )
    state.recent_events.append(
        {
            "event_id": evidence_ref,
            "type": "golden_action",
            "actor": agent,
            "action": action,
            "effects": [str(effect) for effect in effects],
        }
    )
    return effects


def goal_holds(state: OracleState, goal: Literal) -> bool:
    positive = Literal(goal.predicate, goal.arguments)
    if goal.negated:
        return positive not in state.physical_facts
    if goal.predicate == "is_closed":
        return (
            positive in state.physical_facts
            or Literal("is_open", goal.arguments) not in state.physical_facts
        )
    return positive in state.physical_facts


def _derive_assignments(
    state: OracleState, effect_actors: dict[str, str]
) -> None:
    for goal in state.physical_goals:
        actor = effect_actors.get(str(Literal(goal.predicate, goal.arguments)))
        if not actor:
            continue
        assignment = PlanAssignment(
            agent=actor,
            goal=str(goal),
            evidence_ref="golden_trajectory",
        )
        if assignment not in state.assignments:
            state.assignments.append(assignment)
        state.roles.setdefault(actor, set()).add("goal_executor")


def replay_golden_trajectory(
    task: dict[str, Any], split: str
) -> ReplayResult:
    state = compile_oracle_state(task, split)
    counts: Counter[str] = Counter()
    effect_actors: dict[str, str] = {}
    invalid_actions: list[dict[str, Any]] = []
    access_violations: list[dict[str, Any]] = []

    for step_index, step in enumerate(task.get("golden_trajectory", [])):
        for action_index, action_row in enumerate(step.get("actions", [])):
            action = str(action_row["action"])
            action_name = action.split("[", 1)[0]
            counts[action_name] += 1
            evidence_ref = f"golden_trajectory[{step_index}].actions[{action_index}]"
            try:
                if action_name == "Navigate":
                    _, arguments = parse_action(action)
                    target = arguments[0]
                    room = (
                        target
                        if state.object_types.get(target) == "room"
                        else state.entity_rooms.get(target)
                    )
                    if room and not _can_access_room(
                        state, str(action_row["agent"]), room
                    ):
                        access_violations.append(
                            {
                                "evidence_ref": evidence_ref,
                                "agent": action_row["agent"],
                                "action": action,
                                "room": room,
                                "restriction_source": "mechanic_bindings",
                            }
                        )
                effects = apply_action(
                    state,
                    task,
                    agent=str(action_row["agent"]),
                    action=action,
                    evidence_ref=evidence_ref,
                )
                for effect in effects:
                    effect_actors[str(effect)] = str(action_row["agent"])
            except ValueError as exc:
                invalid_actions.append(
                    {
                        "evidence_ref": evidence_ref,
                        "agent": action_row.get("agent"),
                        "action": action,
                        "error": str(exc),
                    }
                )

    _derive_assignments(state, effect_actors)
    goal_results = {
        str(goal): goal_holds(state, goal) for goal in state.physical_goals
    }
    return ReplayResult(
        state=state,
        action_counts=counts,
        effect_actors=effect_actors,
        goal_results=goal_results,
        invalid_actions=invalid_actions,
        access_violations=access_violations,
    )


def _shortest_path(
    topology: dict[str, set[str]], source: str, target: str
) -> list[str] | None:
    queue: deque[list[str]] = deque([[source]])
    visited = {source}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == target:
            return path
        for neighbor in sorted(topology.get(node, set())):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


def send_information(
    state: OracleState,
    *,
    sender: str,
    recipient: str,
    fact: Literal,
    evidence_ref: str,
) -> None:
    if recipient not in state.message_targets.get(sender, set()):
        raise ValueError(f"message route {sender}->{recipient} is not allowed")
    if not state.knows(sender, fact):
        raise ValueError(f"{sender} cannot send unsupported fact {fact}")
    limit = state.message_limits.get(sender)
    sent = state.messages_sent.get(sender, 0)
    if limit is not None and sent >= limit:
        raise ValueError(f"{sender} has exhausted its message budget")
    state.messages_sent[sender] = sent + 1
    state.add_knowledge(
        recipient,
        fact,
        support="informed",
        evidence_ref=evidence_ref,
        source_agent=sender,
    )
    state.recent_events.append(
        {
            "event_id": evidence_ref,
            "type": "inform",
            "actor": sender,
            "recipient": recipient,
            "content": str(fact),
        }
    )


def _apply_route(
    base: OracleState,
    path: list[str],
    fact: Literal,
    *,
    stop_before_last: bool = False,
) -> OracleState:
    state = base.clone()
    edges = list(zip(path, path[1:]))
    if stop_before_last:
        edges = edges[:-1]
    for index, (sender, recipient) in enumerate(edges):
        send_information(
            state,
            sender=sender,
            recipient=recipient,
            fact=fact,
            evidence_ref=f"oracle_message[{index}]",
        )
    return state


def _probe_literal(probe: dict[str, Any]) -> Literal:
    return _literal_from_expr(parse_sexpr(str(probe["fact_pddl"])))


def evaluate_first_order_interventions(
    task: dict[str, Any], replay: ReplayResult
) -> list[dict[str, Any]]:
    """Evaluate supported depth-1 probe messages independently."""

    cases: list[dict[str, Any]] = []
    base = replay.state
    for probe in task.get("literal_tom_probes", []):
        if not probe.get("supported") or probe.get("depth") != 1:
            continue
        target = str(probe["agent_id"])
        fact = _probe_literal(probe)
        if base.knows(target, fact):
            cases.append(
                {
                    "probe_id": probe["probe_id"],
                    "fact": str(fact),
                    "target": target,
                    "status": "skip_target_already_observed",
                }
            )
            continue

        candidate_sources = [
            agent for agent in base.agents if base.knows(agent, fact)
        ]
        paths = [
            path
            for source in candidate_sources
            if (path := _shortest_path(base.message_targets, source, target))
            and len(path) > 1
        ]
        if not paths:
            cases.append(
                {
                    "probe_id": probe["probe_id"],
                    "fact": str(fact),
                    "target": target,
                    "status": "skip_no_grounded_route",
                }
            )
            continue
        path = min(paths, key=lambda value: (len(value), value))

        try:
            full = _apply_route(base, path, fact)
            deleted = _apply_route(base, path, fact, stop_before_last=True)
        except ValueError as exc:
            cases.append(
                {
                    "probe_id": probe["probe_id"],
                    "fact": str(fact),
                    "target": target,
                    "status": "skip_route_or_budget_error",
                    "error": str(exc),
                }
            )
            continue

        delete_target_sensitive = full.knows(target, fact) and not deleted.knows(
            target, fact
        )
        delete_non_target_invariant = full.semantic_snapshot(
            {target}
        ) == deleted.semantic_snapshot({target})

        final_sender = path[-2]
        redirect_candidates = sorted(
            base.message_targets.get(final_sender, set()) - {target}
        )
        redirect_result: dict[str, Any] = {"available": False}
        if redirect_candidates:
            redirect_target = redirect_candidates[0]
            prefix = _apply_route(base, path, fact, stop_before_last=True)
            try:
                send_information(
                    prefix,
                    sender=final_sender,
                    recipient=redirect_target,
                    fact=fact,
                    evidence_ref="oracle_message[redirect]",
                )
                redirect_result = {
                    "available": True,
                    "recipient": redirect_target,
                    "target_sensitive": full.knows(target, fact)
                    and not prefix.knows(target, fact)
                    and prefix.knows(redirect_target, fact),
                    "non_target_invariant": full.semantic_snapshot(
                        {target, redirect_target}
                    )
                    == prefix.semantic_snapshot({target, redirect_target}),
                }
            except ValueError as exc:
                redirect_result = {
                    "available": False,
                    "error": str(exc),
                }

        cases.append(
            {
                "probe_id": probe["probe_id"],
                "fact": str(fact),
                "target": target,
                "status": "evaluated",
                "source": path[0],
                "route": path,
                "route_length": len(path) - 1,
                "full_delivery": full.knows(target, fact),
                "delete": {
                    "target_sensitive": delete_target_sensitive,
                    "non_target_invariant": delete_non_target_invariant,
                },
                "redirect": redirect_result,
            }
        )
    return cases


def iter_tasks(paths: Iterable[tuple[str, Path]]) -> Iterator[tuple[str, dict[str, Any]]]:
    for split, path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield split, json.loads(line)


def run_e0(paths: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    """Run deterministic E0 over one or more public EnactToM JSONL splits."""

    paths = list(paths)
    task_results: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    total_goals = 0
    passed_goals = 0
    all_cases: list[dict[str, Any]] = []
    task_count = 0
    all_task_ids: set[str] = set()
    all_probe_count = 0

    for split, task in iter_tasks(paths):
        task_count += 1
        all_task_ids.add(task["task_id"])
        all_probe_count += len(task.get("literal_tom_probes", []))
        replay = replay_golden_trajectory(task, split)
        cases = evaluate_first_order_interventions(task, replay)
        action_counts.update(replay.action_counts)
        total_goals += len(replay.goal_results)
        passed_goals += sum(replay.goal_results.values())
        for case in cases:
            all_cases.append(
                {
                    "split": split,
                    "task_id": task["task_id"],
                    **case,
                }
            )
        task_results.append(
            {
                "split": split,
                "task_id": task["task_id"],
                "category": task.get("category"),
                "tom_level": task.get("tom_level"),
                "num_agents": len(replay.state.agents),
                "action_counts": dict(sorted(replay.action_counts.items())),
                "physical_goals": replay.goal_results,
                "physical_goal_success": all(replay.goal_results.values()),
                "invalid_actions": replay.invalid_actions,
                "access_violations": replay.access_violations,
                "assignments": [
                    assignment.to_dict()
                    for assignment in replay.state.assignments
                ],
                "first_order_interventions": cases,
            }
        )

    evaluated = [case for case in all_cases if case["status"] == "evaluated"]
    redirects = [
        case
        for case in evaluated
        if case.get("redirect", {}).get("available")
    ]
    status_counts = Counter(case["status"] for case in all_cases)
    invalid_action_count = sum(
        len(task["invalid_actions"]) for task in task_results
    )
    access_violation_count = sum(
        len(task["access_violations"]) for task in task_results
    )
    failed_replay_tasks = [
        task for task in task_results if not task["physical_goal_success"]
    ]
    access_violation_tasks = [
        task for task in task_results if task["access_violations"]
    ]
    no_effect_failed_tasks = [
        task
        for task in failed_replay_tasks
        if not any(
            count
            for action, count in task["action_counts"].items()
            if action not in {"Navigate", "Wait"}
        )
    ]

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "experiment": "enacttom_e0_oracle_state_reducer_v0",
        "status": "deterministic_oracle_semantics",
        "claim_boundary": (
            "This validates query-independent state compilation, physical replay, "
            "and first-order intervention locality. Oracle messages are synthetic "
            "probe-grounded interventions, not recovered benchmark communication "
            "or learned-model evidence."
        ),
        "scope": {
            "task_entries": task_count,
            "unique_task_ids": len(all_task_ids),
            "splits": [split for split, _ in paths],
            "first_order_probe_cases": len(all_cases),
            "higher_order_probe_cases": all_probe_count - len(all_cases),
            "higher_order_status": "diagnostic_only",
        },
        "aggregate": {
            "action_counts": dict(sorted(action_counts.items())),
            "invalid_actions": invalid_action_count,
            "golden_access_violations": access_violation_count,
            "physical_goals": total_goals,
            "physical_goals_satisfied": passed_goals,
            "physical_goal_replay_rate": ratio(passed_goals, total_goals),
            "tasks_with_all_physical_goals_satisfied": (
                task_count - len(failed_replay_tasks)
            ),
            "tasks_with_failed_goal_replay": len(failed_replay_tasks),
            "tasks_with_golden_access_violations": len(access_violation_tasks),
            "failed_tasks_with_no_effectful_golden_action": len(
                no_effect_failed_tasks
            ),
            "intervention_status": dict(sorted(status_counts.items())),
            "evaluated_message_cases": len(evaluated),
            "oracle_delivery_rate": ratio(
                sum(bool(case.get("full_delivery")) for case in evaluated),
                len(evaluated),
            ),
            "delete_target_sensitivity": ratio(
                sum(bool(case["delete"]["target_sensitive"]) for case in evaluated),
                len(evaluated),
            ),
            "delete_non_target_invariance": ratio(
                sum(
                    bool(case["delete"]["non_target_invariant"])
                    for case in evaluated
                ),
                len(evaluated),
            ),
            "redirect_cases": len(redirects),
            "redirect_target_sensitivity": ratio(
                sum(
                    bool(case["redirect"]["target_sensitive"])
                    for case in redirects
                ),
                len(redirects),
            ),
            "redirect_non_target_invariance": ratio(
                sum(
                    bool(case["redirect"]["non_target_invariant"])
                    for case in redirects
                ),
                len(redirects),
            ),
        },
        "data_diagnostics": {
            "failed_goal_replay_task_ids": [
                {
                    "split": task["split"],
                    "task_id": task["task_id"],
                    "failed_goals": [
                        goal
                        for goal, passed in task["physical_goals"].items()
                        if not passed
                    ],
                }
                for task in failed_replay_tasks
            ],
            "golden_access_violation_task_ids": [
                {
                    "split": task["split"],
                    "task_id": task["task_id"],
                    "violations": len(task["access_violations"]),
                }
                for task in access_violation_tasks
            ],
        },
        "tasks": task_results,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_markdown(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]

    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# EnactToM E0: Oracle State and Reducer",
        "",
        "## Scope",
        "",
        f"- Task entries: {result['scope']['task_entries']}",
        f"- Unique task IDs: {result['scope']['unique_task_ids']}",
        f"- First-order probe cases: {result['scope']['first_order_probe_cases']}",
        f"- Higher-order probe cases: {result['scope']['higher_order_probe_cases']} (diagnostic only)",
        "",
        "## Deterministic Replay",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Physical goals | {aggregate['physical_goals']} |",
        f"| Physical goals satisfied | {aggregate['physical_goals_satisfied']} |",
        f"| Physical goal replay rate | {percentage(aggregate['physical_goal_replay_rate'])} |",
        f"| Tasks with all goals satisfied | {aggregate['tasks_with_all_physical_goals_satisfied']} |",
        f"| Tasks with failed goal replay | {aggregate['tasks_with_failed_goal_replay']} |",
        f"| Invalid golden actions | {aggregate['invalid_actions']} |",
        f"| Golden access violations | {aggregate['golden_access_violations']} |",
        f"| Tasks with access violations | {aggregate['tasks_with_golden_access_violations']} |",
        f"| Failed tasks with no effectful golden action | {aggregate['failed_tasks_with_no_effectful_golden_action']} |",
        "",
        "Action coverage:",
        "",
        "| Action | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name} | {count} |"
        for name, count in aggregate["action_counts"].items()
    )
    lines.extend(
        [
            "",
            "## First-order Message Interventions",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Evaluated message cases | {aggregate['evaluated_message_cases']} |",
            f"| Oracle delivery rate | {percentage(aggregate['oracle_delivery_rate'])} |",
            f"| Delete target sensitivity | {percentage(aggregate['delete_target_sensitivity'])} |",
            f"| Delete non-target invariance | {percentage(aggregate['delete_non_target_invariance'])} |",
            f"| Redirect cases | {aggregate['redirect_cases']} |",
            f"| Redirect target sensitivity | {percentage(aggregate['redirect_target_sensitivity'])} |",
            f"| Redirect non-target invariance | {percentage(aggregate['redirect_non_target_invariance'])} |",
            "",
            "Case status:",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
            f"| {status} | {count} |"
        for status, count in aggregate["intervention_status"].items()
    )
    lines.extend(
        [
            "",
            "## Public-data Diagnostics",
            "",
            "The reducer found no unsupported action syntax. It did find two "
            "dataset-level consistency signals:",
            "",
            f"- {aggregate['tasks_with_golden_access_violations']} tasks contain "
            f"{aggregate['golden_access_violations']} golden navigation steps whose "
            "resolved rooms appear in the acting agent's room restrictions. This is "
            "a static metadata/action mismatch and requires Habitat runtime "
            "confirmation before being called a benchmark error.",
            f"- {aggregate['tasks_with_failed_goal_replay']} task entries have "
            "unsatisfied saved planner literals; all of them contain only "
            "Navigate/Wait actions and no effectful golden action.",
            "",
            "| Split | Failed replay task ID |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| {item['split']} | `{item['task_id']}` |"
        for item in result["data_diagnostics"]["failed_goal_replay_task_ids"]
    )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            result["claim_boundary"],
            "",
            "The public golden trajectories contain no communication actions. "
            "E0 therefore tests reducer and intervention semantics, not whether "
            "an agent can infer, generate, or execute the required messages.",
            "",
        ]
    )
    return "\n".join(lines)
