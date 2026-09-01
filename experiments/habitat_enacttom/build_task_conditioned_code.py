#!/usr/bin/env python3
"""Build fair task-conditioned Spatial/Social Code task copies.

The compiler reads the public task, mechanic configuration, and each agent's
own private facts.  It deliberately never reads ``golden_trajectory`` and never
emits action strings.  Consequently Qwen remains responsible for planning and
execution in the normal Habitat runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PREDICATE_RE = re.compile(r"\(([a-zA-Z_][\w-]*)\s+([^()]+?)\)")
ENTITY_RE = re.compile(r"\b[a-z][a-z0-9_]*_\d+\b")


def _agents(task: dict[str, Any]) -> list[str]:
    actions = task.get("agent_actions", {})
    if isinstance(actions, dict) and actions:
        return sorted(str(agent) for agent in actions)
    return [f"agent_{index}" for index in range(int(task.get("num_agents", 0)))]


def _functional_atoms(task: dict[str, Any]) -> list[tuple[str, list[str]]]:
    goal = str(task.get("functional_goal_pddl", ""))
    atoms: list[tuple[str, list[str]]] = []
    for predicate, raw_args in PREDICATE_RE.findall(goal):
        if predicate == "and":
            continue
        atoms.append((predicate, raw_args.split()))
    return atoms


def _init_atoms(task: dict[str, Any]) -> list[tuple[str, list[str]]]:
    problem = str(task.get("problem_pddl", ""))
    init_match = re.search(r"\(:init(.*?)\)\s*\(:goal", problem, re.DOTALL)
    if not init_match:
        return []
    atoms = []
    for predicate, raw_args in PREDICATE_RE.findall(init_match.group(1)):
        atoms.append((predicate, raw_args.split()))
    return atoms


def _room_restrictions(task: dict[str, Any], agents: list[str]) -> dict[str, set[str]]:
    restrictions = {agent: set() for agent in agents}
    for binding in task.get("mechanic_bindings", []):
        if not isinstance(binding, dict) or binding.get("mechanic_type") != "room_restriction":
            continue
        for agent in binding.get("for_agents", []):
            restrictions.setdefault(str(agent), set()).update(
                str(room) for room in binding.get("restricted_rooms", [])
            )
    return restrictions


def _message_limits(task: dict[str, Any], agents: list[str]) -> dict[str, str]:
    limits = {agent: "UNLIMITED" for agent in agents}
    for binding in task.get("mechanic_bindings", []):
        if not isinstance(binding, dict) or binding.get("mechanic_type") != "limited_bandwidth":
            continue
        for agent, limit in binding.get("message_limits", {}).items():
            limits[str(agent)] = str(limit)
    return limits


def _message_targets(task: dict[str, Any], agents: list[str]) -> dict[str, list[str]]:
    raw = task.get("message_targets")
    if isinstance(raw, dict) and raw:
        return {
            agent: [str(target) for target in raw.get(agent, []) if str(target) != agent]
            for agent in agents
        }
    for binding in task.get("mechanic_bindings", []):
        if not isinstance(binding, dict) or binding.get("mechanic_type") != "restricted_communication":
            continue
        allowed = binding.get("allowed_targets")
        if isinstance(allowed, dict):
            return {
                agent: [str(target) for target in allowed.get(agent, []) if str(target) != agent]
                for agent in agents
            }
    return {agent: [peer for peer in agents if peer != agent] for agent in agents}


def _shortest_path(
    source: str, target: str, adjacency: dict[str, list[str]]
) -> list[str]:
    queue: list[list[str]] = [[source]]
    visited = {source}
    while queue:
        path = queue.pop(0)
        if path[-1] == target:
            return path
        for neighbor in adjacency.get(path[-1], []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []


def _coordination_roles(task: dict[str, Any], agents: list[str]) -> dict[str, str]:
    restrictions = _room_restrictions(task, agents)
    targets = _message_targets(task, agents)
    integrator = min(agents, key=lambda item: (len(restrictions.get(item, set())), item))
    paths = {
        source: _shortest_path(source, integrator, targets)
        for source in agents
        if source != integrator
    }
    roles: dict[str, str] = {}
    for agent in agents:
        if agent == integrator:
            roles[agent] = (
                f"primary_integrator={integrator}; BEFORE spending your only message, use received "
                "exact facts and your own observations; while waiting, begin only executable work "
                "supported by exact facts you already know"
            )
            continue
        own_path = paths.get(agent, [])
        next_hop = own_path[1] if len(own_path) > 1 else "UNREACHABLE"
        upstream = [
            source
            for source, path in paths.items()
            if agent in path[1:-1]
        ]
        if upstream:
            roles[agent] = (
                f"relay_to={next_hop}; FIRST DECISION: Wait[None] until the initial message from "
                f"{','.join(upstream)} arrives; then immediately combine it with all task-relevant "
                f"private facts you know and send one message directly to {next_hop}"
            )
        else:
            roles[agent] = (
                f"fact_source_to={next_hop}; FIRST DECISION: send all task-relevant exact private "
                f"facts in one message directly to {next_hop}, before searching or moving"
            )
    return roles


def _known_entities(task: dict[str, Any], agent: str) -> set[str]:
    # Exact IDs present in this agent's own private facts are fair game.  Exact
    # IDs that only occur in another agent's secret remain UNKNOWN.
    own = " ".join(str(item) for item in task.get("agent_secrets", {}).get(agent, []))
    public = str(task.get("task", ""))
    return set(ENTITY_RE.findall(own + " " + public)) | {agent}


def _render_spatial(task: dict[str, Any], agent: str) -> str:
    goals = _functional_atoms(task)
    init = _init_atoms(task)
    known = _known_entities(task, agent)
    goal_entities = {arg for _, args in goals for arg in args if ENTITY_RE.fullmatch(arg)}

    goal_lines = []
    for predicate, args in goals:
        rendered = [arg if arg in known else f"UNKNOWN_{arg.rsplit('_', 1)[0]}" for arg in args]
        goal_lines.append(f"  - {predicate.upper()}({', '.join(rendered)})")

    relation_lines = []
    for predicate, args in init:
        exact_args = [arg for arg in args if ENTITY_RE.fullmatch(arg)]
        if not args or not exact_args or not all(arg in known for arg in exact_args):
            continue
        if predicate == "agent_in_room" and args[0] != agent:
            continue
        # Do not introduce an exact task target that this agent cannot derive
        # from public text or its own private facts.
        if any(arg in goal_entities and arg not in known for arg in args):
            continue
        relation_lines.append(f"  - {predicate.upper()}({', '.join(args)})")

    if not relation_lines:
        relation_lines.append("  - UNKNOWN (use FindObjectTool/FindReceptacleTool; never guess an ID)")
    return "\n".join(
        [
            "SPATIAL_STATE {",
            " task_goal:",
            *goal_lines,
            " known_current_relations:",
            *relation_lines,
            " type_safety: Pick only movable objects; Open/Close only articulated furniture.",
            " interaction_precondition: Navigate to the exact object before Pick/Open/Close; Navigate to the exact destination before Place; after distance/occlusion failure, Navigate and retry.",
            " completion_rule: a goal changes only after a successful runtime observation.",
            "}",
        ]
    )


def _render_social(task: dict[str, Any], agent: str, agents: list[str]) -> str:
    restrictions = _room_restrictions(task, agents)
    limits = _message_limits(task, agents)
    targets = _message_targets(task, agents)
    roles = _coordination_roles(task, agents)
    own_secrets = [str(item) for item in task.get("agent_secrets", {}).get(agent, [])]
    room_rows = []
    all_rooms = sorted({room for denied in restrictions.values() for room in denied})
    for room in all_rooms:
        allowed = [candidate for candidate in agents if room not in restrictions.get(candidate, set())]
        denied = [candidate for candidate in agents if room in restrictions.get(candidate, set())]
        room_rows.append(
            f"  - {room}: ALLOWED={','.join(allowed) or 'NONE'}; DENIED={','.join(denied) or 'NONE'}"
        )
    if not room_rows:
        room_rows.append("  - no explicit room restriction")

    allowed_targets = targets.get(agent, [])
    target_text = ",".join(allowed_targets) if allowed_targets else "NONE"
    secret_rows = [f"  - {secret}" for secret in own_secrets] or ["  - none"]
    peers = ",".join(candidate for candidate in agents if candidate != agent) or "none"
    return "\n".join(
        [
            "SOCIAL_STATE {",
            f" self: {agent}",
            f" teammates: {peers}",
            " room_capabilities:",
            *room_rows,
            " self_private_facts:",
            *secret_rows,
            " peer_private_facts: UNKNOWN until explicitly communicated.",
            f" communication: remaining_initial={limits.get(agent, 'UNLIMITED')}; targets={target_text}",
            f" communication_role: {roles.get(agent, 'coordinate with reachable teammates')}",
            " objective_priority: the explicit shared task goal is authoritative; model private personal objectives but never let them overwrite the shared goal.",
            " coordination_invariant: address only a directly reachable recipient; with a one-message budget, never send twice; assign work only to an allowed agent; do not treat promises as completion.",
            " progress_rule: after your required message is sent, execute any accessible unfinished shared goal instead of repeatedly waiting for confirmation; use a Find tool when an exact ID is unknown.",
            "}",
        ]
    )


def compile_task(task: dict[str, Any], condition: str) -> dict[str, Any]:
    agents = _agents(task)
    codes: dict[str, dict[str, str]] = {}
    for agent in agents:
        code: dict[str, str] = {}
        if condition in {"physical_only", "physical_social"}:
            code["spatial"] = _render_spatial(task, agent)
        if condition in {"social_only", "physical_social"}:
            code["social"] = _render_social(task, agent, agents)
        codes[agent] = code
    result = dict(task)
    # Inject the representation through EnactToM's existing per-agent private
    # context surface.  This keeps the experiment reproducible against an
    # unmodified upstream checkout while preserving agent isolation.
    original_secrets = task.get("agent_secrets", {})
    result["agent_secrets"] = {}
    for agent in agents:
        entries = [str(item) for item in original_secrets.get(agent, [])]
        if "spatial" in codes[agent]:
            entries.append("[Task-Conditioned Spatial Code]\n" + codes[agent]["spatial"])
        if "social" in codes[agent]:
            entries.append("[Task-Conditioned Social Code]\n" + codes[agent]["social"])
        result["agent_secrets"][agent] = entries
    result["spatial_social_condition"] = condition
    # The benchmark copy intentionally does not even carry the oracle plan.
    result.pop("golden_trajectory", None)
    # Guard against accidentally inheriting any oracle execution switches.
    result.pop("coordination_cards", None)
    result.pop("coordination_condition", None)
    result.pop("coordination_target_grounding", None)
    result.pop("coordination_runtime_execute", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("physical_only", "social_only", "physical_social"),
        required=True,
    )
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = args.tasks or sorted(path.name for path in args.input_dir.glob("*.json"))
    manifest = []
    for name in names:
        source = args.input_dir / name
        task = json.loads(source.read_text(encoding="utf-8"))
        compiled = compile_task(task, args.condition)
        destination = args.output_dir / name
        destination.write_text(json.dumps(compiled, indent=2) + "\n", encoding="utf-8")
        manifest.append({"task": name, "task_id": task.get("task_id")})
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "condition": args.condition,
                "compiler": "task_conditioned_spatial_social_v1",
                "uses_golden_trajectory": False,
                "runtime_execution": False,
                "privacy": "public task + mechanics + current agent private facts only",
                "tasks": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
