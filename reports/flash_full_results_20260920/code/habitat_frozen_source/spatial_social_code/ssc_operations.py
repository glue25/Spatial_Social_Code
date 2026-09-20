"""Executable bridge operations for Spatial--Social Code.

This module is deliberately small and deterministic.  It formalizes the three
operations that distinguish a joint state from two adjacent descriptions:

1. project a physical event into agent-indexed evidence;
2. update an agent's information state from observation or communication;
3. compose current location, information, access, goal, and role constraints
   when deciding whether an action is executable.

The implementation is useful for controlled mechanism tests.  It is not a
claim that visibility, attention, or human belief can be recovered perfectly
from real video.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable


UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AgentGeometry:
    room: str
    facing: str
    occluded_targets: frozenset[str] = frozenset()
    accessible_rooms: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MoveEvent:
    event_id: str
    actor: str
    object_name: str
    source_location: str
    destination: str
    event_room: str


@dataclass(frozen=True)
class InformationRecord:
    value: str
    support: str
    evidence_ref: str
    source_agent: str | None = None


@dataclass
class SpatialSocialState:
    """Minimal executable state used by the controlled mechanism study."""

    object_locations: dict[str, str]
    agents: dict[str, AgentGeometry]
    information: dict[str, dict[str, InformationRecord]]
    goals: dict[str, str]
    roles: dict[str, frozenset[str]]
    events: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "SpatialSocialState":
        return copy.deepcopy(self)

    def spatial_code(self) -> dict[str, Any]:
        return {
            "object_locations": dict(sorted(self.object_locations.items())),
            "agents": {
                name: {
                    "room": geometry.room,
                    "facing": geometry.facing,
                    "occluded_targets": sorted(geometry.occluded_targets),
                    "accessible_rooms": sorted(geometry.accessible_rooms),
                }
                for name, geometry in sorted(self.agents.items())
            },
            "events": copy.deepcopy(self.events),
        }

    def social_code(self) -> dict[str, Any]:
        return {
            "information": {
                agent: {
                    key: {
                        "value": record.value,
                        "support": record.support,
                        "evidence_ref": record.evidence_ref,
                        "source_agent": record.source_agent,
                    }
                    for key, record in sorted(records.items())
                }
                for agent, records in sorted(self.information.items())
            },
            "goals": dict(sorted(self.goals.items())),
            "roles": {
                agent: sorted(roles) for agent, roles in sorted(self.roles.items())
            },
        }


def can_observe(state: SpatialSocialState, agent: str, event: MoveEvent) -> bool:
    """Project room, orientation, and occlusion into visual event access."""

    geometry = state.agents[agent]
    return (
        geometry.room == event.event_room
        and geometry.facing == event.object_name
        and event.object_name not in geometry.occluded_targets
    )


def project_observation(
    state: SpatialSocialState, event: MoveEvent
) -> dict[str, InformationRecord]:
    """Return only the agents for whom the move supplies observable evidence."""

    result: dict[str, InformationRecord] = {}
    for agent in state.agents:
        if agent == event.actor or can_observe(state, agent, event):
            result[agent] = InformationRecord(
                value=event.destination,
                support="observed",
                evidence_ref=event.event_id,
                source_agent=event.actor,
            )
    return result


def apply_move(state: SpatialSocialState, event: MoveEvent) -> None:
    """Update the shared world, then update only agents with projected access."""

    current = state.object_locations.get(event.object_name, UNKNOWN)
    if current != event.source_location:
        raise ValueError(
            f"move source mismatch for {event.object_name}: "
            f"state={current!r}, event={event.source_location!r}"
        )
    state.object_locations[event.object_name] = event.destination
    observations = project_observation(state, event)
    for agent, record in observations.items():
        state.information.setdefault(agent, {})[event.object_name] = record
    state.events.append(
        {
            "event_id": event.event_id,
            "type": "move",
            "actor": event.actor,
            "object": event.object_name,
            "source": event.source_location,
            "destination": event.destination,
            "observed_by": sorted(observations),
        }
    )


def inform(
    state: SpatialSocialState,
    *,
    sender: str,
    recipient: str,
    object_name: str,
    evidence_ref: str,
) -> None:
    """Update a recipient only when the sender has supported current evidence."""

    sender_record = state.information.get(sender, {}).get(object_name)
    current = state.object_locations.get(object_name)
    if sender_record is None or sender_record.value != current:
        raise ValueError(f"{sender} lacks supported current information for {object_name}")
    state.information.setdefault(recipient, {})[object_name] = InformationRecord(
        value=sender_record.value,
        support="informed",
        evidence_ref=evidence_ref,
        source_agent=sender,
    )
    state.events.append(
        {
            "event_id": evidence_ref,
            "type": "inform",
            "sender": sender,
            "recipient": recipient,
            "object": object_name,
            "value": sender_record.value,
        }
    )


def last_known_location(
    state: SpatialSocialState, agent: str, object_name: str
) -> str:
    record = state.information.get(agent, {}).get(object_name)
    return record.value if record else UNKNOWN


def can_retrieve(state: SpatialSocialState, agent: str, object_name: str) -> bool:
    """Compose information, current world, access, goal, and role constraints."""

    current = state.object_locations.get(object_name)
    known = last_known_location(state, agent, object_name)
    geometry = state.agents[agent]
    has_goal = state.goals.get(agent) == f"retrieve:{object_name}"
    has_role = "retriever" in state.roles.get(agent, frozenset())
    return bool(
        current
        and known == current
        and current in geometry.accessible_rooms
        and has_goal
        and has_role
    )


def eligible_retrievers(
    state: SpatialSocialState, object_name: str
) -> tuple[str, ...]:
    return tuple(
        agent for agent in sorted(state.agents) if can_retrieve(state, agent, object_name)
    )


def answer_queries(
    state: SpatialSocialState,
    *,
    agent: str,
    object_name: str,
) -> dict[str, Any]:
    """Answer several question families without recompiling or mutating state."""

    known = last_known_location(state, agent, object_name)
    return {
        "world_location": state.object_locations.get(object_name, UNKNOWN),
        "agent_last_known_location": known,
        "agent_next_search_location": known,
        "agent_can_retrieve": can_retrieve(state, agent, object_name),
        "eligible_retrievers": list(eligible_retrievers(state, object_name)),
    }


def changed_paths(left: Any, right: Any, prefix: str = "") -> tuple[str, ...]:
    """Return leaf paths changed by a counterfactual replay."""

    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(changed_paths(left[key], right[key], path))
        return tuple(paths)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if list(left) == list(right):
            return ()
        return (prefix,)
    return () if left == right else (prefix,)


def seed_information(
    agents: Iterable[str], object_name: str, location: str
) -> dict[str, dict[str, InformationRecord]]:
    """Create a common, explicitly observed initial location record."""

    return {
        agent: {
            object_name: InformationRecord(
                value=location,
                support="observed",
                evidence_ref="initial_observation",
            )
        }
        for agent in agents
    }
