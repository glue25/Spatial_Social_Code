"""Deterministic online reducers for action feedback and delivered messages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .action_interface import CanonicalAction, ActionOutcome
from .state_store import Fact


ENTITY_RE = re.compile(r"\b[a-z][a-z0-9_]*_\d+\b")


@dataclass(frozen=True)
class RuntimeEvent:
    event_type: str
    agent_id: str
    turn: int
    payload: Dict[str, Any]


class SpatialUpdater:
    def reduce_action(self, agent_id: str, turn: int, action: CanonicalAction, outcome: ActionOutcome) -> List[Fact]:
        if not action.valid or not outcome.success:
            return []
        args = action.arguments
        evidence = outcome.feedback
        common = dict(source_type="action_feedback", source_agent=agent_id, evidence=evidence, turn=turn)
        if action.name == "Navigate" and args:
            return [Fact("self.near", args[0], **common)]
        if action.name == "Pick" and args:
            return [Fact("self.holding", args[0], **common), Fact(f"object.{args[0]}.held", True, **common)]
        if action.name == "Place" and len(args) >= 3:
            obj, relation, receptacle = args[:3]
            return [
                Fact("self.holding", None, **common),
                Fact(f"object.{obj}.held", False, **common),
                Fact(f"object.{obj}.{relation}", receptacle, **common),
            ]
        if action.name in {"Open", "Close"} and args:
            return [Fact(f"joint.{args[0]}.state", action.name.lower(), **common)]
        return []


class SocialUpdater:
    def reduce_action(self, agent_id: str, turn: int, action: CanonicalAction, outcome: ActionOutcome) -> List[Fact]:
        evidence = outcome.feedback
        common = dict(source_type="action_feedback", source_agent=agent_id, evidence=evidence, turn=turn)
        facts = [Fact(f"action.turn_{turn}.status", outcome.status, **common)]
        if action.name == "Communicate":
            facts.append(Fact(f"message.turn_{turn}.attempted", True, **common))
            if not outcome.success:
                facts.append(Fact(f"message.turn_{turn}.delivered", False, **common))
        return facts

    def reduce_delivery_confirmation(self, sender: str, receivers: List[str], turn: int, message: str) -> List[Fact]:
        common = dict(
            source_type="action_feedback", source_agent=sender,
            evidence=f"environment delivered message to {','.join(receivers)}", turn=turn,
        )
        return [
            Fact(f"message.turn_{turn}.sent", True, **common),
            Fact(f"message.turn_{turn}.delivered", receivers, **common),
        ]

    def reduce_delivered_message(self, receiver: str, sender: str, turn: int, message: str) -> List[Fact]:
        common = dict(
            source_type="delivered_message", source_agent=sender,
            evidence=message, turn=turn, confidence=1.0,
        )
        facts = [Fact(f"message.turn_{turn}.from.{sender}", message, **common)]
        # Mentioned entities are recorded as message claims, never promoted to observations.
        for entity in sorted(set(ENTITY_RE.findall(message))):
            facts.append(Fact(f"claim.{sender}.{entity}", message, status="reported", **common))
        return facts
