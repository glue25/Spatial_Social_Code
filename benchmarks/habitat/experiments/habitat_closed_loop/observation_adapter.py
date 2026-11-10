"""Access-control boundary for online state update events."""

from __future__ import annotations

from typing import Iterable, List

from .action_interface import classify_outcome, parse_action
from .updaters import RuntimeEvent


class ObservationAdapter:
    """Build updater events only from data available to the named agent."""

    def action_feedback(self, agent_id: str, turn: int, action_text: str, feedback: str) -> RuntimeEvent:
        action = parse_action(action_text)
        outcome = classify_outcome(feedback, action=action)
        return RuntimeEvent(
            event_type="action_feedback",
            agent_id=agent_id,
            turn=turn,
            payload={"action": action, "outcome": outcome},
        )

    def delivered_messages(self, sender: str, receivers: Iterable[str], turn: int, message: str) -> List[RuntimeEvent]:
        return [
            RuntimeEvent(
                event_type="delivered_message",
                agent_id=receiver,
                turn=turn,
                payload={"sender": sender, "message": message},
            )
            for receiver in receivers
        ]
