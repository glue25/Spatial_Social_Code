"""Opt-in feedback memory and a fact-matched prose control.

No action is filtered, substituted, or sent to another agent.  Only actual
private tool feedback is used; an inaccessible target is not a world location.
The prose control uses the exact same state reducer as the structured condition.
"""
from __future__ import annotations

import hashlib
import json
import re

from .state_store import Fact
from .updaters import SocialUpdater


VARIANTS = {"v2", "feedback_v3", "nl_feedback_v3"}
DENIED_NAVIGATION = re.compile(
    r"^You cannot enter ([a-z][a-z0-9_]*_\d+) to reach "
    r"([a-z][a-z0-9_]*_\d+)\. The area is off-limits to you\."
)


class FeedbackSocialUpdater(SocialUpdater):
    def __init__(self):
        self.failure_counts = {}

    def reduce_action(self, agent_id, turn, action, outcome):
        facts = super().reduce_action(agent_id, turn, action, outcome)
        # Empty/ambiguous feedback is not evidence of a new inability.
        if not action.valid or outcome.status not in {"failed", "succeeded"}:
            return facts
        identity = json.dumps([action.name, action.arguments], ensure_ascii=False)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        key = (agent_id, identity)
        if outcome.success:
            self.failure_counts[key] = 0
        else:
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        common = dict(source_type="action_feedback", source_agent=agent_id,
                      evidence=outcome.feedback, turn=turn)
        # Put the actual reason in the rendered value.  v2 rendered only the
        # status and metadata, dropping Fact.evidence from the state prompt.
        facts.append(Fact(f"feedback.action_{digest}", {
            "action": action.serialized,
            "last_outcome": outcome.status,
            "failures_since_last_success": self.failure_counts[key],
            "feedback": outcome.feedback,
        }, **common))
        denied = DENIED_NAVIGATION.match(outcome.feedback.strip())
        if (not outcome.success and action.name == "Navigate" and denied
                and action.arguments == [denied.group(2)]):
            facts.append(Fact(f"access.target.{action.arguments[0]}", {
                "currently_blocked_for_self": True,
                "reported_denied_room": denied.group(1),
                "basis": "explicit private navigation denial; not a location observation",
            }, **common))
        elif outcome.success and action.name == "Navigate":
            facts.append(Fact(f"access.target.{action.arguments[0]}", {
                "currently_blocked_for_self": False,
                "basis": "latest own navigation succeeded; does not grant other agents access",
            }, **common))
        return facts


def render_prose(store, agent_id):
    """Lossless sentence rendering of the fields exposed by StateStore.render.

The common marker is transport metadata needed to replace stale prompt blocks;
it is not a learned natural-language baseline or an independent compiler.
"""
    state = store.snapshot(agent_id)
    lines = [f"SSC_RUNTIME_STATE agent={agent_id} version={state['version']}"]
    for layer in ("spatial", "social"):
        if not state[layer]:
            lines.append(f"There are no supported {layer} records; their values are unknown.")
        for key, fact in sorted(state[layer].items()):
            value = json.dumps(fact['value'], ensure_ascii=False, sort_keys=True)
            lines.append(
                f"The {layer} record {key} has value {value}. "
                f"Its status is {fact['status']}; its source is {fact['source_type']} "
                f"from {fact['source_agent']} at turn {fact['turn']}, "
                f"with confidence {fact['confidence']:.2f}."
            )
    lines.append("END_SSC_RUNTIME_STATE")
    return "\n".join(lines)
