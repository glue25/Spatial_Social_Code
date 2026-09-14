"""Online SSC controller used by the EnactToM benchmark runner hook."""

from __future__ import annotations

import os
import platform
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from .observation_adapter import ObservationAdapter
from .state_store import Fact, StateStore
from .trace_recorder import TraceRecorder
from .updaters import SocialUpdater, SpatialUpdater


CONDITIONS = {"direct", "static_ssc", "dynamic_ssc"}
COMMON_GUIDANCE = """[Common execution guidance]
Use only the task information, your own observations, successful tool feedback, and messages actually delivered to you. Do not guess entity IDs. Navigate before manipulation when needed. Do not repeat an unchanged failed action. A promise or attempted action is not task completion."""

PRIVATE_RELATION_RE = re.compile(
    r"\b([a-z][a-z0-9_]*_\d+)\s+(?:starts\s+|is\s+)?(on|inside|in)\s+"
    r"(?:the\s+)?([a-z][a-z0-9_]*_\d+)\b",
    re.IGNORECASE,
)


class ClosedLoopRuntime:
    def __init__(self, condition: str, agent_ids: Iterable[str], output_dir: str, task: Any):
        condition = str(condition).lower()
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported closed-loop condition: {condition}")
        self.condition = condition
        self.store = StateStore(agent_ids)
        self.trace = TraceRecorder(output_dir)
        self.spatial = SpatialUpdater()
        self.social = SocialUpdater()
        self.observations = ObservationAdapter()
        self._last_injected: Dict[str, int] = {}
        self._initialize(task)
        self.trace.write_json("manifest.json", {
            "schema_version": "habitat_closed_loop_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "condition": condition,
            "task_id": getattr(task, "task_id", "unknown"),
            "scene_id": getattr(task, "scene_id", "unknown"),
            "episode_id": getattr(task, "episode_id", "unknown"),
            "agent_ids": self.store.agent_ids,
            "common_guidance_sha256": self.trace.digest(COMMON_GUIDANCE),
            "model": os.environ.get("SSC_MODEL", "unspecified"),
            "model_checkpoint": os.environ.get("SSC_MODEL_CHECKPOINT", "unspecified"),
            "seed": os.environ.get("SSC_SEED", "unspecified"),
            "config_sha256": os.environ.get("SSC_CONFIG_SHA256", "unspecified"),
            "budgets": {
                "environment_turns": int(os.environ.get("SSC_MAX_TURNS", "20")),
                "communication": "task-defined",
                "aggregate_api_calls": "measured_not_assumed",
                "tokens": "provider_usage_log",
            },
            "python": platform.python_version(),
            "status": "initialized_not_yet_complete",
        })

    @property
    def dynamic(self) -> bool:
        return self.condition == "dynamic_ssc"

    def _initialize(self, task: Any) -> None:
        public_task = str(getattr(task, "task", "") or "")
        restrictions: Dict[str, list] = {agent: [] for agent in self.store.agent_ids}
        for binding in getattr(task, "mechanic_bindings", []) or []:
            if getattr(binding, "mechanic_type", None) == "room_restriction":
                for agent in getattr(binding, "for_agents", None) or []:
                    restrictions.setdefault(str(agent), []).extend(getattr(binding, "restricted_rooms", None) or [])
        for agent_id in self.store.agent_ids:
            own_secrets = list((getattr(task, "agent_secrets", {}) or {}).get(agent_id, []))
            spatial = [Fact("task.shared_goal", public_task, "task_public", agent_id, public_task, 0, status="task_specified")]
            for secret in own_secrets:
                for entity, relation, target in PRIVATE_RELATION_RE.findall(str(secret)):
                    normalized_relation = {"on": "on", "inside": "inside", "in": "in"}[relation.lower()]
                    spatial.append(Fact(
                        f"object.{entity}.{normalized_relation}", target,
                        "task_private", agent_id, str(secret), 0,
                        status="task_specified",
                    ))
            social = [
                Fact("task.self", agent_id, "task_public", agent_id, public_task, 0, status="task_specified"),
                Fact("access.denied_rooms", sorted(set(restrictions.get(agent_id, []))), "task_private", agent_id, "task mechanic visible to self", 0, status="task_specified"),
                Fact("task.private_facts", own_secrets, "task_private", agent_id, "own private task input", 0, status="task_specified"),
            ]
            before = self.store.snapshot(agent_id)
            delta_s = self.store.update(agent_id, "spatial", spatial)
            delta_t = self.store.update(agent_id, "social", social)
            self.trace.append("state_before_after.jsonl", {"turn": 0, "agent_id": agent_id, "before": before, "updates": [delta_s, delta_t], "after": self.store.snapshot(agent_id), "reason": "initial_private_compile"})

    def planner_context(self, agent_id: str, turn: int) -> tuple[str, int]:
        version = self.store.version(agent_id)
        if self.condition == "direct":
            block = COMMON_GUIDANCE
        else:
            block = COMMON_GUIDANCE + "\n\n" + self.store.render(agent_id)
        self._last_injected[agent_id] = version
        self.trace.append("plans.jsonl", {
            "turn": turn, "agent_id": agent_id, "phase": "planner_input",
            "condition": self.condition, "state_version_read": version,
            "state_included": self.condition != "direct",
            "context_sha256": self.trace.digest(block), "context": block,
        })
        return block, version

    def private_observation(self, agent_id: str, turn: int, observation_text: str) -> None:
        """Persist only the acting agent's own partial world-graph rendering."""
        self.trace.append("events.jsonl", {
            "turn": turn, "agent_id": agent_id, "event_type": "private_world_observation",
            "observation": observation_text,
        })
        if not self.dynamic:
            return
        before = self.store.snapshot(agent_id)
        fact = Fact(
            "observation.latest_private_world", observation_text,
            "observation", agent_id, observation_text, turn,
            status="observed",
        )
        delta = self.store.update(agent_id, "spatial", [fact])
        self.trace.append("state_before_after.jsonl", {
            "turn": turn, "agent_id": agent_id, "before": before,
            "updates": [delta], "after": self.store.snapshot(agent_id),
            "reason": "private_world_observation",
        })

    def record_plan(self, agent_id: str, turn: int, action: str, thought: Optional[str], prompt: str) -> None:
        marker = f"agent={agent_id} version={self._last_injected.get(agent_id, -1)}"
        self.trace.append("plans.jsonl", {
            "turn": turn, "agent_id": agent_id, "phase": "planner_output",
            "state_version_read": self._last_injected.get(agent_id),
            "state_marker_found_in_actual_prompt": self.condition == "direct" or marker in str(prompt),
            "runtime_state_block_count": str(prompt).count("SSC_RUNTIME_STATE agent="),
            "actual_prompt_sha256": self.trace.digest(str(prompt)),
            "action": action, "thought": thought,
        })
        self.trace.append("predictions.jsonl", {"turn": turn, "agent_id": agent_id, "action": action, "thought": thought})
        self.trace.append("usage.jsonl", {
            "turn": turn, "agent_id": agent_id, "component": "planner",
            "calls": 1, "prompt_characters": len(str(prompt)),
            "input_tokens": None, "output_tokens": None,
            "token_status": "read from provider api_usage log during aggregation",
        })

    def action_feedback(self, agent_id: str, turn: int, action_text: str, feedback: str) -> None:
        event = self.observations.action_feedback(agent_id, turn, action_text, feedback)
        action = event.payload["action"]
        outcome = event.payload["outcome"]
        self.trace.append("events.jsonl", {"turn": turn, "agent_id": agent_id, "event_type": "action_feedback", "action": action.to_dict(), "outcome": outcome.to_dict()})
        if not self.dynamic:
            return
        before = self.store.snapshot(agent_id)
        updates = [
            self.store.update(agent_id, "spatial", self.spatial.reduce_action(agent_id, turn, action, outcome)),
            self.store.update(agent_id, "social", self.social.reduce_action(agent_id, turn, action, outcome)),
        ]
        self.trace.append("state_before_after.jsonl", {"turn": turn, "agent_id": agent_id, "before": before, "updates": updates, "after": self.store.snapshot(agent_id), "reason": "action_feedback"})

    def message_delivery(self, sender: str, receivers: Iterable[str], turn: int, message: str) -> None:
        receivers = list(receivers)
        self.trace.append("events.jsonl", {"turn": turn, "agent_id": sender, "event_type": "message_delivered", "sender": sender, "receivers": receivers, "message": message})
        if not self.dynamic:
            return
        sender_before = self.store.snapshot(sender)
        sender_delta = self.store.update(
            sender, "social",
            self.social.reduce_delivery_confirmation(sender, receivers, turn, message),
        )
        self.trace.append("state_before_after.jsonl", {
            "turn": turn, "agent_id": sender, "before": sender_before,
            "updates": [sender_delta], "after": self.store.snapshot(sender),
            "reason": "message_delivery_confirmation",
        })
        for event in self.observations.delivered_messages(sender, receivers, turn, message):
            receiver = event.agent_id
            before = self.store.snapshot(receiver)
            delta = self.store.update(receiver, "social", self.social.reduce_delivered_message(receiver, sender, turn, message))
            self.trace.append("state_before_after.jsonl", {"turn": turn, "agent_id": receiver, "before": before, "updates": [delta], "after": self.store.snapshot(receiver), "reason": "delivered_message"})

    def finalize(self, summary: Dict[str, Any]) -> None:
        payload = dict(summary)
        payload.update({"condition": self.condition, "final_states": {agent: self.store.snapshot(agent) for agent in self.store.agent_ids}})
        self.trace.write_json("summary.json", payload)
        manifest_path = self.trace.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        self.trace.write_json("manifest.json", manifest)
