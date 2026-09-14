"""Per-agent, provenance-preserving state store for online SSC."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


UNKNOWN = "unknown"


@dataclass
class Fact:
    key: str
    value: Any
    source_type: str
    source_agent: str
    evidence: str
    turn: int
    confidence: float = 1.0
    status: str = "observed"
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    agent_id: str
    version: int = 0
    spatial: Dict[str, Fact] = field(default_factory=dict)
    social: Dict[str, Fact] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "spatial": {key: fact.to_dict() for key, fact in sorted(self.spatial.items())},
            "social": {key: fact.to_dict() for key, fact in sorted(self.social.items())},
        }


class StateStore:
    """Owns disjoint states; cross-agent reads are intentionally unsupported."""

    def __init__(self, agent_ids: Iterable[str]):
        self._states = {agent_id: AgentState(agent_id=agent_id) for agent_id in agent_ids}

    @property
    def agent_ids(self) -> List[str]:
        return sorted(self._states)

    def snapshot(self, agent_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._require(agent_id).to_dict())

    def version(self, agent_id: str) -> int:
        return self._require(agent_id).version

    def update(self, agent_id: str, layer: str, facts: Iterable[Fact]) -> Dict[str, Any]:
        state = self._require(agent_id)
        if layer not in {"spatial", "social"}:
            raise ValueError(f"invalid state layer: {layer}")
        target: Dict[str, Fact] = getattr(state, layer)
        changes = []
        for fact in facts:
            if fact.source_agent != agent_id and fact.source_type not in {"delivered_message", "task_public"}:
                raise ValueError("cross-agent fact lacks delivered-message provenance")
            previous = target.get(fact.key)
            if previous is not None:
                fact.supersedes = f"v{state.version}:{layer}:{fact.key}"
            target[fact.key] = copy.deepcopy(fact)
            changes.append({
                "layer": layer,
                "key": fact.key,
                "before": previous.to_dict() if previous else None,
                "after": fact.to_dict(),
            })
        # Empty updates are versioned too so the trace can prove they occurred.
        state.version += 1
        return {"agent_id": agent_id, "version": state.version, "changes": changes}

    def render(self, agent_id: str, layers: Iterable[str] = ("spatial", "social")) -> str:
        state = self._require(agent_id)
        selected = set(layers)
        lines = [f"SSC_RUNTIME_STATE agent={agent_id} version={state.version}"]
        for layer in ("spatial", "social"):
            if layer not in selected:
                continue
            lines.append(f"{layer.upper()}:")
            facts: Dict[str, Fact] = getattr(state, layer)
            if not facts:
                lines.append("- unknown (no supported records yet)")
            for key, fact in sorted(facts.items()):
                value = json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
                lines.append(
                    f"- {key}={value}; status={fact.status}; source={fact.source_type}; "
                    f"source_agent={fact.source_agent}; turn={fact.turn}; confidence={fact.confidence:.2f}"
                )
        lines.append("END_SSC_RUNTIME_STATE")
        return "\n".join(lines)

    def _require(self, agent_id: str) -> AgentState:
        if agent_id not in self._states:
            raise KeyError(f"unknown agent: {agent_id}")
        return self._states[agent_id]
