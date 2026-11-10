"""Versioned strict physical scorer with stable-window semantics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


SCORER_VERSION = "strict_physical_v2"
SUPPORTED = {"is_on_top", "is_inside", "is_open", "is_closed", "is_in_room", "is_on_floor", "is_next_to"}
ATOM_RE = re.compile(r"\((is_[a-z_]+)\s+([^()]+?)\)")


@dataclass(frozen=True)
class PhysicalGoal:
    predicate: str
    entity: str
    target: Optional[str] = None

    @property
    def goal_id(self) -> str:
        return ":".join(item for item in (self.predicate, self.entity, self.target) if item)


@dataclass(frozen=True)
class PhysicalSample:
    relation_ok: bool
    held_by: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None


def parse_functional_goals(pddl: str) -> List[PhysicalGoal]:
    goals: List[PhysicalGoal] = []
    for predicate, raw_args in ATOM_RE.findall(str(pddl or "")):
        if predicate not in SUPPORTED:
            continue
        args = raw_args.split()
        if not args:
            continue
        goals.append(PhysicalGoal(predicate, args[0], args[1] if len(args) > 1 else None))
    return goals


class StrictPhysicalScorer:
    """Scores snapshots; placement requires relation, release, and stability."""

    def __init__(self, stable_samples: int = 3, open_threshold: float = 0.10):
        if stable_samples < 2:
            raise ValueError("stable_samples must be at least 2")
        self.stable_samples = stable_samples
        self.open_threshold = float(open_threshold)

    def score(self, goals: Iterable[PhysicalGoal], samples: Dict[str, List[PhysicalSample]]) -> Dict[str, Any]:
        checks = []
        for goal in goals:
            observed = list(samples.get(goal.goal_id, []))
            stable_window_measured = len(observed) >= self.stable_samples
            window = observed[-self.stable_samples:]
            relation_stable = stable_window_measured and all(item.relation_ok for item in window)
            release_required = goal.predicate in {"is_on_top", "is_inside"}
            released_stable = (
                stable_window_measured
                and all(item.held_by is None for item in window)
            ) if release_required else True
            satisfied = relation_stable and released_stable
            checks.append({
                "goal_id": goal.goal_id,
                "predicate": goal.predicate,
                "entity": goal.entity,
                "target": goal.target,
                "satisfied": satisfied,
                "relation_stable": relation_stable,
                "released_stable": released_stable,
                "stable_window_measured": stable_window_measured,
                "samples_required": self.stable_samples,
                "samples_observed": len(observed),
                "samples": [asdict(item) for item in observed],
            })
        completed = sum(bool(check["satisfied"]) for check in checks)
        return {
            "scorer_version": SCORER_VERSION,
            "open_threshold": self.open_threshold,
            "stable_samples": self.stable_samples,
            "strict_success": bool(checks) and completed == len(checks),
            "completed_goals": completed,
            "total_goals": len(checks),
            "strict_percent_complete": completed / len(checks) if checks else 0.0,
            "checks": checks,
        }

    def measure(
        self,
        goals: Iterable[PhysicalGoal],
        probe: Callable[[PhysicalGoal], PhysicalSample],
        advance: Callable[[], bool],
    ) -> Dict[str, Any]:
        goals = list(goals)
        samples = {goal.goal_id: [] for goal in goals}
        advance_results: List[bool] = []
        for index in range(self.stable_samples):
            if index:
                advance_results.append(bool(advance()))
            for goal in goals:
                samples[goal.goal_id].append(probe(goal))
        result = self.score(goals, samples)
        result["stability_advance_ok"] = all(advance_results) if advance_results else False
        result["measurement_valid"] = result["stability_advance_ok"]
        result["measurement_status"] = "measured" if result["measurement_valid"] else "invalid"
        if not result["stability_advance_ok"]:
            # Preserve the instantaneous calculation only as diagnostic data.
            # It must not enter aggregates as a valid strict measurement.
            result["unvalidated_strict_success"] = result["strict_success"]
            result["unvalidated_completed_goals"] = result["completed_goals"]
            result["unvalidated_strict_percent_complete"] = result["strict_percent_complete"]
            result["strict_success"] = None
            result["completed_goals"] = None
            result["strict_percent_complete"] = None
            result["stability_error"] = "fixed physics stability window could not be advanced"
        return result


def score_exported_audit(goals: Iterable[PhysicalGoal], snapshots: Dict[str, Dict[str, Any]], stable_samples: int = 3) -> Dict[str, Any]:
    """Score historical audit exports without inventing missing stable samples."""
    scorer = StrictPhysicalScorer(stable_samples=stable_samples)
    by_goal: Dict[str, List[PhysicalSample]] = {}
    for goal in goals:
        row = snapshots.get(goal.entity, {})
        if goal.predicate in {"is_on_top", "is_inside"}:
            relation_ok = bool(row.get("expected_support_matched", False))
            held = "an_agent" if row.get("target_is_grasped") or row.get("grasped_objects") and any(
                item.get("object_handle") == row.get("handles", {}).get(goal.entity)
                for item in row.get("grasped_objects", [])
            ) else None
        elif goal.predicate == "is_open":
            joints = row.get("joint_positions") or []
            relation_ok = bool(joints and max(abs(float(x)) for x in joints) >= scorer.open_threshold)
            held = None
        elif goal.predicate == "is_closed":
            joints = row.get("joint_positions") or []
            relation_ok = bool(joints and max(abs(float(x)) for x in joints) < scorer.open_threshold)
            held = None
        else:
            relation_ok, held = False, None
        # Historical files contain only one snapshot. This intentionally leaves
        # stable_window_measured=false and never upgrades them to strict success.
        by_goal[goal.goal_id] = [PhysicalSample(relation_ok, held, {"historical_export": True})]
    result = scorer.score(goals, by_goal)
    result["instantaneous_completed_goals"] = sum(
        bool(rows and rows[-1].relation_ok and rows[-1].held_by is None)
        for rows in by_goal.values()
    )
    result["historical_stability_status"] = "not_measured"
    return result


def evaluate_habitat_runner(
    runner: Any,
    functional_goal_pddl: str,
    *,
    stable_samples: int = 3,
    physics_steps_per_sample: int = 10,
    physics_dt: float = 1.0 / 60.0,
    open_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Evaluate only through the scorer-side simulator handle.

    This function must never be called by a planner or state updater.  It takes
    multiple physical samples and rejects a placement while any agent grasps
    the object, even if Habitat's ``above`` predicate is momentarily true.
    """
    from enacttom.evaluation import _resolve_handle

    goals = parse_functional_goals(functional_goal_pddl)
    scorer = StrictPhysicalScorer(stable_samples=stable_samples, open_threshold=open_threshold)
    sim = runner.env_interface.sim
    world_graph = getattr(runner.env_interface, "full_world_graph", None)

    def held_by(entity: str) -> Optional[str]:
        wanted = _resolve_handle(entity, sim, world_graph)
        wanted_base = str(wanted).split(":")[0].strip().rstrip("_")
        rom = sim.get_rigid_object_manager()
        for uid in sorted(runner.agents):
            grasp = sim.agents_mgr[uid].grasp_mgr
            if not grasp.is_grasped:
                continue
            try:
                handle = rom.get_object_handle_by_id(grasp.snap_idx)
            except Exception:
                continue
            base = str(handle).split(":")[0].strip().rstrip("_")
            if handle == wanted or base == wanted_base:
                return f"agent_{uid}"
        return None

    def probe(goal: PhysicalGoal) -> PhysicalSample:
        if goal.predicate in {"is_open", "is_closed"}:
            handle = _resolve_handle(goal.entity, sim, world_graph)
            from habitat.sims.habitat_simulator.sim_utilities import get_ao_default_link, link_is_open

            ao = sim.get_articulated_object_manager().get_object_by_handle(handle)
            default_link = get_ao_default_link(ao, compute_if_not_found=True) if ao is not None else None
            relation_ok = bool(
                ao is not None
                and default_link is not None
                and link_is_open(ao, default_link, threshold=open_threshold)
            )
            if goal.predicate == "is_closed":
                relation_ok = bool(ao is not None and default_link is not None and not relation_ok)
            joint_positions = getattr(ao, "joint_positions", None) if ao is not None else None
            detail = {
                "resolved_handle": handle,
                "articulated_object_found": ao is not None,
                "default_link_found": default_link is not None,
                "joint_positions": list(joint_positions) if joint_positions is not None else [],
                "joint_threshold": open_threshold,
            }
        else:
            prop = {"property": goal.predicate, "entity": goal.entity}
            if goal.target:
                prop["target"] = goal.target
            raw = runner.evaluate_task({"required_states": [prop]})
            relation_ok = bool(raw and raw.get("success", False))
            detail = {"benchmark_predicate_result": raw}
        holder = held_by(goal.entity) if goal.predicate in {"is_on_top", "is_inside"} else None
        return PhysicalSample(relation_ok=relation_ok, held_by=holder, detail=detail)

    def advance() -> bool:
        if not hasattr(sim, "step_physics"):
            return False
        try:
            for _ in range(max(1, int(physics_steps_per_sample))):
                sim.step_physics(float(physics_dt))
            return True
        except Exception:
            return False

    result = scorer.measure(goals, probe, advance)
    result.update({
        "physics_steps_per_sample": int(physics_steps_per_sample),
        "physics_dt": float(physics_dt),
    })
    return result
