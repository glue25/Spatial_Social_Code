from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from experiments.habitat_closed_loop.action_interface import (
    classify_outcome,
    communication_payload,
    parse_action,
)
from experiments.habitat_closed_loop.runtime import ClosedLoopRuntime
from experiments.habitat_closed_loop.state_store import Fact, StateStore
from experiments.habitat_closed_loop.strict_scorer import (
    PhysicalGoal,
    PhysicalSample,
    StrictPhysicalScorer,
    parse_functional_goals,
    score_exported_audit,
)


@dataclass
class Binding:
    mechanic_type: str
    for_agents: list = field(default_factory=list)
    restricted_rooms: list = field(default_factory=list)


@dataclass
class Task:
    task_id: str = "test"
    scene_id: str = "scene"
    episode_id: str = "episode"
    task: str = "Put cup on table."
    agent_secrets: dict = field(default_factory=lambda: {"agent_0": ["cup_1 starts on shelf_2"], "agent_1": ["private B"]})
    mechanic_bindings: list = field(default_factory=lambda: [Binding("room_restriction", ["agent_0"], ["kitchen_1"])])


def test_public_action_interface_normalizes_place_without_guessing():
    action = parse_action("Place[cup_1, on, table_2]")
    assert action.valid
    assert action.serialized == "Place[cup_1, on, table_2, None, None]"
    assert not parse_action("Place[cup_1, table_2]").valid
    assert not parse_action('Communicate["hello"]') .valid


def test_public_action_interface_quotes_message_and_preserves_explicit_recipient():
    action = parse_action(
        "Communicate[I found cup_1, please move it, agent_1]"
    )
    assert action.valid
    assert action.arguments == ["I found cup_1, please move it", "agent_1"]
    assert action.serialized == (
        'Communicate["I found cup_1, please move it", agent_1]'
    )
    assert communication_payload(action) == (
        "I found cup_1, please move it", ["agent_1"]
    )
    assert not parse_action("Communicate[message without a recipient]").valid


@pytest.mark.parametrize("feedback", [
    "Object is too far", "Object is occluded", "Message blocked: illegal recipient", "Action timed out"
])
def test_failures_have_no_physical_effect(feedback):
    outcome = classify_outcome(feedback, action=parse_action("Pick[cup_1]"))
    assert not outcome.success
    assert outcome.had_physical_effect is False


def test_ambiguous_failure_keeps_physical_effect_unknown():
    outcome = classify_outcome("Unexpected failure!", action=parse_action("Open[cabinet_1]"))
    assert not outcome.success
    assert outcome.had_physical_effect is None


def test_state_is_agent_isolated_and_unknown_is_not_false():
    store = StateStore(["agent_0", "agent_1"])
    store.update("agent_0", "spatial", [Fact("object.cup_1.location", "unknown", "observation", "agent_0", "not visible", 1, status="unknown")])
    assert "cup_1" in json.dumps(store.snapshot("agent_0"))
    assert "cup_1" not in json.dumps(store.snapshot("agent_1"))
    with pytest.raises(ValueError):
        store.update("agent_1", "spatial", [Fact("leak", True, "observation", "agent_0", "hidden", 1)])


def test_dynamic_state_enters_next_planner_version(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0", "agent_1"], str(tmp_path), Task())
    first, v1 = runtime.planner_context("agent_0", 1)
    runtime.action_feedback("agent_0", 1, "Pick[cup_1]", "Successfully picked cup_1")
    second, v2 = runtime.planner_context("agent_0", 2)
    assert v2 > v1
    assert "object.cup_1.held=true" in second
    assert f"version={v2}" in second
    assert "private B" not in second
    assert first != second
    runtime.record_plan("agent_0", 2, "Pick[cup_1]", "use current state", second)
    rows = [json.loads(line) for line in (tmp_path / "closed_loop_artifacts/plans.jsonl").read_text().splitlines()]
    assert rows[-1]["state_version_read"] == v2
    assert rows[-1]["state_marker_found_in_actual_prompt"] is True
    assert rows[-1]["runtime_state_block_count"] == 1


def test_successful_motion_moves_conflicting_relations_to_history(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0"], str(tmp_path), Task(agent_secrets={
        "agent_0": ["cup_1 starts on shelf_2"],
    }))
    runtime.action_feedback("agent_0", 1, "Pick[cup_1]", "Successfully picked cup_1")
    picked = runtime.store.snapshot("agent_0")
    assert "object.cup_1.on" not in picked["spatial"]
    assert picked["spatial"]["object.cup_1.held"]["value"] is True
    assert any(
        row["key"] == "object.cup_1.on"
        and row["inactive_reason"] == "superseded_by_successful_motion"
        for row in picked["history"]
    )

    runtime.action_feedback(
        "agent_0", 2, "Place[cup_1, inside, cabinet_3, None, None]", "Successfully placed cup_1"
    )
    placed = runtime.store.snapshot("agent_0")
    assert placed["spatial"]["object.cup_1.inside"]["value"] == "cabinet_3"
    assert "object.cup_1.on" not in runtime.store.render("agent_0")


def test_failed_motion_does_not_invalidate_current_relation(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0"], str(tmp_path), Task(agent_secrets={
        "agent_0": ["cup_1 starts on shelf_2"],
    }))
    runtime.action_feedback("agent_0", 1, "Pick[cup_1]", "Failed to pick: too far")
    state = runtime.store.snapshot("agent_0")
    assert state["spatial"]["object.cup_1.on"]["value"] == "shelf_2"


def test_static_state_is_not_updated_after_feedback(tmp_path):
    runtime = ClosedLoopRuntime("static_ssc", ["agent_0"], str(tmp_path), Task(agent_secrets={"agent_0": []}))
    before = runtime.store.snapshot("agent_0")
    runtime.action_feedback("agent_0", 1, "Pick[cup_1]", "Successfully picked cup_1")
    assert runtime.store.snapshot("agent_0") == before


def test_failed_action_does_not_create_completion_fact(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0"], str(tmp_path), Task(agent_secrets={"agent_0": []}))
    runtime.action_feedback("agent_0", 1, "Place[cup_1, on, table_2, None, None]", "Failed to place: too far")
    state = runtime.store.snapshot("agent_0")
    assert "object.cup_1.on" not in state["spatial"]
    assert state["social"]["action.turn_1.status"]["value"] == "failed"


def test_only_delivered_message_updates_receiver(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0", "agent_1"], str(tmp_path), Task())
    before = runtime.store.snapshot("agent_1")
    runtime.action_feedback("agent_0", 1, 'Communicate["cup_1 is on table_2", agent_1]', "Message blocked: recipient is not allowed")
    assert runtime.store.snapshot("agent_1") == before
    runtime.message_delivery("agent_0", ["agent_1"], 2, "cup_1 is on table_2")
    assert "message.turn_2.from.agent_0" in runtime.store.snapshot("agent_1")["social"]


def test_successful_send_attempt_without_delivery_does_not_update_receiver(tmp_path):
    runtime = ClosedLoopRuntime("dynamic_ssc", ["agent_0", "agent_1"], str(tmp_path), Task())
    before = runtime.store.snapshot("agent_1")
    runtime.action_feedback("agent_0", 1, 'Communicate["cup_1 is on table_2", agent_1]', "Executed.")
    assert runtime.store.snapshot("agent_1") == before
    sender = runtime.store.snapshot("agent_0")["social"]
    assert sender["message.turn_1.attempted"]["value"] is True
    assert "message.turn_1.delivered" not in sender


def test_strict_scorer_valid_held_wrong_and_reclosed():
    place = PhysicalGoal("is_on_top", "cup_1", "table_2")
    opened = PhysicalGoal("is_open", "cabinet_3")
    scorer = StrictPhysicalScorer(stable_samples=3)
    valid = {place.goal_id: [PhysicalSample(True, None) for _ in range(3)]}
    held = {place.goal_id: [PhysicalSample(True, "agent_0") for _ in range(3)]}
    wrong = {place.goal_id: [PhysicalSample(False, None) for _ in range(3)]}
    reclosed = {opened.goal_id: [PhysicalSample(True), PhysicalSample(True), PhysicalSample(False)]}
    assert scorer.score([place], valid)["strict_success"]
    assert not scorer.score([place], held)["strict_success"]
    assert not scorer.score([place], wrong)["strict_success"]
    assert not scorer.score([opened], reclosed)["strict_success"]


def test_failed_stability_advance_is_invalid_not_full_completion():
    scorer = StrictPhysicalScorer(stable_samples=3)
    goal = PhysicalGoal("is_on_top", "cup_1", "table_2")
    result = scorer.measure(
        [goal],
        probe=lambda _goal: PhysicalSample(True, None),
        advance=lambda: False,
    )
    assert result["measurement_valid"] is False
    assert result["measurement_status"] == "invalid"
    assert result["strict_success"] is None
    assert result["completed_goals"] is None
    assert result["strict_percent_complete"] is None
    assert result["unvalidated_strict_percent_complete"] == 1.0


def test_budget_exhaustion_is_a_failure():
    result = classify_outcome("agent_0 has used all messages", action=parse_action('Communicate["x", agent_1]'))
    assert result.status == "failed"


def test_illegal_or_missing_message_recipient_is_a_no_effect_failure():
    action = parse_action('Communicate["status", agent_9]')
    outcome = classify_outcome(
        "No valid recipients specified. Message was not delivered.",
        action=action,
    )
    assert outcome.status == "failed"
    assert outcome.had_physical_effect is False


def test_restricted_recipient_feedback_is_a_no_effect_failure():
    action = parse_action('Communicate["status", agent_1]')
    outcome = classify_outcome(
        "You can only send messages to: agent_2. Message to agent_1 was blocked.",
        action=action,
    )
    assert outcome.status == "failed"
    assert outcome.success is False
    assert outcome.had_physical_effect is False


def test_1566_and_772_historical_audits_are_not_promoted_to_stable(tmp_path):
    root = Path(__file__).resolve().parents[2]
    case_dir = root / "reports" / "cases"
    if not case_dir.exists():
        pytest.skip("historical result fixtures omitted from source-only release")
    goals_1566 = parse_functional_goals("(and (is_on_top spray_bottle_3 couch_13) (is_on_top box_5 counter_25) (is_open fridge_34))")
    direct = {g.entity: json.loads((case_dir / f"habitat_ep1566_direct_final_{g.entity}_state.json").read_text()) for g in goals_1566}
    ssc = {g.entity: json.loads((case_dir / f"habitat_ep1566_ssc_final_{g.entity}_state.json").read_text()) for g in goals_1566}
    direct_score = score_exported_audit(goals_1566, direct)
    ssc_score = score_exported_audit(goals_1566, ssc)
    assert direct_score["instantaneous_completed_goals"] == 2
    assert ssc_score["instantaneous_completed_goals"] == 3
    assert not direct_score["strict_success"] and not ssc_score["strict_success"]
    assert ssc_score["historical_stability_status"] == "not_measured"

    goal_772 = PhysicalGoal("is_on_top", "picture_frame_0", "counter_25")
    row_772 = json.loads((case_dir / "habitat_ep772_turn14_state.json").read_text())
    score_772 = score_exported_audit([goal_772], {"picture_frame_0": row_772})
    assert score_772["instantaneous_completed_goals"] == 0
    assert not score_772["checks"][0]["released_stable"]
