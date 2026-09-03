from __future__ import annotations

from build_task_conditioned_code import compile_task


def _task() -> dict:
    return {
        "task": "Put the cup on the kitchen table.",
        "num_agents": 2,
        "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
        "agent_secrets": {
            "agent_0": ["The cup is cup_1 in bedroom_1."],
            "agent_1": ["The correct table is table_2 in kitchen_1."],
        },
        "mechanic_bindings": [
            {
                "mechanic_type": "restricted_communication",
                "allowed_targets": {"agent_0": ["agent_1"], "agent_1": ["agent_0"]},
            }
        ],
        "functional_goal_pddl": "(and (is_on_top cup_1 table_2))",
        "problem_pddl": """(define (problem p) (:domain enacttom)
        (:objects agent_0 agent_1 - agent bedroom_1 kitchen_1 - room
        cup_1 - object table_2 - furniture)
        (:init (agent_in_room agent_0 bedroom_1) (is_in_room cup_1 bedroom_1)
        (is_in_room table_2 kitchen_1))
        (:goal (and (is_on_top cup_1 table_2))))""",
        "golden_trajectory": [{"actions": [{"agent": "agent_0", "action": "Pick[cup_1]"}]}],
    }


def test_compiler_never_copies_oracle_actions_or_other_agents_private_values() -> None:
    result = compile_task(_task(), "physical_social")
    agent_0 = "\n".join(result["agent_secrets"]["agent_0"])
    assert "golden_trajectory" not in result
    assert "Pick[cup_1]" not in agent_0
    assert "table_2" not in agent_0
    assert "UNKNOWN_table" in agent_0
    assert "targets=agent_1" in agent_0


def test_conditions_keep_spatial_and_social_separable() -> None:
    physical = "\n".join(compile_task(_task(), "physical_only")["agent_secrets"]["agent_0"])
    social = "\n".join(compile_task(_task(), "social_only")["agent_secrets"]["agent_0"])
    assert "[Task-Conditioned Spatial Code]" in physical
    assert "[Task-Conditioned Social Code]" not in physical
    assert "[Task-Conditioned Social Code]" in social
    assert "[Task-Conditioned Spatial Code]" not in social


def test_v2_spatial_code_gates_unknowns_and_bounds_failed_retries() -> None:
    result = compile_task(_task(), "physical_social", spatial_version="v2")
    agent_0 = "\n".join(result["agent_secrets"]["agent_0"])
    assert "UNKNOWN_* denotes a type-level unknown" in agent_0
    assert "never repeat an identical failed or unconfirmed action" in agent_0
    assert "if that room is denied, delegate" in agent_0
    assert "golden_trajectory" not in result
