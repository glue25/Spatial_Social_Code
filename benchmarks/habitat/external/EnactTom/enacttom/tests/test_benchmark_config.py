from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from enacttom.examples.run_habitat_benchmark import ensure_benchmark_observation_config
from enacttom.runner.benchmark import (
    _coordination_action_entity_names,
    _is_synthetic_remote_effect,
    task_to_instruction,
)
from enacttom.task_gen.task_generator import GeneratedTask


def test_ensure_benchmark_observation_config_forces_private_partial_obs() -> None:
    config = OmegaConf.create(
        {
            "world_model": {"partial_obs": False},
            "agent_asymmetry": False,
        }
    )

    ensure_benchmark_observation_config(config)

    # Standard mode (default) gets private per-agent world graphs.
    assert config.world_model.partial_obs is True
    assert config.agent_asymmetry is True
    assert config.benchmark_observation_mode == "text"
    assert config.benchmark_run_mode == "standard"
    assert config.benchmark_vision.selector_prompt_name == "enacttom_frame_selector"


def test_baseline_gets_shared_world_graph() -> None:
    config = OmegaConf.create(
        {
            "world_model": {"partial_obs": True},
            "agent_asymmetry": True,
            "benchmark_run_mode": "baseline",
        }
    )

    ensure_benchmark_observation_config(config)

    # Baseline mode uses the full-observability benchmark setting.
    assert config.world_model.partial_obs is False
    assert config.agent_asymmetry is False


def test_standard_instruction_includes_private_state_coordination_protocol() -> None:
    task = GeneratedTask(
        task_id="coordination_test",
        title="Coordination test",
        task="Move the object to the correct table.",
        category="cooperative",
        num_agents=2,
        scene_id="scene",
        episode_id="episode",
        active_mechanics=[],
        mechanic_bindings=[],
        agent_actions={"agent_0": ["Wait"], "agent_1": ["Wait"]},
        agent_secrets={
            "agent_0": ["The object is cup_1."],
            "agent_1": ["The destination is table_2."],
        },
    )

    instructions = task_to_instruction(task, run_mode="standard")

    assert "[Private-State Coordination Protocol]:" in instructions["agent_0"]
    assert "teammates do not know it" in instructions["agent_0"]
    assert "Private-State Coordination Protocol" not in task_to_instruction(
        task, run_mode="baseline"
    )["agent_0"]


def test_oracle_coordination_card_is_agent_indexed() -> None:
    task = GeneratedTask(
        task_id="card_test",
        title="Card test",
        task="Move the object.",
        category="cooperative",
        num_agents=2,
        scene_id="scene",
        episode_id="episode",
        active_mechanics=[],
        mechanic_bindings=[],
        agent_actions={"agent_0": ["Wait"], "agent_1": ["Wait"]},
        agent_secrets={"agent_0": [], "agent_1": []},
        coordination_cards={
            "agent_0": ["Navigate[object_1]", "Pick[object_1]"],
            "agent_1": ["Navigate[table_2]"],
        },
    )

    instructions = task_to_instruction(task, run_mode="standard")

    assert "Oracle Agent-Indexed Execution Card" in instructions["agent_0"]
    assert "Current Oracle Scheduler Directive" in instructions["agent_0"]
    assert "Pick[object_1]" not in instructions["agent_0"]
    assert "Pick[object_1]" not in instructions["agent_1"]


def test_coordination_action_entity_names_are_target_scoped() -> None:
    assert _coordination_action_entity_names("Navigate[plate_0]") == ["plate_0"]
    assert _coordination_action_entity_names(
        "Place[plate_0, on, table_2, None, None]"
    ) == ["plate_0", "table_2"]
    assert _coordination_action_entity_names("Wait[None]") == []


def test_synthetic_remote_effect_bypasses_physical_articulation() -> None:
    assert _is_synthetic_remote_effect(
        SimpleNamespace(applies=True, effects=["remote_effect=drawer_1.is_open=True"])
    )
    assert not _is_synthetic_remote_effect(
        SimpleNamespace(applies=True, effects=["inverted=Open->Close"])
    )
