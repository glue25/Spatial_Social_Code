"""Tests for the deterministic EnactToM E0 state compiler and reducer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spatial_social_code.enacttom import (
    Literal,
    _shortest_path,
    compile_oracle_state,
    evaluate_first_order_interventions,
    goal_holds,
    parse_action,
    parse_sexpr,
    replay_golden_trajectory,
    send_information,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_task() -> dict:
    return {
        "task_id": "synthetic-e0",
        "category": "cooperative",
        "problem_pddl": """
        (define (problem synthetic)
          (:domain emtom)
          (:objects
            agent_0 agent_1 agent_2 - agent
            room_a room_b - room
            cup_0 - object
            table_0 cabinet_0 - furniture)
          (:init
            (agent_in_room agent_0 room_a)
            (agent_in_room agent_1 room_a)
            (agent_in_room agent_2 room_b)
            (is_in_room cup_0 room_a)
            (is_in_room table_0 room_b)
            (is_in_room cabinet_0 room_b)
            (is_on_top cup_0 table_0))
          (:goal (and
            (is_on_top cup_0 table_0)
            (is_open cabinet_0)
            (K agent_0 (is_open cabinet_0)))))
        """,
        "functional_goal_pddl": (
            "(and (is_on_top cup_0 table_0) (is_open cabinet_0))"
        ),
        "agent_actions": {
            "agent_0": [],
            "agent_1": [],
            "agent_2": [],
        },
        "agent_secrets": {
            "agent_0": ["You need confirmation."],
            "agent_1": ["You are the only agent allowed to relay."],
            "agent_2": ["Open cabinet_0."],
        },
        "mechanic_bindings": [
            {
                "mechanic_type": "restricted_communication",
                "allowed_targets": {
                    "agent_0": ["agent_1"],
                    "agent_1": ["agent_0", "agent_2"],
                    "agent_2": ["agent_1"],
                },
            },
            {
                "mechanic_type": "limited_bandwidth",
                "message_limits": {
                    "agent_0": 2,
                    "agent_1": 2,
                    "agent_2": 2,
                },
            },
        ],
        "message_targets": {},
        "golden_trajectory": [
            {
                "actions": [
                    {"agent": "agent_0", "action": "Wait[]"},
                    {"agent": "agent_1", "action": "Wait[]"},
                    {"agent": "agent_2", "action": "Open[cabinet_0]"},
                ]
            }
        ],
        "literal_tom_probes": [
            {
                "probe_id": "k_probe_0",
                "agent_id": "agent_0",
                "subject_agents": [],
                "fact_pddl": "(is_open cabinet_0)",
                "depth": 1,
                "supported": True,
            }
        ],
    }


class EnactToME0Test(unittest.TestCase):
    def test_sexpr_parser_preserves_nested_knowledge(self) -> None:
        parsed = parse_sexpr("(K agent_0 (K agent_1 (is_open cabinet_0)))")
        self.assertEqual(parsed[0], "K")
        self.assertEqual(parsed[2][2][0], "is_open")

    def test_action_parser(self) -> None:
        self.assertEqual(
            parse_action("Place[cup_0, on, table_0, None, None]"),
            ("Place", ["cup_0", "on", "table_0", "None", "None"]),
        )

    def test_compiler_is_query_independent(self) -> None:
        task = sample_task()
        first = compile_oracle_state(task, "test").to_code_dict()
        task["literal_tom_probes"][0]["question"] = "A different query"
        second = compile_oracle_state(task, "test").to_code_dict()
        self.assertEqual(first, second)

    def test_replay_satisfies_physical_goal(self) -> None:
        replay = replay_golden_trajectory(sample_task(), "test")
        self.assertTrue(replay.goal_results["(is_open cabinet_0)"])
        self.assertEqual(replay.invalid_actions, [])
        self.assertEqual(replay.access_violations, [])

    def test_actor_observes_own_effect(self) -> None:
        replay = replay_golden_trajectory(sample_task(), "test")
        self.assertTrue(
            replay.state.knows("agent_2", Literal("is_open", ("cabinet_0",)))
        )

    def test_shortest_path_uses_relay(self) -> None:
        state = compile_oracle_state(sample_task(), "test")
        self.assertEqual(
            _shortest_path(state.message_targets, "agent_2", "agent_0"),
            ["agent_2", "agent_1", "agent_0"],
        )

    def test_message_requires_supported_sender_knowledge(self) -> None:
        state = compile_oracle_state(sample_task(), "test")
        with self.assertRaises(ValueError):
            send_information(
                state,
                sender="agent_1",
                recipient="agent_0",
                fact=Literal("is_open", ("cabinet_0",)),
                evidence_ref="test",
            )

    def test_delete_and_redirect_are_local(self) -> None:
        task = sample_task()
        replay = replay_golden_trajectory(task, "test")
        cases = evaluate_first_order_interventions(task, replay)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["status"], "evaluated")
        self.assertTrue(case["delete"]["target_sensitive"])
        self.assertTrue(case["delete"]["non_target_invariant"])
        self.assertTrue(case["redirect"]["target_sensitive"])
        self.assertTrue(case["redirect"]["non_target_invariant"])

    def test_closed_world_closed_goal(self) -> None:
        state = compile_oracle_state(sample_task(), "test")
        self.assertTrue(goal_holds(state, Literal("is_closed", ("cabinet_0",))))

    @unittest.skipUnless(
        (ROOT / "external_datasets/enacttom/standard.jsonl").is_file(),
        "requires optional EnactToM dataset",
    )
    def test_real_public_task_replays(self) -> None:
        path = ROOT / "external_datasets/enacttom/standard.jsonl"
        with path.open(encoding="utf-8") as handle:
            task = json.loads(next(handle))
        replay = replay_golden_trajectory(task, "standard")
        self.assertEqual(replay.invalid_actions, [])
        self.assertTrue(all(replay.goal_results.values()))


if __name__ == "__main__":
    unittest.main()
