from __future__ import annotations

import json

from experiments.priority_ssc_v1.common import clean_trace, forbidden_target_keys, parse_choice, trace_records
from experiments.priority_ssc_v1.run_four_cell import paired
from experiments.priority_ssc_v1.run_attribution_intervention import cases


def test_social_trace_removes_answer_directed_records() -> None:
    trace = "\n".join(
        json.dumps(row)
        for row in (
            {"kind": "observation", "values": {"agent": "a", "content": "door opened"}},
            {"kind": "posterior", "values": {"A": 0.9, "B": 0.1}},
            {"kind": "query_candidate", "values": {"answer": "A"}},
        )
    )
    cleaned = trace_records(trace, social=True)
    assert [row["kind"] for row in cleaned] == ["observation"]
    assert "posterior" not in clean_trace(trace, social=True)


def test_target_key_guard_is_recursive() -> None:
    assert forbidden_target_keys({"safe": [{"gold": "x"}]}) == ["gold"]
    assert not forbidden_target_keys({"kind": "belief", "values": {"agent": "a"}})


def test_choice_parser_and_paired_direction() -> None:
    assert parse_choice('{"answer":"B"}', "AB") == 1
    assert parse_choice('{"answer":"A","reasoning":"truncated', "AB") == 0
    assert parse_choice("", "AB") == -1
    rows = [
        {"correct": {"left": False, "right": True}},
        {"correct": {"left": True, "right": True}},
    ]
    result = paired(rows, "left", "right")
    assert result["net_right_wins"] == 1


def test_attribution_pairs_hold_physical_state_fixed() -> None:
    examples = cases(100)
    assert len(examples) == 100
    assert {row["intervention_type"] for row in examples} == {"observer", "message_recipient", "presence"}
    for row in examples:
        assert row["variants"]["base"]["answer_index"] != row["variants"]["target_intervention"]["answer_index"]
        assert row["variants"]["base"]["answer_index"] == row["variants"]["negative_control"]["answer_index"]
