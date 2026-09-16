import copy
import json
from pathlib import Path

import pytest

from experiments.astra_world_demo_v1.compile_world import compile_payloads
from experiments.astra_world_demo_v1.world_dsl import WorldProgramError, execute_world


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "experiments/astra_world_demo_v1/fixtures"


def test_developer_fixture_cross_runtime_compilation() -> None:
    spatial = json.loads((FIXTURES / "developer_spatial.json").read_text())
    social = json.loads((FIXTURES / "developer_social.json").read_text())
    result = compile_payloads(spatial, social)
    assert result["normalized"]["statement_count"] == 18
    assert result["normalized"]["supported_subset"]["relations"] == ["inside", "near", "on_top_of"]


def test_world_runtime_rejects_nonliteral_python() -> None:
    with pytest.raises(WorldProgramError, match="unsupported syntax|every line"):
        execute_world("value = __import__('os')\n")


def test_world_runtime_rejects_unsupported_relation() -> None:
    code = "\n".join([
        "W.entity('a', 'bowl', source='observed', evidence_id='E1')",
        "W.entity('b', 'drawer', source='observed', evidence_id='E2')",
        "W.relation('behind', 'a', 'b', source='observed', evidence_id='E3')",
        "W.build()",
    ])
    with pytest.raises(WorldProgramError, match="unsupported relation"):
        execute_world(code)


def test_world_runtime_rejects_conflicting_placement() -> None:
    code = "\n".join([
        "W.entity('a', 'bowl', source='observed', evidence_id='E1')",
        "W.entity('b', 'drawer', source='observed', evidence_id='E2')",
        "W.entity('c', 'counter', source='observed', evidence_id='E3')",
        "W.entity('d', 'agent', source='observed', evidence_id='E4')",
        "W.entity('e', 'bowl', source='observed', evidence_id='E5')",
        "W.entity('f', 'surface', source='observed', evidence_id='E6')",
        "W.relation('inside', 'a', 'b', source='observed', evidence_id='E7')",
        "W.relation('on_top_of', 'a', 'c', source='intervention', evidence_id='I1')",
        "W.build()",
    ])
    with pytest.raises(WorldProgramError, match="conflicting placement"):
        execute_world(code)


def test_social_evidence_must_reference_spatial_trace() -> None:
    spatial = json.loads((FIXTURES / "developer_spatial.json").read_text())
    social = json.loads((FIXTURES / "developer_social.json").read_text())
    corrupted = copy.deepcopy(social)
    changed = False
    for line_index, line in enumerate(corrupted["social_code"]):
        if "evidence_ids=" in line:
            corrupted["social_code"][line_index] = line.replace(
                "evidence_ids='", "evidence_ids='NOT_IN_SPATIAL_TRACE,"
            )
            changed = True
            break
    assert changed
    with pytest.raises(Exception, match="NOT_IN_SPATIAL_TRACE|evidence"):
        compile_payloads(spatial, corrupted)
