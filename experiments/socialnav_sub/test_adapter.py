from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spatial_social_code"))

from socialnav_sub import canonicalize_official_scene, compile_scene  # noqa: E402
from experiments.socialnav_sub.run_pilot_jdcloud import (  # noqa: E402
    condition_uses_raw_frames,
)


SCENE = {
    "scene_id": "pilot_scene_01",
    "goal": {"x": 5.0, "y": 0.0},
    "samples": [
        {
            "t": 0,
            "robot": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "people": {"person_1": {"x": 2.0, "y": 0.2}},
        },
        {
            "t": 4,
            "robot": {"x": 1.0, "y": 0.0, "yaw": 0.0},
            "people": {"person_1": {"x": 2.2, "y": 0.1}},
        },
    ],
}


def test_compile_scene_is_executable_and_query_independent() -> None:
    compiled = compile_scene(SCENE)

    assert compiled.scene_id == "pilot_scene_01"
    assert '"obstructs_goal_path"' in compiled.spatial_code
    assert '"maintain clearance from person_1"' in compiled.social_code
    assert "spatial_fact" in compiled.spatial_trace
    assert "intent" in compiled.social_trace


def test_scene_hash_and_code_change_only_with_scene_evidence() -> None:
    first = compile_scene(SCENE)
    duplicate = compile_scene(copy.deepcopy(SCENE))
    changed = copy.deepcopy(SCENE)
    changed["samples"][-1]["people"]["person_1"]["y"] = 3.0

    assert duplicate.source_sha256 == first.source_sha256
    assert duplicate.spatial_code == first.spatial_code
    assert compile_scene(changed).source_sha256 != first.source_sha256


def test_canonicalize_official_scene_preserves_agent_ids() -> None:
    raw = {
        "imagepath": "/source/frame.jpg",
        "robot_global_poses": [
            {"center_x": "10", "center_y": "20", "yaw": "0"},
            {"center_x": "11", "center_y": "20", "yaw": "0"},
        ],
        "goal_position": {"center_x": "5", "center_y": "0"},
        "tracking_id_to_annotated_label_id": {"12": 3},
        "track_id_to_object_trajectory": {
            "12": [
                {"timestep": 0, "center_x": "2", "center_y": "0"},
                {"timestep": 1, "center_x": "2.5", "center_y": "0"},
            ]
        },
    }

    scene = canonicalize_official_scene(raw, "scene")

    assert scene["samples"][1]["robot"]["x"] == 1.0
    assert scene["samples"][0]["people"]["person_3"]["x"] == 2.0


def test_canonicalize_ignores_unselected_tracks_instead_of_aliasing_person_one() -> None:
    raw = {
        "imagepath": "/source/frame.jpg",
        "robot_global_poses": [
            {"center_x": "0", "center_y": "0", "yaw": "0"},
            {"center_x": "1", "center_y": "0", "yaw": "0"},
            {"center_x": "2", "center_y": "0", "yaw": "0"},
        ],
        "goal_position": {"center_x": "5", "center_y": "0"},
        "tracking_id_to_annotated_label_id": {"selected": 1},
        "track_id_to_object_trajectory": {
            "selected": [
                {"timestep": 0, "center_x": "3", "center_y": "1"},
                {"timestep": 1, "center_x": "3", "center_y": "1"},
            ],
            "unused": [
                {"timestep": 2, "center_x": "9", "center_y": "9", "value": 1},
            ],
        },
    }

    scene = canonicalize_official_scene(raw, "scene")
    compiled = compile_scene(scene)

    assert scene["samples"][2]["people"] == {}
    assert 'S.fact(1, "relative_position", "person_1"' in compiled.spatial_code
    assert 'S.fact(2, "relative_position", "person_1"' not in compiled.spatial_code


def test_fair_code_condition_keeps_the_same_raw_frames_as_direct() -> None:
    assert condition_uses_raw_frames("direct")
    assert condition_uses_raw_frames("visual_predicted_ps_plus_frames")
    assert not condition_uses_raw_frames("visual_predicted_ps")
    assert not condition_uses_raw_frames("oracle_code")
