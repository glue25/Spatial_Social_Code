import gzip
import json
from pathlib import Path

from experiments.habitat_closed_loop.compile_ssc_environment import compile_environment
from experiments.habitat_closed_loop.instantiate_ssc_environment import instantiate


ROOT = Path(__file__).resolve().parents[2]


def test_compiled_code_is_self_contained_and_excludes_answers():
    code = compile_environment(
        ROOT / "workdirs/habitat_enacttom/frozen12/tasks/hard_108294897_176710602_1166.json",
        ROOT / "workdirs/habitat_enacttom/frozen12/episodes.json.gz",
    )
    assert code["accepted"]
    assert len(code["spatial_code"]["rigid_asset_instances"]) == 5
    assert code["habitat_episode_blueprint"]["scene_id"] == code["scene"]["scene_id"]
    assert "golden_trajectory" not in code["task_blueprint"]
    assert "literal_tom_probes" not in code["task_blueprint"]


def test_code_compiles_to_distinct_episode_without_source_files():
    code = compile_environment(
        ROOT / "workdirs/habitat_enacttom/frozen12/tasks/hard_108294897_176710602_1166.json",
        ROOT / "workdirs/habitat_enacttom/frozen12/episodes.json.gz",
    )
    generated, episode, task, manifest = instantiate(
        code, episode_id="911166", asset_index=0, dx=-0.08, dz=0.06, yaw_deg=35.0,
    )
    assert episode["episode_id"] == task["episode_id"] == "911166"
    assert manifest["pose_changed"]
    assert episode["rigid_objs"][0][1] != code["spatial_code"]["rigid_asset_instances"][0]["transform_4x4"]
    assert generated["status"] == "compiled_new_episode_instance_using_existing_hssd_assets"
    assert not manifest["answer_bearing_fields_present"]
