import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "experiments" / "iclr_ablations_v1" / "run_equal_call_direct.py"
    spec = importlib.util.spec_from_file_location("equal_call_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_execution_runner():
    path = ROOT / "experiments" / "iclr_ablations_v1" / "run_mmtom_execution_effect.py"
    spec = importlib.util.spec_from_file_location("execution_effect_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_equal_call_schedule_is_fixed_and_exact():
    schedule = load_runner().call_schedule(list(range(600)))
    assert len(schedule) == 600
    assert sum(schedule.values()) == 1550
    assert list(schedule.values()).count(2) == 250
    assert list(schedule.values()).count(3) == 350


def test_majority_and_tie_break_are_label_independent():
    majority = load_runner().majority
    assert majority(["a", "b"]) == "a"
    assert majority(["b", "a"]) == "b"
    assert majority(["a", "b", "b"]) == "b"
    assert majority(["", "a"]) == "a"


def test_offline_reconstruction_and_counts():
    path = ROOT / "workdirs" / "iclr_ablations_v1" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    ablation = summary["strict_cached_trace_ablation"]
    assert ablation["examples"] == 600
    assert ablation["reconstruction_mismatches_vs_saved_full"] == []
    assert ablation["maximum_reconstruction_probability_error"] == 0.0
    assert ablation["conditions"]["spatial_only_executed"]["correct"] == 444
    assert ablation["conditions"]["social_only_executed"]["correct"] == 450
    assert summary["frozen_primary_same_model_comparison"]["conditions"]["direct"]["correct"] == 406
    assert summary["frozen_primary_same_model_comparison"]["conditions"]["frozen_spatial_social"]["correct"] == 470


def test_execution_effect_changes_only_trace_slots():
    runner = load_execution_runner()
    cached = {
        "codes": {"spatial": "SPATIAL_CODE", "social": "SOCIAL_CODE"},
        "traces": {"spatial": "SPATIAL_TRACE", "social": "SOCIAL_TRACE"},
    }
    raw = runner.prompt_for("raw_code_only", "QUESTION", cached)
    executed = runner.prompt_for("executed_trace", "QUESTION", cached)
    for shared in ("SPATIAL_CODE", "SOCIAL_CODE", "QUESTION"):
        assert shared in raw and shared in executed
    assert "SPATIAL_TRACE" not in raw and "SOCIAL_TRACE" not in raw
    assert "SPATIAL_TRACE" in executed and "SOCIAL_TRACE" in executed


def test_execution_effect_manifest_has_exact_call_count(tmp_path):
    runner = load_execution_runner()
    cached = tmp_path / "cached.jsonl"
    cached.write_text("{}\n", encoding="utf-8")
    protocol = runner.manifest(cached, Path("/model"), 600)
    assert protocol["new_extraction_calls"] == 0
    assert protocol["new_final_answer_calls"] == 1200
    assert protocol["label_access_during_inference"] is False
