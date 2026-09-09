#!/usr/bin/env python3
"""Export the audited Habitat episode-1566 Direct/SSC contrast case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = (
    "emtom-108294897_176710602-1566-"
    "mixed-safety-prep-with-a-private-checklist-ite-2016757d"
)
BENCHMARK_ROOT = ROOT / (
    "workdirs/habitat_enacttom/frozen12/"
    "benchmark_qwen8b_task_conditioned_repeats"
)
REPLAY_ROOT = ROOT / "workdirs/habitat_enacttom/frozen12/replay_episode1566"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "cases"
FONT = Path(
    "/opt/conda/pkgs/font-ttf-dejavu-sans-mono-2.37-"
    "hab24e00_0/fonts/DejaVuSans.ttf"
)

RUN_DIRS = {
    "direct": BENCHMARK_ROOT
    / "results/repeat_3/direct/standard_108294897_176710602_1566",
    "physical_social": BENCHMARK_ROOT
    / "results/repeat_3/physical_social/standard_108294897_176710602_1566",
}
PLANNER_LOGS = {
    "direct": RUN_DIRS["direct"]
    / "results"
    / TASK_ID
    / "planner-log/planner-log-20260902_102147.json",
    "physical_social": RUN_DIRS["physical_social"]
    / "results"
    / TASK_ID
    / "planner-log/planner-log-20260902_104345.json",
}
SUMMARIES = {
    condition: run_dir / "results/benchmark_summary.json"
    for condition, run_dir in RUN_DIRS.items()
}
SOURCE_TASKS = {
    "direct": ROOT
    / "workdirs/habitat_enacttom/frozen12/tasks/standard_108294897_176710602_1566.json",
    "physical_social": BENCHMARK_ROOT
    / "tasks/physical_social/standard_108294897_176710602_1566.json",
}
REPLAY_RUNS = {
    "direct": REPLAY_ROOT / "direct_output/results" / TASK_ID,
    "physical_social": REPLAY_ROOT / "physical_social_output_v2/results" / TASK_ID,
}
FINAL_TURNS = {"direct": 20, "physical_social": 18}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def draw_header(
    canvas: Image.Image,
    lines: list[tuple[str, int, str]],
    *,
    x: int = 24,
    y: int = 12,
) -> None:
    draw = ImageDraw.Draw(canvas)
    for label, size, color in lines:
        draw.text((x, y), label, font=font(size), fill=color)
        y += size + 12


def export_single_with_zoom(
    source: Path,
    target: Path,
    lines: list[tuple[str, int, str]],
    crop_box: tuple[int, int, int, int],
    zoom_lines: tuple[str, ...],
) -> None:
    image = Image.open(source).convert("RGB")
    if image.size != (1024, 1024):
        raise ValueError(f"Unexpected source size for {source}: {image.size}")
    header = 152
    canvas = Image.new("RGB", (1536, header + 1024), "black")
    canvas.paste(image, (0, header))
    crop = image.crop(crop_box).resize((512, 512), Image.Resampling.NEAREST)
    canvas.paste(crop, (1024, header))
    draw_header(canvas, lines)
    draw = ImageDraw.Draw(canvas)
    y = header + 550
    for label in zoom_lines:
        draw.text((1048, y), label, font=font(24), fill="white")
        y += 36
    canvas.save(target, format="PNG", optimize=True)


def export_completed_goals_frame(target: Path) -> None:
    fridge = Image.open(
        REPLAY_RUNS["physical_social"]
        / "replay_audit_views/turn_0018/fridge_34/view_3_fridge_34.png"
    ).convert("RGB")
    box = Image.open(
        REPLAY_RUNS["physical_social"]
        / "replay_audit_views/turn_0018/box_5/view_4_box_5.png"
    ).convert("RGB")
    if fridge.size != (1024, 1024) or box.size != (1024, 1024):
        raise ValueError("Unexpected replay audit image size")

    header = 180
    canvas = Image.new("RGB", (2048, header + 1024), "black")
    canvas.paste(fridge, (0, header))
    canvas.paste(box, (1024, header))
    draw_header(
        canvas,
        [
            ("Replay visualization", 34, "white"),
            ("Episode 1566 | SSC final state after Turn 18 | seed 891003", 25, "white"),
            (
                "Left: fridge_34 open | Right: box_5/counter_25 audit view",
                22,
                "white",
            ),
            (
                "State audit: fridge joint=2.094395; box released; support match=true",
                21,
                "#7CFC98",
            ),
        ],
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((1024, header, 2047, header + 1023), outline="#FFD166", width=4)
    draw.text(
        (1050, header + 945),
        "Box is small; use the paired state JSON for the exact support relation.",
        font=font(19),
        fill="#FFD166",
    )
    canvas.save(target, format="PNG", optimize=True)


def final_state_paths() -> dict[str, dict[str, Path]]:
    return {
        condition: {
            target: REPLAY_RUNS[condition]
            / "replay_audit_views"
            / f"turn_{FINAL_TURNS[condition]:04d}"
            / target
            / "state.json"
            for target in ("spray_bottle_3", "box_5", "fridge_34")
        }
        for condition in ("direct", "physical_social")
    }


def status_records() -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    status_path = BENCHMARK_ROOT / "status.jsonl"
    for raw_line in status_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw_line)
        if (
            row.get("repeat") == 3
            and row.get("seed") == 891003
            and row.get("task") == "standard_108294897_176710602_1566"
            and row.get("condition") in ("direct", "physical_social")
        ):
            matches[row["condition"]] = row
    if set(matches) != {"direct", "physical_social"}:
        raise ValueError(f"Missing status rows: {sorted(matches)}")
    return matches


def extract_codes(task: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, dict[str, str]] = {}
    for agent_id, secrets in task["agent_secrets"].items():
        spatial = next(item for item in secrets if "SPATIAL_STATE" in item)
        social = next(item for item in secrets if "SOCIAL_STATE" in item)
        agents[agent_id] = {
            "spatial_code": spatial,
            "social_code": social,
        }
    return {
        "schema_version": "habitat_spatial_social_code_export_v1",
        "task_id": task["task_id"],
        "episode_id": task["episode_id"],
        "condition": "physical_social",
        "repeat": 3,
        "actual_seed": 891003,
        "agents": agents,
        "provenance": {
            "source_task": relative(SOURCE_TASKS["physical_social"]),
            "source_task_sha256": digest(SOURCE_TASKS["physical_social"]),
            "extraction": "Exact strings from agent_secrets; no rewriting.",
        },
    }


def export_jsons() -> None:
    planners = {condition: load(path) for condition, path in PLANNER_LOGS.items()}
    summaries = {condition: load(path) for condition, path in SUMMARIES.items()}
    tasks = {condition: load(path) for condition, path in SOURCE_TASKS.items()}
    states_paths = final_state_paths()
    states = {
        condition: {target: load(path) for target, path in targets.items()}
        for condition, targets in states_paths.items()
    }

    for condition, planner in planners.items():
        if planner["task_id"] != TASK_ID:
            raise ValueError(f"Wrong planner log for {condition}")
    for condition, task in tasks.items():
        if task["task_id"] != TASK_ID:
            raise ValueError(f"Wrong source task for {condition}")

    code_export = extract_codes(tasks["physical_social"])
    code_path = OUTPUT / "habitat_episode_1566_repeat3_spatial_social_code.json"
    code_path.write_text(
        json.dumps(code_export, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_records: dict[str, Any] = {}
    statuses = status_records()
    for condition in ("direct", "physical_social"):
        summary_result = summaries[condition]["results"][0]
        communication_rows = [
            row
            for row in planners[condition]["action_history"]
            if str(row.get("action", "")).startswith("Communicate[")
        ]
        run_records[condition] = {
            "condition": condition,
            "model": summaries[condition]["model"],
            "llm_provider": summaries[condition]["llm_provider"],
            "observation_mode": summaries[condition]["benchmark_observation_mode"],
            "run_mode": summary_result["run_mode"],
            "repeat": 3,
            "actual_seed": 891003,
            "run_status_record": statuses[condition],
            "benchmark_result": summary_result,
            "action_history": planners[condition]["action_history"],
            "communication_history": communication_rows,
            "communication_metrics": planners[condition].get("communication_metrics"),
            "event_trace": planners[condition].get("event_trace"),
            "audited_final_state": states[condition],
        }

    provenance_files = {
        f"{condition}_{kind}": path
        for condition in ("direct", "physical_social")
        for kind, path in (
            ("source_task", SOURCE_TASKS[condition]),
            ("planner_log", PLANNER_LOGS[condition]),
            ("benchmark_summary", SUMMARIES[condition]),
        )
    }
    for condition, targets in states_paths.items():
        for target, path in targets.items():
            provenance_files[f"{condition}_final_{target}_state"] = path

    export = {
        "schema_version": "habitat_paired_case_export_v1",
        "task_id": TASK_ID,
        "title": tasks["physical_social"]["title"],
        "split": tasks["physical_social"]["split"],
        "scene_id": tasks["physical_social"]["scene_id"],
        "episode_id": tasks["physical_social"]["episode_id"],
        "task_instruction": tasks["physical_social"]["task"],
        "pairing": {
            "same_task": True,
            "same_repeat": True,
            "same_seed": True,
            "same_model": True,
            "repeat": 3,
            "actual_seed": 891003,
            "model": "qwen3-vl-8b-instruct",
        },
        "claim": {
            "direct": "2/3 physical goals; timed out with spray_bottle_3 still grasped.",
            "physical_social": "3/3 physical goals; all targets pass final-state audit.",
            "turns_note": (
                "18 versus 20 is reported descriptively only; this single case does not "
                "establish a general efficiency improvement."
            ),
        },
        "runs": run_records,
        "spatial_social_code_file": relative(code_path),
        "replay_provenance": {
            "source": "saved learned-agent action histories",
            "uses_model_generation": False,
            "uses_golden_trajectory": False,
            "preserves_recorded_failed_actions_as_noops": True,
            "ssc_replay_directory": relative(REPLAY_ROOT / "physical_social_output_v2"),
            "excluded_directory": relative(REPLAY_ROOT / "physical_social_output"),
            "excluded_reason": "Earlier replay incorrectly executed a recorded failed Pick.",
        },
        "provenance_sha256": {
            key: {"path": relative(path), "sha256": digest(path)}
            for key, path in provenance_files.items()
        },
    }
    case_path = OUTPUT / "habitat_episode_1566_repeat3_direct_vs_ssc.json"
    case_path.write_text(
        json.dumps(export, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for condition, targets in states_paths.items():
        short = "ssc" if condition == "physical_social" else "direct"
        for target, source in targets.items():
            target_path = OUTPUT / f"habitat_ep1566_{short}_final_{target}_state.json"
            target_path.write_bytes(source.read_bytes())


def export_images() -> None:
    direct_source = (
        REPLAY_RUNS["direct"]
        / "replay_audit_views/turn_0020/spray_bottle_3/view_2_spray_bottle_3.png"
    )
    ssc_source = (
        REPLAY_RUNS["physical_social"]
        / "replay_audit_views/turn_0018/spray_bottle_3/view_2_spray_bottle_3.png"
    )
    crop = (315, 300, 765, 750)
    export_single_with_zoom(
        direct_source,
        OUTPUT / "habitat_ep1566_direct_turn20_spray_failure_replay.png",
        [
            ("Replay visualization", 34, "white"),
            ("Episode 1566 | Direct final state after Turn 20 | seed 891003", 25, "white"),
            (
                "FAIL: spray_bottle_3 remains grasped; couch_13 support match=false",
                21,
                "#FF6B6B",
            ),
        ],
        crop,
        (
            "Zoom from the same final frame:",
            "yellow spray bottle remains",
            "attached to Agent 0's gripper.",
            "Final physical score: 2/3.",
        ),
    )
    export_single_with_zoom(
        ssc_source,
        OUTPUT / "habitat_ep1566_ssc_turn18_spray_success_replay.png",
        [
            ("Replay visualization", 34, "white"),
            ("Episode 1566 | Spatial-Social Code final state after Turn 18 | seed 891003", 23, "white"),
            (
                "PASS: spray_bottle_3 released; couch_13 support match=true",
                21,
                "#7CFC98",
            ),
        ],
        crop,
        (
            "Zoom from the same final frame:",
            "yellow spray bottle is released",
            "on couch_13; no agent grasp.",
            "Final physical score: 3/3.",
        ),
    )
    export_completed_goals_frame(
        OUTPUT / "habitat_ep1566_ssc_turn18_box_fridge_replay.png"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    export_jsons()
    export_images()


if __name__ == "__main__":
    main()
