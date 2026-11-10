#!/usr/bin/env python3
"""Fair one-call paired pilot for weak CoMind temporal-social fields."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V2 = load_module("comind_v2_weak_skill", ROOT / "experiments/comind_optimization_v2/comind_v2.py")

METRICS = {
    "scoia": ("cue_type", "action_verb"),
    "handover": ("delivery_flow", "initiator", "initiation_type", "tth_within_0_25s"),
}

COMMON = """Use only observations at or before the prediction timestamp. Predict the future event, not merely the action visible in the final frame. Return JSON only.

Official closed labels:
- cue_type/initiation_type is exactly one of [\"verbal\"], [\"gestural\"], [\"implicit\"], or [\"gestural\", \"verbal\"].
- SCOIA action_verb is one of: grab, receive, pass, open, add, place, turn on, cut, rotate, pour, press, move, turn off, scoop, mix, clean, stretch, roll.
- Handover delivery_flow is ltr or rtl; initiator is left or right.
"""

BASELINE = """Direct prediction rule: inspect the ordered frames and transcript, then make the single most likely prediction under the official labels."""

SKILL = """Temporal-Social Transition Skill:
1. Treat the last image as a boundary: distinguish the currently visible micro-action from the socially conditioned action that happens next.
2. Track who emits a verbal/gestural cue, who receives it, the referenced object, and each person's reachable side.
3. Predict the next state transition within 10 seconds using the official coarse verb taxonomy; do not describe fine-grained visible motion.
4. For handover, infer initiator and flow from the first offer/request evidence and estimate time from motion progress across the ordered frames.
Apply these steps internally, then emit only the required JSON."""


def schema_instruction(task: str) -> str:
    if task == "scoia":
        return 'Required JSON: {"cue_type":["..."],"action_verb":"..."}'
    return (
        'Required JSON: {"delivery_flow":"ltr|rtl","initiator":"left|right",'
        '"initiation_type":["..."],"time_to_handover":0.0}'
    )


def row_prompt(row: dict, condition: str) -> str:
    legend = "\n".join(
        f"{index}. {item['role']} {item['kind']} at {float(item['time']):.3f}s"
        for index, item in enumerate(row["images"], 1)
    )
    task_text = (
        "Anticipate the socially conditioned action the helper will perform within 10 seconds."
        if row["task"] == "scoia"
        else "Anticipate the upcoming handover from the two egocentric views."
    )
    method = BASELINE if condition == "baseline" else SKILL
    return (
        f"You are evaluating CoMind {row['task']}. {task_text}\n\n{COMMON}\n{method}\n\n"
        f"Prediction timestamp: {row['prediction_time']:.3f}s\nTranscript from the preceding 10 seconds: "
        f"{row.get('transcript_10s') or '<none>'}\nOrdered images:\n{legend}\n\n{schema_instruction(row['task'])}"
    )


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def stable_rows(rows: list[dict], per_task: int) -> list[dict]:
    selected = []
    for task in METRICS:
        recordings: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["task"] == task:
                recordings[str(row["recording_id"])].append(row)
        for values in recordings.values():
            values.sort(key=lambda row: hashlib.sha256(f"{row['recording_id']}:{row['sample_id']}".encode()).hexdigest())
        ordered_recordings = sorted(recordings, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        cursor = 0
        while len([row for row in selected if row["task"] == task]) < per_task:
            added = False
            for recording in ordered_recordings:
                if cursor < len(recordings[recording]):
                    selected.append(recordings[recording][cursor])
                    added = True
                    if len([row for row in selected if row["task"] == task]) == per_task:
                        break
            if not added:
                break
            cursor += 1
    return selected


def prediction_for_metric(task: str, metric: str, prediction: dict) -> Any:
    field = V2.FIELD_TO_PREDICTION[task][metric]
    return prediction.get(field)


def prior_for(full_rows: list[dict], annotations: dict, row: dict, metric: str) -> Any:
    task = row["task"]
    values = []
    for other in full_rows:
        if other["task"] != task or int(other["recording_fold"]) == int(row["recording_fold"]):
            continue
        gold = annotations[task].get(str(other["recording_id"]), {}).get(str(other["sample_id"]))
        if not isinstance(gold, dict):
            continue
        value = V2.ground_truth_value(task, metric, gold)
        if metric in {"cue_type", "initiation_type"}:
            values.append(tuple(sorted(V2.values(value))))
        elif metric == "tth_within_0_25s":
            values.append(float(value))
        else:
            values.extend(V2.values(value))
    if metric == "tth_within_0_25s":
        candidates = sorted(set(values))
        return max(candidates, key=lambda candidate: sum(abs(candidate - value) <= 0.25 for value in values))
    winner = Counter(values).most_common(1)[0][0]
    return list(winner) if isinstance(winner, tuple) else winner


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# CoMind weak-field fair paired pilot",
        "",
        "Same frozen samples, Qwen3-VL checkpoint, images, transcript, one generation call, greedy decoding, and 256-token output cap. Both prompts receive the same official label ontology.",
        "",
        "| Field | N | Baseline | Temporal-social skill | Recording-heldout prior | Delta vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, metrics in METRICS.items():
        task_rows = [row for row in rows if row["task"] == task and "error" not in row]
        for metric in metrics:
            n = len(task_rows)
            if n == 0:
                continue
            base = sum(row["scores"]["baseline"][metric] for row in task_rows)
            skill = sum(row["scores"]["temporal_social_skill"][metric] for row in task_rows)
            prior = sum(row["scores"]["recording_heldout_prior"][metric] for row in task_rows)
            lines.append(
                f"| {task}.{metric} | {n} | {base/n:.1%} | {skill/n:.1%} | {prior/n:.1%} | {(skill-base)/n:+.1%} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "workdirs/comind_optimization_v2/gpu_grounding_full_manifest.jsonl")
    parser.add_argument("--annotations", type=Path, default=ROOT / "external_datasets/comind/annotations")
    parser.add_argument("--output", type=Path, default=ROOT / "workdirs/comind_weak_skill_v1/pilot_predictions.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "workdirs/comind_weak_skill_v1/pilot_report.md")
    parser.add_argument("--per-task", type=int, default=48)
    parser.add_argument("--model-path", default="/pfs/xiabin/models/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    os.environ["SPATIAL_SOCIAL_MODEL_PATH"] = args.model_path
    os.environ["SPATIAL_SOCIAL_LOCAL_FILES_ONLY"] = "1"
    sys.path.insert(0, str(ROOT / "third_party/autotom/model"))
    from local_backend import generate, set_image_context

    full_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = stable_rows(full_rows, args.per_task)
    annotations = V2.load_annotations(args.annotations)
    existing = {}
    if args.output.exists():
        existing = {
            (row["task"], str(row["recording_id"]), str(row["sample_id"])): row
            for row in (json.loads(line) for line in args.output.read_text().splitlines() if line.strip())
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in selected:
        item_key = (row["task"], str(row["recording_id"]), str(row["sample_id"]))
        if item_key in existing and "error" not in existing[item_key]:
            continue
        result = {"task": row["task"], "recording_id": row["recording_id"], "sample_id": row["sample_id"], "fold": row["recording_fold"], "predictions": {}}
        try:
            set_image_context([item["path"] for item in row["images"]])
            for condition in ("baseline", "temporal_social_skill"):
                raw = generate(row_prompt(row, condition), max_new_tokens=256)
                result["predictions"][condition] = parse_json(raw)
            gold = annotations[row["task"]][str(row["recording_id"])][str(row["sample_id"])]
            result["scores"] = {"baseline": {}, "temporal_social_skill": {}, "recording_heldout_prior": {}}
            for metric in METRICS[row["task"]]:
                for condition in ("baseline", "temporal_social_skill"):
                    value = prediction_for_metric(row["task"], metric, result["predictions"][condition])
                    result["scores"][condition][metric] = V2.field_correct(row["task"], metric, value, gold)
                prior = prior_for(full_rows, annotations, row, metric)
                result["scores"]["recording_heldout_prior"][metric] = V2.field_correct(row["task"], metric, prior, gold)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        existing[item_key] = result
        args.output.write_text("".join(json.dumps(value) + "\n" for value in existing.values()))
        write_report(args.report, list(existing.values()))
        print(json.dumps({"completed": len(existing), "target": len(selected), "last": item_key, "error": result.get("error")}))


if __name__ == "__main__":
    main()
