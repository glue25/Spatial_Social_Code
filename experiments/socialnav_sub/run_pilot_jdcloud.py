#!/usr/bin/env python3
"""Run a frozen, query-independent SocialNav-SUB pilot on JD Cloud Gemini.

Each scene is visually compiled exactly once before questions are constructed.
The same frozen representation is then reused for one spatial, one
spatiotemporal, and one social question.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MA-EgoQA"))

from egomas.src.jdcloud_gemini import (  # noqa: E402
    JDCloudGeminiClient,
    load_service_config,
)


QUESTION_SPECS = (
    ("Spatial reasoning", "q_person_spatial_position_end", "movement_instructions"),
    ("Spatiotemporal reasoning", "q_person_distance_change", "movement_instructions"),
    ("Social reasoning", "q_robot_affected", "navigation_affected_instructions"),
)

VISUAL_CODE_CONDITIONS = (
    "visual_predicted_ps",
    "visual_predicted_ps_plus_frames",
)


def condition_uses_raw_frames(condition: str) -> bool:
    """Keep raw visual evidence for Direct and the fair code-augmented arm."""
    return condition in {"direct", "visual_predicted_ps_plus_frames"}


def json_value(text: str) -> Any:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", stripped):
            try:
                value, _ = decoder.raw_decode(stripped[match.start() :])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("response did not contain valid JSON")


def normalize_answer(raw: str, choices: list[str]) -> str | None:
    try:
        value = json_value(raw)
        answer = value.get("answer") if isinstance(value, dict) else value
    except ValueError:
        answer = raw
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    if not isinstance(answer, str):
        return None
    normalized = re.sub(r"\s+", " ", answer.strip().lower())
    for choice in choices:
        if normalized == choice.lower():
            return choice
    matches = [choice for choice in choices if choice.lower() in normalized]
    return matches[0] if len(matches) == 1 else None


def image_parts(paths: list[Path]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.extend(
            [
                {"text": f"Ordered scene frame t={index}:"},
                {"inline_data": {"mimeType": "image/png", "data": encoded}},
            ]
        )
    return parts


def compile_prompt(scene_id: str) -> str:
    return f"""You are compiling a reusable Spatial-Social Code for robot navigation scene {scene_id}.
The ten supplied images are ordered t=0 through t=9. They combine the robot camera view with an annotated bird's-eye view. Circled labels identify people and G identifies the robot goal.

Important experimental constraint: no question or answer choices are available. Describe the complete scene once so that several later questions can reuse it. Do not guess hidden mental states; use \"unknown\" when visual evidence is insufficient.

Return JSON only with this schema:
{{
  "scene_id": "{scene_id}",
  "robot": {{"motion": ["moving ahead|turning left|turning right"], "goal_begin": "ahead|to the left|to the right|unknown", "goal_end": "ahead|to the left|to the right|unknown"}},
  "people": {{
    "person_1": {{
      "begin_position": "ahead of|to the left of|to the right of|behind|unknown",
      "end_position": "ahead of|to the left of|to the right of|behind|unknown",
      "distance_change": "closer to|further away from|about the same distance to|unknown",
      "path_obstruction": "yes|no|unknown",
      "end_position_obstruction": "yes|no|unknown",
      "robot_affected": "yes|no|unknown",
      "robot_action": "avoiding|overtaking|not considering|following|yielding to|unknown",
      "person_affected": "yes|no|unknown",
      "person_action": "avoiding|overtaking|not considering|following|yielding to|unknown",
      "evidence_frames": [0, 9]
    }}
  }},
  "uncertainties": []
}}
Include one people entry for every visible numbered person. Never include prose outside JSON."""


def question_record(
    scene_id: str,
    group: str,
    base_key: str,
    instructions_key: str,
    survey: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    person_key = f"{base_key}_p1"
    template = survey[base_key]
    question = template["question"].replace("{PERSON}", "person 1")
    choices = list(template["choices"])
    human_answers = labels[person_key]
    human_probabilities = labels[f"{person_key}_probabilities"]
    return {
        "scene_id": scene_id,
        "reasoning_group": group,
        "question_key": person_key,
        "instructions": "\n".join(survey[instructions_key]),
        "question": question,
        "choices": choices,
        "human_distribution": dict(zip(human_answers, human_probabilities)),
        "human_n": labels[f"{person_key}_n_answers"],
    }


def answer_prompt(record: dict[str, Any], representation: str | None = None) -> str:
    context = ""
    if representation is not None:
        context = (
            "A query-independent scene representation was frozen before this question:\n"
            f"{representation}\n\n"
        )
    choices = json.dumps(record["choices"], ensure_ascii=False)
    return (
        context
        + record["instructions"]
        + "\n\n"
        + record["question"]
        + f"\nPossible answers: {choices}\n"
        + "Return JSON only as {\"answer\": \"one exact possible answer\"}."
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = {
        (row["condition"], row["question_key"], row["scene_id"]): row
        for row in rows
    }
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    modal_correct: dict[str, list[bool]] = defaultdict(list)
    invalid: dict[str, int] = defaultdict(int)
    for row in latest.values():
        condition = row["condition"]
        answer = row.get("answer")
        if answer is None:
            invalid[condition] += 1
            continue
        agreement = float(row["human_distribution"].get(answer, 0.0))
        human_mode = max(row["human_distribution"], key=row["human_distribution"].get)
        by_condition[condition].append(agreement)
        by_group[condition][row["reasoning_group"]].append(agreement)
        modal_correct[condition].append(answer == human_mode)
    result: dict[str, Any] = {}
    for condition, values in sorted(by_condition.items()):
        result[condition] = {
            "n": len(values),
            "invalid": invalid.get(condition, 0),
            "mean_human_probability_agreement": sum(values) / len(values) if values else None,
            "human_mode_accuracy": sum(modal_correct[condition]) / len(modal_correct[condition]),
            "by_group": {
                group: sum(group_values) / len(group_values)
                for group, group_values in sorted(by_group[condition].items())
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "workdirs/socialnav_sub/pilot_v1",
    )
    parser.add_argument(
        "--survey",
        type=Path,
        default=ROOT / "external/SocialNavSUB/data/survey_prompt.json",
    )
    parser.add_argument("--model", default="Gemini 3-Pro-Preview")
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=(
            "direct",
            "oracle_code",
            "visual_predicted_ps",
            "visual_predicted_ps_plus_frames",
        ),
        default=("direct", "oracle_code", "visual_predicted_ps"),
    )
    args = parser.parse_args()

    manifest = json.loads((args.pilot_root / "manifest.json").read_text(encoding="utf-8"))
    oracle_by_scene = {
        row["scene_id"]: row for row in load_jsonl(args.pilot_root / "compiled_oracle.jsonl")
    }
    survey = json.loads(args.survey.read_text(encoding="utf-8"))
    output = args.pilot_root / "jdcloud_pilot"
    output.mkdir(parents=True, exist_ok=True)
    compiled_path = output / "visual_compiled.jsonl"
    predictions_path = output / "predictions.jsonl"

    compiled_rows = load_jsonl(compiled_path)
    compiled_by_scene = {row["scene_id"]: row for row in compiled_rows if row.get("valid")}
    prediction_rows = load_jsonl(predictions_path)
    completed = {
        (row["condition"], row["question_key"], row["scene_id"])
        for row in prediction_rows
        if row.get("answer") is not None
    }

    config = load_service_config(model_override=args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)

    for selected in manifest["scenes"]:
        scene_id = selected["scene_id"]
        scene_root = args.pilot_root / "official" / scene_id
        images = [scene_root / f"sample_with_bev_{index}.png" for index in range(10)]
        if any(not path.is_file() or path.stat().st_size < 1024 for path in images):
            raise RuntimeError(f"scene {scene_id} images are not materialized")
        parts = image_parts(images)

        if scene_id not in compiled_by_scene:
            for compilation_attempt in range(1, 4):
                raw = client.generate_parts(
                    parts + [{"text": compile_prompt(scene_id)}],
                    max_output_tokens=8192,
                )
                try:
                    representation = json_value(raw)
                    valid = isinstance(representation, dict) and isinstance(
                        representation.get("people"), dict
                    )
                except ValueError:
                    representation, valid = None, False
                compiled_row = {
                    "scene_id": scene_id,
                    "query_independent": True,
                    "model": config.model,
                    "raw": raw,
                    "representation": representation,
                    "valid": valid,
                    "compilation_attempt": compilation_attempt,
                }
                append_jsonl(compiled_path, compiled_row)
                if valid:
                    compiled_by_scene[scene_id] = compiled_row
                    break
            if scene_id not in compiled_by_scene:
                raise RuntimeError(
                    f"invalid visual compilation for {scene_id} after 3 attempts"
                )

        labels = json.loads((scene_root / "common_answers.json").read_text(encoding="utf-8"))
        records = [
            question_record(scene_id, group, key, instructions, survey, labels)
            for group, key, instructions in QUESTION_SPECS
        ]
        visual_text = json.dumps(
            compiled_by_scene[scene_id]["representation"], ensure_ascii=False, sort_keys=True
        )
        oracle_text = None
        if "oracle_code" in args.conditions:
            oracle = oracle_by_scene[scene_id]
            if oracle.get("oracle_available") is False:
                raise RuntimeError(
                    f"oracle_code requested for uncompilable scene {scene_id}: "
                    f"{oracle.get('compile_error')}"
                )
            oracle_text = json.dumps(
                {
                    "spatial_code": oracle["spatial_code"],
                    "spatial_trace": oracle["spatial_trace"],
                    "social_code": oracle["social_code"],
                    "social_trace": oracle["social_trace"],
                },
                ensure_ascii=False,
            )

        for record in records:
            for condition in args.conditions:
                key = (condition, record["question_key"], scene_id)
                if key in completed:
                    continue
                representation = None
                if condition == "oracle_code":
                    assert oracle_text is not None
                    representation = oracle_text
                elif condition in VISUAL_CODE_CONDITIONS:
                    representation = visual_text
                prompt = answer_prompt(record, representation)
                if condition_uses_raw_frames(condition):
                    # Gemini 3 may spend part of this budget on internal
                    # reasoning even for a one-token answer; 256 occasionally
                    # truncates before the JSON value is emitted.
                    raw = client.generate_parts(parts + [{"text": prompt}], max_output_tokens=1024)
                else:
                    raw = client.generate(prompt, max_output_tokens=1024)
                if args.request_delay_seconds > 0:
                    time.sleep(args.request_delay_seconds)
                answer = normalize_answer(raw, record["choices"])
                row = {
                    **record,
                    "condition": condition,
                    "answer": answer,
                    "raw_response": raw,
                    "query_independent_code": condition != "direct",
                    "model": config.model,
                }
                append_jsonl(predictions_path, row)
                prediction_rows.append(row)
                completed.add(key)

        summary = summarize(prediction_rows)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"finished_scene": scene_id, "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
