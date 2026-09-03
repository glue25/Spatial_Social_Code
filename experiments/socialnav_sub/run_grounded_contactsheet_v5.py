#!/usr/bin/env python3
"""Evaluate query-independent grounded Spatial-Social Code on SocialNav-SUB.

The same two contact sheets (ten ordered RGB views and ten ordered BEV views)
are used by the matched Direct arm and the scene compiler.  The compiler runs
once per scene without seeing any question or choices; its typed fields are
then executed deterministically for all questions in the scene.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MA-EgoQA"))

from egomas.src.jdcloud_gemini import JDCloudGeminiClient, load_service_config  # noqa: E402

from experiments.socialnav_sub.run_pilot_jdcloud import (  # noqa: E402
    QUESTION_SPECS,
    append_jsonl,
    json_value,
    load_jsonl,
    normalize_answer,
    question_record,
    summarize,
)


GROUNDING_RULES = """Evidence-reading rules:
1. The upper contact sheet contains the robot-camera views; the lower contact sheet contains bird's-eye views. Columns are ordered t=0..4 on row one and t=5..9 on row two.
2. In the bird's-eye view, R is the robot, G is its goal, numbered circles are people, and arrows show heading. Track a person by the printed number, never by clothing alone.
3. For begin/end questions, use the first/last frame in which that numbered person and R co-occur. Do not invent a reappearance after the person leaves view.
4. Express left/right/ahead/behind in the robot's facing coordinate frame at that evidence frame. Use the R arrow, not page coordinates or the camera's apparent side alone. Apply a corridor rule: if the person is laterally outside a corridor about one marker-diameter wide through R, choose left/right even when the person is also somewhat ahead; choose ahead only inside that forward corridor. Choose behind when the person is clearly opposite the R arrow.
5. Compute distance change from the R-to-person separation at the first and last co-visible frames.
6. The robot is affected only when the person's path or occupancy plausibly changes the robot's path toward G; mere visibility or proximity is insufficient.
7. Before returning JSON, check that each position value agrees with the stated social explanation (for example, an explanation saying "on the left side" must not be paired with "ahead of").
"""


def _sheet(images: list[Image.Image], crop: str, tile_width: int = 320) -> bytes:
    tiles: list[Image.Image] = []
    for index, image in enumerate(images):
        width, height = image.size
        if crop == "camera":
            region = image.crop((0, 0, width, int(height * 0.36)))
        elif crop == "bev":
            region = image.crop((0, int(height * 0.36), width, height))
        else:
            raise ValueError(crop)
        tile_height = round(region.height * tile_width / region.width)
        tile = region.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 44, 22), fill="white")
        draw.text((4, 3), f"t={index}", fill="black")
        tiles.append(tile)

    tile_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (tile_width * 5, tile_height * 2), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 5) * tile_width, (index // 5) * tile_height))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue()


def contact_sheet_parts(paths: list[Path]) -> list[dict[str, Any]]:
    images = [Image.open(path).convert("RGB") for path in paths]
    parts: list[dict[str, Any]] = []
    for label, crop in (("ordered robot-camera contact sheet", "camera"), ("ordered bird's-eye contact sheet", "bev")):
        encoded = base64.b64encode(_sheet(images, crop)).decode("ascii")
        parts.extend(
            [
                {"text": label},
                {"inline_data": {"mimeType": "image/jpeg", "data": encoded}},
            ]
        )
    return parts


def compiler_prompt(scene_id: str) -> str:
    return f"""Compile reusable Spatial-Social Code for scene {scene_id}.
No question or answer choices are available. Analyze every numbered person so the result can answer later spatial, temporal, and social questions.

{GROUNDING_RULES}
Return JSON only:
{{
  "scene_id": "{scene_id}",
  "people": {{
    "person_1": {{
      "first_visible_frame": 0,
      "last_visible_frame": 9,
      "begin_position": {{"value": "ahead of|to the left of|to the right of|behind|unknown", "confidence": 0.0}},
      "end_position": {{"value": "ahead of|to the left of|to the right of|behind|unknown", "confidence": 0.0}},
      "distance_change": {{"value": "closer to|further away from|about the same distance to|unknown", "confidence": 0.0}},
      "robot_affected": {{"value": "yes|no|unknown", "confidence": 0.0}},
      "social_explanation": "short evidence-grounded description"
    }}
  }},
  "uncertainties": []
}}
Include one entry for every visible numbered person. Confidence must be between 0 and 1. Do not emit prose outside JSON."""


def direct_prompt(record: dict[str, Any]) -> str:
    choices = json.dumps(record["choices"], ensure_ascii=False)
    return (
        GROUNDING_RULES
        + "\n"
        + record["instructions"]
        + "\n\n"
        + record["question"]
        + f"\nPossible answers: {choices}\n"
        + 'Return JSON only as {"answer": "one exact possible answer"}.'
    )


def _value(person: dict[str, Any], field: str) -> str | None:
    value = person.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, str) else None


def execute_code(record: dict[str, Any], representation: dict[str, Any]) -> str | None:
    person = representation.get("people", {}).get("person_1", {})
    key = record["question_key"]
    if "spatial_position_end" in key:
        answer = _value(person, "end_position")
    elif "distance_change" in key:
        answer = _value(person, "distance_change")
    elif "robot_affected" in key:
        answer = _value(person, "robot_affected")
    else:
        answer = None
    return answer if answer in record["choices"] else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--request-delay-seconds", type=float, default=70.0)
    parser.add_argument("--model", default="Gemini 3-Pro-Preview")
    args = parser.parse_args()

    manifest = json.loads((args.pilot_root / "manifest.json").read_text())
    scenes = manifest["scenes"][: args.max_scenes]
    survey = json.loads((ROOT / "external/SocialNavSUB/data/survey_prompt.json").read_text())
    output = args.pilot_root / "jdcloud_grounded_v5"
    compiled_path = output / "compiled.jsonl"
    predictions_path = output / "predictions.jsonl"
    compiled = {r["scene_id"]: r for r in load_jsonl(compiled_path) if r.get("valid")}
    rows = load_jsonl(predictions_path)
    completed = {(r["condition"], r["scene_id"], r["question_key"]) for r in rows}

    config = load_service_config(model_override=args.model)
    client = JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5)

    for selected in scenes:
        scene_id = selected["scene_id"]
        scene_root = args.pilot_root / "official" / scene_id
        paths = [scene_root / f"sample_with_bev_{index}.png" for index in range(10)]
        parts = contact_sheet_parts(paths)
        if scene_id not in compiled:
            # Gemini 3 may consume a substantial part of the response budget
            # as internal reasoning even though the emitted JSON is short.
            raw = client.generate_parts(parts + [{"text": compiler_prompt(scene_id)}], max_output_tokens=8192)
            try:
                representation = json_value(raw)
                valid = isinstance(representation, dict) and isinstance(representation.get("people"), dict)
            except ValueError:
                representation, valid = None, False
            row = {
                "scene_id": scene_id,
                "condition": "grounded_ps_runtime",
                "query_independent": True,
                "model": config.model,
                "representation": representation,
                "raw_response": raw,
                "valid": valid,
            }
            append_jsonl(compiled_path, row)
            if not valid:
                raise RuntimeError(f"invalid compiler response for {scene_id}")
            compiled[scene_id] = row
            time.sleep(args.request_delay_seconds)

        labels = json.loads((scene_root / "common_answers.json").read_text())
        records = [
            question_record(scene_id, group, key, instructions, survey, labels)
            for group, key, instructions in QUESTION_SPECS
        ]
        representation = compiled[scene_id]["representation"]
        for record in records:
            runtime_key = ("grounded_ps_runtime", scene_id, record["question_key"])
            if runtime_key not in completed:
                row = {
                    **record,
                    "condition": "grounded_ps_runtime",
                    "answer": execute_code(record, representation),
                    "query_independent_code": True,
                    "model": config.model,
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed.add(runtime_key)

            direct_key = ("grounded_direct", scene_id, record["question_key"])
            if direct_key not in completed:
                raw = client.generate_parts(parts + [{"text": direct_prompt(record)}], max_output_tokens=1024)
                answer = normalize_answer(raw, record["choices"])
                row = {
                    **record,
                    "condition": "grounded_direct",
                    "answer": answer,
                    "raw_response": raw,
                    "query_independent_code": False,
                    "model": config.model,
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed.add(direct_key)
                time.sleep(args.request_delay_seconds)

        summary = summarize(rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"finished_scene": scene_id, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
