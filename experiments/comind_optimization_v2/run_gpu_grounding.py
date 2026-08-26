#!/usr/bin/env python3
"""Run the frozen identity-preserving grounding probe when a GPU is available.

This deliberately emits candidate predictions only. Evaluation against train
annotations is a separate post-generation step, preventing gold-conditioned
inference. It uses the repository's existing local Qwen backend and checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_social_code.run_comind_codex_pilot import call_local_qwen, write_jsonl


def schema(task: str) -> dict:
    box = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    properties = {"object_identity": {"type": "string"}}
    if task == "joint_attention":
        properties.update({"left_bbox_xyxy": box, "right_bbox_xyxy": box})
    else:
        properties["bbox_xyxy"] = box
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def prompt(row: dict) -> str:
    legend = "\n".join(f"{i}. {image['role']} {image['kind']} at {image['time']:.3f}s" for i, image in enumerate(row["images"], 1))
    target = {
        "joint_attention": "the concrete object jointly attended at the prediction frame",
        "scoia": "the concrete future object the helper will interact with within 10 seconds",
        "handover": "the concrete object in the upcoming handover",
    }[row["task"]]
    return f"""Ground {target}. Use the attached images and transcript only.
Preserve the concrete object identity (for example plate, bowl, rolling pin, or
oven mitt). Never replace it with an abstract superclass such as food,
dishware, container, tool, utensil, kitchenware, or vegetable. Return bounding
boxes as [x1,y1,x2,y2] in ORIGINAL 1408x1408 frame pixels, not resized-model
coordinates and not [x,y,width,height].

Images in order:
{legend}

Preceding transcript:
{row.get('transcript_10s') or '[no speech]'}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("workdirs/comind_optimization_v2/gpu_grounding_predictions.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"))
    args = parser.parse_args()
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise SystemExit("No NVIDIA GPU is currently available; manifest remains frozen and no predictions were written.")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = []
    if args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {(row["task"], row["recording_id"], row["sample_id"]) for row in existing}
    raw_root = args.output.parent / "gpu_grounding_raw"
    for row in rows:
        key = (row["task"], row["recording_id"], row["sample_id"])
        if key in done:
            continue
        images = [Path(image["path"]).resolve() for image in row["images"]]
        output = raw_root / f"{row['task']}_{row['recording_id']}_{row['sample_id']}.json"
        prediction = call_local_qwen(prompt(row), images, schema(row["task"]), output, args.model_path)
        existing.append({"task": row["task"], "recording_id": row["recording_id"], "sample_id": row["sample_id"], "coordinate_space": "original_frame_pixels_1408x1408", "bbox_format": "xyxy", "prediction": prediction})
        write_jsonl(args.output, existing)
        done.add(key)


if __name__ == "__main__":
    main()
