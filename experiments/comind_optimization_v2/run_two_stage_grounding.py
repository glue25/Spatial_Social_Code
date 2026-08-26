#!/usr/bin/env python3
"""Identity-first CoMind grounding with independent SAM3 view proposals.

The VLM is used only to identify the concrete target object.  A text-prompted
SAM3 service then localizes that identity in each required prediction frame.
Generation is label-free; train annotations are read only by the separate
``evaluate_gpu_grounding.py`` scorer.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_social_code.run_comind_codex_pilot import call_local_qwen, write_jsonl


IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {"object_identity": {"type": "string"}},
    "required": ["object_identity"],
    "additionalProperties": False,
}


def prediction_frames(row: dict) -> dict[str, Path]:
    """Return exactly one prediction-frame path per available ego view."""
    result: dict[str, Path] = {}
    for image in row["images"]:
        if image.get("kind") == "prediction":
            role = str(image["role"])
            if role in result:
                raise ValueError(f"duplicate {role} prediction frame")
            result[role] = Path(image["path"]).resolve()
    required = {"helper"} if row["task"] == "scoia" else {"leader", "helper"}
    if not required.issubset(result):
        raise ValueError(f"missing prediction views: {sorted(required - set(result))}")
    return result


def identity_prompt(row: dict) -> str:
    legend = "\n".join(
        f"{index}. {image['role']} {image['kind']} frame at {float(image['time']):.3f}s"
        for index, image in enumerate(row["images"], 1)
    )
    target = {
        "joint_attention": "the concrete object jointly attended at the prediction frame",
        "scoia": "the concrete object the helper will interact with within 10 seconds",
        "handover": "the concrete object in the upcoming handover",
    }[row["task"]]
    return f"""Identify {target}. Use all frames and transcript, but output only its
concrete visible noun phrase. Preserve identity such as plate, bowl, rolling
pin, salt shaker, or oven mitt. Never replace it with an abstract superclass
such as food, dishware, container, tool, utensil, kitchenware, or vegetable.
Do not estimate coordinates: a separate detector will ground the noun in each
ego view.

Images in exact order:
{legend}

Preceding transcript:
{row.get('transcript_10s') or '[no speech]'}
"""


def encode_png(path: Path) -> str:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sam3_request(
    image_path: Path,
    text_prompt: str,
    service_url: str,
    timeout_seconds: float = 180.0,
) -> list[dict]:
    payload = json.dumps(
        {"image_base64": encode_png(image_path), "text_prompt": text_prompt}
    ).encode("utf-8")
    request = urllib.request.Request(
        service_url.rstrip("/") + "/segment",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"SAM3 request failed for {image_path}: {error}") from error
    proposals = result.get("results")
    if not isinstance(proposals, list):
        raise ValueError("SAM3 response has no results list")
    valid = []
    for proposal in proposals:
        box = proposal.get("box") if isinstance(proposal, dict) else None
        if isinstance(box, list) and len(box) == 4:
            valid.append(
                {
                    "box_xyxy": [float(value) for value in box],
                    "score": float(proposal.get("score", 0.0)),
                    "label": str(proposal.get("label", text_prompt)),
                }
            )
    return sorted(valid, key=lambda item: item["score"], reverse=True)


def select_box(proposals: list[dict]) -> list[float] | None:
    """Select the highest-confidence valid SAM proposal without labels."""
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["box_xyxy"]
        if 0 <= x1 < x2 <= 1408 and 0 <= y1 < y2 <= 1408:
            return proposal["box_xyxy"]
    return None


def ground_row(
    row: dict,
    identity: str,
    service_url: str,
    sam_fn: Callable[[Path, str, str], list[dict]] = sam3_request,
) -> tuple[dict, dict]:
    """Ground each view independently; never reuse one view's box."""
    frames = prediction_frames(row)
    proposals = {
        role: sam_fn(path, identity, service_url) for role, path in frames.items()
    }
    boxes = {role: select_box(items) for role, items in proposals.items()}
    prediction: dict[str, Any] = {"object_identity": identity}
    if row["task"] == "joint_attention":
        # CoMind's existing local protocol maps left/leader and right/helper.
        # The critical constraint is that each value comes from its own frame.
        prediction["left_bbox_xyxy"] = boxes["leader"]
        prediction["right_bbox_xyxy"] = boxes["helper"]
    elif row["task"] == "scoia":
        prediction["bbox_xyxy"] = boxes["helper"]
    else:
        # Ground both views.  The current benchmark adapter calls the requested
        # handover box the sender view but does not expose sender role in the
        # frozen manifest.  Preserve both proposals and use the best SAM score
        # as a label-free fallback for the single-field scorer.
        ranked = [
            (items[0]["score"], role, boxes[role])
            for role, items in proposals.items()
            if items and boxes[role] is not None
        ]
        best = max(ranked, default=(0.0, "missing", None))
        prediction["bbox_xyxy"] = best[2]
        prediction["selected_view"] = best[1]
        prediction["leader_bbox_xyxy"] = boxes.get("leader")
        prediction["helper_bbox_xyxy"] = boxes.get("helper")
    return prediction, proposals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("workdirs/comind_optimization_v2/gpu_grounding_manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workdirs/comind_optimization_v2/two_stage_grounding_predictions.jsonl"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("workdirs/comind_optimization_v2/two_stage_raw"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8114")
    parser.add_argument("--limit", type=int, default=10, help="Pilot row limit; use 0 for the full frozen manifest")
    parser.add_argument(
        "--per-task",
        type=int,
        help="Balanced pilot: select the first N frozen rows from each task; overrides --limit",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.per_task is not None:
        if args.per_task <= 0:
            parser.error("--per-task must be positive")
        rows = [
            row
            for task in ("joint_attention", "scoia", "handover")
            for row in [item for item in rows if item["task"] == task][: args.per_task]
        ]
    elif args.limit > 0:
        rows = rows[: args.limit]
    existing = []
    if args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {(str(row["task"]), str(row["recording_id"]), str(row["sample_id"])) for row in existing if not row.get("error")}
    args.raw_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = (str(row["task"]), str(row["recording_id"]), str(row["sample_id"]))
        if key in done:
            continue
        stem = "_".join(key)
        try:
            identity_result = call_local_qwen(
                identity_prompt(row),
                [Path(image["path"]).resolve() for image in row["images"]],
                IDENTITY_SCHEMA,
                args.raw_root / f"{stem}_identity.json",
                args.model_path,
            )
            identity = str(identity_result["object_identity"]).strip()
            if not identity:
                raise ValueError("empty object_identity")
            prediction, proposals = ground_row(row, identity, args.sam3_url)
            result = {
                "task": row["task"], "recording_id": row["recording_id"], "sample_id": row["sample_id"],
                "coordinate_space": "per_view_original_frame_pixels_1408x1408",
                "joint_scoring_space": "leader/helper half-scale concatenated 1408x704 mosaic",
                "bbox_format": "xyxy",
                "prediction": prediction, "sam3_proposals": proposals,
            }
        except Exception as error:
            result = {
                "task": row["task"], "recording_id": row["recording_id"], "sample_id": row["sample_id"],
                "error": f"{type(error).__name__}: {error}",
            }
        existing.append(result)
        write_jsonl(args.output, existing)
        if not result.get("error"):
            done.add(key)


if __name__ == "__main__":
    main()
