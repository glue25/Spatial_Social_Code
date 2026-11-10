"""Run a paired direct vs Spatial-Social CoMind pilot through Codex CLI.

This is an exploratory same-model comparison. The paper's Qwen3-VL results
remain the official reference; this runner does not relabel Codex as an
official CoMind baseline.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spatial_social_code.runtime import UnsafeProgram, execute_social, execute_spatial


TASK_INSTRUCTIONS = {
    "joint_attention": """Predict the jointly attended object at the prediction
frame. Return its [x,y,width,height] bounding box separately in the leader and
helper views, all eliciting cue types, and the level-1 object category. Frames
are 1408x1408; boxes use original-frame pixels and are [x,y,width,height], not
[x1,y1,x2,y2]. Cue types must be selected only from gaze, action, verbal, and
gestural.""",
    "scoia": """From the helper's observations up to the prediction frame,
anticipate the socially conditioned action the helper will perform within 10
seconds. Predict the future interacted object's [x,y,width,height] box, cue
types, action verb, and level-1 action noun. Frames are 1408x1408; boxes use
original-frame pixels and are [x,y,width,height], not [x1,y1,x2,y2]. Cue types
must be selected only from verbal, gestural, and implicit.""",
    "handover": """Anticipate the upcoming handover. Predict the object's
[x,y,width,height] box in the sender view, delivery flow (ltr or rtl),
initiator (left or right), initiation cue types, level-1 category, and seconds
from the prediction frame to handover. Frames are 1408x1408; boxes use
original-frame pixels and are [x,y,width,height], not [x1,y1,x2,y2]. Initiation
types must be selected only from verbal, gestural, and implicit.""",
}


def prediction_schema(task: str) -> dict[str, Any]:
    bbox = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": (
            "Original 1408x1408 pixel box [x,y,width,height]. The third and "
            "fourth values are size, never bottom-right x2/y2 coordinates."
        ),
    }
    cues = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    if task == "joint_attention":
        properties = {
            "left_bbox": bbox,
            "right_bbox": bbox,
            "cue_type": cues,
            "category_l1": {"type": "string"},
        }
    elif task == "scoia":
        properties = {
            "bbox": bbox,
            "cue_type": cues,
            "action_verb": {"type": "string"},
            "action_noun_l1": {"type": "string"},
        }
    elif task == "handover":
        properties = {
            "bbox": bbox,
            "delivery_flow": {"type": "string"},
            "initiator": {"type": "string"},
            "initiation_type": cues,
            "category_l1": {"type": "string"},
            "time_to_handover": {"type": "number"},
        }
    else:
        raise ValueError(task)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def code_schema(name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
            }
        },
        "required": [name],
        "additionalProperties": False,
    }


def image_legend(row: dict) -> str:
    lines = ["Attached images, in exact order:"]
    for index, image in enumerate(row["images"], start=1):
        lines.append(
            f"{index}. {image['role']} {image['kind']} frame at {image['time']:.3f}s"
        )
    return "\n".join(lines)


def evidence_prompt(row: dict) -> str:
    return f"""Task: {row['task']}
{TASK_INSTRUCTIONS[row['task']]}

{image_legend(row)}

Transcript from the preceding 10 seconds:
{row.get('transcript_10s') or '[no speech]'}
"""


def normalize_code(lines: Any, variable: str) -> str:
    allowed = (
        {"fact", "event", "visible", "unknown"}
        if variable == "S"
        else {
            "observe",
            "belief",
            "goal",
            "intent",
            "action_evidence",
            "social_goal",
            "contradiction",
            "unknown",
        }
    )
    executor = execute_spatial if variable == "S" else execute_social
    kept: list[str] = []
    for raw in lines if isinstance(lines, list) else []:
        line = str(raw).strip()
        try:
            tree = ast.parse(line)
            call = tree.body[0].value
            valid = (
                len(tree.body) == 1
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == variable
                and call.func.attr in allowed
            )
            if not valid:
                continue
            executor(line)
        except (SyntaxError, IndexError, UnsafeProgram, TypeError, ValueError):
            continue
        if line not in kept:
            kept.append(line)
    if kept:
        return "\n".join(kept)
    return (
        "S.unknown(0, 'relevant_relation', 'scene', 'target')"
        if variable == "S"
        else "M.unknown('multi_agent_group', 'social_state')"
    )


def call_codex(
    prompt: str,
    images: list[Path],
    schema_path: Path,
    output_path: Path,
    sandbox_root: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    max_retries: int = 5,
    backoff_base_seconds: float = 2.0,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    command = [
        "codex",
        "exec",
        prompt,
        "--ephemeral",
        "--ignore-rules",
        "--disable",
        "apps",
        "--disable",
        "goals",
        "--disable",
        "shell_snapshot",
        "--disable",
        "plugins",
        "--disable",
        "unified_exec",
        "--disable",
        "code_mode_host",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(sandbox_root),
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    for image in images:
        command.extend(["--image", str(image)])
    attempt_log = output_path.with_suffix(output_path.suffix + ".attempts.log")
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            last_error = (completed.stderr or completed.stdout)[-4000:]
            attempt_log.parent.mkdir(parents=True, exist_ok=True)
            with attempt_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "attempt": attempt + 1,
                            "returncode": completed.returncode,
                            "tail": last_error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if completed.returncode == 0 and output_path.is_file():
                return json.loads(output_path.read_text(encoding="utf-8"))
            transient = completed.returncode in {429, 500, 502, 503, 504} or any(
                marker in last_error.lower()
                for marker in (
                    "429",
                    "rate limit",
                    "too many requests",
                    "temporar",
                    "timeout",
                    "timed out",
                    "overloaded",
                    "connection reset",
                    "502",
                    "503",
                    "504",
                )
            )
            if not transient or attempt >= max_retries:
                break
        except subprocess.TimeoutExpired as error:
            last_error = f"timeout after {error.timeout}s"
            with attempt_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"attempt": attempt + 1, "timeout": error.timeout}) + "\n")
            if attempt >= max_retries:
                break
        delay = min(60.0, backoff_base_seconds * (2**attempt))
        time.sleep(delay)

    # Some providers complete simple structured responses but disconnect on
    # schemas containing several heterogeneous fields.  Preserve the exact
    # model, images, and task prompt, then fall back to plain JSON text and
    # validate it locally against the same required-key contract.  This is a
    # transport fallback, not a different inference condition.
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fallback_output = output_path.with_suffix(output_path.suffix + ".unstructured.txt")
    fallback_prompt = (
        prompt
        + "\n\nReturn only one JSON object matching this schema; no markdown or prose:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    fallback_command = [
        "codex",
        "exec",
        fallback_prompt,
        "--ephemeral",
        "--ignore-rules",
        "--disable",
        "apps",
        "--disable",
        "goals",
        "--disable",
        "shell_snapshot",
        "--disable",
        "plugins",
        "--disable",
        "unified_exec",
        "--disable",
        "code_mode_host",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(sandbox_root),
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-last-message",
        str(fallback_output),
    ]
    for image in images:
        fallback_command.extend(["--image", str(image)])
    for attempt in range(max_retries + 1):
        try:
            completed = subprocess.run(
                fallback_command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            last_error = (completed.stderr or completed.stdout)[-4000:]
            with attempt_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "mode": "unstructured_json_fallback",
                            "attempt": attempt + 1,
                            "returncode": completed.returncode,
                            "tail": last_error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if completed.returncode == 0 and fallback_output.is_file():
                result = _json_object(fallback_output.read_text(encoding="utf-8"))
                required = set(schema.get("required", []))
                if set(result) != required:
                    raise ValueError(
                        f"fallback JSON keys {sorted(result)} do not match {sorted(required)}"
                    )
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                return result
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = f"invalid fallback JSON: {error}"
        except subprocess.TimeoutExpired as error:
            last_error = f"fallback timeout after {error.timeout}s"
        if attempt < max_retries:
            time.sleep(min(60.0, backoff_base_seconds * (2**attempt)))
    raise RuntimeError(
        f"codex exec structured and JSON fallback failed after "
        f"{2 * (max_retries + 1)} attempts: {last_error[-2000:]}"
    )


def _json_object(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("local model did not return a JSON object")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("local model response is not a JSON object")
    return value


def call_local_qwen(
    prompt: str,
    images: list[Path],
    schema: dict,
    output_path: Path,
    model_path: Path,
) -> dict:
    """Run the same paired protocol through the shared local Qwen3-VL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))

    os.environ["SPATIAL_SOCIAL_MODEL_PATH"] = str(model_path)
    os.environ["SPATIAL_SOCIAL_LOCAL_FILES_ONLY"] = "1"
    shared_model = Path(__file__).resolve().parents[1] / "third_party/autotom" / "model"
    if str(shared_model) not in sys.path:
        sys.path.insert(0, str(shared_model))
    from local_backend import generate, set_image_context

    set_image_context(images)
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    instruction = (
        prompt
        + "\n\nReturn only one valid JSON object matching this exact schema. "
        + "Do not use markdown or add explanations.\nSchema:\n"
        + schema_text
    )
    last_error: Exception | None = None
    raw = ""
    for attempt in range(3):
        raw = generate(instruction, max_new_tokens=512)
        output_path.with_suffix(output_path.suffix + f".attempt{attempt + 1}.txt").write_text(
            raw + "\n", encoding="utf-8"
        )
        try:
            result = _json_object(raw)
            required = set(schema.get("required", []))
            if set(result) != required:
                raise ValueError(
                    f"JSON keys {sorted(result)} do not match {sorted(required)}"
                )
            output_path.write_text(
                json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return result
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = error
            instruction += (
                "\n\nYour previous response was invalid:\n"
                + raw
                + "\nFix it and return only the required JSON object."
            )
    raise ValueError("local model failed structured output after 3 attempts") from last_error


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def prediction_key(row: dict) -> tuple[str, str, str, str]:
    """Stable identity used to resume without accumulating duplicate failures."""
    return (
        str(row["condition"]),
        str(row["task"]),
        str(row["recording_id"]),
        str(row["sample_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workdirs/comind_pilot/codex_predictions.jsonl"),
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--backend", choices=("codex", "local-qwen"), default="codex")
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base-seconds", type=float, default=2.0)
    args = parser.parse_args()

    def invoke(prompt: str, images: list[Path], schema_name: str, output: Path):
        if args.backend == "local-qwen":
            return call_local_qwen(
                prompt, images, schemas[schema_name], output, args.local_model_path
            )
        return call_codex(
            prompt,
            images,
            schema_paths[schema_name],
            output,
            sandbox_root,
            args.model,
            args.reasoning_effort,
            args.timeout_seconds,
            args.max_retries,
            args.backoff_base_seconds,
        )

    work_root = args.output.parent / "codex_work"
    sandbox_root = work_root / "sandbox"
    schema_root = work_root / "schemas"
    raw_root = work_root / "raw"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    schema_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    schemas = {
        "spatial": code_schema("spatial_code"),
        "social": code_schema("social_code"),
        **{task: prediction_schema(task) for task in TASK_INSTRUCTIONS},
    }
    schema_paths = {}
    for name, schema in schemas.items():
        path = schema_root / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        schema_paths[name] = path.resolve()

    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    existing = []
    if args.output.is_file():
        existing = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # Keep only the newest record for each condition/sample. Failed records remain
    # retryable, and a successful retry replaces the prior error in-place.
    rows_by_key = {prediction_key(row): row for row in existing}
    completed_keys = {
        key for key, row in rows_by_key.items() if "error" not in row
    }

    def save(row: dict) -> None:
        key = prediction_key(row)
        rows_by_key[key] = row
        if "error" not in row:
            completed_keys.add(key)
        write_jsonl(args.output, list(rows_by_key.values()))

    for sample in manifest:
        task = sample["task"]
        stem = f"{task}_{sample['recording_id']}_{sample['sample_id']}"
        images = [Path(item["path"]).resolve() for item in sample["images"]]
        evidence = evidence_prompt(sample)
        base = {
            "task": task,
            "recording_id": sample["recording_id"],
            "sample_id": sample["sample_id"],
            "model": (
                args.model
                if args.backend == "codex"
                else f"local:{args.local_model_path.name}"
            ),
        }
        direct_key = ("direct", task, sample["recording_id"], sample["sample_id"])
        if direct_key not in completed_keys:
            try:
                prediction = invoke(
                    "You are a direct CoMind vision-language baseline. Use only the "
                    "attached frames and transcript. Do not call tools, inspect files, "
                    "or produce explanations.\n\n" + evidence,
                    images,
                    task,
                    raw_root / f"{stem}_direct.json",
                )
                save({**base, "condition": "direct", "prediction": prediction})
            except Exception as error:  # keep resumable experiment records
                save({**base, "condition": "direct", "error": str(error)})

        method_key = (
            "spatial_social",
            task,
            sample["recording_id"],
            sample["sample_id"],
        )
        if method_key in completed_keys:
            continue
        try:
            spatial_result = invoke(
                "You are the grounded Spatial Code compiler. Use only visible facts "
                "from the attached ordered frames. Do not infer intent or an answer. "
                "Do not call tools. Every spatial_code array item must be one complete "
                "executable literal call, never a bare method name or prose. Use exact "
                "signatures: S.fact(t, relation, subject, object, source='video', "
                "confidence=0.8); S.event(t, actor, action, target, source='video', "
                "confidence=0.8); S.visible(t, observer, object, value, "
                "source='video', confidence=0.8); S.unknown(t, relation, subject, "
                "object). Here t is an integer frame index. In S.fact the second "
                "argument is always a relation such as 'near' or 'holds', followed "
                "by subject and object. Fill arguments only from the current evidence; "
                "do not copy placeholder entities.\n\n" + evidence,
                images,
                "spatial",
                raw_root / f"{stem}_spatial.json",
            )
            spatial_code = normalize_code(spatial_result.get("spatial_code"), "S")
            spatial_trace = execute_spatial(spatial_code)
            social_result = invoke(
                "You are the agent-indexed Social Code compiler. Infer observations, "
                "goals, intents, roles, action evidence, and uncertainty without "
                "choosing the task answer. Do not call tools. Every social_code array "
                "item must be one complete executable literal call, never 'M.method:' "
                "prose. Use exact signatures: M.observe(t, agent, fact, "
                "confidence=0.8); M.belief(t, agent, fact, confidence, evidence); "
                "M.goal(agent, goal, confidence, evidence); M.intent(t, agent, intent, "
                "confidence, evidence); M.action_evidence(t, agent, action, hypothesis, "
                "confidence, evidence); M.social_goal(agent, target_agent, goal, "
                "confidence, evidence); M.contradiction(agent, claim, source_a, "
                "source_b); M.unknown(agent, variable). Fill all arguments only from "
                "the current evidence and Spatial Trace.\n\n"
                + evidence
                + "\nExecuted Spatial Trace:\n"
                + spatial_trace,
                images,
                "social",
                raw_root / f"{stem}_social.json",
            )
            social_code = normalize_code(social_result.get("social_code"), "M")
            social_trace = execute_social(social_code)
            prediction = invoke(
                "Use the attached raw evidence and executable Spatial-Social traces "
                "to make the final CoMind prediction. Raw evidence overrides an "
                "incorrect intermediate record. Do not call tools or explain.\n\n"
                + evidence
                + "\nSpatial Code:\n"
                + spatial_code
                + "\nSpatial Trace:\n"
                + spatial_trace
                + "\nSocial Code:\n"
                + social_code
                + "\nSocial Trace:\n"
                + social_trace,
                images,
                task,
                raw_root / f"{stem}_answer.json",
            )
            save(
                {
                    **base,
                    "condition": "spatial_social",
                    "spatial_code": spatial_code,
                    "spatial_trace": spatial_trace,
                    "social_code": social_code,
                    "social_trace": social_trace,
                    "prediction": prediction,
                }
            )
        except Exception as error:  # keep resumable experiment records
            save({**base, "condition": "spatial_social", "error": str(error)})


if __name__ == "__main__":
    main()
