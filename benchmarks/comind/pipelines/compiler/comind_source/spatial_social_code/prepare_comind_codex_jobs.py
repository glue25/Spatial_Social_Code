"""Prepare resumable Codex CLI jobs for shell-direct execution.

The managed environment permits the Codex CLI as a direct shell child but not
as a Python subprocess.  This helper materializes prompts between the four
paired stages; it never performs inference itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_social_code.run_comind_codex_pilot import (
    code_schema,
    evidence_prompt,
    normalize_code,
    prediction_schema,
)
from spatial_social_code.runtime import execute_social, execute_spatial


SPATIAL_PREFIX = """You are the grounded Spatial Code compiler. Use only visible facts
from the attached ordered frames. Do not infer intent or an answer. Do not call tools.
Every spatial_code array item must be one complete executable literal call. Use exact
signatures: S.fact(t, relation, subject, object, source='video', confidence=0.8);
S.event(t, actor, action, target, source='video', confidence=0.8);
S.visible(t, observer, object, value, source='video', confidence=0.8);
S.unknown(t, relation, subject, object). Fill arguments only from current evidence.\n\n"""

SOCIAL_PREFIX = """You are the agent-indexed Social Code compiler. Infer observations,
goals, intents, roles, action evidence, and uncertainty without choosing the answer.
Do not call tools. Every social_code item must be one executable literal call. Use:
M.observe(t, agent, fact, confidence=0.8); M.belief(t, agent, fact, confidence, evidence);
M.goal(agent, goal, confidence, evidence); M.intent(t, agent, intent, confidence, evidence);
M.action_evidence(t, agent, action, hypothesis, confidence, evidence);
M.social_goal(agent, target_agent, goal, confidence, evidence);
M.contradiction(agent, claim, source_a, source_b); M.unknown(agent, variable).\n\n"""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem(row: dict) -> str:
    return f"{row['task']}_{row['recording_id']}_{row['sample_id']}"


def write_jobs(work_root: Path, phase: str, manifest: list[dict]) -> dict:
    raw_root = work_root / "codex_work" / "raw"
    schema_root = work_root / "codex_work" / "schemas"
    prompt_root = work_root / "codex_work" / "prompts" / phase
    for directory in (raw_root, schema_root, prompt_root):
        directory.mkdir(parents=True, exist_ok=True)
    schemas = {"spatial": code_schema("spatial_code"), "social": code_schema("social_code")}
    schemas.update({task: prediction_schema(task) for task in ("joint_attention", "scoia", "handover")})
    for name, schema in schemas.items():
        (schema_root / f"{name}.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    jobs = []
    skipped = failed_dependency = 0
    for row in manifest:
        item_stem = stem(row)
        evidence = evidence_prompt(row)
        schema_name = row["task"]
        if phase == "direct":
            prompt = "You are a direct CoMind vision-language baseline. Use only the attached frames and transcript. Do not call tools, inspect files, or explain.\n\n" + evidence
            output = raw_root / f"{item_stem}_direct.json"
        elif phase == "spatial":
            prompt = SPATIAL_PREFIX + evidence
            output = raw_root / f"{item_stem}_spatial.json"
            schema_name = "spatial"
        elif phase == "social":
            spatial_output = raw_root / f"{item_stem}_spatial.json"
            if not spatial_output.is_file():
                failed_dependency += 1
                continue
            try:
                spatial = json.loads(spatial_output.read_text(encoding="utf-8"))
                spatial_code = normalize_code(spatial.get("spatial_code"), "S")
                spatial_trace = execute_spatial(spatial_code)
            except Exception:
                failed_dependency += 1
                continue
            prompt = SOCIAL_PREFIX + evidence + "\nExecuted Spatial Trace:\n" + spatial_trace
            output = raw_root / f"{item_stem}_social.json"
            schema_name = "social"
        elif phase == "answer":
            spatial_output = raw_root / f"{item_stem}_spatial.json"
            social_output = raw_root / f"{item_stem}_social.json"
            if not spatial_output.is_file() or not social_output.is_file():
                failed_dependency += 1
                continue
            try:
                spatial = json.loads(spatial_output.read_text(encoding="utf-8"))
                social = json.loads(social_output.read_text(encoding="utf-8"))
                spatial_code = normalize_code(spatial.get("spatial_code"), "S")
                social_code = normalize_code(social.get("social_code"), "M")
                spatial_trace = execute_spatial(spatial_code)
                social_trace = execute_social(social_code)
            except Exception:
                failed_dependency += 1
                continue
            prompt = "Use the attached evidence and executable Spatial-Social traces to make the final CoMind prediction. Raw evidence overrides an incorrect intermediate record. Do not call tools or explain.\n\n" + evidence + "\nSpatial Code:\n" + spatial_code + "\nSpatial Trace:\n" + spatial_trace + "\nSocial Code:\n" + social_code + "\nSocial Trace:\n" + social_trace
            output = raw_root / f"{item_stem}_answer.json"
        else:
            raise ValueError(phase)
        if output.is_file():
            skipped += 1
            continue
        prompt_path = prompt_root / f"{item_stem}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        jobs.append({
            "task": row["task"],
            "recording_id": row["recording_id"],
            "sample_id": row["sample_id"],
            "prompt": str(prompt_path.resolve()),
            "schema": str((schema_root / f"{schema_name}.json").resolve()),
            "output": str(output.resolve()),
            "images": [str(Path(image["path"]).resolve()) for image in row["images"]],
        })
    jobs_path = work_root / f"codex_jobs_{phase}.jsonl"
    jobs_path.write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs), encoding="utf-8")
    shell_jobs_path = work_root / f"codex_jobs_{phase}.tsv"
    shell_jobs_path.write_text(
        "".join(
            "\t".join((job["prompt"], job["schema"], job["output"], *job["images"])) + "\n"
            for job in jobs
        ),
        encoding="utf-8",
    )
    return {"phase": phase, "jobs": len(jobs), "already_complete": skipped, "failed_dependency": failed_dependency, "jobs_path": str(jobs_path), "shell_jobs_path": str(shell_jobs_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("direct", "spatial", "social", "answer"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = load_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(json.dumps(write_jobs(args.work_root, args.phase, rows), indent=2))


if __name__ == "__main__":
    main()
