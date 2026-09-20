#!/usr/bin/env python3
"""Run Direct and Dynamic SSC on executable asset-linked environment cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("direct", "dynamic_ssc")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=892156)
    parser.add_argument("--base-url", default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run only this case ID (repeatable).")
    parser.add_argument("--condition", action="append", choices=CONDITIONS, dest="conditions")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already contain one summary artifact.")
    args = parser.parse_args()
    tasks_dir = args.tasks.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = sorted(path for path in tasks_dir.glob("*.json") if path.name != "manifest.json")
    if len(tasks) != 4:
        raise SystemExit(f"expected four generated cases, found {len(tasks)}")
    if args.case_ids:
        requested = set(args.case_ids)
        known = {path.stem for path in tasks}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"unknown case IDs: {sorted(unknown)}; available={sorted(known)}")
        tasks_to_run = [path for path in tasks if path.stem in requested]
    else:
        tasks_to_run = tasks
    conditions = tuple(args.conditions or CONDITIONS)

    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer local-qwen"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        models = [row["id"] for row in json.loads(response.read()).get("data", [])]
    if args.model not in models:
        raise SystemExit(f"model service missing {args.model}: {models}")

    checkpoint = Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct")
    config = {
        "protocol": "ssc_generated_environment_cases_v1",
        "conditions": list(CONDITIONS),
        "selected_conditions": list(conditions),
        "selected_cases": [path.stem for path in tasks_to_run],
        "seed": args.seed,
        "model": args.model,
        "model_checkpoint": str(checkpoint),
        "tasks": [{"path": str(path), "sha256": sha256(path)} for path in tasks],
        "same_visual_environment": True,
        "max_environment_turns": 20,
        "strict_scorer": "strict_physical_v1",
    }
    config_path = output / "frozen_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    env_base = os.environ.copy()
    env_base.update({
        "PYTHONPATH": f"{ROOT}:{ROOT / 'external/EnactTom'}" + (f":{env_base['PYTHONPATH']}" if env_base.get("PYTHONPATH") else ""),
        "ENACTTOM_PYTHON": str(ROOT / ".venv-habitat/bin/python"),
        "ENACTTOM_EPISODES_PATH": str(ROOT / "workdirs/habitat_enacttom/frozen12/episodes.json.gz"),
        "OPENAI_API_KEY": "local-qwen",
        "OPENAI_BASE_URL": args.base_url,
        "ENACTTOM_BENCHMARK_SEED": str(args.seed),
        "SSC_MODEL": args.model,
        "SSC_MODEL_CHECKPOINT": str(checkpoint),
        "SSC_SEED": str(args.seed),
        "SSC_CONFIG_SHA256": sha256(config_path),
        "SSC_MAX_TURNS": "20",
        "SSC_STABLE_SAMPLES": "3",
        "SSC_PHYSICS_STEPS_PER_SAMPLE": "10",
        "SSC_PHYSICS_DT": str(1 / 60),
        "SSC_OPEN_THRESHOLD": "0.10",
    })
    status_path = output / "status.jsonl"
    for task in tasks_to_run:
        task_payload = json.loads(task.read_text(encoding="utf-8"))
        case_id = task_payload["ssc_generated_case"]["case_id"]
        for condition in conditions:
            run_dir = output / "results" / condition / case_id
            run_dir.mkdir(parents=True, exist_ok=True)
            completed = list(run_dir.glob("results/*/closed_loop_artifacts/summary.json"))
            if args.resume and len(completed) == 1:
                append(status_path, {
                    "case_id": case_id, "condition": condition, "seed": args.seed,
                    "status": "skipped_complete", "run_dir": str(run_dir),
                    "finished": datetime.now(timezone.utc).isoformat(),
                })
                continue
            env = dict(env_base)
            env["SSC_CLOSED_LOOP_CONDITION"] = condition
            cmd = [
                str(ROOT / "external/EnactTom/enacttom/run.sh"), "benchmark",
                "--task", str(task), "--output-dir", str(run_dir),
                "--model", args.model, "--llm", "openai_chat", "--num-times", "1",
                "--max-llm-calls", "20", "--benchmark-run-mode", "standard",
                "--observation-mode", "text", "--no-calibration", "--video",
            ]
            started = datetime.now(timezone.utc).isoformat()
            with (run_dir / "raw_console.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    cmd, cwd=ROOT / "external/EnactTom", env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            row = {
                "case_id": case_id, "condition": condition, "seed": args.seed,
                "returncode": process.returncode, "started": started,
                "finished": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
            }
            append(status_path, row)
            if process.returncode:
                raise SystemExit(f"failed {case_id}/{condition}: {run_dir / 'raw_console.log'}")


if __name__ == "__main__":
    main()
