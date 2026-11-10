#!/usr/bin/env python3
"""Run planners inside a Habitat episode compiled from SSC code."""

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
CONDITIONS = ("direct", "static_ssc", "dynamic_ssc")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=893116)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    task = bundle / "task.json"
    episodes = bundle / "episodes.json.gz"
    code = bundle / "generated_ssc_code.json"
    manifest = json.loads((bundle / "compile_manifest.json").read_text(encoding="utf-8"))
    for path in (task, episodes, code):
        if not path.is_file():
            raise SystemExit(f"missing generated environment component: {path}")
    if not manifest.get("accepted") or not manifest.get("pose_changed"):
        raise SystemExit("generated environment did not pass compile checks")

    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer local-qwen"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        models = [row["id"] for row in json.loads(response.read()).get("data", [])]
    if args.model not in models:
        raise SystemExit(f"model service missing {args.model}: {models}")

    conditions = tuple(args.condition or CONDITIONS)
    checkpoint = Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct")
    config = {
        "protocol": "ssc_code_to_environment_execution_v1",
        "generated_episode_id": manifest["generated_episode_id"],
        "conditions": list(conditions),
        "seed": args.seed,
        "max_turns": args.max_turns,
        "model": args.model,
        "bundle_hashes": {p.name: sha256(p) for p in (task, episodes, code)},
        "strict_scorer": "strict_physical_v1",
    }
    config_path = output / "frozen_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    env_base = os.environ.copy()
    env_base.update({
        "PYTHONPATH": f"{ROOT}:{ROOT / 'external/EnactTom'}" + (f":{env_base['PYTHONPATH']}" if env_base.get("PYTHONPATH") else ""),
        "ENACTTOM_PYTHON": str(ROOT / ".venv-habitat/bin/python"),
        "ENACTTOM_EPISODES_PATH": str(episodes),
        "OPENAI_API_KEY": "local-qwen",
        "OPENAI_BASE_URL": args.base_url,
        "ENACTTOM_BENCHMARK_SEED": str(args.seed),
        "SSC_MODEL": args.model,
        "SSC_MODEL_CHECKPOINT": str(checkpoint),
        "SSC_SEED": str(args.seed),
        "SSC_CONFIG_SHA256": sha256(config_path),
        "SSC_MAX_TURNS": str(args.max_turns),
        "SSC_STABLE_SAMPLES": "3",
        "SSC_PHYSICS_STEPS_PER_SAMPLE": "10",
        "SSC_PHYSICS_DT": str(1 / 60),
        "SSC_OPEN_THRESHOLD": "0.10",
    })
    status_path = output / "status.jsonl"
    for condition in conditions:
        run_dir = output / "results" / condition
        run_dir.mkdir(parents=True, exist_ok=True)
        completed = list(run_dir.glob("results/*/closed_loop_artifacts/summary.json"))
        if args.resume and len(completed) == 1:
            append(status_path, {"condition": condition, "status": "skipped_complete"})
            continue
        env = dict(env_base)
        env["SSC_CLOSED_LOOP_CONDITION"] = condition
        cmd = [
            str(ROOT / "external/EnactTom/enacttom/run.sh"), "benchmark",
            "--task", str(task), "--output-dir", str(run_dir),
            "--model", args.model, "--llm", "openai_chat", "--num-times", "1",
            "--max-llm-calls", str(args.max_turns), "--benchmark-run-mode", "standard",
            "--observation-mode", "text", "--no-calibration", "--video",
        ]
        started = datetime.now(timezone.utc).isoformat()
        with (run_dir / "raw_console.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                cmd, cwd=ROOT / "external/EnactTom", env=env,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        append(status_path, {
            "condition": condition, "seed": args.seed, "returncode": process.returncode,
            "started": started, "finished": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
        })
        if process.returncode:
            raise SystemExit(f"generated environment run failed: {condition}; see {run_dir / 'raw_console.log'}")


if __name__ == "__main__":
    main()
