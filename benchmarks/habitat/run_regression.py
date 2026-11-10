#!/usr/bin/env python3
"""Run the frozen D/T/C Habitat smoke or 36-rollout engineering regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("direct", "static_ssc", "dynamic_ssc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def service_ready(base_url: str, model: str) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer local-qwen"})
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        names = [str(item.get("id")) for item in payload.get("data", [])]
        return model in names, f"available_models={names}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "dev36"), default="smoke")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--model-checkpoint", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--seed", type=int, default=891011)
    parser.add_argument("--allow-missing-preflight", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "workdirs/habitat_closed_loop" / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = sorted((root / "workdirs/habitat_enacttom/frozen12/tasks").glob("*.json"))
    if args.mode == "smoke":
        tasks = [path for path in tasks if path.name.endswith("_1566.json")][:1]
    if (args.mode == "dev36" and len(tasks) != 12) or not tasks:
        raise SystemExit(f"expected {'12' if args.mode == 'dev36' else '1'} tasks, found {len(tasks)}")

    ready, service_detail = service_ready(args.base_url, args.model)
    gpu_probe = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    checkpoint = args.model_checkpoint.resolve()
    checkpoint_files = [checkpoint / "config.json", checkpoint / "model.safetensors.index.json"]
    checkpoint_ready = checkpoint.is_dir() and all(path.is_file() for path in checkpoint_files)
    candidate = {
        "protocol": "habitat_closed_loop_engineering_v2",
        "frozen": args.mode == "dev36",
        "mode": args.mode,
        "conditions": list(CONDITIONS),
        "task_count": len(tasks),
        "seed": args.seed,
        "model": args.model,
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_metadata_hashes": {
            path.name: sha256(path) for path in checkpoint_files if path.is_file()
        },
        "model_base_url": args.base_url,
        "observation_mode": "text",
        "max_environment_turns": 20,
        "max_llm_calls_cli_note": "EnactToM --max-llm-calls maps to max environment turns and per-agent replanning threshold; it is not a total API-call cap.",
        "spatial_contract": {
            "version": "spatial_evidence_v2",
            "profile": "habitat_execution",
            "metric_pose_required": False,
            "message_success_requires_transport_confirmation": True,
            "initial_relation_requires_explicit_starts_marker": True,
        },
        "strict_scorer": {"version": "strict_physical_v2", "stable_samples": 3, "physics_steps_per_sample": 10, "physics_dt": 1 / 60, "open_threshold": 0.10},
        "code_hashes": {
            str(path.relative_to(root)): sha256(path)
            for path in [
                root / "benchmarks/habitat/action_interface.py",
                root / "benchmarks/habitat/state_store.py",
                root / "benchmarks/habitat/observation_adapter.py",
                root / "benchmarks/habitat/updaters.py",
                root / "benchmarks/habitat/runtime.py",
                root / "benchmarks/habitat/strict_scorer.py",
                root / "spatial_social_code/runtime.py",
                root / "spatial_social_code/spatial_contract.py",
                root / "external/EnactTom/enacttom/runner/base.py",
                root / "external/EnactTom/enacttom/runner/benchmark.py",
                root / "external/EnactTom/enacttom/planner.py",
                root / "external/EnactTom/habitat_llm/agent/env/environment_interface.py",
                root / "external/EnactTom/habitat_llm/tools/perception/communication_tool.py",
            ]
        },
        "tasks": [{"path": str(path), "sha256": sha256(path)} for path in tasks],
        "service_preflight": {"ready": ready, "detail": service_detail},
        "gpu_preflight": {"ready": gpu_probe.returncode == 0, "stdout": gpu_probe.stdout, "stderr": gpu_probe.stderr},
        "checkpoint_preflight": {"ready": checkpoint_ready},
    }
    candidate_path = output / ("frozen_config.json" if args.mode == "dev36" else "candidate_config.json")
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    candidate_sha256 = sha256(candidate_path)
    all_ready = ready and gpu_probe.returncode == 0 and checkpoint_ready
    if not all_ready and not args.allow_missing_preflight:
        append_jsonl(output / "status.jsonl", {
            "time": datetime.now(timezone.utc).isoformat(), "status": "environment_blocked",
            "reason": "gpu_model_service_or_checkpoint_unavailable",
            "detail": {"service": service_detail, "gpu_returncode": gpu_probe.returncode, "checkpoint_ready": checkpoint_ready},
        })
        raise SystemExit(f"preflight blocked: service={service_detail}; gpu_rc={gpu_probe.returncode}; checkpoint={checkpoint_ready}")

    env_base = os.environ.copy()
    env_base.update({
        "PYTHONPATH": f"{root}:{root / 'external/EnactTom'}" + (f":{env_base['PYTHONPATH']}" if env_base.get("PYTHONPATH") else ""),
        "ENACTTOM_PYTHON": str(root / ".venv-habitat/bin/python"),
        "ENACTTOM_EPISODES_PATH": str(root / "workdirs/habitat_enacttom/frozen12/episodes.json.gz"),
        "OPENAI_API_KEY": "local-qwen",
        "OPENAI_BASE_URL": args.base_url,
        "ENACTTOM_BENCHMARK_SEED": str(args.seed),
        "SSC_MODEL": args.model,
        "SSC_MODEL_CHECKPOINT": str(checkpoint),
        "SSC_SEED": str(args.seed),
        "SSC_CONFIG_SHA256": candidate_sha256,
        "SSC_MAX_TURNS": "20",
        "SSC_STABLE_SAMPLES": "3",
        "SSC_PHYSICS_STEPS_PER_SAMPLE": "10",
        "SSC_PHYSICS_DT": str(1 / 60),
        "SSC_OPEN_THRESHOLD": "0.10",
    })
    status_path = output / "status.jsonl"
    for task in tasks:
        for condition in CONDITIONS:
            run_dir = output / "results" / condition / task.stem
            run_dir.mkdir(parents=True, exist_ok=True)
            env = dict(env_base)
            env["SSC_CLOSED_LOOP_CONDITION"] = condition
            cmd = [
                str(root / "external/EnactTom/enacttom/run.sh"), "benchmark",
                "--task", str(task), "--output-dir", str(run_dir),
                "--model", args.model, "--llm", "openai_chat", "--num-times", "1",
                "--max-llm-calls", "20", "--benchmark-run-mode", "standard",
                "--observation-mode", "text", "--no-calibration", "--video",
            ]
            started = datetime.now(timezone.utc).isoformat()
            with (run_dir / "raw_console.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(cmd, cwd=root / "external/EnactTom", env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            append_jsonl(status_path, {
                "task": task.stem, "condition": condition, "seed": args.seed,
                "started": started, "finished": datetime.now(timezone.utc).isoformat(),
                "returncode": process.returncode, "run_dir": str(run_dir),
            })
            if process.returncode != 0:
                raise SystemExit(f"real rollout failed: {condition}/{task.stem}; inspect {run_dir / 'raw_console.log'}")


if __name__ == "__main__":
    main()
