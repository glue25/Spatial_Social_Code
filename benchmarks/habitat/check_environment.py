#!/usr/bin/env python3
"""Write a reproducible environment/model preflight report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def command(argv):
    try:
        run = subprocess.run(argv, text=True, capture_output=True, timeout=20, check=False)
        return {"command": argv, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except Exception as exc:
        return {"command": argv, "error": f"{type(exc).__name__}: {exc}"}


def http_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local-qwen"}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return {"ok": True, "status": response.status, "body": body}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--model-checkpoint", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-8B-Instruct"))
    args = parser.parse_args()
    root = args.root.resolve()
    task_dir = root / "workdirs/habitat_enacttom/frozen12/tasks"
    episodes = root / "workdirs/habitat_enacttom/frozen12/episodes.json.gz"
    checkpoint_files = [args.model_checkpoint / "config.json", args.model_checkpoint / "model.safetensors.index.json"]
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "python": platform.python_version(),
        "packages": {},
        "gpu": command(["nvidia-smi", "-L"]),
        "processes": command(["pgrep", "-af", "qwen|vllm|sglang|gemini"]),
        "assets": {
            "episodes_path": str(episodes), "episodes_exists": episodes.is_file(),
            "episodes_bytes": episodes.stat().st_size if episodes.is_file() else None,
            "task_dir": str(task_dir), "task_count": len(list(task_dir.glob("*.json"))),
        },
        "model_checkpoint": {
            "path": str(args.model_checkpoint),
            "exists": args.model_checkpoint.is_dir(),
            "metadata_hashes": {path.name: sha256(path) for path in checkpoint_files if path.is_file()},
        },
        "service_models": http_json(args.base_url.rstrip("/") + "/models"),
    }
    for package in ("habitat-sim", "habitat-lab", "torch", "openai"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    report["model_call"] = http_json(args.base_url.rstrip("/") + "/chat/completions", {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: ENV_OK"}],
        "temperature": 0,
        "max_tokens": 8,
    })
    report["ready_for_real_rollout"] = bool(
        report["gpu"].get("returncode") == 0
        and report["assets"]["episodes_exists"]
        and report["assets"]["task_count"] == 12
        and report["model_checkpoint"]["exists"]
        and report["service_models"].get("ok")
        and report["model_call"].get("ok")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ready_for_real_rollout"] else 2)


if __name__ == "__main__":
    main()
