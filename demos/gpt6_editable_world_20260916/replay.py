#!/usr/bin/env python3
"""Offline verification/replay of the published model outputs. No API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE / "source"
HERE = SOURCE / "experiments/astra_world_demo_v1"
sys.path.insert(0, str(SOURCE))
from experiments.astra_world_demo_v1.compile_world import compile_payloads
from experiments.astra_world_demo_v1.world_dsl import execute_world

CASES = ("base", "intervention_move_bowl", "intervention_toggle_drawer", "intervention_remove_bowl")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = read(BUNDLE / "SHA256SUMS.json")
    for name, expected in manifest.items():
        path = (BUNDLE / name).resolve()
        require(BUNDLE in path.parents, f"Invalid manifest path: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"SHA-256 mismatch: {name}")
    result = compile_payloads(read(BUNDLE / "inputs/spatial.json"), read(BUNDLE / "inputs/social.json"))
    require(result["world_code"] == (BUNDLE / "compiled/world_code.py").read_text(), "Saved model output no longer compiles to published World code")
    require(result["normalized"] == read(BUNDLE / "compiled/normalized_scene.json"), "Normalized compilation mismatch")
    for case in CASES:
        scene = read(BUNDLE / "cases" / case / "normalized_scene.json")
        normalized = execute_world((BUNDLE / "cases" / case / "world_code.py").read_text())
        require(normalized["statements"] == scene["statements"], f"World code mismatch: {case}")
    summary = read(BUNDLE / "analysis/summary.json")
    require(all(c["generation_accepted"] and c["geometry_all_passed"] and c["preview_nonblank"] for c in summary["cases"].values()), "Published case check failed")
    require(all(c["semantic_single_factor_verified"] and c["expected_geometric_effect_observed"] and not c["unexpected_primary_geometry_changes"] for c in summary["interventions"].values()), "Published intervention check failed")
    require(summary["determinism"]["geometry_snapshot_byte_identical"], "Published determinism check failed")
    for case in (*CASES, "base_repeat"):
        require(read(BUNDLE / "rendered" / case / "geometry_checks.json")["all_passed"], f"Published geometry failure: {case}")
    return len(manifest)


def run(command, log):
    with log.open("w") as handle:
        subprocess.run(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true", help="Check hashes, compile payloads and inspect recorded checks; no Blender")
    p.add_argument("--output", type=Path, help="New output directory; never overwrites a previous run")
    p.add_argument("--blender", default="blender")
    p.add_argument("--no-xvfb", action="store_true", help="Use an existing graphical display instead of xvfb-run")
    args = p.parse_args()
    count = verify()
    print(f"Verified {count} file hashes and unchanged model-to-World compilation.", flush=True)
    if args.verify_only:
        return
    if args.output is None:
        p.error("--output is required unless --verify-only is supplied")
    out = args.output.resolve()
    require(out != BUNDLE and BUNDLE not in out.parents, "Write replay output outside the published bundle")
    out.mkdir(parents=True, exist_ok=False)
    status = {"state": "started", "api_calls": 0, "evidence_class": "offline replay of saved GPT-6 outputs; proxy geometry only", "seed": 17}
    try:
        cases = out / "cases"
        run([sys.executable, str(HERE / "make_interventions.py"), "--base-scene", str(BUNDLE / "compiled/normalized_scene.json"), "--base-code", str(BUNDLE / "compiled/world_code.py"), "--output", str(cases)], out / "interventions.log")
        for case in CASES:
            require((cases / case / "world_code.py").read_bytes() == (BUNDLE / "cases" / case / "world_code.py").read_bytes(), f"Intervention code mismatch: {case}")
        snapshots_match = {}
        for case in (*CASES, "base_repeat"):
            scene = cases / ("base" if case == "base_repeat" else case) / "normalized_scene.json"
            destination = out / "rendered" / case
            destination.mkdir(parents=True)
            prefix = [] if args.no_xvfb else ["xvfb-run", "-a"]
            run(prefix + [args.blender, "-b", "--python-exit-code", "1", "--python", str(HERE / "blender_generate.py"), "--", str(scene), str(destination), "17"], destination / "generate.log")
            run([args.blender, "-b", str(destination / "scene.blend"), "--python-exit-code", "1", "--python", str(HERE / "blender_check.py"), "--", str(scene), str(destination / "geometry_checks.json")], destination / "check.log")
            require(read(destination / "geometry_checks.json")["all_passed"], f"Fresh geometry checks failed: {case}")
            require(read(destination / "generation_trace.json")["accepted"], f"Fresh generation rejected: {case}")
            snapshots_match[case] = (destination / "geometry_snapshot.json").read_bytes() == (BUNDLE / "rendered" / case / "geometry_snapshot.json").read_bytes()
        run([sys.executable, str(HERE / "analyze_outputs.py"), "--cases", str(cases), "--renders", str(out / "rendered"), "--output", str(out / "analysis"), "--evidence-class", status["evidence_class"]], out / "analysis.log")
        status.update(state="complete", published_geometry_snapshot_matches=snapshots_match)
        print(json.dumps(status, indent=2))
    except Exception as error:
        status.update(state="failed_preserved", error=str(error))
        raise
    finally:
        (out / "replay_status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
