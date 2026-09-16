"""Offline integrity verification and optional Blender replay; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check_hash(relative, expected):
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError("Invalid bundle path: " + relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("SHA256 mismatch: " + relative)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    hashes = read("SHA256SUMS.json")
    for relative, expected in hashes.items():
        check_hash(relative, expected)
    for relative, expected in read("artifact_hashes.json").items():
        check_hash(relative, expected)
    manifest = read("observations_manifest.json")
    frames = manifest["frames"]
    require(len(frames) == 51, "Expected 51 real observations")
    for index, frame in enumerate(frames):
        require(not frame["interpolated"], "Observation cannot be interpolated")
        require(frame["source_frame_index_zero_based"] == 31772 + index * 6,
                "Source-frame sequence mismatch")
        require(abs(frame["timestamp_seconds"] - (31772 + index * 6) / 30) < 1e-6,
                "Source timestamp mismatch")
        check_hash(frame["path"], frame["sha256"])
        check_hash(frame["model_image_path"], frame["model_image_sha256"])
    code = read("temporal_code.json")
    require(runpy.run_path(str(ROOT / "temporal_code.py"))["T"].data == code,
            "Executable Code does not replay exact JSON")
    require(len(code["states"]) == code["observation_count"] == 51,
            "Temporal state count mismatch")
    require(len(code["entity_registry"]) == 12, "Entity count mismatch")
    require(code["entity_registry"] == read("entity_registry.json"),
            "Entity registry mismatch")
    require(not any(s["social_updates"] for s in code["states"]),
            "Unexpected timed Social Code update")
    require(len(read("render_interpolation.json")["frames"]) == 301,
            "Render frame count mismatch")
    print(json.dumps({"verified_files": len(hashes), "observations": 51,
                      "persistent_ids": 12, "timed_social_updates": 0,
                      "python_json_exact": True, "all_passed": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--output", type=Path, help="New directory for Blender replay")
    args = parser.parse_args()
    verify()
    if args.output is None:
        return
    blender = shutil.which("blender")
    xvfb = shutil.which("xvfb-run")
    if not blender:
        parser.error("Blender 3.x must be on PATH")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("temporal_code.py", "temporal_code.json", "entity_registry.json",
                 "blender_animation_tracks.json", "render_interpolation.json"):
        shutil.copyfile(ROOT / name, out / name)
    prefix = [xvfb, "-a"] if xvfb else []
    subprocess.run(prefix + [blender, "-b", "--python-exit-code", "1", "--python",
                            str(ROOT / "source_v3/blender_temporal.py"), "--", str(out)],
                   check=True)
    subprocess.run(prefix + [blender, "-b", str(out / "blender/temporal_scene.blend"),
                            "--python-exit-code", "1", "--python",
                            str(ROOT / "source_v2/check_temporal_blend.py"), "--", str(out)],
                   check=True)
    print("Replay and saved-scene track validation completed: " + str(out))


if __name__ == "__main__":
    main()
