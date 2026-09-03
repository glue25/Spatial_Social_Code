#!/usr/bin/env python3
"""Freeze and download a small, non-cherry-picked SocialNav-SUB pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spatial_social_code"))

from socialnav_sub import canonicalize_official_scene, compile_scene  # noqa: E402


TARGET_PEOPLE_COUNTS = (1, 3, 5, 7, 9, 13)
DEFAULT_MIRROR = "https://hf-mirror.com/datasets/michaelmunje/SocialNav-SUB/resolve/main"


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def materialize(url: str, destination: Path, local_source: Path | None = None) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if local_source is not None and local_source.is_file() and not is_lfs_pointer(local_source):
        shutil.copy2(local_source, temporary)
        temporary.replace(destination)
        return
    # The mirror's bare resolver currently redirects LFS files through an
    # unavailable ``cdn.hf-mirror.org`` hostname.  ``download=true`` selects
    # the working official CDN while preserving the frozen repository commit.
    separator = "&" if "?" in url else "?"
    download_url = f"{url}{separator}download=true"
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "spatial-social-code-pilot/1.0"},
    )
    last_error: OSError | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            break
        except OSError as error:
            last_error = error
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt - 1)))
    if last_error is not None and not temporary.is_file():
        raise last_error
    temporary.replace(destination)


def select_scenes(
    sample_info: dict, n_scenes: int | None = None, scene_start: int = 0
) -> list[dict]:
    rows = list(sample_info["samples"])
    if n_scenes is not None:
        if scene_start < 0 or n_scenes < 1 or scene_start + n_scenes > len(rows):
            raise ValueError(
                f"scene slice [{scene_start}:{scene_start + n_scenes}] is outside "
                f"the {len(rows)} official scenes"
            )
        # The official metadata order is frozen.  Taking its prefix gives a
        # deterministic expansion without looking at labels or model outputs.
        return [
            {"scene_id": row["folderpath"], "n_people": int(row["n_people"])}
            for row in rows[scene_start : scene_start + n_scenes]
        ]
    selected = []
    for count in TARGET_PEOPLE_COUNTS:
        match = next(row for row in rows if int(row["n_people"]) == count)
        selected.append({"scene_id": match["folderpath"], "n_people": count})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "workdirs/socialnav_sub/official_metadata/sample_info.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "workdirs/socialnav_sub/pilot_v1",
    )
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "external/SocialNavSUB/data/official_dataset",
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=None,
        help="Use the first N scenes in frozen official metadata order.",
    )
    parser.add_argument(
        "--scene-start",
        type=int,
        default=0,
        help="Zero-based start in frozen official metadata order.",
    )
    parser.add_argument(
        "--oracle-optional",
        action="store_true",
        help="Keep scenes even when the canonical oracle adapter cannot compile them.",
    )
    args = parser.parse_args()

    sample_info = json.loads(args.metadata.read_text(encoding="utf-8"))
    scenes = select_scenes(sample_info, args.n_scenes, args.scene_start)
    manifest = {
        "schema_version": 1,
        "selection": (
            f"official metadata slice [{args.scene_start}:{args.scene_start + args.n_scenes}]"
            if args.n_scenes is not None
            else "first official scene at each frozen people-count stratum"
        ),
        "target_people_counts": (
            sorted({scene["n_people"] for scene in scenes})
            if args.n_scenes is not None
            else list(TARGET_PEOPLE_COUNTS)
        ),
        "query_independent": True,
        "scenes": scenes,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    canonical_rows = []
    compiled_rows = []
    for selected in scenes:
        scene_id = selected["scene_id"]
        scene_root = args.output_root / "official" / scene_id
        relative_files = [
            f"prompts/{scene_id}/{scene_id}_localized_obj_coords.json",
            f"prompts/{scene_id}/n_people.json",
            f"human_labels/{scene_id}/common_answers.json",
        ] + [f"prompts/{scene_id}/sample_with_bev_{index}.png" for index in range(10)]
        for relative in relative_files:
            materialize(
                f"{args.mirror}/{relative}",
                scene_root / Path(relative).name,
                args.dataset_root / relative,
            )

        raw = json.loads(
            (scene_root / f"{scene_id}_localized_obj_coords.json").read_text(encoding="utf-8")
        )
        canonical = canonicalize_official_scene(raw, scene_id)
        canonical_rows.append(canonical)
        try:
            compiled = compile_scene(canonical)
            compiled_rows.append(compiled.to_dict())
        except ValueError as error:
            if not args.oracle_optional:
                raise
            compiled_rows.append(
                {
                    "scene_id": scene_id,
                    "oracle_available": False,
                    "compile_error": str(error),
                }
            )

    for name, rows in (("scenes.jsonl", canonical_rows), ("compiled_oracle.jsonl", compiled_rows)):
        path = args.output_root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    main()
