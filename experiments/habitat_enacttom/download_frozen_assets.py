#!/usr/bin/env python3
"""Download only assets referenced by the frozen EnactToM canary."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


HSSD_REPO = "hssd/hssd-hab"
HSSD_REVISION = "partnr"
OVMM_REPO = "ai-habitat/OVMM_objects"


def strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def download(repo: str, revision: str, name: str, local_dir: Path) -> Path:
    target = local_dir / name
    if target.is_file():
        return target
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=name,
            local_dir=local_dir,
        )
    )


def unique_basename(files: list[str], basename: str, preferred_prefixes: tuple[str, ...] = ()) -> str:
    matches = [name for name in files if posixpath.basename(name) == basename]
    preferred = [name for name in matches if name.startswith(preferred_prefixes)] if preferred_prefixes else []
    matches = preferred or matches
    config_matches = [name for name in matches if "/configs/" in name]
    matches = config_matches or matches
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {basename}, got {matches}")
    return matches[0]


def json_asset_paths(config_path: Path, repo_name: str, repo_files: set[str]) -> set[str]:
    data = json.loads(config_path.read_text())
    result = set()
    parent = posixpath.dirname(repo_name)
    for value in strings(data):
        cleaned = value.replace("\\", "/")
        if not cleaned.lower().endswith(
            (
                ".glb",
                ".gltf",
                ".obj",
                ".mtl",
                ".ply",
                ".navmesh",
                ".urdf",
                ".json",
                ".png",
                ".jpg",
                ".jpeg",
            )
        ):
            continue
        candidates = [posixpath.normpath(posixpath.join(parent, cleaned)), cleaned.lstrip("./")]
        for candidate in candidates:
            if candidate in repo_files:
                result.add(candidate)
                break
    return result


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, default=Path("workdirs/habitat_enacttom/frozen12/manifest.json"))
    parser.add_argument("--hssd-dir", type=Path, default=Path("external/EnactTom/data/versioned_data/hssd-hab"))
    parser.add_argument("--ovmm-dir", type=Path, default=Path("external/EnactTom/data/objects_ovmm"))
    parser.add_argument("--status", type=Path, default=Path("workdirs/habitat_enacttom/frozen12/asset_download_status.json"))
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    frozen = json.loads(args.frozen_manifest.read_text())
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    status = {"phase": "listing_hssd", "complete": False}
    write_status(args.status, status)

    hssd_files = api.list_repo_files(HSSD_REPO, repo_type="dataset", revision=HSSD_REVISION)
    hssd_set = set(hssd_files)
    hssd_selected = {
        "hssd-hab-partnr.scene_dataset_config.json",
        "semantics/hssd-hab_semantic_lexicon.json",
        "semantics/objects.csv",
    }
    hssd_selected.update(name for name in hssd_files if name.startswith("metadata/"))
    hssd_selected.update(name for name in hssd_files if name.startswith("pbr/"))
    for scene_id in frozen["scene_ids"]:
        hssd_selected.update(
            name
            for name in hssd_files
            if scene_id in name
            and name.startswith(
                ("stages/", "semantics/", "scenes-partnr-filtered/", "scene_filter_files/")
            )
        )

    scene_configs = []
    for scene_id in frozen["scene_ids"]:
        name = f"scenes-partnr-filtered/{scene_id}.scene_instance.json"
        scene_configs.append((name, download(HSSD_REPO, HSSD_REVISION, name, args.hssd_dir)))
    object_handles = set()
    articulated_handles = set()
    for _, path in scene_configs:
        scene = json.loads(path.read_text())
        object_handles.update(item["template_name"] for item in scene.get("object_instances", []))
        articulated_handles.update(item["template_name"] for item in scene.get("articulated_object_instances", []))

    hssd_configs = set()
    for handle in object_handles:
        hssd_configs.add(unique_basename(hssd_files, f"{handle}.object_config.json", ("objects/",)))
    for handle in articulated_handles:
        hssd_selected.update(name for name in hssd_files if name.startswith(f"urdf/{handle}/"))
    for scene_id in frozen["scene_ids"]:
        stage_config = unique_basename(hssd_files, f"{scene_id}.stage_config.json", ("stages/",))
        hssd_configs.add(stage_config)
    hssd_selected.update(hssd_configs)

    status.update({"phase": "downloading_hssd_configs", "hssd_planned": len(hssd_selected), "hssd_configs": len(hssd_configs)})
    write_status(args.status, status)
    for name in sorted(hssd_configs | {"hssd-hab-partnr.scene_dataset_config.json"}):
        path = download(HSSD_REPO, HSSD_REVISION, name, args.hssd_dir)
        hssd_selected.update(json_asset_paths(path, name, hssd_set))
    status.update({"phase": "downloading_hssd_assets", "hssd_planned": len(hssd_selected)})
    write_status(args.status, status)
    for index, name in enumerate(sorted(hssd_selected), 1):
        download(HSSD_REPO, HSSD_REVISION, name, args.hssd_dir)
        if index % 25 == 0:
            status["hssd_downloaded"] = index
            write_status(args.status, status)

    status.update({"phase": "listing_ovmm", "hssd_downloaded": len(hssd_selected)})
    write_status(args.status, status)
    ovmm_files = api.list_repo_files(OVMM_REPO, repo_type="dataset")
    ovmm_set = set(ovmm_files)
    ovmm_configs = set()
    for basename in frozen["rigid_object_config_names"]:
        ovmm_configs.add(unique_basename(ovmm_files, basename, ("train_val/",)))
    # Keep direct config references strict: Habitat cannot register the object
    # template without these files.  Extra files from an OBJ bundle are best
    # effort because the upstream repository index currently contains a small
    # number of dangling entries (for example ``W6.obj``).
    ovmm_required = set(ovmm_configs)
    ovmm_bundle_extras = set()
    status.update({"phase": "downloading_ovmm_configs", "ovmm_configs": len(ovmm_configs)})
    write_status(args.status, status)
    for name in sorted(ovmm_configs):
        path = download(OVMM_REPO, "main", name, args.ovmm_dir)
        referenced_assets = json_asset_paths(path, name, ovmm_set)
        ovmm_required.update(referenced_assets)
        # Google Scanned render assets are OBJ bundles: model.obj references an
        # MTL file and sibling textures that are not named in object_config.json.
        # Keep the download minimal by including only the bundle directories of
        # OBJ files referenced by the selected frozen-task object configs.
        for asset in referenced_assets:
            if not asset.lower().endswith(".obj"):
                continue
            if "/meshes/" in asset:
                bundle_prefix = asset.split("/meshes/", 1)[0] + "/"
            else:
                bundle_prefix = posixpath.dirname(asset) + "/"
            ovmm_bundle_extras.update(
                candidate for candidate in ovmm_files if candidate.startswith(bundle_prefix)
            )
    # Dataset configs are tiny and register the selected object templates.
    ovmm_required.update(name for name in ovmm_files if name.endswith(".scene_dataset_config.json"))
    ovmm_selected = ovmm_required | ovmm_bundle_extras
    status.update({"phase": "downloading_ovmm_assets", "ovmm_planned": len(ovmm_selected)})
    write_status(args.status, status)
    downloaded = 0
    skipped_optional = []
    for index, name in enumerate(sorted(ovmm_selected), 1):
        try:
            download(OVMM_REPO, "main", name, args.ovmm_dir)
            downloaded += 1
        except EntryNotFoundError:
            if name in ovmm_required:
                raise
            skipped_optional.append(name)
        if index % 25 == 0:
            status["ovmm_downloaded"] = downloaded
            status["ovmm_optional_missing"] = len(skipped_optional)
            write_status(args.status, status)

    status.update(
        {
            "phase": "complete",
            "complete": True,
            "hssd_downloaded": len(hssd_selected),
            "ovmm_downloaded": downloaded,
            "ovmm_optional_missing": len(skipped_optional),
            "ovmm_optional_missing_files": skipped_optional,
            "hssd_files": sorted(hssd_selected),
            "ovmm_files": sorted(ovmm_selected - set(skipped_optional)),
        }
    )
    write_status(args.status, status)
    print(json.dumps({key: value for key, value in status.items() if not key.endswith("_files")}, indent=2))


if __name__ == "__main__":
    main()
