#!/usr/bin/env python3
"""Validate the local datasets used by Spatial Social Code."""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: pathlib.Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def git_revision(path: pathlib.Path) -> str:
    if not (path / ".git").is_dir():
        return "archive"
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
        text=True,
    ).strip()


def validate_code() -> None:
    expected = {
        "EgoTaskQA": ROOT / "external/EgoTaskQA",
        "PyHoloAssist": ROOT / "external/PyHoloAssist",
        "holoassist-release": ROOT / "external/holoassist-release",
        "MA-EgoQA": ROOT / "external/MA-EgoQA",
        "EnactTom": ROOT / "external/EnactTom",
    }
    for name, path in expected.items():
        require(path.is_dir(), f"{name} code exists ({git_revision(path)})")
    comind_downloader = ROOT / "external/CoMind/comind_download.py"
    require(
        comind_downloader.is_file()
        and sha256(comind_downloader)
        == "ef9fa39c5f0b39c47b8790278df211e6e354b1cf8026bdbd5e1f442a06f1ea49",
        "CoMind official downloader exists and matches the frozen SHA-256",
    )


def validate_holoassist() -> None:
    base = ROOT / "external_datasets/holoassist"
    annotations = base / "annotations/data-annotation-trainval-v1_1.json"
    splits_zip = base / "splits/data-splits-v1_2.zip"
    require(
        sha256(annotations)
        == "cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c",
        "HoloAssist annotation checksum",
    )
    require(
        sha256(splits_zip)
        == "e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621",
        "HoloAssist split checksum",
    )
    records = json.loads(annotations.read_text(encoding="utf-8"))
    labels = collections.Counter(
        event["label"] for record in records for event in record["events"]
    )
    require(len(records) == 1758, "HoloAssist has 1,758 recording records")
    require(sum(labels.values()) == 195945, "HoloAssist has 195,945 labeled events")
    expected_splits = {"train": 1466, "val": 207, "test": 438}
    for split, expected_count in expected_splits.items():
        path = base / f"splits/{split}-v1_2.txt"
        require(
            line_count(path) == expected_count,
            f"HoloAssist {split} split has {expected_count:,} entries",
        )


def validate_ma_egoqa() -> None:
    base = ROOT / "external_datasets/ma_egoqa"
    questions = json.loads((base / "MA-EgoQA.json").read_text(encoding="utf-8"))
    categories = collections.Counter(item["category"] for item in questions)
    require(len(questions) == 1741, "MA-EgoQA has 1,741 questions")
    require(len(categories) == 5, "MA-EgoQA covers five question categories")
    caption_files = list((base / "caption").rglob("*.json"))
    require(len(caption_files) == 127, "MA-EgoQA has 127 caption JSON files")
    require(
        (base / "10min_shared_memory.json").is_file(),
        "MA-EgoQA shared-memory baseline exists",
    )


def validate_enacttom() -> None:
    base = ROOT / "external_datasets/enacttom"
    for split in ("standard", "hard"):
        records = [
            json.loads(line)
            for line in (base / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(records) == 150, f"EnactToM {split} split has 150 tasks")
        require(
            all("problem_pddl" in item and "golden_trajectory" in item for item in records),
            f"EnactToM {split} tasks contain PDDL and golden trajectories",
        )


def validate_manifest_files(
    base: pathlib.Path,
    manifest: dict,
    *,
    prefix: str | None = None,
) -> int:
    selected = [
        entry
        for entry in manifest["files"]
        if prefix is None or entry["path"].startswith(prefix)
    ]
    for entry in selected:
        path = base / entry["path"]
        require(
            path.is_file()
            and path.stat().st_size == entry["size"]
            and sha256(path) == entry["hash"],
            f"CoMind verified file {entry['path']}",
        )
    return len(selected)


def validate_comind() -> None:
    base = ROOT / "external_datasets/comind"
    recording_id = "f0b49334-565a-4d30-a010-15d92e511c9a"

    pilot_ids = [
        line.strip()
        for line in (ROOT / "experiments/configs/comind_pilot_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]
    require(pilot_ids == [recording_id], "CoMind pilot ID is frozen")

    annotations_dir = base / "annotations"
    annotations_manifest = json.loads(
        (annotations_dir / "healthcheck.json").read_text(encoding="utf-8")
    )
    require(
        validate_manifest_files(annotations_dir, annotations_manifest) == 3,
        "CoMind has three verified annotation files",
    )

    recording_dir = base / f"recordings/{recording_id}"
    recording_manifest = json.loads(
        (recording_dir / "healthcheck.json").read_text(encoding="utf-8")
    )
    require(
        validate_manifest_files(
            recording_dir, recording_manifest, prefix="transcripts/"
        )
        == 8,
        "CoMind pilot has eight verified transcript files",
    )

    expected_annotation_rows = {
        "dataset_handover_consolidated.json": 77,
        "dataset_joint_attention_consolidated.json": 78,
        "dataset_scoia_consolidated.json": 79,
    }
    for filename, expected_rows in expected_annotation_rows.items():
        payload = json.loads(
            (annotations_dir / filename).read_text(encoding="utf-8")
        )
        require(
            len(payload["data"]) == expected_rows,
            f"CoMind {filename} has {expected_rows} records",
        )

    transcript_dir = recording_dir / "transcripts"
    expected_transcripts = {
        "helper_trimmed_sync_transcript.json": (439, 2644),
        "leader_trimmed_sync_transcript.json": (345, 2769),
    }
    for filename, (segments, words) in expected_transcripts.items():
        payload = json.loads(
            (transcript_dir / filename).read_text(encoding="utf-8")
        )
        require(
            len(payload["segments"]) == segments
            and len(payload["word_segments"]) == words,
            f"CoMind {filename} has {segments} segments and {words} words",
        )


def print_pending() -> None:
    egotask_data = ROOT / "external_datasets/egotaskqa"
    if not egotask_data.exists():
        print("WAIT  EgoTaskQA: dataset license/request form must be completed")
    print("INFO  Large raw videos and Habitat/HSSD assets are intentionally excluded")


def main() -> int:
    try:
        validate_code()
        validate_holoassist()
        validate_ma_egoqa()
        validate_enacttom()
        validate_comind()
    except (AssertionError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        return 1
    print_pending()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
