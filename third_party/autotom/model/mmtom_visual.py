"""MMToM-QA metadata and official RGB-frame sampling helpers."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                record = json.loads(line)
                if "episode" not in record or "end_time" not in record:
                    raise ValueError(
                        f"Missing episode/end_time in MMToM metadata line {line_number}"
                    )
                records.append(record)
    return records


def _episode_root(video_root: str | Path, episode: int) -> Path:
    root = Path(video_root).resolve()
    direct = root / f"task_{episode}"
    if direct.is_dir():
        return direct

    matches = list(root.glob(f"*/task_{episode}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find task_{episode} under MMToM video root {root}"
        )
    raise RuntimeError(f"Ambiguous task_{episode} directories under {root}: {matches}")


def sample_rgb_frames(
    video_root: str | Path,
    record: dict,
    num_frames: int = 8,
) -> tuple[list[Path], dict]:
    """Apply the benchmark's official uniform sampling through end_time."""
    if num_frames < 2:
        raise ValueError("MMToM multimodal sampling requires at least two frames")

    episode = int(record["episode"])
    end_time = int(record["end_time"])
    episode_root = _episode_root(video_root, episode)
    interval_path = episode_root / "frame_intervals.pik"
    with interval_path.open("rb") as source:
        intervals = pickle.load(source)

    times = [action[1] for action in intervals]
    if not 0 <= end_time < len(times):
        raise IndexError(
            f"end_time {end_time} is outside episode {episode}'s {len(times)} intervals"
        )
    end_frame = int(times[end_time])
    step_size = int(end_frame / (num_frames - 1))
    frame_numbers = [index * step_size for index in range(num_frames)]
    frame_root = episode_root / "script" / "0"
    requested_paths = [
        frame_root / f"Action_{frame_number:04d}_0_normal.png"
        for frame_number in frame_numbers
    ]
    if all(path.is_file() for path in requested_paths):
        selected_numbers = frame_numbers
        paths = requested_paths
    else:
        available = {}
        pattern = re.compile(r"^Action_(\d+)_0_normal\.png$")
        for candidate in frame_root.glob("Action_*_0_normal.png"):
            match = pattern.match(candidate.name)
            if match:
                available[int(match.group(1))] = candidate
        if not available:
            raise FileNotFoundError(f"No MMToM RGB frames found in {frame_root}")
        selected_numbers = [
            min(available, key=lambda number: (abs(number - target), number))
            for target in frame_numbers
        ]
        paths = [available[number] for number in selected_numbers]

    details = {
        "episode": episode,
        "end_time": end_time,
        "end_frame": end_frame,
        "requested_frame_numbers": frame_numbers,
        "frame_numbers": selected_numbers,
    }
    return paths, details
