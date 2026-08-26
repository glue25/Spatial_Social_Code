"""Extract the official CoMind context-frame protocol from local ego videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import av

from spatial_social_code.comind import TASK_FILES, iter_labeled_samples
from spatial_social_code.http_range_reader import HTTPRangeReader


def sample_times(prediction_time: float, context_frames: int = 5) -> list[float]:
    """Five uniform frames in the preceding 10 s, followed by the target frame."""
    if context_frames < 1:
        raise ValueError("context_frames must be positive")
    start = max(0.0, prediction_time - 10.0)
    step = (prediction_time - start) / context_frames
    return [start + step * index for index in range(context_frames)] + [
        prediction_time
    ]


def uniformly_select(rows: list[tuple[str, str, dict]], limit: int) -> list:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    positions = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    return [rows[position] for position in positions]


class VideoFrames:
    def __init__(self, path: Path, remote_url: str | None = None, size: int = 0) -> None:
        self.path = path
        self.reader = None
        if path.is_file():
            self.container = av.open(str(path))
        elif remote_url and size:
            self.reader = HTTPRangeReader(remote_url, size)
            self.container = av.open(self.reader, mode="r")
        else:
            raise FileNotFoundError(path)
        self.stream = self.container.streams.video[0]

    def close(self) -> None:
        self.container.close()
        if self.reader is not None:
            self.reader.close()

    def frame_at(self, seconds: float):
        timestamp = max(0, int(seconds / float(self.stream.time_base)))
        self.container.seek(timestamp, stream=self.stream, backward=True)
        candidate = None
        for frame in self.container.decode(self.stream):
            candidate = frame
            if frame.time is not None and frame.time >= seconds:
                break
        if candidate is None:
            raise RuntimeError(f"no frame decoded from {self.path} at {seconds:.3f}s")
        return candidate.to_image()


def _views(task: str) -> tuple[str, ...]:
    return ("helper",) if task == "scoia" else ("leader", "helper")


def prepare(
    data_root: Path,
    annotation_root: Path,
    recording_id: str,
    output_root: Path,
    limit_per_task: int,
    context_frames: int,
    image_size: int,
    remote_base_url: str | None,
    selected_samples: dict[str, set[str]] | None = None,
) -> list[dict]:
    recording_root = data_root / "recordings" / recording_id
    manifest = json.loads(
        (recording_root / "healthcheck.json").read_text(encoding="utf-8")
    )
    assets = {item["path"]: item for item in manifest["files"]}
    videos = {
        role: VideoFrames(
            recording_root / "mp4s" / f"{role}_trimmed_sync.mp4",
            (
                f"{remote_base_url.rstrip('/')}/{recording_id}/mp4s/"
                f"{role}_trimmed_sync.mp4"
                if remote_base_url
                else None
            ),
            int(assets[f"mp4s/{role}_trimmed_sync.mp4"]["size"]),
        )
        for role in ("leader", "helper")
    }
    records: list[dict] = []
    try:
        for task in TASK_FILES:
            available = sorted(
                iter_labeled_samples(annotation_root, task, [recording_id]),
                key=lambda item: (float(item[2]["start_time"]), item[1]),
            )
            if selected_samples is not None:
                wanted = selected_samples.get(task, set())
                available = [item for item in available if item[1] in wanted]
            for _, sample_id, sample in uniformly_select(available, limit_per_task):
                prediction_time = float(sample["start_time"])
                times = sample_times(prediction_time, context_frames)
                sample_root = output_root / task / sample_id
                images: list[dict] = []
                for role in _views(task):
                    for position, timestamp in enumerate(times):
                        kind = "prediction" if position == len(times) - 1 else "context"
                        destination = sample_root / f"{role}_{position:02d}_{kind}.jpg"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.is_file():
                            image = videos[role].frame_at(timestamp)
                            image.thumbnail((image_size, image_size))
                            image.save(destination, quality=90)
                        images.append(
                            {
                                "role": role,
                                "kind": kind,
                                "time": timestamp,
                                "path": str(destination.resolve()),
                            }
                        )
                records.append(
                    {
                        "task": task,
                        "recording_id": recording_id,
                        "sample_id": sample_id,
                        "prediction_time": prediction_time,
                        "transcript_10s": sample.get("transcript_10s", ""),
                        "images": images,
                    }
                )
    finally:
        for video in videos.values():
            video.close()
    manifest = output_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-id", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("external_datasets/comind")
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("external_datasets/comind/annotations"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("workdirs/comind_pilot/frames")
    )
    parser.add_argument("--limit-per-task", type=int, default=10)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument(
        "--remote-base-url",
        default=None,
        help="Use seekable HTTP Range reads when complete local MP4s are absent",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="TASK:SAMPLE_ID",
        help="Select an exact labeled sample; may be repeated",
    )
    args = parser.parse_args()
    selected_samples: dict[str, set[str]] | None = None
    if args.sample:
        selected_samples = {task: set() for task in TASK_FILES}
        for value in args.sample:
            task, separator, sample_id = value.partition(":")
            if not separator or task not in TASK_FILES or not sample_id:
                parser.error(f"invalid --sample {value!r}; expected TASK:SAMPLE_ID")
            selected_samples[task].add(sample_id)
    records = prepare(
        args.data_root,
        args.annotations,
        args.recording_id,
        args.output_root,
        args.limit_per_task,
        args.context_frames,
        args.image_size,
        args.remote_base_url,
        selected_samples,
    )
    counts = {task: sum(row["task"] == task for row in records) for task in TASK_FILES}
    print(json.dumps({"samples": len(records), "by_task": counts}, indent=2))


if __name__ == "__main__":
    main()
