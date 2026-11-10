"""Append-only artifact writer for auditable Habitat closed-loop runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _json_default(value: Any):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)


class TraceRecorder:
    STREAMS = (
        "events.jsonl", "state_before_after.jsonl", "plans.jsonl",
        "physical_checks.jsonl", "predictions.jsonl", "usage.jsonl",
        "labels_SCORER_ONLY.jsonl",
    )

    def __init__(self, output_dir: str):
        self.root = Path(output_dir) / "closed_loop_artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        for name in self.STREAMS:
            (self.root / name).touch(exist_ok=True)

    def append(self, stream: str, payload: Dict[str, Any]) -> None:
        if stream not in self.STREAMS:
            raise ValueError(f"unknown trace stream: {stream}")
        with (self.root / stream).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        with (self.root / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")

    @staticmethod
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
