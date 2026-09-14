#!/usr/bin/env python3
"""Render an auditable Dynamic-SSC overview for Habitat episode 1166.

The figure is generated only from saved closed-loop artifacts.  It does not
rerun Habitat or call a model.  Completion labels come from strict_physical_v1,
not from visual inspection of the recorded RGB frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "workdirs/habitat_closed_loop/dev36_final_v3/results"
OUTPUT_PNG = ROOT / "reports/cases/habitat_ep1166_dynamic_ssc_visualization.png"
OUTPUT_JSON = ROOT / "reports/cases/habitat_ep1166_dynamic_ssc_visualization.json"

COLORS = {
    "bg": "#F6F7FB",
    "ink": "#172033",
    "muted": "#5A6475",
    "line": "#CBD2E0",
    "spatial": "#DDF2FF",
    "spatial_dark": "#1677A5",
    "social": "#F3E7FF",
    "social_dark": "#7851A9",
    "success": "#DDF7E6",
    "success_dark": "#167A45",
    "warning": "#FFF0D6",
    "white": "#FFFFFF",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = ROOT / ".venv-habitat/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf" / name
    return ImageFont.truetype(str(path), size)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def find_run(condition: str) -> Path:
    matches = sorted((RUN_ROOT / condition).glob("**/*1166*/closed_loop_artifacts"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one episode-1166 {condition} run, got {len(matches)}")
    return matches[0]


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str,
            outline: str = COLORS["line"], radius: int = 18, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
         *, bold: bool = False, fill: str = COLORS["ink"], anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, size: int, width: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    face = font(size)
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=face)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
              width: int, *, fill: str = COLORS["ink"], spacing: int = 6,
              bold: bool = False) -> int:
    y = xy[1]
    for line in wrap(draw, value, size, width):
        text(draw, (xy[0], y), line, size, bold=bold, fill=fill)
        y += size + spacing
    return y


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line((start, end), fill=color, width=5)
    x, y = end
    draw.polygon([(x, y), (x - 13, y - 8), (x - 13, y + 8)], fill=color)


def strict_summary(run: Path) -> dict[str, Any]:
    checks = load_jsonl(run / "physical_checks.jsonl")
    final = checks[-1]
    return {
        "turns": int(final["turn"]),
        "strict_success": bool(final["strict_success"]),
        "completed_goals": int(final["completed_goals"]),
        "total_goals": int(final["total_goals"]),
        "strict_percent_complete": float(final["strict_percent_complete"]),
        "scorer_version": final["scorer_version"],
        "stable_samples": int(final["stable_samples"]),
        "final_checks": final["checks"],
    }


def turn_end_frame(run: Path, turn: int, agent: str) -> Path:
    index = load_json(run / "frames/index.json")["turns"][str(turn)][agent]
    record = next((item for item in reversed(index) if item["kind"] == "turn_end"), index[-1])
    return Path(record["path"])


def state_versions(plans: Iterable[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for row in plans:
        if row.get("phase") != "planner_input":
            continue
        result.setdefault(row["agent_id"], []).append((int(row["turn"]), int(row["state_version_read"])))
    return result


def render() -> dict[str, Any]:
    runs = {condition: find_run(condition) for condition in ("direct", "static_ssc", "dynamic_ssc")}
    strict = {condition: strict_summary(path) for condition, path in runs.items()}
    dynamic = runs["dynamic_ssc"]
    manifest = load_json(dynamic / "manifest.json")
    plans = load_jsonl(dynamic / "plans.jsonl")
    predictions = load_jsonl(dynamic / "predictions.jsonl")
    events = load_jsonl(dynamic / "events.jsonl")
    versions = state_versions(plans)

    selected = [
        (4, "agent_2", "vase_0 ON table_15"),
        (5, "agent_1", "box_3 ON table_12"),
        (7, "agent_2", "fridge_34 OPEN"),
    ]
    frames = [(turn, agent, label, turn_end_frame(dynamic, turn, agent)) for turn, agent, label in selected]

    canvas = Image.new("RGB", (2000, 1450), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    text(draw, (70, 48), "Dynamic Spatial-Social Code in Closed-Loop Habitat", 42, bold=True)
    text(draw, (70, 102), "Episode 1166 | real saved rollout | model qwen3-vl-8b-instruct | seed 891011", 23,
         fill=COLORS["muted"])
    text(draw, (1930, 77), "MECHANISM CASE", 20, bold=True, fill=COLORS["social_dark"], anchor="ra")

    # Closed-loop controller panel.
    rounded(draw, (60, 145, 1940, 360), COLORS["white"])
    text(draw, (90, 170), "Online control loop", 27, bold=True)
    labels = [
        ("Private observation", COLORS["spatial"]),
        ("Spatial / Social state\n(agent-specific, versioned)", COLORS["social"]),
        ("Planner action", COLORS["warning"]),
        ("Environment feedback", COLORS["success"]),
    ]
    xs = [95, 535, 1050, 1470]
    widths = [330, 400, 300, 380]
    for (label, fill), x, w in zip(labels, xs, widths):
        rounded(draw, (x, 230, x + w, 325), fill, radius=14)
        parts = label.split("\n")
        for i, part in enumerate(parts):
            text(draw, (x + w // 2, 268 + i * 27), part, 21, bold=i == 0, anchor="mm")
    for x1, x2 in ((425, 535), (935, 1050), (1350, 1470)):
        arrow(draw, (x1 + 10, 278), (x2 - 12, 278), COLORS["muted"])
    draw.arc((80, 305, 1905, 405), 0, 180, fill=COLORS["social_dark"], width=4)
    draw.polygon([(92, 350), (112, 340), (111, 363)], fill=COLORS["social_dark"])
    text(draw, (995, 337), "successful / failed action result updates the next state version", 18,
         fill=COLORS["social_dark"], anchor="mm")

    # Spatial and social panels.
    rounded(draw, (60, 390, 990, 800), COLORS["spatial"])
    text(draw, (90, 420), "Spatial Code: what is where, and what actually changed", 27, bold=True,
         fill=COLORS["spatial_dark"])
    nodes = {
        "vase_0": (175, 545), "table_15": (770, 545),
        "box_3": (175, 645), "table_12": (770, 645),
        "fridge_34": (175, 745), "OPEN": (770, 745),
    }
    for name, (x, y) in nodes.items():
        fill = COLORS["white"] if name not in {"OPEN"} else COLORS["success"]
        rounded(draw, (x - 80, y - 32, x + 110, y + 32), fill, radius=12)
        text(draw, (x + 15, y), name, 20, bold=True, anchor="mm")
    for y, turn in ((545, 4), (645, 5), (745, 7)):
        arrow(draw, (290, y), (675, y), COLORS["spatial_dark"])
        text(draw, (480, y - 17), f"verified relation | turn {turn}", 18,
             fill=COLORS["spatial_dark"], anchor="mm")
    text(draw, (90, 475), "Only successful runtime feedback writes completion; failed attempts remain failed.", 19,
         fill=COLORS["muted"])

    rounded(draw, (1010, 390, 1940, 800), COLORS["social"])
    text(draw, (1040, 420), "Social Code: who may access, receive, and know", 27, bold=True,
         fill=COLORS["social_dark"])
    rows = [
        ("agent_0", "DENIED kitchen_1", "messages only to agent_2"),
        ("agent_1", "DENIED bedroom_2", "messages only to agent_2"),
        ("agent_2", "no denied room", "may message agent_0 / agent_1"),
    ]
    for i, (agent, access, comm) in enumerate(rows):
        y = 500 + i * 83
        rounded(draw, (1045, y - 28, 1195, y + 28), COLORS["white"], radius=12)
        text(draw, (1120, y), agent, 20, bold=True, anchor="mm")
        text(draw, (1230, y - 14), access, 18, bold=True)
        text(draw, (1230, y + 12), comm, 17, fill=COLORS["muted"])
    text(draw, (1045, 742), "Delivered claim != observed world fact; blocked messages update sender only.", 18,
         bold=True, fill=COLORS["social_dark"])

    # Version timeline and condition comparison.
    rounded(draw, (60, 830, 1940, 1025), COLORS["white"])
    text(draw, (90, 855), "State versions read by the planner", 25, bold=True)
    agent_colors = {"agent_0": "#1677A5", "agent_1": "#C06B1A", "agent_2": "#7851A9"}
    for row, agent in enumerate(("agent_0", "agent_1", "agent_2")):
        y = 930 + row * 36
        text(draw, (95, y), agent, 18, bold=True, fill=agent_colors[agent])
        points = versions[agent]
        for turn, version in points:
            x = 270 + (turn - 1) * 155
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=agent_colors[agent])
            text(draw, (x, y - 20), f"v{version}", 15, fill=agent_colors[agent], anchor="mm")
            if turn < 7:
                draw.line((x + 10, y, x + 145, y), fill=agent_colors[agent], width=3)
    for turn in range(1, 8):
        x = 270 + (turn - 1) * 155
        text(draw, (x, 882), f"T{turn}", 16, fill=COLORS["muted"], anchor="mm")

    for i, condition in enumerate(("direct", "static_ssc", "dynamic_ssc")):
        result = strict[condition]
        x = 1425 + i * 165
        fill = COLORS["success"] if result["strict_success"] else COLORS["warning"]
        rounded(draw, (x, 885, x + 145, 992), fill, radius=12)
        text(draw, (x + 72, 910), condition.replace("_", " "), 16, bold=True, anchor="mm")
        text(draw, (x + 72, 946), f'{result["completed_goals"]}/{result["total_goals"]}', 30,
             bold=True, anchor="mm")
        text(draw, (x + 72, 976), f'{result["turns"]} turns', 15, fill=COLORS["muted"], anchor="mm")

    # Real frames.
    text(draw, (60, 1055), "Recorded agent views at key turns", 27, bold=True)
    frame_w, frame_h = 570, 285
    for idx, (turn, agent, label, path) in enumerate(frames):
        x = 60 + idx * 625
        rounded(draw, (x, 1100, x + frame_w, 1410), COLORS["white"], radius=14)
        source = Image.open(path).convert("RGB")
        source.thumbnail((frame_w - 20, frame_h - 70))
        px = x + (frame_w - source.width) // 2
        py = 1110
        canvas.paste(source, (px, py))
        text(draw, (x + 18, 1345), f"T{turn} | {agent} | {label}", 18, bold=True)
        text(draw, (x + 18, 1373), "RGB is contextual; strict scorer is authoritative.", 15,
             fill=COLORS["muted"])

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PNG, optimize=True)

    messages = [
        {key: event.get(key) for key in ("turn", "sender", "receivers", "message")}
        for event in events if event.get("event_type") == "message_delivered"
    ]
    payload = {
        "schema_version": "habitat_dynamic_ssc_visualization_v1",
        "status": "mechanism_case_not_aggregate_evidence",
        "task_id": manifest["task_id"],
        "episode_id": manifest["episode_id"],
        "scene_id": manifest["scene_id"],
        "model": manifest["model"],
        "model_checkpoint": manifest["model_checkpoint"],
        "seed": int(manifest["seed"]),
        "conditions": strict,
        "dynamic_state_versions_read": versions,
        "dynamic_actions": predictions,
        "delivered_messages": messages,
        "selected_frames": [
            {"turn": turn, "agent": agent, "label": label, "source_path": str(path.relative_to(ROOT)),
             "sha256": digest(path)}
            for turn, agent, label, path in frames
        ],
        "sources": {
            condition: {
                "artifact_dir": str(path.relative_to(ROOT)),
                "manifest_sha256": digest(path / "manifest.json"),
                "plans_sha256": digest(path / "plans.jsonl"),
                "physical_checks_sha256": digest(path / "physical_checks.jsonl"),
            }
            for condition, path in runs.items()
        },
        "visualization": str(OUTPUT_PNG.relative_to(ROOT)),
        "completion_semantics": "strict_physical_v1; three stable physics samples; RGB not used as the completion judge",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = render()
    print(json.dumps({
        "png": str(OUTPUT_PNG),
        "json": str(OUTPUT_JSON),
        "episode_id": result["episode_id"],
        "conditions": {
            name: f'{row["completed_goals"]}/{row["total_goals"]}'
            for name, row in result["conditions"].items()
        },
    }, indent=2))
