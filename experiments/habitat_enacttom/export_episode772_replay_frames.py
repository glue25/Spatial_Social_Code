#!/usr/bin/env python3
"""Export labeled, full-frame replay evidence for Habitat episode 772."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN = Path(
    "workdirs/habitat_enacttom/frozen12/replay_episode772/output_audit/results/"
    "emtom-108294897_176710602-772-cooperative-exhibit-reset-with-split-"
    "object-target-k-41a5f8ab"
)
OUTPUT = Path("reports/cases")
FONT = "/opt/conda/pkgs/font-ttf-dejavu-sans-mono-2.37-hab24e00_0/fonts/DejaVuSans.ttf"


def export_frame(
    source: Path,
    target: Path,
    lines: list[tuple[str, int, str]],
    crop_box: tuple[int, int, int, int] | None = None,
    zoom_lines: tuple[str, ...] = (),
) -> None:
    image = Image.open(source).convert("RGB")
    if image.size != (1024, 1024):
        image = image.resize((1024, 1024), Image.Resampling.NEAREST)

    bar_height = 48 + 38 * len(lines)
    width = 1536 if crop_box else 1024
    canvas = Image.new("RGB", (width, 1024 + bar_height), "black")
    canvas.paste(image, (0, bar_height))
    if crop_box:
        crop = image.crop(crop_box).resize((512, 512), Image.Resampling.NEAREST)
        canvas.paste(crop, (1024, bar_height))
    draw = ImageDraw.Draw(canvas)
    y = 12
    for text, size, color in lines:
        draw.text((24, y), text, font=ImageFont.truetype(FONT, size), fill=color)
        y += 42 if size >= 30 else 36
    if crop_box:
        y = bar_height + 560
        for text in zoom_lines:
            draw.text((1050, y), text, font=ImageFont.truetype(FONT, 25), fill="white")
            y += 38
    canvas.save(target, format="PNG", optimize=True)


def export_turn14() -> None:
    """Keep the full post-turn audit view and add a zoom of the held frame."""
    source = RUN / "replay_audit_views/turn_0014/view_4_picture_frame_0.png"
    image = Image.open(source).convert("RGB")
    crop = image.crop((300, 250, 800, 750)).resize(
        (512, 512), Image.Resampling.NEAREST
    )
    header_height = 160
    canvas = Image.new("RGB", (1536, header_height + 1024), "black")
    canvas.paste(image, (0, header_height))
    canvas.paste(crop, (1024, header_height))
    draw = ImageDraw.Draw(canvas)
    lines = [
        ("Replay visualization", 34, "white"),
        ("Episode 772 | after Turn 14 | Agent 2 | seed 891003", 25, "white"),
        ("picture_frame_0 remains grasped above counter_25", 22, "white"),
        ("No Place action; benchmark 4/4 is an evaluator false positive", 22, "#FFD166"),
    ]
    y = 12
    for text, size, color in lines:
        draw.text((24, y), text, font=ImageFont.truetype(FONT, size), fill=color)
        y += 42 if size >= 30 else 36
    side_lines = [
        "Zoom of the same",
        "post-turn frame:",
        "frame is attached",
        "to Agent 2's gripper.",
    ]
    y = header_height + 560
    for text in side_lines:
        draw.text((1050, y), text, font=ImageFont.truetype(FONT, 25), fill="white")
        y += 38
    canvas.save(
        OUTPUT / "habitat_ep772_turn14_agent2_replay.png",
        format="PNG",
        optimize=True,
    )


def main() -> None:
    specs = [
        (
            RUN / "replay_audit_views/turn_0008/view_1_chest_of_drawers_30.png",
            OUTPUT / "habitat_ep772_turn08_agent1_replay.png",
            [
                ("Replay visualization", 34, "white"),
                ("Episode 772 | Turn 8 | Agent 1 | seed 891003", 25, "white"),
                (
                    "chest_of_drawers_30 open | joint position 0.38270077",
                    22,
                    "white",
                ),
            ],
            (470, 300, 780, 610),
            ("Zoom of the same", "post-turn frame:", "opened drawer joint."),
        ),
        (
            RUN / "replay_audit_views/turn_0011/view_2_vase_1.png",
            OUTPUT / "habitat_ep772_turn11_agent0_replay.png",
            [
                ("Replay visualization", 34, "white"),
                ("Episode 772 | Turn 11 | Agent 0 | seed 891003", 25, "white"),
                ("vase_1 released above table_12 | no agent grasp", 22, "white"),
            ],
            (370, 340, 710, 680),
            ("Zoom of the same", "post-turn frame:", "vase_1 above table_12."),
        ),
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source, target, lines, crop_box, zoom_lines in specs:
        export_frame(source, target, lines, crop_box, zoom_lines)
    export_turn14()

    for turn in ("0008", "0011", "0014"):
        source = RUN / f"replay_audit_views/turn_{turn}/state.json"
        target = OUTPUT / f"habitat_ep772_turn{turn[-2:]}_state.json"
        target.write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
