#!/usr/bin/env python3
"""Export evidence that an SSC-compiled episode was loaded and executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle, runs, output = args.bundle.resolve(), args.runs.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compile_manifest = json.loads((bundle / "compile_manifest.json").read_text(encoding="utf-8"))

    rows = []
    selected_frames = []
    panels_to_render = []
    summaries = sorted(runs.glob("results/*/results/*/closed_loop_artifacts/summary.json"))
    if len(summaries) != 3:
        raise SystemExit(f"expected three generated-environment runs, found {len(summaries)}")
    for summary_path in summaries:
        artifact = summary_path.parent
        run_manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        checks = read_jsonl(artifact / "physical_checks.jsonl")
        final = checks[-1]
        frames = sorted((artifact / "frames").glob("**/*.png"))
        if not frames:
            raise SystemExit(f"no rendered frames in {artifact}")
        condition = run_manifest["condition"]
        events = read_jsonl(artifact / "events.jsonl")
        if condition == "dynamic_ssc":
            target_entities = {str(check["entity"]) for check in final["checks"]}
            key_actions = []
            seen_entities = set()
            for event in events:
                action = event.get("action") or {}
                outcome = event.get("outcome") or {}
                arguments = action.get("arguments") or []
                if (
                    event.get("event_type") == "action_feedback"
                    and outcome.get("success")
                    and action.get("name") in {"Place", "Open"}
                    and arguments
                    and str(arguments[0]) in target_entities
                    and str(arguments[0]) not in seen_entities
                ):
                    key_actions.append((int(event["turn"]), str(event["agent_id"]), str(arguments[0]), action["name"]))
                    seen_entities.add(str(arguments[0]))
            if len(key_actions) < len(target_entities):
                raise SystemExit(f"missing successful goal actions in {artifact}: {key_actions}")
            for turn, agent, entity, action_name in key_actions:
                actor_candidates = sorted((artifact / "frames" / agent / f"turn_{turn:04d}").glob("*_in_action.png"))
                chosen = max(actor_candidates, key=lambda path: path.stat().st_size) if actor_candidates else None
                if chosen is None or chosen.stat().st_size < 50_000:
                    candidates = sorted((artifact / "frames").glob(f"*/turn_{turn:04d}/*_in_action.png"))
                    candidates += sorted((artifact / "frames").glob(f"*/turn_{turn + 1:04d}/*_in_action.png"))
                    if not candidates:
                        candidates = sorted((artifact / "frames" / agent / f"turn_{turn:04d}").glob("*.png"))
                    chosen = max(candidates, key=lambda path: path.stat().st_size)
                frame_agent = chosen.parents[1].name
                frame_out = output / f"generated_environment_dynamic_t{turn:04d}_{entity}.png"
                shutil.copy2(chosen, frame_out)
                record = {
                    "condition": condition, "turn": turn, "agent": agent,
                    "frame_agent": frame_agent,
                    "action": action_name, "entity": entity,
                    "source": str(chosen.relative_to(ROOT)),
                    "export": str(frame_out.relative_to(ROOT)), "sha256": sha256(frame_out),
                }
                selected_frames.append(record)
                panels_to_render.append((record, f"Dynamic SSC · T{turn} · {action_name} {entity}"))
        else:
            final_turn = int(final["turn"])
            candidates = sorted((artifact / "frames").glob(f"*/turn_{final_turn:04d}/*_in_action.png"))
            chosen = candidates[-1] if candidates else frames[-1]
            frame_out = output / f"generated_environment_{condition}_final.png"
            shutil.copy2(chosen, frame_out)
            record = {
                "condition": condition, "turn": final_turn,
                "source": str(chosen.relative_to(ROOT)),
                "export": str(frame_out.relative_to(ROOT)), "sha256": sha256(frame_out),
            }
            selected_frames.append(record)
            panels_to_render.append((record, f"{condition} · final · {final['completed_goals']}/{final['total_goals']} goals"))
        rows.append({
            "condition": condition,
            "task_id": run_manifest["task_id"],
            "episode_id": run_manifest["episode_id"],
            "seed": int(run_manifest["seed"]),
            "turns": int(final["turn"]),
            "strict_success": bool(final["strict_success"]),
            "completed_goals": int(final["completed_goals"]),
            "total_goals": int(final["total_goals"]),
            "frame_count": len(frames),
            "artifact_dir": str(artifact.relative_to(ROOT)),
        })

    thumbs = []
    for frame, title in panels_to_render:
        image = Image.open(ROOT / frame["export"]).convert("RGB")
        image.thumbnail((640, 420))
        panel = Image.new("RGB", (660, 470), "white")
        panel.paste(image, ((660 - image.width) // 2, 40))
        ImageDraw.Draw(panel).text((18, 12), title, fill="black")
        thumbs.append(panel)
    columns = 2
    rows_count = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (660 * columns, 470 * rows_count), "white")
    for index, panel in enumerate(thumbs):
        sheet.paste(panel, (660 * (index % columns), 470 * (index // columns)))
    sheet_path = output / "ssc_code_generated_environment_execution.png"
    sheet.save(sheet_path)

    evidence = {
        "schema_version": "ssc_code_generated_environment_execution_v1",
        "status": "executed_in_habitat",
        "claim_boundary": "New episode/layout instance from SSC using existing HSSD stage and meshes; no novel mesh synthesis.",
        "compile_manifest": compile_manifest,
        "runs": sorted(rows, key=lambda row: row["condition"]),
        "selected_frames": selected_frames,
        "contact_sheet": str(sheet_path.relative_to(ROOT)),
        "contact_sheet_sha256": sha256(sheet_path),
        "checks": {
            "all_runs_use_generated_episode": all(row["episode_id"] == compile_manifest["generated_episode_id"] for row in rows),
            "all_runs_have_real_frames": all(row["frame_count"] > 0 for row in rows),
            "pose_differs_from_source": compile_manifest["pose_changed"],
            "answer_bearing_fields_present": compile_manifest["answer_bearing_fields_present"],
        },
    }
    evidence["accepted"] = all((
        evidence["checks"]["all_runs_use_generated_episode"],
        evidence["checks"]["all_runs_have_real_frames"],
        evidence["checks"]["pose_differs_from_source"],
        not evidence["checks"]["answer_bearing_fields_present"],
    ))
    json_out = output / "execution_evidence.json"
    json_out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# SSC → Habitat 环境实例化与闭环执行", "",
        "本实验不是只把现有环境转写成 JSON。编译器仅以 Spatial–Social Code 为输入，",
        "生成独立的 Habitat episode 与任务文件，再由 Habitat 实际加载、渲染和执行。", "",
        "## 生成边界", "",
        "- 生成了新的 episode ID、资产实例位姿、任务绑定、agent 配置与社会信息边界。",
        "- 使用已有 HSSD 房间壳和资产网格；没有声称生成全新 mesh。",
        "- 输入中不含 golden trajectory、Oracle schedule 或目标答案。", "",
        "## 执行结果", "",
        "| 条件 | 生成 episode | 严格完成 | 轮数 | 实际帧数 |", "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["condition"]):
        lines.append(
            f"| {row['condition']} | {row['episode_id']} | "
            f"{row['completed_goals']}/{row['total_goals']} | {row['turns']} | {row['frame_count']} |"
        )
    lines.extend([
        "", f"编译与执行验收：`accepted={str(evidence['accepted']).lower()}`。", "",
        "- [真实执行帧对照](ssc_code_generated_environment_execution.png)",
        "- [生成后的 Spatial–Social Code](generated_ssc_code.json)",
        "- [编译 manifest](compile_manifest.json)",
        "- [可直接加载的 Habitat episodes](episodes.json.gz)", "",
    ])
    (output / "README_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
