#!/usr/bin/env python3
"""Validate determinism/interventions and export traceable result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


CASES = ["base", "intervention_move_bowl", "intervention_toggle_drawer", "intervention_remove_bowl"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statement_without_id(row: dict[str, Any]) -> str:
    return json.dumps({key: value for key, value in row.items() if key != "statement_id"}, sort_keys=True)


def refs(row: dict[str, Any], entity_id: str) -> bool:
    values = row.get("values", {})
    return entity_id in {
        values.get("entity_id"), values.get("subject"), values.get("object"),
        values.get("agent"), values.get("target"), values.get("speaker"),
    }


def expected_variant_statements(
    base: dict[str, Any], case: str, intervention: dict[str, Any]
) -> list[str]:
    rows = base["statements"]
    if case == "intervention_move_bowl":
        subject = intervention["subject"]
        before_relation, before_object = intervention["before"]
        after_relation, after_object = intervention["after"]
        result = []
        for row in rows:
            copy = json.loads(json.dumps(row))
            values = copy["values"]
            if copy["kind"] == "relation" and values == {
                "relation": before_relation, "subject": subject, "object": before_object
            }:
                copy["values"] = {
                    "relation": after_relation, "subject": subject, "object": after_object
                }
                copy.update(source="intervention", evidence_id="I1_MOVE_BOWL", confidence=1.0,
                            basis=f"single-factor intervention: replace {before_relation}({subject},{before_object}) with {after_relation}({subject},{after_object})")
            result.append(statement_without_id(copy))
        return result
    if case == "intervention_toggle_drawer":
        entity_id = intervention["entity_id"]
        result = []
        for row in rows:
            copy = json.loads(json.dumps(row))
            if (copy["kind"] == "state"
                    and copy["values"]["entity_id"] == entity_id
                    and copy["values"]["state"] == "open"):
                copy["values"]["value"] = intervention["after"]
                copy.update(source="intervention", evidence_id="I2_TOGGLE_DRAWER", confidence=1.0,
                            basis="single-factor intervention: toggle drawer open state")
            result.append(statement_without_id(copy))
        return result
    if case == "intervention_remove_bowl":
        removed = intervention["removed_entity"]
        return [statement_without_id(row) for row in rows if not refs(row, removed)]
    raise ValueError(case)


def allowed_geometry_changes(base: dict[str, Any], intervention: dict[str, Any]) -> set[str]:
    kind = intervention["type"]
    if kind == "relation_replacement":
        return {intervention["subject"]}
    if kind == "entity_removal":
        return {intervention["removed_entity"]}
    if kind != "state_toggle":
        raise ValueError(f"unsupported intervention type {kind!r}")

    changed = {intervention["entity_id"]}
    # Contents follow a toggled container. Agents constrained by a pose toward
    # it are also repositioned by the procedural solver.
    while True:
        descendants = {
            row["values"]["subject"]
            for row in base["statements"]
            if row["kind"] == "relation"
            and row["values"]["relation"] == "inside"
            and row["values"]["object"] in changed
        }
        if descendants <= changed:
            break
        changed.update(descendants)
    changed.update(
        row["values"]["agent"]
        for row in base["statements"]
        if row["kind"] == "pose" and row["values"]["target"] in changed
    )
    return changed


def make_montage(render_root: Path, output: Path) -> None:
    panels = []
    for case in CASES:
        image = Image.open(render_root / case / "render.png").convert("RGB")
        image.thumbnail((720, 560))
        panel = Image.new("RGB", (720, 600), "white")
        panel.paste(image, ((720 - image.width) // 2, 40))
        ImageDraw.Draw(panel).text((14, 12), case.replace("intervention_", "Intervention: "), fill="black")
        panels.append(panel)
    montage = Image.new("RGB", (1440, 1200), "white")
    for index, panel in enumerate(panels):
        montage.paste(panel, ((index % 2) * 720, (index // 2) * 600))
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-class",
        default="human-authored developer fixture; generator validation only; not gpt-6-astra output",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_scene = read(args.cases / "base" / "normalized_scene.json")
    base_snapshot = read(args.renders / "base" / "geometry_snapshot.json")

    summary: dict[str, Any] = {
        "schema_version": "ssc_world_experiment_summary_v1",
        "evidence_class": args.evidence_class,
        "cases": {},
        "determinism": {},
        "interventions": {},
    }
    trace_rows = []
    for case in CASES:
        scene = read(args.cases / case / "normalized_scene.json")
        generation = read(args.renders / case / "generation_trace.json")
        checks = read(args.renders / case / "geometry_checks.json")
        snapshot = read(args.renders / case / "geometry_snapshot.json")
        preview = args.renders / case / "render.png"
        with Image.open(preview) as im:
            extrema = im.convert("RGB").getextrema()
            visible = any(high - low > 8 for low, high in extrema)
            dimensions = list(im.size)
        summary["cases"][case] = {
            "statement_count": scene["statement_count"],
            "generation_accepted": generation["accepted"],
            "unhandled_statement_ids": generation["unhandled_statement_ids"],
            "geometry_all_passed": checks["all_passed"],
            "geometry_failed_check_ids": checks["failed_check_ids"],
            "preview_nonblank": visible,
            "preview_dimensions": dimensions,
            "blend_sha256": sha256(args.renders / case / "scene.blend"),
            "render_sha256": sha256(preview),
        }
        operation_by_id = {row["statement_id"]: row for row in generation["operations"]}
        check_by_id = {row["check_id"]: row for row in checks["checks"]}
        for row in scene["statements"]:
            values = row["values"]
            check = check_by_id.get(row["statement_id"])
            trace_rows.append({
                "case": case,
                "statement_id": row["statement_id"],
                "kind": row["kind"],
                "values_json": json.dumps(values, ensure_ascii=False, sort_keys=True),
                "source": row["source"],
                "evidence_id": row["evidence_id"],
                "basis": row["basis"],
                "generator_operation": operation_by_id.get(row["statement_id"], {}).get("operation", "MISSING"),
                "geometry_check": "not_applicable" if check is None else str(check["passed"]).lower(),
                "geometry_method": "" if check is None else check.get("measured", {}).get("method", "entity count"),
            })
        if case != "base":
            intervention = scene["intervention"]
            expected = sorted(expected_variant_statements(base_scene, case, intervention))
            actual = sorted(statement_without_id(row) for row in scene["statements"])
            if expected != actual:
                raise AssertionError(f"{case} changed statements beyond its declared single-factor intervention")
            allowed_changes = allowed_geometry_changes(base_scene, intervention)
            changed, unchanged = [], []
            all_entities = sorted(set(base_snapshot) | set(snapshot))
            for entity_id in all_entities:
                if base_snapshot.get(entity_id) == snapshot.get(entity_id):
                    unchanged.append(entity_id)
                else:
                    changed.append(entity_id)
            unexpected = sorted(set(changed) - allowed_changes)
            expected_effect = bool(set(changed) & allowed_changes)
            summary["interventions"][case] = {
                "semantic_single_factor_verified": True,
                "changed_primary_geometry": changed,
                "unchanged_primary_geometry": unchanged,
                "allowed_causal_geometry_changes": sorted(allowed_changes),
                "unexpected_primary_geometry_changes": unexpected,
                "expected_geometric_effect_observed": expected_effect,
            }
            if unexpected or not expected_effect:
                raise AssertionError(f"{case} geometry diff failed: unexpected={unexpected}, changed={changed}")

    repeated = args.renders / "base_repeat" / "geometry_snapshot.json"
    summary["determinism"] = {
        "same_seed": 17,
        "geometry_snapshot_byte_identical": repeated.read_bytes() == (args.renders / "base" / "geometry_snapshot.json").read_bytes(),
        "base_snapshot_sha256": sha256(args.renders / "base" / "geometry_snapshot.json"),
        "repeat_snapshot_sha256": sha256(repeated),
    }
    if not summary["determinism"]["geometry_snapshot_byte_identical"]:
        raise AssertionError("same-code same-seed geometry is not deterministic")
    if not all(row["generation_accepted"] and row["geometry_all_passed"] and row["preview_nonblank"] for row in summary["cases"].values()):
        raise AssertionError("at least one case failed generation, geometry, or visible-preview checks")

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output / "evidence_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["case", "statements", "generation_accepted", "geometry_all_passed", "preview_nonblank", "failed_checks"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, row in summary["cases"].items():
            writer.writerow({
                "case": case,
                "statements": row["statement_count"],
                "generation_accepted": row["generation_accepted"],
                "geometry_all_passed": row["geometry_all_passed"],
                "preview_nonblank": row["preview_nonblank"],
                "failed_checks": ";".join(row["geometry_failed_check_ids"]),
            })
    make_montage(args.renders, args.output / "intervention_montage.png")
    print(json.dumps({"accepted": True, "output": str(args.output), "cases": CASES}, indent=2))


if __name__ == "__main__":
    main()
