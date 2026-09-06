#!/usr/bin/env python3
"""Synthetic minimal-pair tests for agent information attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MA-EgoQA") not in sys.path:
    sys.path.insert(0, str(ROOT / "MA-EgoQA"))

from experiments.priority_ssc_v1.common import atomic_json, atomic_jsonl, parse_choice, read_jsonl  # noqa: E402
from experiments.priority_ssc_v1.run_four_cell import one_call  # noqa: E402
from experiments.priority_ssc_v1.run_reasoning_baselines import make_client  # noqa: E402

TEMPLATE = """Use the physical and agent-indexed information states below. Answer whether the queried agent knows the stated final fact. Do not replace an agent's private evidence with the omniscient physical state. Return JSON only: {{"answer":"A","reason":"one short sentence"}}.

Physical state:
{physical}

Agent-indexed information state:
{social}

Question: Does {target} know that the {object_name} is in the {location}?
A. Yes
B. No"""

NAMES = ("Alice", "Bob", "Carol", "David", "Emma", "Farah", "Grace", "Hiro", "Iris", "Jon")
OBJECTS = ("red cup", "blue key", "green book", "silver spoon", "yellow ball", "black phone", "white plate", "orange box", "purple towel", "brown mug")
LOCATIONS = ("cabinet", "drawer", "fridge", "basket", "closet", "desk", "shelf", "toolbox", "pantry", "backpack")


def record(kind: str, **values: Any) -> str:
    return json.dumps({"kind": kind, "values": values}, ensure_ascii=False, sort_keys=True)


def cases(total: int) -> list[dict[str, Any]]:
    result = []
    types = ("observer", "message_recipient", "presence")
    for index in range(total):
        target, other, irrelevant = NAMES[index % 10], NAMES[(index + 3) % 10], NAMES[(index + 6) % 10]
        obj, location = OBJECTS[(index * 3) % 10], LOCATIONS[(index * 7) % 10]
        intervention = types[index % len(types)]
        physical = record("spatial_fact", t=2, relation="in", subject=obj, object=location, source="controlled")
        target_knows_in_base = index % 2 == 0
        knows = record("observation", t=2, agent=target, content=f"{obj} is in {location}", channel="vision" if intervention != "message_recipient" else "message", evidence_ids=["P1"])
        other_knows = record("observation", t=2, agent=other, content=f"{obj} is in {location}", channel="vision" if intervention != "message_recipient" else "message", evidence_ids=["P1"])
        unknown = record("social_unknown", t=2, agent=target, relation="location", subject=obj)
        base_relevant, changed_relevant = (knows, unknown) if target_knows_in_base else (unknown, knows)
        base = "\n".join((base_relevant, record("observation", t=1, agent=irrelevant, content="weather is sunny", channel="vision", evidence_ids=[])))
        changed = "\n".join((changed_relevant, other_knows))
        negative = "\n".join((base_relevant, record("observation", t=1, agent=irrelevant, content="weather is cloudy", channel="vision", evidence_ids=[])))
        variants = {
            "base": {"social": base, "answer_index": 0 if target_knows_in_base else 1},
            "target_intervention": {"social": changed, "answer_index": 1 if target_knows_in_base else 0},
            "negative_control": {"social": negative, "answer_index": 0 if target_knows_in_base else 1},
        }
        result.append({"pair_id": index, "intervention_type": intervention, "target": target, "object_name": obj, "location": location, "physical": physical, "physical_digest": hashlib.sha256(physical.encode()).hexdigest(), "variants": variants})
    return result


def summarize(rows: list[dict[str, Any]], total: int, model: str) -> dict[str, Any]:
    complete = [row for row in rows if all(int(row.get("predictions", {}).get(v, -1)) >= 0 for v in ("base", "target_intervention", "negative_control"))]
    return {
        "schema_version": 1,
        "protocol": "controlled_agent_attribution_minimal_pairs",
        "model": model,
        "selected_pairs": total,
        "complete_pairs": len(complete),
        "physical_terminal_digest_equal_within_every_pair": all(row["physical_digest_equal"] for row in complete),
        "pair_both_correct": sum(row["correct"]["base"] and row["correct"]["target_intervention"] for row in complete) / len(complete) if complete else None,
        "target_update_correct": sum(row["correct"]["target_intervention"] and row["predictions"]["base"] != row["predictions"]["target_intervention"] for row in complete) / len(complete) if complete else None,
        "negative_control_invariant_and_correct": sum(row["correct"]["negative_control"] and row["predictions"]["base"] == row["predictions"]["negative_control"] for row in complete) / len(complete) if complete else None,
        "actual_calls": sum(len(row["call_metrics"]) for row in complete),
        "input_tokens": sum(int(metric.get("usage", {}).get("input_tokens", 0)) for row in complete for metric in row["call_metrics"].values()),
        "output_tokens": sum(int(metric.get("usage", {}).get("output_tokens", 0)) for row in complete for metric in row["call_metrics"].values()),
        "latency_seconds": sum(float(metric.get("elapsed_seconds", 0)) for row in complete for metric in row["call_metrics"].values()),
        "retries": sum(int(metric.get("transport_retries") or 0) for row in complete for metric in row["call_metrics"].values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local-qwen32b", "jdcloud"), default="local-qwen32b")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--credentials-file", type=Path, default=Path("/pfs/yfsong/code/JoyVideo/AVGen-Bench/eval.sh"))
    parser.add_argument("--local-model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--duty-cycle", type=float, default=1.0)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    args = parser.parse_args()
    output = args.output or ROOT / f"workdirs/priority_ssc_v1/attribution_{args.backend}.jsonl"
    examples = cases(args.total)
    existing = {int(row["pair_id"]): row for row in read_jsonl(output)} if output.exists() else {}
    client = make_client(args)
    model = str(client.config.model)
    for example in examples:
        pair_id = int(example["pair_id"])
        old = existing.get(pair_id, {})
        predictions, responses, metrics = dict(old.get("predictions", {})), dict(old.get("responses", {})), dict(old.get("call_metrics", {}))
        correct = {}
        for name, variant in example["variants"].items():
            if int(predictions.get(name, -1)) < 0:
                try:
                    prompt = TEMPLATE.format(physical=example["physical"], social=variant["social"], target=example["target"], object_name=example["object_name"], location=example["location"])
                    responses[name], metrics[name] = one_call(client, prompt)
                    predictions[name] = parse_choice(responses[name], "AB")
                except Exception as error:
                    responses[name] = f"ERROR: {type(error).__name__}: {error}"
                    predictions[name] = -1
            correct[name] = int(predictions[name]) == int(variant["answer_index"])
        existing[pair_id] = {"schema_version": 1, "pair_id": pair_id, "intervention_type": example["intervention_type"], "model": model, "physical_digest": example["physical_digest"], "physical_digest_equal": len({example["physical_digest"] for _ in example["variants"]}) == 1, "predictions": predictions, "correct": correct, "responses": responses, "call_metrics": metrics}
        atomic_jsonl(output, (existing[i] for i in sorted(existing)))
        atomic_json(output.with_name(output.stem + "_summary.json"), summarize([existing[i] for i in sorted(existing)], args.total, model))
        print(f"pair={pair_id} complete={len(existing)}/{args.total}", flush=True)


if __name__ == "__main__":
    main()
