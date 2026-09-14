#!/usr/bin/env python3
"""One-call, same-backbone baselines over a shared frozen perception trace.

``pddl_mind_adapted`` is intentionally named as an adaptation: it applies the
paper's verified-transition idea to the same frozen trace, but is not an
official PDDL-Mind reproduction or its text/state input protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "MA-EgoQA", ROOT / "AutoToM/model", ROOT / "spatial_social_code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.priority_ssc_v1.common import (  # noqa: E402
    CONDITIONS,
    atomic_json,
    atomic_jsonl,
    digest_text,
    exact_mcnemar,
    format_question,
    parse_choice,
    read_jsonl,
    source_traces,
)
from mmtom_visual import load_questions  # noqa: E402


PROMPTS = {
    "cot": """Reason step by step from the frozen physical evidence. Track event order and each person's information access, then compare every option. Return JSON only: {\"answer\":\"A\",\"reasoning\":\"concise chain of thought summary\"}.""",
    "natural_language_scratchpad": """Build a concise natural-language scratchpad containing chronology, physical state, who saw or heard each event, and unknowns. Use it to compare every option. Do not use code or PDDL. Return JSON only: {\"answer\":\"A\",\"scratchpad\":\"concise notes\",\"reason\":\"short conclusion\"}.""",
    "pddl_mind_adapted": """Apply a PDDL-Mind-style verified transition analysis to the frozen physical evidence: identify initial predicates, ordered actions with preconditions/effects, advance the world state only when supported, separately derive each agent's observable state, and compare every option. This is an adapted baseline, not an official reproduction. Return JSON only: {\"answer\":\"A\",\"pddl_state_summary\":\"concise verified transitions\",\"reason\":\"short conclusion\"}.""",
    "spatial_only": """Answer from the frozen Physical trace alone. Missing records are unknown, not false. Compare every option, but do not construct a separate Social state. Return JSON only: {\"answer\":\"A\",\"reason\":\"one short sentence\"}.""",
    "ssc": """Answer from the frozen Physical trace and agent-indexed Social trace. Preserve who observed or heard each event; missing records are unknown, not false. Compare every option. Return JSON only: {\"answer\":\"A\",\"reason\":\"one short sentence\"}.""",
}


def balanced_independent(source: list[dict[str, Any]], dataset: str, total: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        stratum = str(row.get("question_type") if dataset == "mmtom" else row.get("category"))
        groups[stratum].append(row)
    for rows in groups.values():
        rng.shuffle(rows)
    cursors = {name: 0 for name in groups}
    selected: list[int] = []
    used_units: set[str] = set()
    while len(selected) < total:
        progressed = False
        for stratum in sorted(groups):
            rows = groups[stratum]
            while cursors[stratum] < len(rows):
                row = rows[cursors[stratum]]
                cursors[stratum] += 1
                unit = str(row.get("episode", row.get("index"))) if dataset == "mmtom" else str(row.get("index"))
                if unit not in used_units:
                    selected.append(int(row["index"]))
                    used_units.add(unit)
                    progressed = True
                    break
            if len(selected) >= total:
                break
        if not progressed:
            break
    if len(selected) != total:
        raise ValueError(f"only {len(selected)} independent examples available, requested {total}")
    return selected


class CodexResponsesClient:
    """Small Codex-CLI adapter with the metadata shape used by this runner."""

    def __init__(self, model: str, reasoning_effort: str, timeout_seconds: int, max_retries: int):
        self.config = SimpleNamespace(model=model)
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @staticmethod
    def _events(stdout: str) -> tuple[dict[str, int], str]:
        usage: dict[str, int] = {}
        response = ""
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = {
                    str(key): int(value)
                    for key, value in event["usage"].items()
                    if isinstance(value, int)
                }
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
            ):
                response = str(item.get("text", ""))
        return usage, response

    def generate_parts_with_metadata(
        self,
        parts: list[dict[str, Any]],
        max_output_tokens: int = 1536,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        del max_output_tokens, temperature
        prompt = "\n\n".join(str(part.get("text", "")) for part in parts if "text" in part)
        attempts: list[dict[str, Any]] = []
        overall_started = time.monotonic()
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 2):
            with tempfile.TemporaryDirectory(prefix="ssc_gpt56_") as temp_dir:
                output_path = Path(temp_dir) / "response.txt"
                command = [
                    "codex", "exec", prompt, "--ephemeral", "--ignore-rules",
                    "--disable", "apps", "--disable", "goals",
                    "--disable", "shell_snapshot", "--disable", "plugins",
                    "--disable", "unified_exec", "--disable", "code_mode_host",
                    "--skip-git-repo-check", "--sandbox", "read-only", "--cd", "/tmp",
                    "--model", self.config.model,
                    "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                    "--json", "--output-last-message", str(output_path),
                ]
                started = time.monotonic()
                try:
                    completed = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                    elapsed = time.monotonic() - started
                    usage, event_text = self._events(completed.stdout)
                    attempts.append({"attempt": attempt, "returncode": completed.returncode, "elapsed_seconds": elapsed})
                    if completed.returncode == 0:
                        response = (
                            output_path.read_text(encoding="utf-8")
                            if output_path.is_file()
                            else event_text
                        )
                        if response.strip():
                            return {
                                "text": response,
                                "attempts": attempts,
                                "transport_retries": attempt - 1,
                                "elapsed_seconds": time.monotonic() - overall_started,
                                "usage": usage,
                            }
                    last_error = completed.stderr[-2000:] or completed.stdout[-2000:]
                except subprocess.TimeoutExpired as error:
                    last_error = f"timeout after {error.timeout}s"
                    attempts.append({"attempt": attempt, "returncode": None, "error": last_error})
            if attempt <= self.max_retries:
                time.sleep(min(60, 2 ** (attempt - 1)))
        raise RuntimeError(f"Codex call failed after {len(attempts)} attempts: {last_error}")


def make_client(args: argparse.Namespace) -> Any:
    if args.backend == "local-qwen32b":
        from experiments.iclr_ablations_v1.run_mmtom_local_qwen32b import LocalQwenPartsClient

        return LocalQwenPartsClient(args.local_model_path, args.max_pixels, args.duty_cycle)
    if args.backend == "codex":
        return CodexResponsesClient(
            args.model or "gpt-5.6-sol",
            args.reasoning_effort,
            args.timeout_seconds,
            args.max_retries,
        )
    from egomas.src.jdcloud_gemini import JDCloudGeminiClient, load_service_config

    config = load_service_config(args.credentials_file, args.model)
    return JDCloudGeminiClient(config, timeout_seconds=600, max_attempts=5, min_request_interval_seconds=args.min_request_interval)


def call(client: Any, prompt: str, condition: str) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    result = client.generate_parts_with_metadata([{"text": prompt}], max_output_tokens=1536, temperature=0.0)
    return str(result["text"]), {
        "condition": condition,
        "calls": 1,
        "attempts": result.get("attempts"),
        "transport_retries": result.get("transport_retries", 0),
        "elapsed_seconds": result.get("elapsed_seconds", time.monotonic() - started),
        "usage": result.get("usage", {}),
    }


def prompt_for(condition: str, question: str, labels: str, physical: str, social: str) -> str:
    sections = [PROMPTS[condition], f"Valid answer labels: {', '.join(labels)}", f"Frozen Physical trace:\n{physical}"]
    if condition == "ssc":
        sections.append(f"Frozen agent-indexed Social trace (answer-directed records removed):\n{social}")
    sections.append(f"Question:\n{question}")
    return "\n\n".join(sections)


def summarize(rows: list[dict[str, Any]], selected: list[int], model: str) -> dict[str, Any]:
    complete = [row for row in rows if all(row.get("predictions", {}).get(c, -1) >= 0 for c in CONDITIONS)]
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        calls = [row["call_metrics"][condition] for row in complete]
        metrics[condition] = {
            "correct": sum(row["correct"][condition] for row in complete),
            "total": len(complete),
            "accuracy": sum(row["correct"][condition] for row in complete) / len(complete) if complete else None,
            "calls": len(calls),
            "input_tokens": sum(int(c.get("usage", {}).get("input_tokens", 0)) for c in calls),
            "output_tokens": sum(int(c.get("usage", {}).get("output_tokens", 0)) for c in calls),
            "latency_seconds": sum(float(c.get("elapsed_seconds", 0)) for c in calls),
            "retries": sum(int(c.get("transport_retries") or 0) for c in calls),
        }
    comparisons = {}
    for left in CONDITIONS[:-1]:
        left_only = sum(row["correct"][left] and not row["correct"]["ssc"] for row in complete)
        ssc_only = sum(not row["correct"][left] and row["correct"]["ssc"] for row in complete)
        comparisons[f"{left}_vs_ssc"] = {"left_only_correct": left_only, "ssc_only_correct": ssc_only, "mcnemar_exact_p": exact_mcnemar(left_only, ssc_only)}
    return {"schema_version": 1, "protocol": "reasoning_only_shared_frozen_perception_one_call_each", "model": model, "selected": len(selected), "fully_complete": len(complete), "conditions": metrics, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mmtom", "maegoqa"), required=True)
    parser.add_argument("--backend", choices=("local-qwen32b", "jdcloud", "codex"), default="local-qwen32b")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--credentials-file", type=Path, default=ROOT / "credentials.env")
    parser.add_argument("--local-model-path", type=Path, default=Path("/pfs/xiabin/models/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--duty-cycle", type=float, default=1.0)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    default_source = ROOT / ("spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl" if args.dataset == "mmtom" else "MA-EgoQA/data/baselines/video_balanced_role_full.jsonl")
    default_questions = ROOT / ("MMToM-QA/Benchmark/questions.json" if args.dataset == "mmtom" else "MA-EgoQA/data/MA-EgoQA.json")
    source_path = args.source or default_source
    questions_path = args.questions or default_questions
    output = args.output or ROOT / f"workdirs/priority_ssc_v1/{args.dataset}_{args.backend}_reasoning.jsonl"
    source_rows = read_jsonl(source_path)
    source = {int(row["index"]): row for row in source_rows}
    questions = load_questions(questions_path) if args.dataset == "mmtom" else json.loads(questions_path.read_text(encoding="utf-8"))
    selected = balanced_independent(source_rows, args.dataset, args.total, args.seed)
    if args.limit:
        selected = selected[: args.limit]
    manifest = {
        "schema_version": 1,
        "protocol": "reasoning_only_shared_frozen_perception_one_call_each",
        "dataset": args.dataset,
        "backend": args.backend,
        "selected_indices": selected,
        "conditions": list(CONDITIONS),
        "calls_per_condition": 1,
        "pddl_mind_status": "adapted_baseline_not_official_reproduction",
        "target_directed_social_records_removed": True,
        "source": str(source_path),
        "source_sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
    }
    atomic_json(output.with_name(output.stem + "_manifest.json"), manifest)
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return
    existing = {int(row["index"]): row for row in read_jsonl(output)} if output.exists() else {}
    client = make_client(args)
    model_name = str(client.config.model)
    for index in selected:
        previous = existing.get(index, {})
        predictions = dict(previous.get("predictions", {}))
        responses = dict(previous.get("responses", {}))
        call_metrics = dict(previous.get("call_metrics", {}))
        question, labels, answer_index = format_question(args.dataset, questions[index])
        physical, social = source_traces(source[index])
        for condition in CONDITIONS:
            if int(predictions.get(condition, -1)) >= 0:
                continue
            try:
                response, metrics = call(client, prompt_for(condition, question, labels, physical, social), condition)
                responses[condition] = response
                call_metrics[condition] = metrics
                predictions[condition] = parse_choice(response, labels)
            except Exception as error:
                responses[condition] = f"ERROR: {type(error).__name__}: {error}"
                predictions[condition] = -1
        existing[index] = {
            "schema_version": 1,
            "dataset": args.dataset,
            "model": model_name,
            "index": index,
            "unit_id": str(source[index].get("episode", index)),
            "answer_index": answer_index,
            "physical_digest": digest_text(physical),
            "social_digest": digest_text(social),
            "predictions": predictions,
            "correct": {name: int(predictions.get(name, -1)) == answer_index for name in CONDITIONS},
            "responses": responses,
            "call_metrics": call_metrics,
        }
        atomic_jsonl(output, (existing[i] for i in sorted(existing)))
        atomic_json(output.with_name(output.stem + "_summary.json"), summarize([existing[i] for i in selected if i in existing], selected, model_name))
        done = sum(i in existing and all(int(existing[i].get("predictions", {}).get(c, -1)) >= 0 for c in CONDITIONS) for i in selected)
        print(f"index={index} complete={done}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
