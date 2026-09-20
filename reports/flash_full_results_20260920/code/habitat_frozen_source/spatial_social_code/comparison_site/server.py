#!/usr/bin/env python3
"""Live local report for Spatial-Social Code, AutoToM, and CaP-X-style Spatial Code."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
EXP = ROOT / "spatial_social_code"
QUESTIONS_PATH = ROOT / "MMToM-QA" / "Benchmark" / "questions.json"
FRAMES_ROOT = ROOT / "MMToM-QA-data" / "required_frames"
METRICS_ROOT = ROOT / "AutoToM" / "results" / "metrics"
V2_NATIVE_PATH = EXP / "mmtom_candidate_aligned_full.jsonl"
V1_NATIVE_PATH = EXP / "mmtom_native_hybrid_full.jsonl"
CAPX_PILOT_PATH = EXP / "mmtom_fair70_merged.jsonl"
V2_STATUS_PATH = EXP / "mmtom_candidate_aligned_full.status"
V1_STATUS_PATH = EXP / "mmtom_native_hybrid_full.status"


def native_path() -> Path:
    """Prefer the candidate-aligned run as soon as it has produced a checkpoint."""
    return V2_NATIVE_PATH if V2_NATIVE_PATH.exists() else V1_NATIVE_PATH


def status_path() -> Path:
    return V2_STATUS_PATH if V2_STATUS_PATH.exists() else V1_STATUS_PATH

MODEL_META = {
    "spatial_social": {
        "name": "Spatial-Social Code",
        "short": "Ours",
        "note": "Candidate-aligned Spatial→Social bridge + persistent-belief Bayesian fusion",
    },
    "autotom": {
        "name": "AutoToM",
        "short": "AutoToM",
        "note": "47 candidate-mismatched cases use aligned reruns",
    },
    "capx": {
        "name": "CaP-X-style Spatial Code",
        "short": "CaP-X",
        "note": "Standalone spatial/world-model baseline; currently 70-case pilot coverage",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def indexed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["index"]): row for row in rows}


def prediction_entry(prediction: dict[str, Any], answer: str) -> dict[str, Any]:
    pred = str(prediction["prediction"]).lower()
    prob_a = float(prediction.get("prob_a", 0.0))
    prob_b = float(prediction.get("prob_b", 0.0))
    return {
        "prediction": pred,
        "correct": pred == answer,
        "confidence": max(prob_a, prob_b),
        "prob_a": prob_a,
        "prob_b": prob_b,
    }


def autotom_metric(index: int) -> tuple[dict[str, Any] | None, bool]:
    suffix = f"MMToM-QA_{index}_back1_reduce1_seed42_metrics.json"
    aligned = METRICS_ROOT / f"automated_qwen3vl_mm_required_aligned_{suffix}"
    fallback = METRICS_ROOT / f"automated_qwen3vl_mm_required_{suffix}"
    path = aligned if aligned.exists() else fallback
    if not path.exists():
        return None, False
    return json.loads(path.read_text(encoding="utf-8")), path == aligned


def autotom_entry(index: int, answer: str) -> dict[str, Any] | None:
    metric, aligned = autotom_metric(index)
    if metric is None:
        return None
    p_correct = float(metric.get("Likelihood of correct answer", 0.5))
    p_incorrect = 1.0 - p_correct
    other = "b" if answer == "a" else "a"
    correct = bool(metric.get("Correctness", p_correct >= p_incorrect))
    pred = answer if correct else other
    prob_a = p_correct if answer == "a" else p_incorrect
    prob_b = p_correct if answer == "b" else p_incorrect
    return {
        "prediction": pred,
        "correct": pred == answer,
        "confidence": max(prob_a, prob_b),
        "prob_a": prob_a,
        "prob_b": prob_b,
        "aligned_rerun": aligned,
        "total_time": metric.get("Total time"),
        "total_api": metric.get("Total API"),
    }


def has_execution_error(record: dict[str, Any] | None) -> bool:
    return bool(record and "execution_error" in json.dumps(record.get("traces", {})))


class ReportData:
    def __init__(self) -> None:
        self.questions = load_jsonl(QUESTIONS_PATH)

    def sources(self):
        native = indexed(load_jsonl(native_path()))
        pilot = indexed(load_jsonl(CAPX_PILOT_PATH))
        return native, pilot

    def methods_for_case(
        self,
        index: int,
        native: dict[int, dict[str, Any]],
        pilot: dict[int, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        answer = str(self.questions[index]["answer"]).lower()
        methods: dict[str, dict[str, Any]] = {}
        native_record = native.get(index)
        if native_record and "spatial_social" in native_record.get("predictions", {}):
            methods["spatial_social"] = prediction_entry(
                native_record["predictions"]["spatial_social"], answer
            )
        auto = autotom_entry(index, answer)
        if auto:
            methods["autotom"] = auto
        capx_record = native_record if (
            native_record and "spatial" in native_record.get("predictions", {})
        ) else pilot.get(index)
        if capx_record and "spatial" in capx_record.get("predictions", {}):
            methods["capx"] = prediction_entry(
                capx_record["predictions"]["spatial"], answer
            )
            methods["capx"]["source"] = (
                "full" if capx_record is native_record else "pilot70"
            )
        return methods

    def compact_cases(self):
        native, pilot = self.sources()
        cases = []
        model_rows: dict[str, dict[int, dict[str, Any]]] = {
            model: {} for model in MODEL_META
        }
        for index, question in enumerate(self.questions):
            methods = self.methods_for_case(index, native, pilot)
            for model, result in methods.items():
                model_rows[model][index] = result
            query = " ".join(str(question["question"]).split())
            cases.append(
                {
                    "index": index,
                    "episode": question["episode"],
                    "question_type": str(question["question_type"]),
                    "answer": str(question["answer"]).lower(),
                    "preview": query[:220] + ("…" if len(query) > 220 else ""),
                    "methods": methods,
                    "execution_error": has_execution_error(native.get(index)),
                }
            )
        return cases, model_rows, native, pilot

    def score(self, rows: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
        selected = [rows[index] for index in sorted(ids) if index in rows]
        correct = sum(row["correct"] for row in selected)
        return {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else None,
        }

    def by_type(
        self, rows: dict[int, dict[str, Any]], ids: set[int]
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for question_type in sorted({str(q["question_type"]) for q in self.questions}):
            typed = {
                index
                for index in ids
                if str(self.questions[index]["question_type"]) == question_type
            }
            result[question_type] = self.score(rows, typed)
        return result

    def summary(self) -> dict[str, Any]:
        cases, model_rows, _, _ = self.compact_cases()
        available = {model: set(rows) for model, rows in model_rows.items()}
        common = set.intersection(*available.values()) if all(available.values()) else set()
        ours_auto = available["spatial_social"] & available["autotom"]
        all_ids = set(range(len(self.questions)))
        scopes = {}
        for scope, scope_ids in (
            ("common", common),
            ("ours_auto", ours_auto),
            ("available", all_ids),
        ):
            model_ids = {
                model: (
                    set() if scope == "ours_auto" and model == "capx" else scope_ids
                )
                for model in model_rows
            }
            scopes[scope] = {
                "case_count": len(scope_ids),
                "models": {
                    model: {
                        **self.score(rows, model_ids[model]),
                        "by_type": self.by_type(rows, model_ids[model]),
                    }
                    for model, rows in model_rows.items()
                },
            }
        current_status = status_path()
        status = current_status.read_text().strip() if current_status.exists() else "UNKNOWN"
        current_native = native_path()
        mtimes = [
            path.stat().st_mtime
            for path in (current_native, CAPX_PILOT_PATH, current_status)
            if path.exists()
        ]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_updated_at": datetime.fromtimestamp(
                max(mtimes), timezone.utc
            ).isoformat() if mtimes else None,
            "run_status": status,
            "target_cases": len(self.questions),
            "models": MODEL_META,
            "coverage": {model: len(rows) for model, rows in model_rows.items()},
            "common_coverage": len(common),
            "ours_auto_coverage": len(ours_auto),
            "scopes": scopes,
            "cases": cases,
        }

    def detail(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.questions):
            raise IndexError(index)
        native, pilot = self.sources()
        question = self.questions[index]
        methods = self.methods_for_case(index, native, pilot)
        native_record = native.get(index)
        pilot_record = pilot.get(index)
        capx_record = native_record if (
            native_record and "spatial" in native_record.get("predictions", {})
        ) else pilot_record
        frame_record = native_record or pilot_record
        frame_count = len(frame_record.get("image_details", {}).get("frame_numbers", [])) if frame_record else 0
        auto, aligned = autotom_metric(index)
        return {
            "index": index,
            "episode": question["episode"],
            "question_type": str(question["question_type"]),
            "question": question["question"],
            "answer": str(question["answer"]).lower(),
            "methods": methods,
            "frames": [f"/api/frame/{index}/{slot}" for slot in range(frame_count)],
            "spatial_social": {
                "codes": native_record.get("codes", {}) if native_record else {},
                "traces": native_record.get("traces", {}) if native_record else {},
                "components": native_record.get("social_components", []) if native_record else [],
                "reasoning": native_record.get("social_reasoning", {}) if native_record else {},
            },
            "capx": {
                "code": capx_record.get("codes", {}).get("spatial", "") if capx_record else "",
                "trace": capx_record.get("traces", {}).get("spatial", "") if capx_record else "",
                "source": methods.get("capx", {}).get("source"),
            },
            "autotom": {"metrics": auto or {}, "aligned_rerun": aligned},
        }

    def frame_path(self, index: int, slot: int) -> Path:
        native, pilot = self.sources()
        record = native.get(index) or pilot.get(index)
        if record is None:
            raise FileNotFoundError(index)
        details = record.get("image_details", {})
        numbers = details.get("frame_numbers", [])
        if not 0 <= slot < len(numbers):
            raise FileNotFoundError(slot)
        episode = int(details.get("episode", self.questions[index]["episode"]))
        direct = FRAMES_ROOT / f"task_{episode}"
        if direct.is_dir():
            episode_root = direct
        else:
            matches = list(FRAMES_ROOT.glob(f"*/task_{episode}"))
            if len(matches) != 1:
                raise FileNotFoundError(f"task_{episode}")
            episode_root = matches[0]
        path = episode_root / "script" / "0" / f"Action_{int(numbers[slot]):04d}_0_normal.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


DATA = ReportData()


class Handler(BaseHTTPRequestHandler):
    def json_response(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def file_response(self, path: Path, cache: bool = True) -> None:
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in {"/", "/index.html"}:
                return self.file_response(SITE / "index.html", cache=False)
            if self.path in {"/app.js", "/styles.css"}:
                return self.file_response(SITE / self.path[1:], cache=False)
            if self.path == "/api/summary":
                return self.json_response(DATA.summary())
            detail = re.fullmatch(r"/api/case/(\d+)", self.path)
            if detail:
                return self.json_response(DATA.detail(int(detail.group(1))))
            frame = re.fullmatch(r"/api/frame/(\d+)/(\d+)", self.path)
            if frame:
                return self.file_response(
                    DATA.frame_path(int(frame.group(1)), int(frame.group(2)))
                )
            self.json_response({"error": "not found"}, 404)
        except (FileNotFoundError, IndexError) as error:
            self.json_response({"error": str(error)}, 404)
        except Exception as error:  # Keep the live dashboard debuggable.
            self.json_response({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Spatial-Social comparison site: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
