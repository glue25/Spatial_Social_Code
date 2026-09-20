"""Optional AutoToM-to-Social-Code adapter.

Nothing in the default Spatial-Social Code path imports this module.  It is
loaded lazily only for the Bayesian source or an explicitly Bayesian hybrid.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _option_texts(question: str) -> dict[str, str]:
    match = re.search(
        r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*Please respond with either a or b\.?\s*$",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("could not extract A/B option text from MMToM question")
    return {"a": match.group(1).strip(), "b": match.group(2).strip()}


def _semantic_text(text: str) -> str:
    replacements = (
        ("refrigerator", "fridge"),
        ("bag of chips", "chips"),
        ("bottle of wine", "wine"),
        ("dishbowl", "dish bowl"),
        ("1st", "first"),
        ("2nd", "second"),
        ("3rd", "third"),
        ("4th", "fourth"),
        ("5th", "fifth"),
        ("6th", "sixth"),
    )
    normalized = text.lower()
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def load_autotom_social(
    index: int, question: str, prefixes: list[str]
) -> tuple[str, str, dict[str, float | str]]:
    parsed_root = ROOT / "AutoToM" / "results" / "parsed_result"
    metrics_root = ROOT / "AutoToM" / "results" / "metrics"
    source_prefix = None
    parsed_path = None
    metrics_path = None
    for prefix in prefixes:
        candidate_parsed = parsed_root / f"{prefix}_MMToM-QA_{index}.json"
        candidates = list(
            metrics_root.glob(
                f"{prefix}_MMToM-QA_{index}_back1_reduce1_seed42_metrics*.json"
            )
        )
        if candidate_parsed.is_file() and candidates:
            source_prefix = prefix
            parsed_path = candidate_parsed
            metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
            break
    if source_prefix is None or parsed_path is None or metrics_path is None:
        raise FileNotFoundError(
            f"missing AutoToM result for index {index}; tried prefixes {prefixes}"
        )

    parsed = json.loads(parsed_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    choices = parsed["choices"]
    probabilities = [float(value) for value in metrics["probs"]]
    if len(choices) != 2 or len(probabilities) != 2:
        raise ValueError(f"index {index}: expected two AutoToM choices/probabilities")

    raw_options = _option_texts(question)
    normalized_options = {
        label: _semantic_text(option) for label, option in raw_options.items()
    }
    mapped: dict[str, float] = {}
    for choice, probability in zip(choices, probabilities, strict=True):
        normalized = _semantic_text(str(choice))
        labels = [
            label for label, option in normalized_options.items() if normalized == option
        ]
        if len(labels) != 1:
            raise ValueError(
                f"index {index}: AutoToM choice does not align with A/B: {choice!r}"
            )
        mapped[labels[0]] = probability

    agent = str(parsed["inf_agent_name"])
    variable = str(parsed["inf_var_name"])
    evidence = "AutoToM Bayesian inverse-planning posterior"
    lines = []
    for choice, probability in zip(choices, probabilities, strict=True):
        if variable == "Belief":
            lines.append(
                f"M.belief(-1, {agent!r}, {str(choice)!r}, {probability!r}, {evidence!r})"
            )
        elif variable == "Goal":
            lines.append(
                f"M.goal({agent!r}, {str(choice)!r}, {probability!r}, {evidence!r})"
            )
        else:
            lines.append(f"M.unknown({agent!r}, {variable!r})")
        lines.append(
            f"M.likelihood({agent!r}, 'observed action sequence', {str(choice)!r}, {probability!r})"
        )
    predicted = "a" if mapped["a"] >= mapped["b"] else "b"
    return "\n".join(lines), "", {
        "prediction": predicted,
        "prob_a": mapped["a"],
        "prob_b": mapped["b"],
        "source_prefix": source_prefix,
    }
