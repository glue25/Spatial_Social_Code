"""Build an AutoToM CSV whose answer candidates match the evaluation JSONL."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def option_texts(question: str) -> list[str]:
    match = re.search(
        r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*Please respond with either a or b\.?\s*$",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("could not extract A/B options")
    return [match.group(1).strip(), match.group(2).strip()]


def autotom_text(text: str) -> str:
    replacements = (
        ("fridge", "refrigerator"),
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
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def semantic_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", autotom_text(text.lower())).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "MMToM-QA" / "Benchmark" / "questions.json",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=ROOT / "AutoToM" / "benchmarks" / "full_data_formatted" / "MMToM-QA.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_autotom_aligned.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "spatial_social_code" / "mmtom_autotom_alignment.json",
    )
    parser.add_argument("--print-indices", action="store_true")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.open() if line.strip()]
    with args.source_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None or len(rows) != len(questions):
        raise ValueError(f"row mismatch: CSV={len(rows)} questions={len(questions)}")

    changed_indices = []
    for index, (row, question) in enumerate(zip(rows, questions, strict=True)):
        raw_choices = option_texts(question["question"])
        aligned_choices = [autotom_text(choice) for choice in raw_choices]
        old_choices = ast.literal_eval(row["answer_choices"])
        if {semantic_text(choice) for choice in old_choices} != {
            semantic_text(choice) for choice in aligned_choices
        }:
            changed_indices.append(index)
        row["answer_choices"] = repr(aligned_choices)
        answer_index = 0 if question["answer"].lower() == "a" else 1
        row["gt_answer"] = aligned_choices[answer_index]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "questions": str(args.questions),
        "source_csv": str(args.source_csv),
        "output_csv": str(args.output_csv),
        "total_questions": len(rows),
        "candidate_mismatch_count": len(changed_indices),
        "candidate_mismatch_indices": changed_indices,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.print_indices:
        print("\n".join(str(index) for index in changed_indices))
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
