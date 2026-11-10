import json
import unittest

from egomas.src.run_spatial_social_jdcloud import (
    normalize_social_code,
    normalize_spatial_code,
    question_premise,
    summarize,
)


class SpatialSocialJDCloudTest(unittest.TestCase):
    def test_inline_temporal_options_are_hidden(self) -> None:
        row = {
            "question": "Which happened last? A) SECRET_A B) SECRET_B C) C D) D E) E",
            "options": ["A", "B", "C", "D", "E"],
        }
        premise = question_premise(row)
        self.assertEqual(premise, "Which happened last?")
        self.assertNotIn("SECRET", premise)

    def test_normalizers_keep_only_safe_runtime_calls(self) -> None:
        spatial = normalize_spatial_code(
            'S.event(0, "Alice", "opens", "door", "caption", 0.9)\n'
            'import os\nM.goal("Alice", "leave", 0.8, "guess")'
        )
        social = normalize_social_code(
            'M.observe(0, "Alice", "door open", "caption", 0.8)\n'
            'S.event(0, "Alice", "opens", "door")'
        )
        self.assertIn("S.event", spatial)
        self.assertNotIn("import", spatial)
        self.assertNotIn("M.goal", spatial)
        self.assertIn("M.observe", social)
        self.assertNotIn("S.event", social)

    def test_paired_summary(self) -> None:
        rows = [
            {
                "category": "a",
                "subcategory": "x",
                "baseline_correct": True,
                "correct": False,
            },
            {
                "category": "a",
                "subcategory": "x",
                "baseline_correct": False,
                "correct": True,
            },
        ]
        report = summarize(rows, "model")
        self.assertEqual(report["overall"]["baseline"]["accuracy"], 0.5)
        self.assertEqual(report["overall"]["spatial_social"]["accuracy"], 0.5)
        self.assertEqual(report["contingency"]["net_spatial_social_wins"], 0)


if __name__ == "__main__":
    unittest.main()
