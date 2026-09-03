import unittest

from experiments.iclr_ablations_v1.run_mmtom_sam_ablation import (
    exact_mcnemar,
    extract_terms,
    summarize,
)


class MMToMSAMAblationTest(unittest.TestCase):
    def test_extracts_all_choice_terms_without_gold(self) -> None:
        question = """Initial state mentions many items.
Question: Which is more likely? (a) Ana wants a dishbowl. (b) Ana wants a
plate inside the 2nd kitchen cabinet."""
        self.assertEqual(
            extract_terms(question), ["dish bowl", "plate", "2nd kitchen cabinet"]
        )

    def test_summary_uses_paired_direction(self) -> None:
        rows = [
            {
                "question_type": "1.1",
                "vlm_only_correct": True,
                "sam_correct": False,
                "sam_term_coverage": {"plate": True},
            },
            {
                "question_type": "1.1",
                "vlm_only_correct": False,
                "sam_correct": True,
                "sam_term_coverage": {"plate": False},
            },
        ]
        report = summarize(rows, [0, 1], "model")
        self.assertEqual(report["contingency"]["net_sam_wins"], 0)
        self.assertEqual(report["contingency"]["mcnemar_exact_p"], 1.0)
        self.assertEqual(report["sam_target_term_coverage"], 0.5)
        self.assertEqual(exact_mcnemar(1, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
