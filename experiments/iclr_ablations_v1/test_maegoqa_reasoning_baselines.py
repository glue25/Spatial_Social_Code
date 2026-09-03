import unittest

from experiments.iclr_ablations_v1.run_maegoqa_reasoning_baselines import (
    NEW_CONDITIONS,
    comparison,
    exact_mcnemar,
    summarize,
)


def _row(index: int, winner: str) -> dict:
    names = (
        *NEW_CONDITIONS,
        "saved_raw_only_rerun",
        "saved_raw_spatial",
        "saved_raw_social",
        "saved_raw_spatial_social",
        "saved_trace_only_spatial_social",
        "saved_saved_direct",
        "saved_saved_full_primary",
    )
    correct = {name: name == winner for name in names}
    return {
        "index": index,
        "category": "test",
        "predictions": {name: 0 if value else 1 for name, value in correct.items()},
        "correct": correct,
    }


class ReasoningBaselineTest(unittest.TestCase):
    def test_summary_registers_no_code_baselines_and_call_counts(self) -> None:
        rows = [_row(0, "option_decomposition"), _row(1, "saved_raw_only_rerun")]
        report = summarize(rows, [0, 1], "model")
        self.assertEqual(
            report["logical_calls_per_question"],
            {
                "option_decomposition": 1,
                "natural_language_scratchpad": 1,
                "three_call_natural_language": 3,
            },
        )
        self.assertEqual(report["conditions"]["option_decomposition"]["correct"], 1)
        self.assertFalse(report["label_access_in_prompt"])
        self.assertTrue(report["task_question_conditioned"])

    def test_comparison_direction_and_exact_test(self) -> None:
        rows = [_row(0, "option_decomposition"), _row(1, "saved_raw_only_rerun")]
        result = comparison(rows, "saved_raw_only_rerun", "option_decomposition")
        self.assertEqual(result["left_only_correct"], 1)
        self.assertEqual(result["right_only_correct"], 1)
        self.assertEqual(result["net_right_wins"], 0)
        self.assertEqual(exact_mcnemar(1, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
