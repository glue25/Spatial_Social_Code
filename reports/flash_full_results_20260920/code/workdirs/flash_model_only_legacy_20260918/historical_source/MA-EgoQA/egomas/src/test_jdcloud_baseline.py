import unittest

from egomas.src.jdcloud_gemini import _extract_text, _export_value
from egomas.src.run_jdcloud_baseline import sanitize_selection, summarize


class JDCloudBaselineTest(unittest.TestCase):
    def test_extract_text(self) -> None:
        payload = {
            "candidates": [
                {"content": {"parts": [{"text": "A"}, {"text": "B"}]}}
            ]
        }
        self.assertEqual(_extract_text(payload), "A\nB")

    def test_export_value(self) -> None:
        text = 'export GEMINI_API_KEY="secret"\nexport GEMINI_MODEL=Model-X\n'
        self.assertEqual(_export_value(text, "GEMINI_API_KEY"), "secret")
        self.assertEqual(_export_value(text, "GEMINI_MODEL"), "Model-X")

    def test_sanitize_selection(self) -> None:
        selection = [{"name": "jake", "query": "dessert"}, {"name": "bad"}]
        self.assertEqual(
            sanitize_selection(selection, "question", 0),
            [{"name": "Jake", "query": "dessert"}],
        )

    def test_summary(self) -> None:
        rows = [
            {"category": "a", "subcategory": "x", "correct": True},
            {"category": "a", "subcategory": "x", "correct": False},
        ]
        report = summarize(rows, "model")
        self.assertEqual(report["overall"]["correct"], 1)
        self.assertEqual(report["overall"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
