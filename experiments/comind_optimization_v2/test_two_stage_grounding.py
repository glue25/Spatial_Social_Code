import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_two_stage_grounding.py")
SPEC = importlib.util.spec_from_file_location("two_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TwoStageGroundingTest(unittest.TestCase):
    def row(self, task="joint_attention"):
        roles = ("helper",) if task == "scoia" else ("leader", "helper")
        return {
            "task": task,
            "images": [
                {"role": role, "kind": "prediction", "time": 1.0, "path": f"/{role}.jpg"}
                for role in roles
            ],
        }

    def test_joint_views_are_grounded_independently(self):
        calls = []

        def fake(path, identity, url):
            calls.append(path.name)
            box = [10, 20, 30, 40] if path.name == "leader.jpg" else [50, 60, 80, 100]
            return [{"box_xyxy": box, "score": 0.9, "label": identity}]

        prediction, proposals = MODULE.ground_row(self.row(), "plate", "local", fake)
        self.assertEqual(calls, ["leader.jpg", "helper.jpg"])
        self.assertEqual(prediction["left_bbox_xyxy"], [10, 20, 30, 40])
        self.assertEqual(prediction["right_bbox_xyxy"], [50, 60, 80, 100])
        self.assertNotEqual(prediction["left_bbox_xyxy"], prediction["right_bbox_xyxy"])
        self.assertEqual(set(proposals), {"leader", "helper"})

    def test_scoia_uses_helper_prediction_frame(self):
        prediction, _ = MODULE.ground_row(
            self.row("scoia"), "bowl", "local",
            lambda path, identity, url: [{"box_xyxy": [1, 2, 11, 22], "score": 0.8, "label": identity}],
        )
        self.assertEqual(prediction["bbox_xyxy"], [1, 2, 11, 22])

    def test_handover_preserves_both_views_and_selects_by_score(self):
        def fake(path, identity, url):
            score = 0.4 if path.name == "leader.jpg" else 0.7
            return [{"box_xyxy": [1, 2, 11, 22], "score": score, "label": identity}]

        prediction, _ = MODULE.ground_row(self.row("handover"), "salt shaker", "local", fake)
        self.assertEqual(prediction["selected_view"], "helper")
        self.assertIn("leader_bbox_xyxy", prediction)
        self.assertIn("helper_bbox_xyxy", prediction)

    def test_invalid_high_score_box_is_skipped(self):
        proposals = [
            {"box_xyxy": [-1, 2, 3, 4], "score": 0.99},
            {"box_xyxy": [1, 2, 3, 4], "score": 0.8},
        ]
        self.assertEqual(MODULE.select_box(proposals), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
