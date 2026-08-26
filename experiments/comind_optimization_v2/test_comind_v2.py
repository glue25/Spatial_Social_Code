import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("comind_v2.py")
SPEC = importlib.util.spec_from_file_location("comind_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComindV2Test(unittest.TestCase):
    def test_coordinate_hypotheses_are_explicit(self):
        self.assertEqual(MODULE.convert_box([100, 100, 200, 300], "xywh"), [100, 100, 200, 300])
        self.assertEqual(MODULE.convert_box([100, 100, 200, 300], "xyxy"), [100, 100, 100, 200])
        self.assertEqual(MODULE.convert_box([0.5, 0.5, 0.75, 1.0], "xyxy"), [704, 704, 352, 704])

    def test_invalid_boxes_are_rejected(self):
        self.assertIsNone(MODULE.convert_box([1, 2, 3], "xywh"))
        self.assertIsNone(MODULE.convert_box([100, 100, -2, 5], "xywh"))
        self.assertIsNone(MODULE.convert_box([100, 100, 50, 50], "xyxy"))

    def test_trace_identity_and_bbox_extraction(self):
        row = {"spatial_trace": "\n".join([
            '{"kind":"spatial_fact","values":{"relation":"level_1_object_category","subject":"wooden spoon","object":"utensil","confidence":0.9}}',
            '{"kind":"spatial_fact","values":{"relation":"bounding_box","object":"[10,20,30,40]","confidence":0.8}}',
        ])}
        candidates = MODULE.trace_candidates(row, "handover")
        self.assertEqual(candidates["category_l1"][0][0], "wooden spoon")
        self.assertEqual(candidates["category_l1"][0][2], "trace_subject_identity")
        self.assertEqual(candidates["iou_at_0_5"][0][0], [10.0, 20.0, 30.0, 40.0])

    def test_recording_fold_is_deterministic(self):
        self.assertEqual(MODULE.fold_for("recording-a", 5), MODULE.fold_for("recording-a", 5))
        self.assertTrue(0 <= MODULE.fold_for("recording-a", 5) < 5)

    def test_scalar_mapping_requires_replicated_support(self):
        policy = {"maps": {"trace_relation": {"cabbage": {"label": "vegetable", "support": 2, "purity": 1.0}}}, "reliability": {"trace_relation": 0.8, "direct": 0.1, "spatial_social": 0.1}}
        item = {"task": "handover", "direct": {"category_l1": "unknown"}, "spatial_social": {"category_l1": "food"}, "trace_candidates": {"category_l1": [("cabbage", 0.9, "trace_relation")]}}
        value, info = MODULE.route_scalar(item, "category_l1", policy)
        self.assertEqual(value, "vegetable")
        self.assertEqual(info["source"], "trace_relation")


if __name__ == "__main__":
    unittest.main()
