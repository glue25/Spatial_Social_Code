import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_gpu_grounding.py")
SPEC = importlib.util.spec_from_file_location("evaluate_grounding", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class JointCoordinateTransformTest(unittest.TestCase):
    def test_leader_is_half_scaled_into_left_mosaic(self):
        value = MODULE.individual_xyxy_to_joint_mosaic_xywh([600, 800, 800, 1000], "leader")
        self.assertEqual(value, [300, 400, 100, 100])

    def test_helper_is_half_scaled_and_offset_into_right_mosaic(self):
        value = MODULE.individual_xyxy_to_joint_mosaic_xywh([600, 800, 800, 1000], "helper")
        self.assertEqual(value, [1004, 400, 100, 100])

    def test_transform_is_also_valid_for_handover_selected_view(self):
        leader = MODULE.individual_xyxy_to_joint_mosaic_xywh([200, 400, 300, 600], "leader")
        helper = MODULE.individual_xyxy_to_joint_mosaic_xywh([200, 400, 300, 600], "helper")
        self.assertEqual(leader, [100, 200, 50, 100])
        self.assertEqual(helper, [804, 200, 50, 100])


if __name__ == "__main__":
    unittest.main()
