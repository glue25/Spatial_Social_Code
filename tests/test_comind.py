from __future__ import annotations

import unittest
import io

from spatial_social_code.comind import box_iou, evaluate_paired_rows, score_sample
from spatial_social_code.prepare_comind_pilot import sample_times, uniformly_select
from spatial_social_code.run_comind_codex_pilot import (
    code_schema,
    normalize_code,
    prediction_schema,
)
from spatial_social_code.http_range_reader import HTTPRangeReader


class MemoryRangeReader(HTTPRangeReader):
    def __init__(self, payload: bytes, block_size: int = 4) -> None:
        self.payload = payload
        super().__init__("memory://test", len(payload), block_size=block_size, max_blocks=2)

    def _fetch_block(self, index: int) -> bytes:
        start = index * self.block_size
        data = self.payload[start : start + self.block_size]
        self.bytes_downloaded += len(data)
        return data


class CoMindMetricsTest(unittest.TestCase):
    def test_http_range_reader_seek_and_cache(self) -> None:
        reader = MemoryRangeReader(b"abcdefghijkl")
        self.assertEqual(reader.read(3), b"abc")
        reader.seek(-4, io.SEEK_END)
        self.assertEqual(reader.read(), b"ijkl")
        reader.seek(0)
        self.assertEqual(reader.read(4), b"abcd")
        self.assertLessEqual(len(reader._cache), 2)
        reader.close()

    def test_official_context_times(self) -> None:
        self.assertEqual(sample_times(20.0), [10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        self.assertEqual(sample_times(5.0), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_uniform_selection_includes_endpoints(self) -> None:
        rows = list(range(11))
        self.assertEqual(uniformly_select(rows, 3), [0, 5, 10])

    def test_codex_schemas_are_strict(self) -> None:
        for task in ("joint_attention", "scoia", "handover"):
            schema = prediction_schema(task)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(code_schema("spatial_code")["required"], ["spatial_code"])

    def test_code_normalization(self) -> None:
        code = normalize_code(
            [
                "S.fact(0, 'near', 'cup', 'plate')",
                "print('unsafe')",
                "S.fact(0, 'near', 'cup', 'plate')",
            ],
            "S",
        )
        self.assertEqual(code, "S.fact(0, 'near', 'cup', 'plate')")

    def test_box_iou_xywh(self) -> None:
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertAlmostEqual(
            box_iou([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3
        )

    def test_joint_attention_aliases(self) -> None:
        truth = {
            "bounding_boxes": [[0, 0, 10, 10], [5, 5, 8, 8]],
            "cue_types": ["communication", "gesture"],
            "object_category_level_1": "mixing bowl",
        }
        prediction = {
            "left_bbox": [0, 0, 10, 10],
            "right_bbox": [5, 5, 8, 8],
            "cue_type": ["verbal", "gestural"],
            "category_l1": "mixing_bowl",
        }
        self.assertTrue(all(score_sample("joint_attention", truth, prediction).values()))

    def test_scoia_label_mapping(self) -> None:
        truth = {
            "bbox": [1, 2, 3, 4],
            "annot_type": "AG",
            "verbs": ["grab", "take"],
            "nouns": ["cup"],
        }
        prediction = {
            "bbox": [1, 2, 3, 4],
            "cue_type": ["gestural", "verbal"],
            "action_verb": "take",
            "action_noun_l1": "cup",
        }
        self.assertTrue(all(score_sample("scoia", truth, prediction).values()))

    def test_handover_tth_and_flow(self) -> None:
        truth = {
            "bbox": [1, 2, 3, 4],
            "delivering_flow": "ltr",
            "initiator": "left",
            "initiation_type": ["gestural"],
            "object_category_level_1": "spoon",
            "start_time": 10.0,
            "end_time": 12.0,
        }
        prediction = {
            "bbox": [1, 2, 3, 4],
            "delivery_flow": "left_to_right",
            "initiator": "left",
            "initiation_type": ["gesture"],
            "category_l1": "spoon",
            "time_to_handover": 2.2,
        }
        self.assertTrue(all(score_sample("handover", truth, prediction).values()))

    def test_paired_evaluation_excludes_unpaired_failures(self) -> None:
        annotation_root = "external_datasets/comind/annotations"
        import json
        from pathlib import Path

        data = json.loads(
            (Path(annotation_root) / "dataset_joint_attention_consolidated.json").read_text()
        )["data"]
        recording_id = next(
            rid for rid, samples in data.items() if any(isinstance(v, dict) for v in samples.values())
        )
        sample_id, truth = next(
            (sid, value)
            for sid, value in data[recording_id].items()
            if isinstance(value, dict)
        )
        prediction = {
            "left_bbox": truth["bounding_boxes"][0],
            "right_bbox": truth["bounding_boxes"][1],
            "cue_type": truth["cue_types"],
            "category_l1": truth["object_category_level_1"],
        }
        base = {
            "task": "joint_attention",
            "recording_id": recording_id,
            "sample_id": sample_id,
        }
        report = evaluate_paired_rows(
            annotation_root,
            [
                {**base, "condition": "direct", "prediction": prediction},
                {**base, "condition": "spatial_social", "error": "bad json"},
            ],
        )
        self.assertEqual(report["paired_samples"]["joint_attention"], 0)
        self.assertEqual(
            report["failed_or_missing_by_condition"]["spatial_social"]["joint_attention"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
