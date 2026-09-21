import json
import unittest

from egomas.src.run_video_provenance_v3_jdcloud import (
    build_claim_plan,
    covered_claim_ids,
    final_answer_index,
    normalize_social_code,
    normalize_spatial_code,
    spatial_records,
    summarize,
)
from runtime import execute_social, execute_spatial


class ProvenanceVideoV3Test(unittest.TestCase):
    def test_final_answer_parser_does_not_treat_a_check_as_the_answer(self) -> None:
        truncated = '{"checks":[{"option":"A","status":"contradicted"}'
        self.assertEqual(final_answer_index(truncated)[0], -1)
        self.assertEqual(final_answer_index('{"answer":"D"}')[0], 3)

    def test_claim_plan_and_spatial_coverage(self) -> None:
        claims = build_claim_plan(
            '{"claims":[{"option":"A","claim":"Alice laughs"},'
            '{"option":"B","claim":"Bob dances"}]}',
            ["Alice laughs", "Bob dances"],
        )
        self.assertEqual([claim["claim_id"] for claim in claims], ["C1", "C2"])
        raw = (
            'S.event(1, "Shure", "observe", "Alice laughs", source="video", '
            'confidence=0.8, evidence_id="S1", '
            'clip_id="DAY3_A6_SHURE_21423000.mp4", observer="Shure", '
            'evidence_span="00:03-00:08", quote="", claim_ids="C1")'
        )
        code = normalize_spatial_code(
            raw,
            {"DAY3_A6_SHURE_21423000.mp4"},
            valid_claim_ids={"C1", "C2"},
        )
        records = spatial_records(execute_spatial(code))
        self.assertEqual(covered_claim_ids(records, {"C1", "C2"}), {"C1"})

    def test_spatial_normalizer_requires_valid_clip_time_observer_and_unique_id(self) -> None:
        clip = "DAY3_A6_SHURE_21423000.mp4"
        raw = "\n".join(
            [
                'S.event(3, "Shure", "say", "use chopsticks", source="video", '
                'confidence=0.8, evidence_id="S1", clip_id="DAY3_A6_SHURE_21423000.mp4", '
                'observer="Shure", evidence_span="00:03-00:08", quote="use chopsticks")',
                'S.event(5, "Jake", "say", "unsupported", source="video", '
                'confidence=0.8, evidence_id="S2", clip_id="DAY3_A6_SHURE_21423000.mp4", '
                'observer="Jake", evidence_span="00:05-00:09", quote="")',
                'S.event(6, "Shure", "repeat", "unsupported", source="video", '
                'confidence=0.8, evidence_id="S1", clip_id="DAY3_A6_SHURE_21423000.mp4", '
                'observer="Shure", evidence_span="00:06-00:09", quote="")',
            ]
        )
        code = normalize_spatial_code(raw, {clip})
        self.assertEqual(len(code.splitlines()), 1)
        trace = execute_spatial(code)
        self.assertEqual(set(spatial_records(trace)), {"S1"})

    def test_spatial_normalizer_maps_literal_t_placeholder_to_sequence(self) -> None:
        clip = "DAY3_A1_JAKE_21483000.mp4"
        raw = (
            'S.event(t, "Jake", "asks about", "conversion rate", source="video", '
            'confidence=0.9, evidence_id="S1", clip_id="DAY3_A1_JAKE_21483000.mp4", '
            'observer="Jake", evidence_span="00:00-00:05", quote="")'
        )
        code = normalize_spatial_code(raw, {clip})
        self.assertIn("S.event(1,", code)
        self.assertEqual(set(spatial_records(execute_spatial(code))), {"S1"})

    def test_spatial_normalizer_preserves_event_role_and_phase(self) -> None:
        clip = "DAY3_A1_JAKE_21483000.mp4"
        raw = (
            'S.event(1, "Jake", "pours", "vinegar", source="video", '
            'confidence=0.9, evidence_id="S1", clip_id="DAY3_A1_JAKE_21483000.mp4", '
            'observer="Jake", evidence_span="00:00-00:05", quote="", '
            'event_role="executor", phase="during")'
        )
        code = normalize_spatial_code(raw, {clip})
        values = json.loads(execute_spatial(code))["values"]
        self.assertEqual(values["event_role"], "executor")
        self.assertEqual(values["phase"], "during")

    def test_social_normalizer_requires_existing_spatial_references(self) -> None:
        raw = "\n".join(
            [
                'M.belief(3, "Katrina", "Shure suggested chopsticks", 0.7, '
                '"heard quote", evidence_ids="S1")',
                'M.intent(3, "Katrina", "guess", 0.7, "unsupported")',
                'M.goal("Katrina", "guess", 0.7, "bad id", evidence_ids="S9")',
            ]
        )
        code = normalize_social_code(raw, {"S1"})
        self.assertEqual(len(code.splitlines()), 1)
        trace = execute_social(code, {"S1"}, require_evidence=True)
        self.assertEqual(json.loads(trace)["values"]["evidence_ids"], ["S1"])

    def test_paired_summary(self) -> None:
        rows = [
            {
                "baseline_correct": True,
                "structured_correct": False,
                "structured_prediction_index": 1,
                "correct": False,
                "category": "a",
                "subcategory": "x",
                "valid_evidence_ids": ["S1"],
            },
            {
                "baseline_correct": False,
                "structured_correct": True,
                "structured_prediction_index": 0,
                "correct": True,
                "category": "a",
                "subcategory": "x",
                "valid_evidence_ids": ["S1"],
            },
        ]
        report = summarize(rows, "model")
        self.assertEqual(report["overall"]["baseline"]["accuracy"], 0.5)
        self.assertEqual(
            report["overall"]["standalone_spatial_social"]["accuracy"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
