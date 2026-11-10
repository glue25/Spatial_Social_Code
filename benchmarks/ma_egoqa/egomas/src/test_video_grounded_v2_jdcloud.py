import json
import unittest

from egomas.src.run_video_grounded_v2_jdcloud import (
    _answer_index,
    _clip_start_seconds,
    _json_value,
    normalize_grounded_records,
    normalize_social_records,
    rerank_clips,
    rescore_records,
    should_switch_to_structured,
)


class GroundedVideoV2Test(unittest.TestCase):
    def test_embedded_json_and_answer(self) -> None:
        text = '```json\n{"answer":"C","evidence_ids":["C1R1"]}\n```'
        self.assertEqual(_json_value(text)["answer"], "C")
        self.assertEqual(_answer_index(text)[0], 2)
        self.assertEqual(_answer_index('```json\n{"answer":"')[0], -1)
        self.assertEqual(_answer_index("")[0], -1)
        prediction, recovered = _answer_index(
            '```json\n{"answer":"D","clip_id":"DAY1_A1_JAKE_12153000.mp4",'
            '"evidence_ids":["C2R1"],"reason":"unfinished'
        )
        self.assertEqual(prediction, 3)
        self.assertEqual(recovered["clip_id"], "DAY1_A1_JAKE_12153000.mp4")
        self.assertEqual(recovered["evidence_ids"], ["C2R1"])

    def test_clip_absolute_time(self) -> None:
        self.assertEqual(_clip_start_seconds("DAY1_A1_JAKE_12153000.mp4"), 12 * 3600 + 15 * 60 + 30)

    def test_grounded_records_are_bound_to_clip_and_time(self) -> None:
        raw = json.dumps(
            {
                "records": [
                    {
                        "kind": "speech",
                        "time_span": "00:03-00:07",
                        "subject": "Jake",
                        "claim": "Jake says the cup is his",
                        "grounding": "audible",
                        "quote": "That is mine",
                        "confidence": 1.0,
                    },
                    {
                        "kind": "belief",
                        "time_span": "00:03-00:07",
                        "subject": "Jake",
                        "claim": "Jake wants the cup",
                        "grounding": "audible",
                    },
                ]
            }
        )
        clip = {"id": "DAY1_A1_JAKE_12153000.mp4"}
        records = normalize_grounded_records(raw, clip, 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["evidence_id"], "C2R1")
        self.assertEqual(records[0]["clip_id"], clip["id"])
        self.assertEqual(records[0]["observer"], "Jake")
        self.assertEqual(records[0]["local_time_span"], "00:03-00:07")
        self.assertEqual(records[0]["absolute_start_seconds"], 12 * 3600 + 15 * 60 + 33)
        self.assertEqual(records[0]["confidence"], 0.9)

    def test_invalid_time_and_unsupported_social_evidence_are_rejected(self) -> None:
        clip = {"id": "DAY1_A1_JAKE_12153000.mp4"}
        grounded = normalize_grounded_records(
            '{"records":[{"kind":"event","time_span":"00:40-00:50",'
            '"subject":"Jake","claim":"walks","grounding":"visible"}]}',
            clip,
            1,
        )
        self.assertEqual(grounded, [])
        social = normalize_social_records(
            '{"records":[{"kind":"belief","agent":"Jake","claim":"x",'
            '"evidence_ids":["missing"],"confidence":0.9}]}',
            {"C1R1"},
        )
        self.assertEqual(social, [])

    def test_reranker_uses_question_relevance_and_wearer_diversity(self) -> None:
        candidates = [
            {"id": "DAY1_A1_JAKE_12000000.mp4", "score": 10.0},
            {"id": "DAY1_A1_JAKE_12003000.mp4", "score": 9.9},
            {"id": "DAY1_A2_ALICE_12000000.mp4", "score": 9.8},
        ]
        captions = {
            candidates[0]["id"]: "Jake walks through the kitchen",
            candidates[1]["id"]: "Jake holds the red water bottle",
            candidates[2]["id"]: "Alice asks who owns the red water bottle",
        }
        ranked = rerank_clips(candidates, "Who owned the red water bottle?", captions, 2)
        self.assertEqual({row["id"] for row in ranked}, {candidates[1]["id"], candidates[2]["id"]})

    def test_switch_gate_requires_explicit_agreement_and_grounding(self) -> None:
        structured = {"evidence_ids": ["C1R1", "C1R2"]}
        verifier = {"favored": "structured"}
        self.assertTrue(
            should_switch_to_structured(
                "task_coordination", "single_span", 2, 2, structured, verifier
            )
        )
        self.assertFalse(
            should_switch_to_structured(
                "task_coordination", "multi_span", 2, 2, structured, verifier
            )
        )
        self.assertFalse(
            should_switch_to_structured(
                "social_interaction", "single_span", 2, 4, structured, verifier
            )
        )
        self.assertFalse(
            should_switch_to_structured(
                "theory_of_mind", "theory_of_mind", 2, 2, structured, {}
            )
        )

    def test_rescore_can_route_only_theory_of_mind(self) -> None:
        base = {
            "direct_prediction_index": 0,
            "structured_prediction_index": 1,
            "answer_index": 1,
            "structured_data": {"evidence_ids": ["C1R1"]},
            "verifier_raw": '{"answer":"B","favored":"structured"}',
        }
        records = {
            1: {**base, "category": "social_interaction", "subcategory": "single_span"},
            2: {**base, "category": "theory_of_mind", "subcategory": "theory_of_mind"},
        }
        rescore_records(records, {"theory_of_mind"})
        self.assertEqual(records[1]["prediction_index"], 0)
        self.assertEqual(records[2]["prediction_index"], 1)
        self.assertTrue(records[2]["correct"])


if __name__ == "__main__":
    unittest.main()
