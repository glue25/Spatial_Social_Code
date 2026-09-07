from __future__ import annotations

import json
import unittest

from runtime import UnsafeProgram, execute_social, execute_social_legacy, execute_spatial
from executable_hypothesis import (
    compile_payload,
    ensure_spatial_evidence_ids,
    infer_query_variable_type,
    score_hypotheses,
)
from social_code import (
    _location_match,
    answer_posterior_probs,
    answer_options,
    compile_rules,
    compile_social_code,
    compile_spatial_bridge,
    normalize_social_code,
    question_without_options,
)


QUESTION = (
    "Actions taken by Jessica: Jessica approaches the cabinet. "
    "She opens the cabinet. She takes the plate. "
    "Question: What is likely? (a) SECRET_A. (b) SECRET_B. "
    "Please respond with either a or b."
)


class SocialCodeTests(unittest.TestCase):
    def test_location_matching_keeps_cabinets_distinct(self) -> None:
        self.assertFalse(_location_match("cabinet", "second kitchen cabinet"))
        self.assertTrue(_location_match("7th cabinet", "kitchen cabinet 7"))

    def test_executable_query_variable_typing_uses_general_semantics(self) -> None:
        self.assertEqual(
            infer_query_variable_type("What did Mia believe was in the box?"),
            "belief",
        )
        self.assertEqual(
            infer_query_variable_type("Which event happened before the door opened?"),
            "temporal",
        )
        with self.assertRaises(UnsafeProgram):
            execute_social(
                "M.candidate('A', 'answer', 'A', 'x', variable_type='benchmark_label')"
            )

    def test_executable_all_requires_every_required_clause(self) -> None:
        spatial = execute_spatial(
            'S.event(1, "Ana", "pick up", "cup", evidence_id="S1")'
        )
        trace = execute_social(
            "\n".join(
                [
                    "M.candidate('Ana', 'answer', 'A', 'pick cup and plate', variable_type='world_state')",
                    "M.clause('A', 'A1', 'Ana picks cup')",
                    "M.clause('A', 'A2', 'Ana picks plate')",
                    "M.constraint('A', 'ALL', 'A1,A2')",
                    "M.support(1, 'A', 'A1', 'S1')",
                    "M.clause_unknown(1, 'A', 'A2', 'not visible')",
                    "M.candidate('Ana', 'answer', 'B', 'pick cup', variable_type='world_state')",
                    "M.clause('B', 'B1', 'Ana picks cup')",
                    "M.constraint('B', 'ALL', 'B1')",
                    "M.support(1, 'B', 'B1', 'S1')",
                ]
            ),
            valid_evidence_ids={"S1"},
            require_evidence=True,
        )
        decision = score_hypotheses(trace, spatial)
        self.assertEqual(decision.choice, "B")
        self.assertGreater(decision.scores["A"], 0.0)
        self.assertLess(decision.scores["A"], 1.0)

    def test_executable_unknown_is_not_negative_evidence(self) -> None:
        spatial = execute_spatial(
            'S.visible(1, "Ana", "cup", False, evidence_id="S1")'
        )
        trace = execute_social(
            "\n".join(
                [
                    "M.candidate('Ana', 'answer', 'A', 'unknown', variable_type='world_state')",
                    "M.clause('A', 'A1', 'unknown fact')",
                    "M.constraint('A', 'ALL', 'A1')",
                    "M.clause_unknown(0, 'A', 'A1')",
                    "M.candidate('Ana', 'answer', 'B', 'false', variable_type='world_state')",
                    "M.clause('B', 'B1', 'cup is visible', subject='Ana', predicate='visible cup')",
                    "M.constraint('B', 'ALL', 'B1')",
                    "M.contradict(1, 'B', 'B1', 'S1')",
                ]
            ),
            valid_evidence_ids={"S1"},
            require_evidence=True,
        )
        decision = score_hypotheses(trace, spatial)
        self.assertEqual(decision.choice, "A")
        self.assertEqual(decision.scores["A"], 0.0)
        self.assertEqual(decision.scores["B"], -1.0)

    def test_executable_belief_requires_viewpoint_chain(self) -> None:
        spatial = execute_spatial(
            'S.visible(1, "Bob", "cup", True, evidence_id="S1")'
        )
        trace = execute_social(
            "\n".join(
                [
                    "M.candidate('Alice', 'answer', 'A', 'Alice saw cup', variable_type='belief')",
                    "M.clause('A', 'A1', 'Alice believes cup exists', subject='Alice')",
                    "M.constraint('A', 'ALL', 'A1')",
                    "M.support(1, 'A', 'A1', 'S1')",
                    "M.candidate('Bob', 'answer', 'B', 'Bob saw cup', variable_type='belief')",
                    "M.clause('B', 'B1', 'Bob believes cup exists', subject='Bob')",
                    "M.constraint('B', 'ALL', 'B1')",
                    "M.support(1, 'B', 'B1', 'S1')",
                ]
            ),
            valid_evidence_ids={"S1"},
            require_evidence=True,
        )
        decision = score_hypotheses(trace, spatial)
        self.assertEqual(decision.choice, "B")
        self.assertEqual(decision.diagnostics["invalid_evidence_links"], 1)

    def test_executor_does_not_entail_responsibility(self) -> None:
        spatial = execute_spatial(
            'S.event(1, "Alice", "pour", "water", evidence_id="S1", event_role="executor")\n'
            'S.event(0, "Bob", "ask", "Alice", evidence_id="S2", event_role="initiator")'
        )
        trace = execute_social(
            "\n".join(
                [
                    "M.candidate('Alice', 'answer', 'A', 'Alice responsible', variable_type='responsibility/causality')",
                    "M.clause('A', 'A1', 'Alice initiated pouring', subject='Alice', predicate='pour')",
                    "M.constraint('A', 'ALL', 'A1')",
                    "M.support(1, 'A', 'A1', 'S1')",
                    "M.candidate('Bob', 'answer', 'B', 'Bob responsible', variable_type='responsibility/causality')",
                    "M.clause('B', 'B1', 'Bob asked Alice to pour', subject='Bob', predicate='ask')",
                    "M.constraint('B', 'ALL', 'B1')",
                    "M.support(0, 'B', 'B1', 'S2')",
                ]
            ),
            valid_evidence_ids={"S1", "S2"},
            require_evidence=True,
        )
        self.assertEqual(score_hypotheses(trace, spatial).choice, "B")

    def test_order_requires_both_endpoints_and_correct_chronology(self) -> None:
        spatial = execute_spatial(
            'S.event(1, "A", "open", "door", evidence_id="S1")\n'
            'S.event(2, "A", "sit", "chair", evidence_id="S2")'
        )
        trace = execute_social(
            "\n".join(
                [
                    "M.candidate('A', 'answer', 'A', 'open then sit', variable_type='temporal', constraint='ORDER')",
                    "M.clause('A', 'A1', 'open')",
                    "M.clause('A', 'A2', 'sit')",
                    "M.constraint('A', 'ORDER', 'A1,A2')",
                    "M.support(1, 'A', 'A1', 'S1')",
                    "M.support(2, 'A', 'A2', 'S2')",
                    "M.candidate('A', 'answer', 'B', 'sit then open', variable_type='temporal', constraint='ORDER')",
                    "M.clause('B', 'B1', 'sit')",
                    "M.clause('B', 'B2', 'open')",
                    "M.constraint('B', 'ORDER', 'B1,B2')",
                    "M.support(2, 'B', 'B1', 'S2')",
                    "M.support(1, 'B', 'B2', 'S1')",
                ]
            ),
            valid_evidence_ids={"S1", "S2"},
            require_evidence=True,
        )
        self.assertEqual(score_hypotheses(trace, spatial).choice, "A")

    def test_payload_answer_field_cannot_override_runtime_choice(self) -> None:
        spatial = execute_spatial(
            'S.event(1, "Ana", "pick up", "cup", evidence_id="S1")'
        )
        payload = {
            "answer": "B",
            "agent": "Ana",
            "variable_type": "world_state",
            "options": [
                {
                    "label": "A",
                    "clauses": [{"id": "A1", "text": "Ana picks cup", "subject": "Ana"}],
                    "evidence": [{"clause_id": "A1", "verdict": "support", "evidence_ids": ["S1"]}],
                },
                {
                    "label": "B",
                    "clauses": [{"id": "B1", "text": "Ana picks plate", "subject": "Ana"}],
                    "evidence": [{"clause_id": "B1", "verdict": "unknown", "evidence_ids": []}],
                },
            ],
        }
        _, _, decision = compile_payload(
            payload, [("A", "cup"), ("B", "plate")], spatial, "What did Ana pick?"
        )
        self.assertEqual(decision.choice, "A")

    def test_missing_spatial_ids_are_stably_indexed(self) -> None:
        trace = execute_spatial('S.fact(0, "in", "cup", "box")')
        indexed, evidence = ensure_spatial_evidence_ids(trace, prefix="M")
        self.assertEqual(set(evidence), {"M1"})
        self.assertEqual(json.loads(indexed)["values"]["evidence_id"], "M1")

    def test_event_role_and_phase_are_closed_vocabularies(self) -> None:
        trace = execute_spatial(
            'S.event(1, "Alice", "pours", "vinegar", '
            'event_role="executor", phase="during")'
        )
        values = json.loads(trace)["values"]
        self.assertEqual(values["event_role"], "executor")
        self.assertEqual(values["phase"], "during")
        with self.assertRaises(UnsafeProgram):
            execute_spatial(
                'S.event(1, "Alice", "pours", "vinegar", '
                'event_role="responsible")'
            )

    def test_spatial_provenance_is_optional_and_executable(self) -> None:
        trace = execute_spatial(
            'S.event(12, "Shure", "use", "chopsticks", source="video", '
            'confidence=0.8, evidence_id="S1", clip_id="clip.mp4", '
            'observer="Katrina", evidence_span="00:10-00:14", quote="use chopsticks")'
        )
        values = json.loads(trace)["values"]
        self.assertEqual(values["evidence_id"], "S1")
        self.assertEqual(values["clip_id"], "clip.mp4")
        self.assertEqual(values["observer"], "Katrina")
        self.assertEqual(values["evidence_span"], "00:10-00:14")

    def test_spatial_claim_ids_are_validated_and_preserved(self) -> None:
        trace = execute_spatial(
            'S.event(1, "Jake", "laughs", "joke", claim_ids="C1,C2")'
        )
        self.assertEqual(json.loads(trace)["values"]["claim_ids"], "C1,C2")
        with self.assertRaises(UnsafeProgram):
            execute_spatial(
                'S.event(1, "Jake", "laughs", "joke", claim_ids="bad")'
            )

    def test_duplicate_spatial_evidence_ids_are_rejected(self) -> None:
        with self.assertRaises(UnsafeProgram):
            execute_spatial(
                '\n'.join(
                    [
                        'S.event(1, "A", "see", "cup", evidence_id="S1")',
                        'S.event(2, "A", "take", "cup", evidence_id="S1")',
                    ]
                )
            )

    def test_strict_social_execution_requires_valid_spatial_references(self) -> None:
        trace = execute_social(
            'M.belief(12, "Katrina", "Shure used chopsticks", 0.7, "video", '
            'evidence_ids="S1")',
            valid_evidence_ids={"S1"},
            require_evidence=True,
        )
        self.assertEqual(json.loads(trace)["values"]["evidence_ids"], ["S1"])
        with self.assertRaises(UnsafeProgram):
            execute_social(
                'M.belief(12, "Katrina", "unsupported", 0.7, "video")',
                valid_evidence_ids={"S1"},
                require_evidence=True,
            )
        with self.assertRaises(UnsafeProgram):
            execute_social(
                'M.intent(12, "Katrina", "unsupported", 0.7, "video", '
                'evidence_ids="S9")',
                valid_evidence_ids={"S1"},
                require_evidence=True,
            )

    def test_answer_aligned_posterior_is_the_native_decision(self) -> None:
        context = "\n".join(
            [
                json.dumps(
                    {
                        "kind": "posterior",
                        "values": {
                            "variable": "answer_goal",
                            "label": "A",
                            "value": 0.7,
                        },
                    }
                ),
                json.dumps(
                    {
                        "kind": "posterior",
                        "values": {
                            "variable": "answer_goal",
                            "label": "B",
                            "value": 0.3,
                        },
                    }
                ),
            ]
        )
        self.assertEqual(answer_posterior_probs(context), {"A": 0.7, "B": 0.3})

    def test_options_are_hidden_from_neural_state_compiler(self) -> None:
        evidence = question_without_options(QUESTION)
        self.assertNotIn("SECRET_A", evidence)
        self.assertNotIn("SECRET_B", evidence)
        seen = {}

        def fake_generate(prompt: str, max_new_tokens: int) -> str:
            seen["prompt"] = prompt
            return 'M.goal("Jessica", "find plate", 0.7, "search actions")'

        result = compile_social_code("neural", QUESTION, generate=fake_generate)
        self.assertNotIn("SECRET_A", seen["prompt"])
        self.assertNotIn("SECRET_B", seen["prompt"])
        self.assertIn("Never choose", seen["prompt"])
        self.assertIn('"kind": "goal"', result.trace)

    def test_rule_compiler_is_standalone(self) -> None:
        result = compile_rules(QUESTION)
        records = [json.loads(line) for line in result.trace.splitlines()]
        kinds = {record["kind"] for record in records}
        self.assertIn("intent", kinds)
        self.assertNotIn("action_evidence", kinds)
        self.assertIn("goal", kinds)
        self.assertNotIn("SECRET_A", result.code)

    def test_default_hybrid_contains_state_not_answer_evidence(self) -> None:
        result = compile_social_code(
            "hybrid",
            QUESTION,
            generate=lambda prompt, max_new_tokens: (
                'M.intent(0, "Jessica", "inspect cabinet", 0.6, "approach")'
            ),
        )
        self.assertEqual(result.components, ("neural",))
        self.assertNotIn("external_bayesian", result.components)
        self.assertNotIn('"kind": "posterior"', result.trace)
        self.assertNotIn("SECRET_A", result.code)
        self.assertNotIn("SECRET_B", result.code)
        self.assertNotIn('"variable": "answer_', result.trace)

    def test_neural_rules_ablation_has_fusion_without_posterior(self) -> None:
        result = compile_social_code(
            "neural_rules",
            QUESTION,
            generate=lambda prompt, max_new_tokens: (
                'M.intent(0, "Jessica", "inspect cabinet", 0.6, "approach")'
            ),
        )
        self.assertEqual(result.components, ("neural", "rules"))
        self.assertNotIn('"kind": "posterior"', result.trace)

    def test_bayesian_source_is_native(self) -> None:
        result = compile_social_code(
            "bayesian",
            QUESTION,
        )
        self.assertEqual(result.components, ("rules", "bayesian"))
        self.assertIn('"kind": "posterior"', result.trace)

    def test_answer_candidates_exist_only_in_explicit_legacy_mode(self) -> None:
        result = compile_social_code(
            "legacy_hybrid",
            QUESTION,
            generate=lambda prompt, max_new_tokens: 'M.unknown("Jessica", "social_state")',
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        candidates = [r for r in records if r["kind"] == "query_candidate"]
        posteriors = [
            r["values"]
            for r in records
            if r["kind"] == "posterior" and r["values"]["variable"] == "answer_belief"
        ]
        self.assertEqual({r["values"]["label"] for r in candidates}, {"A", "B"})
        self.assertEqual({r["label"] for r in posteriors}, {"A", "B"})
        self.assertAlmostEqual(sum(r["value"] for r in posteriors), 1.0)

    def test_option_parser_handles_existential_belief_wording(self) -> None:
        question = (
            "Actions taken by Mark: Mark approaches the fridge. Question: Which? "
            "(a) Mark thinks that there is a salmon inside the fridge. "
            "(b) Mark thinks that there isn't any salmon inside the fridge. "
            "Please respond with either a or b."
        )
        self.assertEqual(len(answer_options(question)), 2)
        result = compile_social_code(
            "legacy_hybrid",
            question,
            generate=lambda prompt, max_new_tokens: 'M.unknown("Mark", "social_state")',
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        self.assertEqual(
            len([r for r in records if r["kind"] == "query_candidate"]),
            2,
        )

    def test_spatial_bridge_only_observes_contents_after_opening(self) -> None:
        question = (
            "Actions taken by Jessica: Jessica approaches the cabinet and opens it. "
            "Question: If Jessica has been trying to get a plate, which is more likely? "
            "(a) Jessica thinks that the plate is inside the cabinet. "
            "(b) Jessica thinks that the plate is not inside the cabinet. "
            "Please respond with either a or b."
        )
        trace = json.dumps(
            {
                "kind": "spatial_fact",
                "values": {
                    "t": 0,
                    "relation": "contains",
                    "subject": "cabinet",
                    "object": "plate",
                    "confidence": 1.0,
                },
            }
        )
        bridge = compile_spatial_bridge(question, trace)
        self.assertIn("plate is inside cabinet", bridge.code)
        social = compile_social_code(
            "hybrid",
            question,
            spatial_trace=trace,
            generate=lambda prompt, max_new_tokens: 'M.unknown("Jessica", "social_state")',
        )
        self.assertEqual(social.components, ("neural", "spatial_bridge"))
        self.assertNotIn("(a)", social.code.lower())
        self.assertNotIn("(b)", social.code.lower())
        self.assertNotIn('"kind": "posterior"', social.trace)

    def test_unopened_container_does_not_reveal_world_contents(self) -> None:
        question = (
            "Actions taken by Jessica: Jessica approaches the cabinet and is about to open it. "
            "Question: If Jessica has been trying to get a plate, which is more likely? "
            "(a) Jessica thinks that the plate is inside the cabinet. "
            "(b) Jessica thinks that the plate is not inside the cabinet. "
            "Please respond with either a or b."
        )
        trace = json.dumps(
            {
                "kind": "spatial_fact",
                "values": {
                    "t": 0,
                    "relation": "contains",
                    "subject": "fridge",
                    "object": "plate",
                    "confidence": 1.0,
                },
            }
        )
        bridge = compile_spatial_bridge(question, trace)
        self.assertNotIn("plate is not inside cabinet", bridge.code)

    def test_missing_spatial_fact_is_unknown_not_negative(self) -> None:
        question = (
            "Actions taken by Jessica: Jessica opens the cabinet and closes it. "
            "Question: If Jessica has been trying to get a plate, which is more likely? "
            "(a) Jessica thinks that the plate is inside the cabinet. "
            "(b) Jessica thinks that the plate is not inside the cabinet. "
            "Please respond with either a or b."
        )
        bridge = compile_spatial_bridge(question, "")
        self.assertNotIn("M.query_evidence", bridge.code)
        self.assertNotIn("plate is not inside cabinet", bridge.code)

    def test_clean_hybrid_does_not_score_goal_candidates(self) -> None:
        question = (
            "Actions taken by John: John approaches the microwave, opens it, and closes it. "
            "Question: Which is more likely? "
            "(a) John has been trying to get a salmon. "
            "(b) John has been trying to get a dish bowl. "
            "Please respond with either a or b."
        )
        trace = "\n".join(
            json.dumps(record)
            for record in [
                {
                    "kind": "spatial_fact",
                    "values": {
                        "t": 0,
                        "relation": "contains",
                        "subject": "microwave",
                        "object": "salmon",
                        "confidence": 1.0,
                    },
                },
                {
                    "kind": "spatial_fact",
                    "values": {
                        "t": 0,
                        "relation": "contains",
                        "subject": "fridge",
                        "object": "dish bowl",
                        "confidence": 1.0,
                    },
                },
            ]
        )
        social = compile_social_code(
            "hybrid",
            question,
            spatial_trace=trace,
            generate=lambda prompt, max_new_tokens: 'M.unknown("John", "social_state")',
        )
        self.assertNotIn('"kind": "posterior"', social.trace)
        self.assertNotIn('"kind": "query_candidate"', social.trace)

    def test_clean_neural_path_does_not_invent_carried_beliefs(self) -> None:
        result = compile_social_code(
            "neural",
            QUESTION,
            generate=lambda prompt, max_new_tokens: "\n".join(
                [
                    'M.observe(0, "Jessica", "plate is in cabinet", "video", 0.8)',
                    'M.intent(3, "Jessica", "walk to kitchen", 0.6, "video")',
                ]
            ),
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        self.assertEqual(sum(r["kind"] == "observation" for r in records), 1)
        self.assertEqual(sum(r["kind"] == "belief" for r in records), 0)
        self.assertFalse(result.diagnostics["hand_weighting"])

    def test_clean_neural_path_rejects_external_answer_evidence(self) -> None:
        with self.assertRaises(ValueError):
            compile_social_code(
                "neural",
                QUESTION,
                generate=lambda prompt, max_new_tokens: (
                    'M.intent(0, "Jessica", "inspect cabinet", 0.6, "approach")'
                ),
                bayesian_loader=lambda: (
                    'M.likelihood("Jessica", "open cabinet", "inspect cabinet", 0.8)',
                    "unused trace",
                ),
                include_external_bayesian=True,
            )

    def test_clean_neural_path_preserves_records_without_weight_fusion(self) -> None:
        result = compile_social_code(
            "neural",
            QUESTION,
            generate=lambda prompt, max_new_tokens: "\n".join(
                [
                    'M.observe(1, "Jessica", "plate is in cabinet", "video", 0.8)',
                    'M.belief(1, "Jessica", "plate is in cabinet", 0.6, "action")',
                ]
            ),
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        beliefs = [record for record in records if record["kind"] == "belief"]
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]["values"]["confidence"], 0.6)
        self.assertFalse(result.diagnostics["hand_weighting"])

    def test_clean_neural_path_does_not_hand_discount_conflicts(self) -> None:
        result = compile_social_code(
            "neural",
            QUESTION,
            generate=lambda prompt, max_new_tokens: "\n".join(
                [
                    'M.belief(0, "Jessica", "plate is in cabinet", 0.8, "old")',
                    'M.belief(1, "Jessica", "plate is not in cabinet", 0.9, "new")',
                ]
            ),
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        self.assertEqual(sum(r["kind"] == "contradiction" for r in records), 0)
        positive = next(
            r for r in records if r["kind"] == "belief" and "not" not in r["values"]["fact"]
        )
        self.assertAlmostEqual(positive["values"]["confidence"], 0.8)

    def test_clean_hybrid_has_no_native_posteriors(self) -> None:
        result = compile_social_code(
            "hybrid",
            QUESTION,
            generate=lambda prompt, max_new_tokens: (
                'M.intent(0, "Jessica", "search cabinet", 0.7, "approach")'
            ),
        )
        records = [json.loads(line) for line in result.trace.splitlines()]
        self.assertFalse(any(r["kind"] == "posterior" for r in records))
        self.assertFalse(result.diagnostics["answer_scoring"])

    def test_clean_hybrid_does_not_convert_confidence_to_log_odds(self) -> None:
        question = (
            "Actions taken by Jessica: Jessica takes the plate. "
            "Question: If Jessica has been trying to get a plate, which is more likely? "
            "(a) Jessica thinks that the plate is inside the cabinet. "
            "(b) Jessica thinks that the plate is not inside the cabinet. "
            "Please respond with either a or b."
        )
        result = compile_social_code(
            "hybrid",
            question,
            generate=lambda prompt, max_new_tokens: (
                'M.goal("Jessica", "get plate", 1.0, "strong duplicate evidence")'
            ),
        )
        self.assertNotIn('"kind": "posterior"', result.trace)
        self.assertFalse(result.diagnostics["hand_weighting"])

    def test_normalizer_rejects_arbitrary_calls(self) -> None:
        code = normalize_social_code(
            'open("/tmp/leak")\nM.goal("A", "find plate", 0.5, "evidence")'
        )
        self.assertNotIn("open", code)
        self.assertIn("M.goal", code)

    def test_query_evidence_is_rejected_outside_legacy_runtime(self) -> None:
        code = 'M.query_evidence(0, "A", "answer", "x", 0.9, 0.9, "rule")'
        self.assertEqual(normalize_social_code(code), "")
        with self.assertRaises(UnsafeProgram):
            execute_social(code)
        self.assertIn('"kind": "query_evidence"', execute_social_legacy(code))

    def test_normalizer_rejects_runtime_calls_with_wrong_signature(self) -> None:
        code = normalize_social_code(
            '\n'.join(
                [
                    'M.belief("Jessica", "plate is hidden", 0.7, "video")',
                    'M.intent(0, "Jessica", "find plate", 0.7, "video")',
                ]
            )
        )
        self.assertNotIn("M.belief", code)
        self.assertIn("M.intent", code)

    def test_normalizer_rejects_non_integer_timestamps(self) -> None:
        code = normalize_social_code(
            '\n'.join(
                [
                    'M.belief(t, "Jessica", "plate is hidden", 0.7, "video")',
                    'M.intent(0, "Jessica", "find plate", 0.7, "video")',
                ]
            )
        )
        self.assertNotIn("M.belief", code)
        self.assertIn("M.intent", code)

    def test_neural_compiler_falls_back_when_all_calls_are_invalid(self) -> None:
        result = compile_social_code(
            "neural",
            QUESTION,
            generate=lambda prompt, max_new_tokens: (
                'M.belief("Jessica", "plate is hidden", 0.7, "video")'
            ),
        )
        self.assertIn("M.unknown", result.code)
        self.assertIn('"kind": "social_unknown"', result.trace)

    def test_runtime_rejects_control_flow(self) -> None:
        with self.assertRaises(UnsafeProgram):
            execute_social('for x in [1]:\n    M.unknown("A", "belief")')

    def test_runtime_rejects_unlisted_memory_methods(self) -> None:
        with self.assertRaises(UnsafeProgram):
            execute_social("M.trace()")

    def test_runtime_rejects_wrong_method_signature_before_execution(self) -> None:
        with self.assertRaisesRegex(UnsafeProgram, "invalid M.belief call"):
            execute_social(
                '\n'.join(
                    [
                        'M.unknown("Jessica", "belief")',
                        'M.belief("Jessica", "plate is hidden", 0.7, "video")',
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
