from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from cpg2.slm_compact_contract import evaluate_contract_output
from cpg2.slm_contract_analysis import CONDITIONS, REFERENCE_STATES, summarize_ablation
from cpg2.slm_contract_sensitivity import (
    CATEGORIES, analyze_existing_outputs, normalize_enclosing_json_fence,
    render_report, summarize_cell, summarize_sensitivity,
)


def _row(condition, index, gold="PRESENT_CURRENT", *, state=None, abstain=False,
         confidence=0.9, reason=None, missing=None, fence=False, runtime_failed=False):
    state = gold if state is None else state
    criterion = f"private_criterion_{index}"
    compact = {"criterion_id": criterion, "state": state, "confidence": confidence, "abstain": abstain}
    contract = condition.split("_", 1)[0]
    raw = json.dumps(compact)
    if contract == "full":
        raw = evaluate_contract_output(raw, contract="compact", expected_criterion_id=criterion)["assembled_output"]
    payload = json.loads(raw)
    if reason is not None:
        payload["reason_code"] = reason
    if missing is not None:
        del payload[missing]
    raw = json.dumps(payload)
    if fence:
        raw = "```json\n" + raw + "\n```"
    if runtime_failed:
        raw = None
    evaluated = evaluate_contract_output(raw, contract=contract, expected_criterion_id=criterion, runtime_failed=runtime_failed)
    parsed = evaluated["parsed"]
    return {"request_id": f"private_request_{index}", "case_id": f"private_case_{index}",
            "domain": "test", "criterion_id": criterion, "expected_state": gold,
            "manifest_id": "test_model", "condition": condition, "raw_output": raw,
            "runtime_failed": runtime_failed, "prompt_tokens": 20, "output_tokens": 30,
            "finish_reason": "stop", **evaluated,
            "state_correct": bool(parsed["logical_valid"] and not parsed["abstained"] and parsed["state"] == gold),
            "state_only_correct": evaluated["state_only_state"] == gold}


def _panel():
    return {"test_model": {condition: [
        _row(condition, 1, fence=condition.endswith("_unconstrained")),
        _row(condition, 2, "ABSENT_EXPLICIT"),
    ] for condition in CONDITIONS}}


class ReleaseConfigSelectionTests(unittest.TestCase):
    def test_default_and_explicit_config_paths_keep_hash_checks(self):
        root = Path(".").resolve()
        default = root / "configs/slm_benchmark/e1_contract_ablation.v1.json"
        local = root / "configs/slm_benchmark/e1_contract_ablation.v1.local.json"
        for supplied, expected in ((None, default), (local, local)):
            with self.subTest(config=supplied), patch(
                "cpg2.slm_contract_sensitivity._read",
                side_effect=[{"experiment_config_sha256": "not-the-config-hash"}, {}],
            ) as read:
                with self.assertRaisesRegex(ValueError, "configuration hash differs"):
                    analyze_existing_outputs(
                        root / "test-only-results", repository_root=root,
                        config_path=supplied,
                    )
                self.assertEqual(read.call_args_list[1].args, (expected,))


class ExactFenceTests(unittest.TestCase):
    def test_only_single_complete_json_object_fences_are_normalized(self):
        body = '  {"nested": {"x": [1, 2]}, "text": "한국어"} \n'
        for opening in ("```json\n", "```\n", "```json\r\n", "```\r\n"):
            with self.subTest(opening=opening):
                raw = " \n\t" + opening + body + "\n```\t \n"
                self.assertEqual(normalize_enclosing_json_fence(raw), (body, True))
                self.assertEqual(normalize_enclosing_json_fence(raw.encode()), (body, True))

    def test_prose_extraction_truncation_and_semantic_repairs_are_forbidden(self):
        rejected = [
            None, b"\xff", "", "{}", " \n{}\n", "```json\n{}", "{}\n```",
            "```json\n{\n```", "```json\n{} {}\n```", "```json\n[]\n```",
            "```json\nnull\n```", "```json\n1\n```", '```json\n"text"\n```',
            "Here is JSON:\n```json\n{}\n```", "```json\n{}\n```\nDone.",
            "```json\n{}\n```\n```json\n{}\n```", "````json\n{}\n````",
            "```JSON\n{}\n```", "```javascript\n{}\n```", "```json {} ```",
            "```json \n{}\n```", "~~~json\n{}\n~~~", "```json\n{'a': 1}\n```",
            '```json\n{"a": 1,}\n```', '```json\n{"text": "```"}\n```',
        ]
        for raw in rejected:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_enclosing_json_fence(raw), (raw, False))

    def test_normalization_preserves_all_json_field_values_and_whitespace(self):
        body = ' { "state":"WRONG_STATE", "confidence":2, "abstain":"true" } '
        normalized, removed = normalize_enclosing_json_fence("```json\n" + body + "\n```")
        self.assertTrue(removed)
        self.assertEqual(normalized, body)
        evaluated = evaluate_contract_output(normalized, contract="compact", expected_criterion_id="id")
        self.assertEqual(evaluated["parsed"]["failure"], "schema_failure")


class DecompositionTests(unittest.TestCase):
    def test_abstention_keeps_generated_state_correctness_separate_from_logic(self):
        rows = [
            _row("compact_constrained", 1, abstain=True, confidence=0),
            _row("compact_constrained", 2, "ABSENT_EXPLICIT", state="NOT_DOCUMENTED", abstain=True, confidence=0),
            _row("compact_constrained", 3, "ABSENT_EXPLICIT", state="PRESENT_CURRENT", abstain=True, confidence=0),
        ]
        self.assertIsNone(rows[0]["parsed"]["state"])
        self.assertEqual(rows[0]["state_only_state"], "PRESENT_CURRENT")
        summary = summarize_cell(rows, normalize_fences=False)
        part = summary["partition"]
        self.assertEqual(part["schema_valid__state_correct__abstain_true__logical_invalid"]["count"], 1)
        self.assertEqual(part["schema_valid__state_wrong__abstain_true__logical_valid"]["count"], 1)
        self.assertEqual(part["schema_valid__state_wrong__abstain_true__logical_invalid"]["count"], 1)
        self.assertEqual(summary["metrics"]["full_record_state_agreement"]["count"], 0)
        self.assertEqual(summary["metrics"]["state_only_agreement"]["state_macro_rate"], 0.5)

    def test_all_failure_classes_and_schema_valid_nonabstentions_are_exhaustive(self):
        rows = [
            _row("full_unconstrained", 1, runtime_failed=True),
            _row("full_unconstrained", 2, fence=True),
            _row("full_unconstrained", 3, missing="confidence"),
            _row("full_unconstrained", 4, "ABSENT_EXPLICIT"),
            _row("full_unconstrained", 5, "ABSENT_EXPLICIT", state="PRESENT_CURRENT"),
            _row("full_unconstrained", 6, reason="explicit_negation"),
            _row("full_unconstrained", 7, "ABSENT_EXPLICIT", state="PRESENT_CURRENT", reason="explicit_negation"),
        ]
        summary = summarize_cell(rows, normalize_fences=False)
        self.assertEqual(set(summary["partition"]), set(CATEGORIES))
        self.assertEqual(sum(value["count"] for value in summary["partition"].values()), 7)
        self.assertAlmostEqual(sum(value["state_macro_rate"] for value in summary["partition"].values()), 1)
        for name in ("runtime_failure", "parse_failure", "schema_failure",
                     "schema_valid__state_correct__abstain_false__logical_valid",
                     "schema_valid__state_wrong__abstain_false__logical_valid",
                     "schema_valid__state_correct__abstain_false__logical_invalid",
                     "schema_valid__state_wrong__abstain_false__logical_invalid"):
            self.assertEqual(summary["partition"][name]["count"], 1)
        self.assertEqual(summary["metrics"]["state_only_correct_with_schema_failure"]["count"], 1)
        self.assertEqual(summary["metrics"]["full_record_state_agreement"]["count"], 1)
        normalized = summarize_cell(rows, normalize_fences=True)
        self.assertEqual(normalized["metrics"]["full_record_state_agreement"]["count"], 2)
        self.assertEqual(normalized["metrics"]["fence_removed"]["count"], 1)
        self.assertEqual(rows[1]["parsed"]["failure"], "parse_failure")

    def test_state_only_requires_complete_object_matching_id_and_recognized_state(self):
        rows = [_row("compact_constrained", 1, missing="confidence"),
                _row("compact_constrained", 2, "ABSENT_EXPLICIT", state="PRESENT_CURRENT", missing="confidence"),
                _row("compact_constrained", 3, "ABSENT_EXPLICIT")]
        payload = json.loads(rows[2]["raw_output"])
        payload["criterion_id"] = "different"
        rows[2]["raw_output"] = json.dumps(payload)
        result = summarize_cell(rows, normalize_fences=False)["metrics"]
        self.assertEqual(result["state_only_eligible"]["count"], 2)
        self.assertEqual(result["state_only_correct_with_schema_failure"]["count"], 1)
        self.assertEqual(result["state_only_wrong_with_schema_failure"]["count"], 1)
        self.assertEqual(result["schema_failure_without_recognized_matching_id_state"]["count"], 1)


class PairedPanelTests(unittest.TestCase):
    def setUp(self):
        self.panel = _panel()
        self.aggregate = summarize_ablation(self.panel, [], bootstrap_replicates=20, bootstrap_seed=1)

    def test_complete_panel_matches_strict_aggregate_and_leaks_no_row_data(self):
        result = summarize_sensitivity(self.panel, self.aggregate)
        self.assertTrue(all(result["verification"].values()))
        cell = result["models"]["test_model"]["conditions"]["compact_unconstrained"]
        self.assertEqual(cell["strict"]["metrics"]["full_record_state_agreement"]["state_macro_rate"], 0.5)
        self.assertEqual(cell["fence_normalized"]["metrics"]["full_record_state_agreement"]["state_macro_rate"], 1)
        for text in (json.dumps(result), render_report(result)):
            for prohibited in ("private_request_", "private_criterion_", "private_case_", "raw_output", "assembled_output"):
                self.assertNotIn(prohibited, text)

    def test_missing_condition_response_model_and_duplicate_responses_fail(self):
        mutations = [
            lambda panel: panel["test_model"].pop("compact_constrained"),
            lambda panel: panel["test_model"]["full_unconstrained"].pop(),
            lambda panel: panel["test_model"]["full_unconstrained"].append(panel["test_model"]["full_unconstrained"][0]),
            lambda panel: panel.clear(),
            lambda panel: panel["test_model"]["full_unconstrained"][0].update(case_id="mismatch"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                panel = copy.deepcopy(self.panel)
                mutation(panel)
                with self.assertRaises(ValueError):
                    summarize_sensitivity(panel, self.aggregate)

    def test_changed_stored_strict_evaluation_is_rejected(self):
        self.panel["test_model"]["full_constrained"][0]["assembled_output"] = "{}"
        with self.assertRaisesRegex(ValueError, "stored strict evaluation"):
            summarize_sensitivity(self.panel, self.aggregate)

    def test_changed_saved_diagnostics_or_point_estimate_is_rejected(self):
        for metric in ("parse_valid", "state_macro_agreement", "state_only_micro_agreement"):
            with self.subTest(metric=metric):
                aggregate = copy.deepcopy(self.aggregate)
                cell = aggregate["models"]["test_model"]["conditions"]["full_constrained"]
                cell[metric]["count" if metric == "parse_valid" else "estimate"] = 0.123
                with self.assertRaisesRegex(ValueError, "saved aggregate"):
                    summarize_sensitivity(self.panel, aggregate)


if __name__ == "__main__":
    unittest.main()
