from __future__ import annotations

from copy import deepcopy
import json
import unittest

from cpg2.slm_contract_analysis import CONDITIONS, summarize_ablation


def _row(request_id: str, case_id: str, state: str, correct: bool = True) -> dict:
    return {
        "request_id": request_id,
        "case_id": case_id,
        "domain": "synthetic",
        "criterion_id": f"criterion_{request_id}",
        "expected_state": state,
        "state_correct": correct,
        "state_only_correct": correct,
        "parsed": {
            "parse_valid": True,
            "schema_valid": True,
            "logical_valid": True,
            "abstained": False,
            "failure": None,
        },
        "output_tokens": 10,
        "prompt_tokens": 100,
        "runtime_failed": False,
    }


def _panel() -> dict:
    rows = [
        _row("r1", "v1", "PRESENT_CURRENT"),
        _row("r2", "v1", "ABSENT_EXPLICIT", False),
        _row("r3", "v2", "PRESENT_CURRENT"),
        _row("r4", "v2", "PRESENT_CURRENT"),
        _row("r5", "v2", "ABSENT_EXPLICIT", False),
    ]
    return {model: {condition: deepcopy(rows) for condition in CONDITIONS} for model in ("a", "b")}


CONTRAST = [{"contrast_id": "b_vs_a", "models": ["a", "b"], "claim_type": "descriptive"}]


class ContractAnalysisTests(unittest.TestCase):
    def summarize(self, panel: dict, contrasts: list | None = None, **kwargs) -> dict:
        return summarize_ablation(panel, CONTRAST if contrasts is None else contrasts, bootstrap_replicates=100, **kwargs)

    def test_macro_and_micro_use_different_denominators(self) -> None:
        result = self.summarize(_panel())
        cell = result["models"]["a"]["conditions"]["full_unconstrained"]
        self.assertEqual(cell["state_macro_agreement"]["estimate"], 0.5)
        self.assertEqual(cell["strict_micro_agreement"]["estimate"], 0.6)
        self.assertEqual(cell["state_only_macro_agreement"]["estimate"], 0.5)
        self.assertEqual(cell["output_tokens"]["total"], 50)
        self.assertEqual(cell["output_tokens"]["mean"], 10)
        self.assertEqual(result["data"]["vignette_count"], 2)

    def test_identical_cells_have_exact_zero_paired_intervals(self) -> None:
        panel = _panel()
        panel["b"]["compact_constrained"].reverse()
        result = self.summarize(panel)
        for model in result["models"].values():
            for metric in model["contract_by_decoder_interaction"].values():
                self.assertEqual((metric["estimate"], metric["ci_low"], metric["ci_high"]), (0, 0, 0))
        for cell in result["contrasts"][0]["conditions"].values():
            for metric in cell.values():
                self.assertEqual((metric["estimate"], metric["ci_low"], metric["ci_high"]), (0, 0, 0))

    def test_case_resampling_retains_all_requests_within_drawn_cases(self) -> None:
        result = self.summarize(_panel())
        cell = result["models"]["a"]["conditions"]["full_unconstrained"]
        # Drawing v1 twice yields 2/4; drawing v2 twice yields 4/6.  A
        # request-level bootstrap could generate values outside that range.
        self.assertEqual(cell["strict_micro_agreement"]["ci_low"], 0.5)
        self.assertEqual(cell["strict_micro_agreement"]["ci_high"], 2 / 3)
        self.assertEqual(cell["state_macro_agreement"]["ci_low"], 0.5)
        self.assertEqual(cell["state_macro_agreement"]["ci_high"], 0.5)

    def test_length_finish_reason_has_all_request_denominator(self) -> None:
        panel = _panel()
        panel["a"]["full_unconstrained"][0]["finish_reason"] = "length"
        panel["a"]["full_unconstrained"][1]["finish_reason"] = "stop"
        cells = self.summarize(panel)["models"]["a"]["conditions"]
        self.assertEqual(cells["full_unconstrained"]["length_truncation"]["count"], 1)
        self.assertEqual(cells["full_unconstrained"]["length_truncation"]["rate"], 0.2)
        self.assertEqual(cells["full_unconstrained"]["length_truncation"]["missing_finish_reason_count"], 3)
        self.assertEqual(cells["full_constrained"]["length_truncation"]["count"], 0)

    def test_direction_interaction_and_descriptive_reversal(self) -> None:
        panel = _panel()
        # A is perfect everywhere; B is zero except for compact-constrained.
        for model in panel:
            for condition in CONDITIONS:
                for row in panel[model][condition]:
                    row["state_correct"] = model == "a" or condition == "compact_constrained"
                    row["state_only_correct"] = row["state_correct"]
        result = self.summarize(panel)
        model = result["models"]["b"]
        self.assertEqual(model["compact_minus_full"]["constrained"]["state_macro_agreement"]["estimate"], 1)
        self.assertEqual(model["contract_by_decoder_interaction"]["state_macro_agreement"]["estimate"], 1)
        contrast = result["contrasts"][0]
        self.assertEqual(contrast["conditions"]["full_unconstrained"]["state_macro_agreement"]["estimate"], -1)
        self.assertEqual(contrast["change_in_decoder_shift_compact_minus_full"]["state_macro_agreement"]["estimate"], 1)
        self.assertFalse(contrast["descriptive_sign_reversals"]["state_macro_agreement"]["across_contract_with_constrained_decoder"])
        for row in panel["a"]["compact_constrained"]:
            row["state_correct"] = row["state_only_correct"] = False
        contrast = self.summarize(panel)["contrasts"][0]
        self.assertTrue(contrast["descriptive_sign_reversals"]["state_macro_agreement"]["across_contract_with_constrained_decoder"])

    def test_state_only_can_be_correct_when_full_record_is_invalid(self) -> None:
        panel = _panel()
        row = panel["a"]["compact_constrained"][0]
        row["state_correct"] = False
        row["parsed"]["logical_valid"] = False
        row["parsed"]["failure"] = "logical_failure"
        cell = self.summarize(panel)["models"]["a"]["conditions"]["compact_constrained"]
        self.assertGreater(cell["state_only_macro_agreement"]["estimate"], cell["state_macro_agreement"]["estimate"])
        self.assertEqual(cell["logical_invalid_among_schema_valid"], {"count": 1, "denominator": 5, "rate": 0.2})

    def test_valid_abstention_count_excludes_invalid_abstentions(self) -> None:
        panel = _panel()
        rows = panel["a"]["full_unconstrained"]
        for index, valid in ((1, True), (4, False)):
            rows[index]["parsed"]["abstained"] = True
            rows[index]["parsed"]["logical_valid"] = valid
            rows[index]["parsed"]["failure"] = None if valid else "logical_failure"
        cell = self.summarize(panel)["models"]["a"]["conditions"]["full_unconstrained"]
        self.assertEqual(cell["abstention_count"], 2)
        self.assertEqual(cell["valid_abstention_count"], 1)

    def test_missing_stratum_resamples_are_excluded_only_from_macro(self) -> None:
        panel = {"a": {condition: [
            _row("r1", "v1", "PRESENT_CURRENT"),
            _row("r2", "v2", "ABSENT_EXPLICIT"),
        ] for condition in CONDITIONS}}
        result = self.summarize(panel, [])
        omitted = result["bootstrap"]["macro_replicates_missing_reference_stratum"]
        self.assertGreater(omitted, 0)
        cell = result["models"]["a"]["conditions"]["full_unconstrained"]
        self.assertEqual(cell["state_macro_agreement"]["replicates_used"], 100 - omitted)
        self.assertEqual(cell["strict_micro_agreement"]["replicates_used"], 100)
        self.assertEqual(cell["state_macro_agreement"]["ci_low"], 1)

    def test_duplicate_missing_condition_missing_request_and_missing_gold_fail(self) -> None:
        panel = _panel()
        panel["a"]["full_unconstrained"].append(deepcopy(panel["a"]["full_unconstrained"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.summarize(panel)
        panel = _panel()
        del panel["a"]["compact_constrained"]
        with self.assertRaisesRegex(ValueError, "four ablation conditions"):
            self.summarize(panel)
        panel = _panel()
        panel["b"]["full_unconstrained"].pop()
        with self.assertRaisesRegex(ValueError, "mismatched request IDs"):
            self.summarize(panel)
        panel = _panel()
        for cells in panel.values():
            for condition in CONDITIONS:
                cells[condition] = [row for row in cells[condition] if row["expected_state"] == "PRESENT_CURRENT"]
        with self.assertRaisesRegex(ValueError, "both reference-state strata"):
            self.summarize(panel)

    def test_pairing_checks_all_alignment_fields(self) -> None:
        for key in ("case_id", "domain", "criterion_id", "expected_state"):
            panel = _panel()
            row = panel["b"]["compact_unconstrained"][0]
            row[key] = "ABSENT_EXPLICIT" if key == "expected_state" else "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "paired metadata differ"):
                self.summarize(panel)

    def test_rejects_contradictory_or_untyped_flags_and_bad_tokens(self) -> None:
        for field, value in (("state_correct", "true"), ("output_tokens", True)):
            panel = _panel()
            panel["a"]["full_unconstrained"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(panel)
        panel = _panel()
        panel["a"]["full_unconstrained"][0]["parsed"]["abstained"] = True
        with self.assertRaisesRegex(ValueError, "correctness flags contradict"):
            self.summarize(panel)

    def test_deterministic_and_strict_json_serializable(self) -> None:
        first = self.summarize(_panel())
        second = self.summarize(_panel())
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)

    def test_runtime_failures_preserve_denominators_and_missing_token_counts(self) -> None:
        panel = _panel()
        for row in panel["a"]["full_unconstrained"]:
            row["runtime_failed"] = True
            row["state_correct"] = row["state_only_correct"] = False
            row["parsed"] = dict(parse_valid=False, schema_valid=False, logical_valid=False, abstained=False, failure="runtime_failure")
            row["output_tokens"] = None
        cell = self.summarize(panel)["models"]["a"]["conditions"]["full_unconstrained"]
        self.assertEqual(cell["strict_micro_agreement"]["estimate"], 0)
        self.assertEqual(cell["runtime_failure_count"], 5)
        self.assertEqual(cell["output_tokens"]["missing_requests"], 5)
        self.assertIsNone(cell["output_tokens"]["mean"])
        self.assertIsNone(cell["logical_invalid_among_schema_valid"]["rate"])
        json.dumps(cell, allow_nan=False)

    def test_invalid_bootstrap_and_contrast_arguments(self) -> None:
        for replicates in (0, -1, True, 1.5):
            with self.subTest(replicates=replicates), self.assertRaises(ValueError):
                summarize_ablation(_panel(), CONTRAST, bootstrap_replicates=replicates)
        with self.assertRaisesRegex(ValueError, "absent from panel"):
            self.summarize(_panel(), [{"contrast_id": "bad", "models": ["a", "missing"]}])


if __name__ == "__main__":
    unittest.main()
