from __future__ import annotations

import json
import unittest

from cpg2.slm_e0 import parse_typed_e0_output


def typed_output(**overrides):
    payload = {
        "criterion_id": "current_fever",
        "state": "PRESENT_CURRENT",
        "value": None,
        "unit": None,
        "evidence": [],
        "reason_code": "explicit_present",
        "confidence": 0.9,
        "abstain": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


class SlmE0ParserTests(unittest.TestCase):
    def test_valid_typed_prediction(self) -> None:
        parsed = parse_typed_e0_output(
            typed_output(), expected_criterion_id="current_fever"
        )
        self.assertTrue(parsed.schema_valid)
        self.assertTrue(parsed.logical_valid)
        self.assertEqual(parsed.state, "PRESENT_CURRENT")

    def test_reason_code_mismatch_is_logical_failure(self) -> None:
        parsed = parse_typed_e0_output(
            typed_output(reason_code="historical"),
            expected_criterion_id="current_fever",
        )
        self.assertTrue(parsed.schema_valid)
        self.assertFalse(parsed.logical_valid)
        self.assertEqual(parsed.failure, "logical_failure")

    def test_model_supplied_evidence_is_schema_failure(self) -> None:
        parsed = parse_typed_e0_output(
            typed_output(evidence=[{"note_id": "x", "start": 0, "end": 1}]),
            expected_criterion_id="current_fever",
        )
        self.assertTrue(parsed.parse_valid)
        self.assertFalse(parsed.schema_valid)

    def test_wrong_json_types_are_schema_failures_not_exceptions(self) -> None:
        parsed = parse_typed_e0_output(
            typed_output(reason_code=["explicit_present"]),
            expected_criterion_id="current_fever",
        )
        self.assertTrue(parsed.parse_valid)
        self.assertEqual(parsed.failure, "schema_failure")

    def test_explicit_abstention_contract(self) -> None:
        parsed = parse_typed_e0_output(
            typed_output(
                state="NOT_DOCUMENTED",
                reason_code="model_abstention",
                confidence=0,
                abstain=True,
            ),
            expected_criterion_id="current_fever",
        )
        self.assertTrue(parsed.logical_valid)
        self.assertTrue(parsed.abstained)
        self.assertIsNone(parsed.state)

    def test_runtime_and_parse_failures_are_separate(self) -> None:
        self.assertEqual(
            parse_typed_e0_output(
                None,
                expected_criterion_id="current_fever",
                runtime_failed=True,
            ).failure,
            "runtime_failure",
        )
        self.assertEqual(
            parse_typed_e0_output(
                "not json", expected_criterion_id="current_fever"
            ).failure,
            "parse_failure",
        )


if __name__ == "__main__":
    unittest.main()
