from __future__ import annotations

import json
import unittest

from cpg2.ext_notes_benchmark import (
    parse_ext_notes_output,
    validate_ext_notes_labels,
)


class ExtNotesOutputContractTests(unittest.TestCase):
    def test_valid_detected_prediction(self) -> None:
        result = parse_ext_notes_output(
            json.dumps({"detection": "yes", "encounter": "yes", "negation": "no"})
        )
        self.assertTrue(result.schema_valid)
        self.assertTrue(result.logical_valid)
        self.assertFalse(result.abstained)
        self.assertIsNone(result.failure)

    def test_valid_rejected_candidate_requires_not_applicable(self) -> None:
        result = parse_ext_notes_output(
            json.dumps(
                {
                    "detection": "no",
                    "encounter": "not_applicable",
                    "negation": "not_applicable",
                }
            )
        )
        self.assertTrue(result.logical_valid)

    def test_gold_unsure_is_not_model_abstention(self) -> None:
        result = parse_ext_notes_output(
            json.dumps(
                {"detection": "yes", "encounter": "yes", "negation": "unsure"}
            )
        )
        self.assertTrue(result.logical_valid)
        self.assertFalse(result.abstained)

    def test_detection_abstention_is_explicit_and_hierarchical(self) -> None:
        result = parse_ext_notes_output(
            json.dumps(
                {
                    "detection": "abstain",
                    "encounter": "abstain",
                    "negation": "abstain",
                }
            )
        )
        self.assertTrue(result.logical_valid)
        self.assertTrue(result.abstained)

    def test_logical_inconsistency_is_separate_from_schema_failure(self) -> None:
        result = parse_ext_notes_output(
            json.dumps({"detection": "no", "encounter": "yes", "negation": "no"})
        )
        self.assertTrue(result.schema_valid)
        self.assertFalse(result.logical_valid)
        self.assertEqual(result.failure, "logical_failure")

    def test_parse_schema_and_runtime_failures_remain_distinct(self) -> None:
        self.assertEqual(parse_ext_notes_output("not json").failure, "parse_failure")
        self.assertEqual(
            parse_ext_notes_output(json.dumps({"detection": "yes"})).failure,
            "schema_failure",
        )
        self.assertEqual(parse_ext_notes_output(None).failure, "runtime_failure")

    def test_wrong_json_types_are_schema_failures_not_exceptions(self) -> None:
        result = parse_ext_notes_output(
            json.dumps({"detection": ["yes"], "encounter": "yes", "negation": "no"})
        )
        self.assertEqual(result.failure, "schema_failure")

    def test_expected_labels_cannot_encode_model_abstention(self) -> None:
        error = validate_ext_notes_labels(
            {"detection": "abstain", "encounter": "abstain", "negation": "abstain"},
            allow_abstention=False,
        )
        self.assertIn("cannot use model abstention", error or "")


if __name__ == "__main__":
    unittest.main()
