from __future__ import annotations

import unittest

from cpg2.slm_e1 import (
    _mcnemar_exact,
    _posthoc_output_diagnostics,
    parse_feature_cell_recursive,
)


class SlmE1Tests(unittest.TestCase):
    def test_recursive_feature_parser_flattens_and_deduplicates(self) -> None:
        value = "[['Severe bone pain', 'Hard DRE'], ['Severe bone pain']]"
        self.assertEqual(
            parse_feature_cell_recursive(value),
            ("Severe bone pain", "Hard DRE"),
        )

    def test_recursive_feature_parser_preserves_plain_scalar(self) -> None:
        self.assertEqual(parse_feature_cell_recursive("Thunderclap headache"), ("Thunderclap headache",))
        self.assertEqual(parse_feature_cell_recursive("[]"), ())

    def test_exact_mcnemar_handles_no_discordance(self) -> None:
        self.assertIsNone(_mcnemar_exact(0, 0))
        self.assertEqual(_mcnemar_exact(0, 5), 0.0625)

    def test_posthoc_diagnostic_tolerates_non_string_reason_code(self) -> None:
        response = {
            "criterion_id": "criterion_a",
            "expected_state": "PRESENT_CURRENT",
            "state_correct": False,
            "finish_reason": "stop",
            "observed_label": "SCHEMA_FAILURE",
            "raw_output": (
                '{"criterion_id":"criterion_a","state":"PRESENT_CURRENT",'
                '"value":null,"unit":null,"evidence":[],"reason_code":'
                '["explicit_present"],"confidence":1,"abstain":false}'
            ),
        }
        result = _posthoc_output_diagnostics([response])
        self.assertEqual(result["recoverable_state_predictions"], 1)
        self.assertEqual(result["recoverable_state_correct"], 1)


if __name__ == "__main__":
    unittest.main()
