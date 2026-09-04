from __future__ import annotations

import json
import unittest

from cpg2.ext_notes_benchmark import parse_ext_notes_output
from cpg2.slm_e3 import (
    FAILURE_LABEL,
    _classification_metrics,
    _hierarchical_correct,
    _holm_adjust,
    _prediction_heads,
    _window_bounds,
)


class SlmE3Tests(unittest.TestCase):
    def test_window_is_shared_character_bounded_and_shifts_at_edges(self) -> None:
        self.assertEqual(_window_bounds(1000, 450, 460, 200), (355, 555))
        self.assertEqual(_window_bounds(1000, 5, 15, 200), (0, 200))
        self.assertEqual(_window_bounds(1000, 985, 995, 200), (800, 1000))
        self.assertEqual(_window_bounds(100, 10, 20, 200), (0, 100))

    def test_detection_remains_scoreable_when_only_downstream_abstains(self) -> None:
        parsed = parse_ext_notes_output(
            json.dumps(
                {
                    "detection": "yes",
                    "encounter": "abstain",
                    "negation": "abstain",
                }
            )
        ).as_dict()
        heads = _prediction_heads(parsed)
        self.assertEqual(heads["detection"], "yes")
        self.assertEqual(heads["encounter"], FAILURE_LABEL)
        self.assertEqual(heads["negation"], FAILURE_LABEL)

    def test_logical_failure_invalidates_all_heads(self) -> None:
        parsed = parse_ext_notes_output(
            json.dumps(
                {"detection": "no", "encounter": "yes", "negation": "no"}
            )
        ).as_dict()
        self.assertEqual(
            _prediction_heads(parsed),
            {
                "detection": FAILURE_LABEL,
                "encounter": FAILURE_LABEL,
                "negation": FAILURE_LABEL,
            },
        )

    def test_joint_ignores_downstream_source_labels_for_gold_detection_no(self) -> None:
        parsed = parse_ext_notes_output(
            json.dumps(
                {
                    "detection": "no",
                    "encounter": "not_applicable",
                    "negation": "not_applicable",
                }
            )
        ).as_dict()
        self.assertTrue(
            _hierarchical_correct(
                {
                    "detection": "no",
                    "encounter": "not_applicable",
                    "negation": "not_applicable",
                },
                parsed,
            )
        )

    def test_invalid_prediction_is_failure_not_a_negative_class(self) -> None:
        metrics = _classification_metrics(
            ["yes", "no"], [FAILURE_LABEL, "no"], ["yes", "no"]
        )
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["per_class"]["yes"]["false_negative"], 1)
        self.assertEqual(metrics["per_class"]["no"]["true_positive"], 1)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = _holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)


if __name__ == "__main__":
    unittest.main()
