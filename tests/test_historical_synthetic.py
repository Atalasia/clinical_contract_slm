from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpg2.historical_synthetic import (
    END_LABEL,
    _csv_signature,
    compute_classification_metrics,
    normalize_terminal_label,
)


class HistoricalSyntheticTest(unittest.TestCase):
    def test_terminal_normalization_and_weighted_metrics(self) -> None:
        self.assertEqual(normalize_terminal_label("End"), END_LABEL)
        self.assertEqual(normalize_terminal_label("No specific action required"), END_LABEL)
        self.assertEqual(normalize_terminal_label("Refer urgently"), "Refer urgently")

        metrics = compute_classification_metrics(
            [
                ("End", "End of Decision Tree"),
                ("Refer urgently", "Refer urgently"),
                ("Refer urgently", "End"),
            ]
        )
        self.assertAlmostEqual(metrics["multiclass"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["multiclass"]["weighted_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["multiclass"]["weighted_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["multiclass"]["weighted_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["binary"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["binary"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["binary"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["binary"]["f1"], 2 / 3)
        self.assertEqual(metrics["binary"]["positive_reference_n"], 2)
        self.assertEqual(metrics["binary"]["negative_reference_n"], 1)
        self.assertEqual(metrics["binary"]["true_positive"], 1)
        self.assertEqual(metrics["binary"]["true_negative"], 1)

    def test_selected_column_signature_ignores_unselected_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text(
                "action,text,diagnosis\nA,case one,A\nB,case two,B\n",
                encoding="utf-8",
            )
            second.write_text(
                "action,text,diagnosis\nA,case one,B\nB,case two,A\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _csv_signature(first, ("action", "text")),
                _csv_signature(second, ("action", "text")),
            )


if __name__ == "__main__":
    unittest.main()
