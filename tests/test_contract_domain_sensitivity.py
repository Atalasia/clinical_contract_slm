from __future__ import annotations

import json
import unittest
from copy import deepcopy

from scripts.analyze_contract_domain_sensitivity import aligned_cell, metrics, reversal, summarize


def fixture():
    rows = []
    for domain, states in (("d1", ["PRESENT_CURRENT"] * 3 + ["ABSENT_EXPLICIT"]),
                           ("d2", ["PRESENT_CURRENT"] + ["ABSENT_EXPLICIT"] * 3)):
        for index, state in enumerate(states):
            rows.append({"request_id": f"private-request-{domain}-{index}", "case_id": f"private-case-{domain}-{index}",
                         "domain": domain, "criterion_id": "private-criterion", "expected_state": state,
                         "state_correct": (domain == "d1") == (state == "PRESENT_CURRENT"),
                         "runtime_failed": False, "parsed": {"abstained": False, "logical_valid": True}})
    return rows


def summarize_rows(rows):
    expected = {r["request_id"]: {k: r[k] for k in ("case_id", "domain", "criterion_id", "expected_state")} for r in rows}
    return summarize({"a": {"u": rows}}, expected, ("u",), {})


class DomainSensitivityTests(unittest.TestCase):
    def test_pooled_state_weights_differ_from_equal_domain(self):
        result = summarize_rows(fixture())
        cell = result["models"]["a"]["u"]
        self.assertEqual(cell["pooled"]["state_macro_agreement"], .75)
        self.assertEqual(cell["equal_domain"]["state_macro_agreement"], .5)
        self.assertEqual(cell["pooled"]["strict_micro_agreement"], .75)
        self.assertEqual(result["data"]["weighting"]["d1"]["pooled_macro_domain_weight_within_state"], {"PRESENT_CURRENT": .75, "ABSENT_EXPLICIT": .25})
        self.assertEqual(result["data"]["domains"]["d1"]["vignettes"], 4)

    def test_missing_domain_state_is_explicit_and_not_dropped(self):
        rows = [r for r in fixture() if not (r["domain"] == "d1" and r["expected_state"] == "ABSENT_EXPLICIT")]
        cell = summarize_rows(rows)["models"]["a"]["u"]
        self.assertEqual(cell["by_domain"]["d1"]["missing_reference_states"], ["ABSENT_EXPLICIT"])
        self.assertIsNone(cell["by_domain"]["d1"]["state_macro_agreement"])
        self.assertIsNone(cell["equal_domain"]["state_macro_agreement"])
        self.assertIsNotNone(cell["pooled"]["state_macro_agreement"])
        self.assertEqual(cell["by_domain"]["d1"]["strict_micro_agreement"], 1)
        json.dumps(cell, allow_nan=False)

    def test_failures_abstentions_preserve_denominator(self):
        rows = fixture()
        for i in (0, 1):
            rows[i]["state_correct"] = False
        rows[0]["runtime_failed"] = True
        rows[0]["parsed"]["logical_valid"] = False
        rows[1]["parsed"]["abstained"] = True
        cell = metrics(rows)
        self.assertEqual((cell["requests"], cell["correct"], cell["strict_micro_agreement"]), (8, 4, .5))
        self.assertEqual((cell["runtime_failure_count"], cell["abstention_count"], cell["contract_invalid_count"]), (1, 1, 1))

    def test_alignment_rejects_missing_duplicate_and_changed_metadata(self):
        rows = fixture()
        expected = {r["request_id"]: r for r in rows}
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                aligned_cell(bad, expected)
        for field in ("case_id", "domain", "criterion_id", "expected_state"):
            bad = deepcopy(rows)
            bad[0][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata differ"):
                aligned_cell(bad, expected)
        self.assertEqual(aligned_cell(list(reversed(rows)), expected), rows)

    def test_no_private_identifiers_in_count_report(self):
        encoded = json.dumps(summarize_rows(fixture()))
        self.assertNotIn("private-", encoded)

    def test_reversal_requires_opposite_nonzero_signs(self):
        self.assertTrue(reversal(-.2, .1))
        self.assertFalse(reversal(0, .1))
        self.assertFalse(reversal(-.2, -.1))
        self.assertIsNone(reversal(None, .1))


if __name__ == "__main__":
    unittest.main()
