from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/analyze_e2_punctuation_policy_v0_7.py"
SPEC = importlib.util.spec_from_file_location("e2_punctuation_policy_v0_7", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class E2PunctuationPolicyAnalysisTests(unittest.TestCase):
    def test_parser_accepts_only_complete_binary_with_optional_period(self) -> None:
        for raw, expected in (
            ("Yes", "yes"),
            (" no ", "no"),
            ("YES.", "yes"),
            ("\nNo.\t", "no"),
        ):
            self.assertEqual(MODULE.parse_punctuation_output(raw), expected)

        for raw in ("No..", "No,", "**No**", "Answer: No", "", None):
            self.assertIsNone(MODULE.parse_punctuation_output(raw))
        self.assertIsNone(
            MODULE.parse_punctuation_output("No.", runtime_failed=True)
        )

    def test_failure_classes_do_not_reproduce_raw_text(self) -> None:
        self.assertEqual(
            MODULE.classify_strict_failure_output("No."),
            "binary_with_single_terminal_period",
        )
        self.assertEqual(
            MODULE.classify_strict_failure_output("**No**"),
            "other_text_with_standalone_binary_token",
        )
        self.assertEqual(
            MODULE.classify_strict_failure_output("Analyze the case"),
            "no_standalone_binary_token",
        )

    def test_replay_accepts_terminal_period_and_rejects_markdown(self) -> None:
        machine = {
            "domain": "test",
            "root_state_id": "test.n01",
            "states": {
                "test.n01": {
                    "state_id": "test.n01",
                    "node_name": "Criterion",
                    "questions": [
                        {
                            "query_id": "test.n01",
                            "criterion": "Criterion",
                            "reference_source": "positive_features",
                        }
                    ],
                    "minimum_yes": 1,
                    "yes_target": {"kind": "terminal", "value": "Action"},
                    "no_target": {
                        "kind": "terminal",
                        "value": "End of Decision Tree",
                    },
                }
            },
        }
        source_case = {
            "case_id": "case-1",
            "domain": "test",
            "expected_action": "End of Decision Tree",
        }

        def decision(raw: str) -> dict:
            return {
                "decision_id": "decision-1",
                "case_id": "case-1",
                "domain": "test",
                "round": 1,
                "state_id": "test.n01",
                "query_id": "test.n01",
                "criterion": "Criterion",
                "raw_output": raw,
                "runtime_failed": False,
                "parsed": MODULE.parse_binary_output(raw),
            }

        accepted = MODULE.replay_case(machine, source_case, [decision("No.")])
        self.assertEqual(accepted["status"], MODULE.STRICT_COMPLETED)
        self.assertTrue(accepted["terminal_correct"])
        self.assertEqual(accepted["decision_calls"], 1)

        rejected = MODULE.replay_case(machine, source_case, [decision("**No**")])
        self.assertEqual(rejected["status"], MODULE.OUTPUT_FAILURE)
        self.assertFalse(rejected["terminal_correct"])
        self.assertEqual(rejected["decision_calls"], 1)

    def test_percentile_interval_matches_frozen_e2_indexing(self) -> None:
        values = list(range(100))
        self.assertEqual(MODULE.percentile_interval(values), [2, 97])


if __name__ == "__main__":
    unittest.main()
