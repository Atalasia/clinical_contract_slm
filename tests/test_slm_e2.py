from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from cpg2.historical_synthetic import END_LABEL
from cpg2.slm_e2 import (
    OUTPUT_FAILURE,
    RUNTIME_FAILURE,
    STRICT_COMPLETED,
    advance_traversal,
    build_historical_machine,
    compute_terminal_metrics,
    current_query,
    new_traversal,
    parse_binary_output,
    prepare_e2_dataset,
    summarize_policy_cases,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
E2_CONFIG = REPOSITORY_ROOT / "configs/slm_benchmark/e2.historical_tree_binary.v1.json"
E2_SOURCE_PATHS = (
    REPOSITORY_ROOT / "data/cpgprompt/headache/merged_decision_tree.json",
    REPOSITORY_ROOT / "data/cpgprompt/headache/vignettes_data.csv",
    REPOSITORY_ROOT
    / "data/cpgprompt/lower_back_pain/decision_tree_lower_back_pain.json",
    REPOSITORY_ROOT
    / "data/cpgprompt/lower_back_pain/lower_back_pain_results_short.csv",
    REPOSITORY_ROOT
    / "data/cpgprompt/prostate_cancer/decision_trees_output_text.json",
    REPOSITORY_ROOT
    / "data/cpgprompt/prostate_cancer/vignettes_data_pc_test.csv",
)


class SlmE2Tests(unittest.TestCase):
    def test_strict_parser_requires_whole_output_but_legacy_uses_first_token(self) -> None:
        for raw, expected in (("Yes", "yes"), ("  no\n", "no"), ("YES", "yes")):
            parsed = parse_binary_output(raw)
            self.assertTrue(parsed["strict_valid"])
            self.assertEqual(parsed["strict_answer"], expected)
            self.assertEqual(parsed["legacy_answer"], expected)
            self.assertFalse(parsed["legacy_coerced"])
            self.assertIsNone(parsed["failure"])

        punctuated = parse_binary_output("Yes.")
        self.assertFalse(punctuated["strict_valid"])
        self.assertIsNone(punctuated["strict_answer"])
        self.assertEqual(punctuated["legacy_answer"], "yes")
        self.assertTrue(punctuated["legacy_coerced"])
        self.assertEqual(punctuated["failure"], OUTPUT_FAILURE)

        first_token = parse_binary_output("No, on reflection yes")
        self.assertFalse(first_token["strict_valid"])
        self.assertEqual(first_token["legacy_answer"], "no")

        unclear = parse_binary_output("The evidence is unclear")
        self.assertFalse(unclear["strict_valid"])
        self.assertEqual(unclear["legacy_answer"], "no")
        self.assertTrue(unclear["legacy_coerced"])

    def test_runtime_failure_is_not_coerced_by_legacy_policy(self) -> None:
        parsed = parse_binary_output(None, runtime_failed=True)
        self.assertTrue(parsed["runtime_failed"])
        self.assertFalse(parsed["strict_valid"])
        self.assertIsNone(parsed["strict_answer"])
        self.assertIsNone(parsed["legacy_answer"])
        self.assertFalse(parsed["legacy_coerced"])
        self.assertEqual(parsed["failure"], RUNTIME_FAILURE)

    def test_fixed_reference_macro_f1_counts_failure_as_false_negative(self) -> None:
        metrics = compute_terminal_metrics(
            ["Action A", "Action A", "Action B"],
            ["Action A", None, "Action B"],
            labels=["Action A", "Action B"],
        )
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["failure_n"], 1)
        self.assertEqual(metrics["label_count"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["per_reference_label"]["Action A"]["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["per_reference_label"]["Action B"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 6)

        predicted_only_label = compute_terminal_metrics(
            ["Action A", "Action B"],
            ["Unrecognized terminal", "Action B"],
            labels=["Action A", "Action B"],
        )
        self.assertEqual(predicted_only_label["label_count"], 2)
        self.assertNotIn(
            "Unrecognized terminal", predicted_only_label["per_reference_label"]
        )

    def test_terminal_aliases_are_normalized_before_scoring(self) -> None:
        metrics = compute_terminal_metrics(
            ["End"], ["No specific action required"], labels=[END_LABEL]
        )
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_domain_balanced_endpoint_is_not_pooled_accuracy(self) -> None:
        rows = [
            {
                "domain": "small_domain",
                "expected_action": "A",
                "strict": {"predicted_action": "A"},
                "legacy": {"predicted_action": "A"},
            },
            *[
                {
                    "domain": "large_domain",
                    "expected_action": "B",
                    "strict": {"predicted_action": "wrong"},
                    "legacy": {"predicted_action": "wrong"},
                }
                for _ in range(3)
            ],
        ]
        summary = summarize_policy_cases(rows, "strict")
        self.assertAlmostEqual(
            summary["primary_domain_balanced_terminal_accuracy"], 0.5
        )
        self.assertAlmostEqual(summary["pooled_terminal"]["accuracy"], 0.25)

    def test_threshold_node_stops_as_soon_as_minimum_is_reached(self) -> None:
        machine = build_historical_machine(
            "threshold_test",
            {
                "nodes": [
                    {
                        "name": "Aggregate finding",
                        "criteria": ["component one", "component two", "component three"],
                        "min_criteria": 2,
                        "yes_action": "Threshold action",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
            "linear_threshold",
        )
        traversal = new_traversal(machine)
        self.assertEqual(current_query(machine, traversal)["criterion"], "component one")
        self.assertIsNone(advance_traversal(machine, traversal, "yes"))
        self.assertEqual(current_query(machine, traversal)["criterion"], "component two")
        branch = advance_traversal(machine, traversal, "yes")
        self.assertEqual(branch["branch"], "yes")
        self.assertEqual(branch["questions_consumed"], 2)
        self.assertEqual(traversal["decision_calls"], 2)
        self.assertEqual(traversal["status"], STRICT_COMPLETED)
        self.assertEqual(traversal["predicted_action"], "Threshold action")

    def test_sectioned_tree_resolves_casefolded_prostate_subtree_targets(self) -> None:
        tree = {
            "symptoms": {
                "nodes": [
                    {
                        "name": "LUTS entry finding",
                        "yes_action": "  ACTIONS FOR PATIENTS WITH LUTS ",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
            "actions for patients with unexplained symptoms of metastatic prostate cancer": {
                "nodes": [
                    {
                        "name": "Metastatic finding",
                        "yes_action": "Metastatic action",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
            "actions for patients with luts": {
                "nodes": [
                    {
                        "name": "LUTS risk finding",
                        "yes_action": "NoMoGrAmS",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
            "actions for patients with incidental elevated psa results": {
                "nodes": [
                    {
                        "name": "Incidental finding",
                        "yes_action": "Incidental action",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
            "nomograms": {
                "nodes": [
                    {
                        "name": "Nomogram finding",
                        "yes_action": "Nomogram action",
                        "no_node": "End of Decision Tree",
                    }
                ]
            },
        }
        machine = build_historical_machine(
            "prostate_test", tree, "sectioned"
        )
        traversal = new_traversal(machine)
        advance_traversal(machine, traversal, "yes")
        self.assertEqual(traversal["current_state_id"], "prostate_test.luts.n01")
        advance_traversal(machine, traversal, "yes")
        self.assertEqual(traversal["current_state_id"], "prostate_test.nomograms.n01")
        advance_traversal(machine, traversal, "yes")
        self.assertEqual(traversal["status"], STRICT_COMPLETED)
        self.assertEqual(traversal["predicted_action"], "Nomogram action")

    @unittest.skipUnless(
        all(path.is_file() for path in E2_SOURCE_PATHS),
        "requires the upstream CPGPrompt source files",
    )
    def test_repository_sources_reconstruct_frozen_reference_and_all_no_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            blueprint_path = root / "blueprint.json"
            manifest_path = root / "manifest.json"
            manifest = prepare_e2_dataset(
                E2_CONFIG,
                output_cases=cases_path,
                output_blueprint=blueprint_path,
                output_manifest=manifest_path,
            )

            self.assertEqual(manifest["cases"], 323)
            self.assertEqual(
                manifest["domain_case_counts"],
                {"headache": 128, "low_back_pain": 99, "prostate_cancer": 96},
            )
            self.assertEqual(manifest["query_definitions"], 68)
            self.assertEqual(manifest["reference_route_queries"], 3778)
            self.assertEqual(
                manifest["domain_reference_route_queries"],
                {"headache": 2459, "low_back_pain": 564, "prostate_cancer": 755},
            )
            self.assertEqual(manifest["reference_terminal_matches"], 323)
            self.assertEqual(
                manifest["reference_terminal_classes"],
                {"headache": 24, "low_back_pain": 7, "prostate_cancer": 11},
            )

            for restricted_path in (cases_path, blueprint_path, manifest_path):
                self.assertEqual(stat.S_IMODE(restricted_path.stat().st_mode), 0o600)

            blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                blueprint["all_no_terminal_by_domain"],
                {
                    "headache": END_LABEL,
                    "low_back_pain": END_LABEL,
                    "prostate_cancer": END_LABEL,
                },
            )
            cases = [
                json.loads(line)
                for line in cases_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(len(case["reference"]["decisions"]) for case in cases), 3778
            )
            baseline_rows = [
                {
                    "domain": case["domain"],
                    "expected_action": case["expected_action"],
                    "strict": {
                        "predicted_action": blueprint["all_no_terminal_by_domain"][
                            case["domain"]
                        ]
                    },
                    "legacy": {
                        "predicted_action": blueprint["all_no_terminal_by_domain"][
                            case["domain"]
                        ]
                    },
                }
                for case in cases
            ]
            baseline = summarize_policy_cases(baseline_rows, "strict")
            self.assertEqual(baseline["by_domain"]["headache"]["correct"], 32)
            self.assertEqual(baseline["by_domain"]["low_back_pain"]["correct"], 0)
            self.assertEqual(baseline["by_domain"]["prostate_cancer"]["correct"], 5)
            self.assertAlmostEqual(
                baseline["primary_domain_balanced_terminal_accuracy"],
                0.10069444444444443,
            )
            self.assertAlmostEqual(
                baseline["pooled_terminal"]["accuracy"], 37 / 323
            )


if __name__ == "__main__":
    unittest.main()
