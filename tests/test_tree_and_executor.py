import copy
import json
import unittest
from pathlib import Path

from cpg2.executor import execute_partial
from cpg2.schemas import CriterionAssessment
from cpg2.states import CriterionState
from cpg2.tree import TreeValidationError, tree_from_dict, validate_tree

from .helpers import ROOT, demo_tree


def assessment(criterion_id, state):
    return CriterionAssessment(criterion_id, state, resolution_method="test")


class TreeAndExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tree = demo_tree()

    def test_fully_resolved_path(self):
        result = execute_partial(
            self.tree,
            {"demo_threshold_signal": assessment("demo_threshold_signal", CriterionState.PRESENT_CURRENT)},
        )
        self.assertEqual(result.reachable_leaf_ids, ("demo_t_a",))
        self.assertEqual(result.compatible_action_ids, ("demo_action_a",))
        self.assertTrue(result.unique_action)

    def test_unresolved_explores_both_and_is_critical(self):
        result = execute_partial(self.tree, {})
        self.assertEqual(set(result.compatible_action_ids), {"demo_action_a", "demo_action_b"})
        self.assertIn("demo_n1", result.decision_critical_unresolved_node_ids)

    def test_multiple_paths_same_action_are_unique(self):
        result = execute_partial(
            self.tree,
            {"demo_threshold_signal": assessment("demo_threshold_signal", CriterionState.ABSENT_EXPLICIT)},
        )
        self.assertEqual(len(result.reachable_paths), 2)
        self.assertEqual(result.compatible_action_ids, ("demo_action_b",))
        self.assertTrue(result.unique_action)
        self.assertNotIn("demo_n2", result.decision_critical_unresolved_node_ids)

    def test_cycle_dangling_unreachable_and_malformed_map_fail(self):
        source = json.loads((ROOT / "examples" / "mechanical_tree.json").read_text())
        cycle = copy.deepcopy(source)
        cycle["nodes"][3]["node_type"] = "criterion"
        cycle["nodes"][3].update(
            {
                "criterion_id": "cycle",
                "question": "cycle",
                "criterion_type": "semantic",
                "temporal_rule": {"cutoff_relation": "at_or_before", "lookback_hours": None},
                "state_branch_map": copy.deepcopy(cycle["nodes"][0]["state_branch_map"]),
                "branches": {"yes": "demo_n1", "no": "demo_n1"},
            }
        )
        cycle["nodes"][3].pop("terminal_action_id")
        cycle["nodes"][3].pop("terminal_action_label")
        self.assertTrue(any("cycle detected" in item for item in validate_tree(cycle)["errors"]))

        dangling = copy.deepcopy(source)
        dangling["nodes"][0]["branches"]["yes"] = "missing"
        self.assertTrue(any("missing node" in item for item in validate_tree(dangling)["errors"]))

        unreachable = copy.deepcopy(source)
        unreachable["nodes"].append(
            {
                "node_id": "orphan", "node_type": "terminal", "terminal_action_id": "x",
                "terminal_action_label": "x", "source_reference": "test"
            }
        )
        self.assertIn("unreachable node: orphan", validate_tree(unreachable)["errors"])

        malformed = copy.deepcopy(source)
        malformed["nodes"][0]["state_branch_map"]["NOT_DOCUMENTED"] = []
        with self.assertRaises(TreeValidationError):
            tree_from_dict(malformed)


if __name__ == "__main__":
    unittest.main()
