from __future__ import annotations

import unittest

from cpg2.target_spec import legacy_feature_name


class TargetSpecTests(unittest.TestCase):
    def test_normalization_metadata_name_takes_precedence(self) -> None:
        node = {
            "node_id": "criterion.test",
            "source_reference": "trees/lbp_tree.json:nodes[0]",
            "normalization_metadata": {"legacy_name": "Preferred source name"},
        }
        self.assertEqual(legacy_feature_name(node), "Preferred source name")

    def test_source_reference_preserves_slashes_in_the_feature(self) -> None:
        node = {
            "node_id": "criterion.test",
            "source_reference": (
                "trees/headache_tree.json:redflags/"
                "headache triggered by cough/exertion"
            ),
        }
        self.assertEqual(
            legacy_feature_name(node), "headache triggered by cough/exertion"
        )


if __name__ == "__main__":
    unittest.main()
