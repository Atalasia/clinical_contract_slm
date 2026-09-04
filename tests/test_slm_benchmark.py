from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from cpg2.slm_benchmark import (
    audit_ext_notes_dataset,
    validate_benchmark_config,
    validate_gate_suite,
    validate_model_manifest,
)

from .helpers import ROOT


class SlmBenchmarkPreflightTests(unittest.TestCase):
    def test_release_benchmark_declares_the_frozen_panel(self) -> None:
        source = ROOT / "configs/slm_benchmark/benchmark.v2.json"
        benchmark = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(len(benchmark["panel"]["core_pilot"]), 6)
        self.assertEqual(len(benchmark["panel"]["contemporary_references"]), 2)
        self.assertEqual(len(benchmark["model_manifests"]), 8)
        self.assertEqual(len(benchmark["contrasts"]), 7)
        for relative_path in benchmark["model_manifests"]:
            manifest = json.loads(
                (source.parent / relative_path).read_text(encoding="utf-8")
            )
            report = validate_model_manifest(manifest, check_local_paths=False)
            self.assertEqual(report["errors"], [], report)

        gate = json.loads(
            (ROOT / "configs/slm_benchmark/gates/compatibility.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(gate["cases"]), 48)

    def test_gate_validator_rejects_duplicate_case_ids(self) -> None:
        source = ROOT / "configs/slm_benchmark/gates/compatibility.v1.json"
        suite = json.loads(source.read_text(encoding="utf-8"))
        broken = copy.deepcopy(suite)
        broken["cases"][1]["case_id"] = broken["cases"][0]["case_id"]
        report = validate_gate_suite(broken)
        self.assertTrue(any("duplicate case_id" in error for error in report["errors"]))

    def test_manifest_validator_rejects_factor_misclassification(self) -> None:
        source = ROOT / "configs/slm_benchmark/models/qwen2_5_1_5b_pt.v1.json"
        manifest = json.loads(source.read_text(encoding="utf-8"))
        broken = copy.deepcopy(manifest)
        broken["factors"]["release_reference"] = True
        report = validate_model_manifest(broken, check_local_paths=False)
        self.assertTrue(
            any("core_pilot models cannot" in error for error in report["errors"])
        )

    def test_manifest_never_treats_missing_hashes_as_verified(self) -> None:
        source = ROOT / "configs/slm_benchmark/models/qwen2_5_1_5b_pt.v1.json"
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["verification"]["artifact_sha256"] = None
        report = validate_model_manifest(manifest, check_local_paths=False)
        self.assertTrue(any("FROZEN manifest" in item for item in report["errors"]))
        self.assertTrue(any("artifact_sha256" in item for item in report["blockers"]))

    def test_ext_notes_audit_emits_only_aggregate_information(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.csv").write_text(
                "row_id,subject_id,hadm_id,text\n1,patient-secret,visit-secret,No cough.\n",
                encoding="utf-8",
            )
            (root / "labels.csv").write_text(
                "row_id,trigger_word,concept,semtypes,start,end,detection,encounter,negation\n"
                "1,cough,Cough,sosy,3,8,no,yes,no\n",
                encoding="utf-8",
            )
            audit = audit_ext_notes_dataset(root)
        serialized = json.dumps(audit)
        self.assertNotIn("patient-secret", serialized)
        self.assertNotIn("visit-secret", serialized)
        self.assertNotIn("No cough", serialized)
        self.assertEqual(
            audit["integrity"]["detection_no_non_dash_downstream_rows"], 1
        )
        self.assertFalse(audit["restricted_text_or_identifiers_emitted"])


if __name__ == "__main__":
    unittest.main()
