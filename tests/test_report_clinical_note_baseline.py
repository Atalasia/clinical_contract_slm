from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cpg2.mimic.csvio import config_sha256


SPEC = importlib.util.spec_from_file_location(
    "report_clinical_note_baseline", Path(__file__).resolve().parents[1] / "scripts/report_clinical_note_baseline.py"
)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class NoteBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_dir = self.root / "configs/slm_benchmark"
        self.config_dir.mkdir(parents=True)
        self.source_dir = self.root / "data/notes"
        self.source_dir.mkdir(parents=True)
        self.input_dir = self.root / "outputs/local_restricted/fixture"
        self.input_dir.mkdir(parents=True)
        self.rows = []
        # These are invented labels and explicit privacy sentinels, not real data.
        for i, (detection, encounter, negation) in enumerate([
            ("yes", "yes", "no"), ("yes", "yes", "no"),
            ("yes", "yes", "no"), ("yes", "no", "yes"),
            ("yes", "no", "unsure"), ("no", "not_applicable", "not_applicable"),
        ]):
            self.rows.append({"request_id": f"private-request-{i}",
                              "note_cluster": "private-note", "admission_cluster": "private-admission",
                              "subject_cluster": "private-subject", "note_context": "private-text-sentinel",
                              "expected": {"detection": detection, "encounter": encounter, "negation": negation}})
        self.primary = {
            "experiment_id": "mock_e3", "dataset": {
                "name": "MIMIC-III-Ext-Notes", "version": "1.0.0", "root": "data/notes",
                "notes_file": "notes.csv", "labels_file": "labels.csv",
                "source_audit": "../../outputs/local_restricted/fixture/source_audit.json",
                "expected_counts": {"requests": 6, "notes": 1, "admissions": 1, "subjects": 1},
            },
        }
        for name in ("notes", "labels"):
            path = self.source_dir / f"{name}.csv"
            path.write_text(f"invented-{name}\n", encoding="utf-8")
            self.primary["dataset"][f"{name}_sha256"] = self.digest(path)
        self.audit = self.input_dir / "source_audit.json"
        self.audit.write_text("{}\n", encoding="utf-8")
        self.primary["dataset"]["source_audit_sha256"] = self.digest(self.audit)
        self.requests = self.input_dir / "requests.jsonl"
        self.manifest_path = self.input_dir / "manifest.json"
        self.config_path = self.config_dir / "paired.json"
        self.save_fixture()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_fixture(self):
        (self.config_dir / "primary.json").write_text(json.dumps(self.primary), encoding="utf-8")
        self.requests.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")
        counts = {head: dict(Counter(row["expected"][field] for row in self.rows
                                    if field == "detection" or row["expected"]["detection"] == "yes"))
                  for head, field in baseline.HEADS.items()}
        self.manifest = {"adapter_stage": "slm_benchmark_e3_prepare", "experiment_id": "mock_e3",
                         "e3_config_sha256": config_sha256(self.primary), "dataset": "MIMIC-III-Ext-Notes",
                         "dataset_version": "1.0.0", "source_files": {
                             key: self.primary["dataset"][key] for key in ("notes_sha256", "labels_sha256", "source_audit_sha256")},
                         "counts": dict(self.primary["dataset"]["expected_counts"]), "label_counts": counts,
                         "output_requests": str(self.requests.relative_to(self.root)),
                         "output_requests_sha256": self.digest(self.requests)}
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.config = {"freeze_status": "FROZEN", "primary_e3_config": "primary.json",
                       "primary_e3_config_sha256": config_sha256(self.primary),
                       "dataset_manifest": "../../outputs/local_restricted/fixture/manifest.json",
                       "dataset_manifest_sha256": self.digest(self.manifest_path),
                       "expected_counts": dict(self.primary["dataset"]["expected_counts"])}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def report(self):
        return baseline.build_report(self.config_path, repository_root=self.root)

    def test_joint_and_fixed_class_macro_f1_preserve_all_candidates(self):
        result = self.report()
        self.assertEqual(result["hierarchical_joint"], {"correct": 3, "n": 6, "exact_match": 0.5})
        self.assertAlmostEqual(result["per_head"]["negation_gold_detection_yes"]["metrics"]["macro_f1"], 0.25)
        self.assertEqual(result["predictions"], baseline.FIXED_PREDICTION)
        serialized = json.dumps(result)
        for private in ("private-request", "private-note", "private-admission", "private-subject", "private-text-sentinel"):
            self.assertNotIn(private, serialized)

    def test_request_hash_change_rejected(self):
        self.requests.write_text(self.requests.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "request file hash"):
            self.report()

    def test_source_file_change_rejected(self):
        (self.source_dir / "notes.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source notes hash"):
            self.report()

    def test_source_audit_change_rejected(self):
        self.audit.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source audit hash"):
            self.report()

    def test_manifest_change_rejected(self):
        self.manifest_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.report()

    def test_primary_config_change_rejected(self):
        (self.config_dir / "primary.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "primary E3 configuration hash"):
            self.report()

    def test_duplicate_request_rejected_without_exposing_id(self):
        self.rows[1]["request_id"] = self.rows[0]["request_id"]
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "identifiers must be unique") as raised:
            self.report()
        self.assertNotIn("private-request", str(raised.exception))

    def test_invalid_expected_labels_rejected(self):
        self.rows[-1]["expected"]["negation"] = "no"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "invalid expected labels"):
            self.report()

    def test_marginals_mismatch_rejected(self):
        self.manifest["label_counts"]["detection"] = {"yes": 4, "no": 2}
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.config["dataset_manifest_sha256"] = self.digest(self.manifest_path)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "marginals differ"):
            self.report()

    def test_tied_or_changed_majority_rejected(self):
        self.rows[2]["expected"]["encounter"] = "no"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "unique per-head majority"):
            self.report()

    def test_cluster_count_mismatch_rejected(self):
        self.rows[0]["subject_cluster"] = "another-private-subject"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "cluster counts differ"):
            self.report()

    def test_restricted_no_overwrite_output(self):
        report = self.report()
        output = self.input_dir / "baseline.json"
        baseline.write_report(output, report, repository_root=self.root)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        baseline.write_report(output, report, repository_root=self.root)
        with self.assertRaisesRegex(ValueError, "existing baseline differs"):
            baseline.write_report(output, {"changed": True}, repository_root=self.root)
        with self.assertRaisesRegex(ValueError, "must remain under"):
            baseline.write_report(self.root / "public.json", report, repository_root=self.root)


if __name__ == "__main__":
    unittest.main()
