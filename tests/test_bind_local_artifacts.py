from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import bind_local_artifacts as binding


def write_fixture(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fixture(root):
    primary = {
        "experiment_id": "invented-test", "benchmark_config_sha256": "benchmark",
        "prompt_sha256": "prompt", "expected_dataset": {"requests": 2, "cases": 1},
        "construction": {"version": "invented-v1"},
    }
    primary_path = write_fixture(root / "primary.json", primary)
    requests = write_fixture(root / "requests.json", [{"invented": True}])
    blueprint = write_fixture(root / "blueprint.json", {"invented": True})
    manifest = {
        "e1_config_sha256": binding.config_sha256(primary),
        "experiment_id": primary["experiment_id"],
        "benchmark_config_sha256": "benchmark", "prompt_sha256": "prompt",
        "adapter_stage": "slm_benchmark_e1_dataset", "requests": 2, "cases": 1,
        "construction_version": "invented-v1", "output_requests": str(requests),
        "output_requests_sha256": binding.digest(requests),
        "output_blueprint": str(blueprint), "output_blueprint_sha256": binding.digest(blueprint),
    }
    manifest_path = write_fixture(root / "manifest.json", manifest)
    source = write_fixture(root / "followup.json", {
        "primary_e1_config": primary_path.name, "primary_e1_config_sha256": "old",
        "dataset_manifest": manifest_path.name, "dataset_manifest_sha256": "old",
        "conditions": ["invented-condition"],
    })
    return primary, primary_path, manifest, manifest_path, source


class LocalBindingTests(unittest.TestCase):
    def test_proposal_changes_only_binding_and_writes_nothing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, primary_path, _, manifest_path, source = fixture(root)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with patch("cpg2.slm_e1._load_e1_config", return_value=(primary_path, primary)):
                bound, audit = binding.propose("compact", source)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            self.assertEqual(bound["primary_e1_config_sha256"], binding.config_sha256(primary))
            self.assertEqual(bound["dataset_manifest_sha256"], binding.digest(manifest_path))
            self.assertEqual(bound["conditions"], ["invented-condition"])
            self.assertEqual(set(audit["changes"]), {"primary_e1_config_sha256", "dataset_manifest_sha256"})
            self.assertEqual(bound["local_binding"], audit)

    def test_manifest_rejects_hash_identity_count_and_content_mismatch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, _, manifest, path, _ = fixture(root)
            self.assertEqual(binding.verified_manifest(path, primary, task="e1"), manifest)
            changes = {"e1_config_sha256": "wrong", "experiment_id": "wrong",
                       "prompt_sha256": "wrong", "requests": 3,
                       "output_requests_sha256": "wrong", "construction_version": "wrong"}
            for key, value in changes.items():
                with self.subTest(key=key):
                    write_fixture(path, {**manifest, key: value})
                    with self.assertRaises(ValueError):
                        binding.verified_manifest(path, primary, task="e1")

    def test_primary_scientific_change_and_already_bound_source_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, _, _, _, source = fixture(root)
            changed = write_fixture(root / "changed.json", {**primary, "scientific_setting": "changed"})
            with self.assertRaisesRegex(ValueError, "scientific settings"):
                binding.propose("compact", source, primary_path=changed)
            payload = binding.read(source)
            write_fixture(source, {**payload, "local_binding": {}})
            with self.assertRaisesRegex(ValueError, "already-bound"):
                binding.propose("compact", source)

    def test_note_manifest_rejects_context_source_and_count_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = write_fixture(root / "requests.json", [{"invented": True}])
            primary = {"experiment_id": "invented-note", "benchmark_config_sha256": "b",
                       "prompt_sha256": "p", "context_policy": {"mode": "invented"},
                       "dataset": {"expected_counts": {"requests": 2}, "version": "test",
                                   "notes_sha256": "n", "labels_sha256": "l", "source_audit_sha256": "a"}}
            manifest = {"e3_config_sha256": binding.config_sha256(primary),
                        "experiment_id": "invented-note", "benchmark_config_sha256": "b",
                        "prompt_sha256": "p", "adapter_stage": "slm_benchmark_e3_prepare",
                        "all_models_fit_context": True, "counts": {"requests": 2},
                        "context_policy": primary["context_policy"], "dataset_version": "test",
                        "source_files": {key: primary["dataset"][key] for key in
                                         ("notes_sha256", "labels_sha256", "source_audit_sha256")},
                        "output_requests": str(requests), "output_requests_sha256": binding.digest(requests)}
            path = write_fixture(root / "manifest.json", manifest)
            self.assertEqual(binding.verified_manifest(path, primary, task="e3"), manifest)
            for key, value in (("all_models_fit_context", False), ("context_policy", {}),
                               ("counts", {"requests": 3}), ("source_files", {})):
                with self.subTest(key=key):
                    write_fixture(path, {**manifest, key: value})
                    with self.assertRaises(ValueError):
                        binding.verified_manifest(path, primary, task="e3")

    def test_write_is_private_idempotent_and_never_replaces_different_content(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.local.json"
            binding.write_new(path, {"invented": True})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            before = path.read_bytes()
            binding.write_new(path, {"invented": True})
            with self.assertRaisesRegex(ValueError, "differs"):
                binding.write_new(path, {"invented": False})
            self.assertEqual(path.read_bytes(), before)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "permissions"):
                binding.write_new(path, {"invented": True})

    def test_cli_defaults_to_preview_and_guards_source_and_output_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            with patch.object(binding, "ROOT", root), patch.object(Path, "cwd", return_value=root), \
                 patch.object(binding, "propose", return_value=({}, {})) as propose, \
                 patch.object(binding, "write_new") as write, redirect_stdout(StringIO()) as out:
                with patch.object(sys, "argv", ["bind", "compact", "--config", str(source)]):
                    binding.main()
                write.assert_not_called()
                self.assertIn("Preview only", out.getvalue())
                self.assertEqual(list(root.iterdir()), [])
                for arguments in (["--output", str(root / "unsafe.json")],
                                  ["--config", str(root.parent / "outside.json")]):
                    propose.reset_mock()
                    with patch.object(sys, "argv", ["bind", "compact", "--config", str(source), *arguments]), \
                         self.assertRaises(ValueError):
                        binding.main()
                    propose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
