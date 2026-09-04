from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/slm_benchmark"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseConfigHashTests(unittest.TestCase):
    def test_portable_design_references_are_in_the_release(self) -> None:
        benchmark = _load(CONFIG_ROOT / "benchmark.v2.json")
        for field in ("design_document", "declaration_review"):
            self.assertTrue((CONFIG_ROOT / benchmark[field]).resolve().is_file())

        tree_config = _load(CONFIG_ROOT / "e2.historical_tree_binary.v1.json")
        self.assertTrue(
            (CONFIG_ROOT / tree_config["design_document"]).resolve().is_file()
        )

    def test_task_configs_match_benchmark_and_prompt_hashes(self) -> None:
        benchmark = _load(CONFIG_ROOT / "benchmark.v2.json")
        benchmark_hash = _config_sha256(benchmark)
        for filename in (
            "e1.synthetic_typed.v1.json",
            "e2.historical_tree_binary.v1.json",
            "e3.ext_notes.v1.json",
        ):
            task = _load(CONFIG_ROOT / filename)
            self.assertEqual(task["benchmark_config_sha256"], benchmark_hash)
            prompt = _load(CONFIG_ROOT / task["prompt"])
            self.assertEqual(task["prompt_sha256"], _config_sha256(prompt))

    def test_structured_sensitivity_matches_release_e1_config(self) -> None:
        primary = _load(CONFIG_ROOT / "e1.synthetic_typed.v1.json")
        sensitivity = _load(CONFIG_ROOT / "e1_s2.structured_output.v1.json")
        self.assertEqual(
            sensitivity["primary_e1_config_sha256"], _config_sha256(primary)
        )
        for override in sensitivity["wrapper_overrides"].values():
            wrapper_path = CONFIG_ROOT / override["wrapper"]
            self.assertEqual(override["wrapper_sha256"], _file_sha256(wrapper_path))

    def test_model_and_tree_wrapper_hashes_are_internal_consistent(self) -> None:
        benchmark = _load(CONFIG_ROOT / "benchmark.v2.json")
        for relative in benchmark["model_manifests"]:
            manifest_path = CONFIG_ROOT / relative
            manifest = _load(manifest_path)
            wrapper_path = manifest_path.parent / manifest["model"]["wrapper_spec"]
            self.assertEqual(
                manifest["verification"]["wrapper_spec_sha256"],
                _file_sha256(wrapper_path),
            )

        tree_config = _load(CONFIG_ROOT / "e2.historical_tree_binary.v1.json")
        for override in tree_config["wrapper_overrides"].values():
            wrapper_path = CONFIG_ROOT / override["wrapper"]
            self.assertEqual(override["wrapper_sha256"], _file_sha256(wrapper_path))


if __name__ == "__main__":
    unittest.main()
