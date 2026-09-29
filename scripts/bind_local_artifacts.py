#!/usr/bin/env python3
"""Verify regenerated local inputs and preview a separate, hash-bound config.

No inference or downloads occur. The default is read-only; --write creates a
sibling *.local.json without replacing an existing file. Scientific settings
and source-content hashes are not changed. Never use this to relabel old runs:
prepare the inputs first, bind once, then use that config for the entire run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg2.mimic.csvio import config_sha256

CONFIGS = {
    "e1-s2": "e1_s2.structured_output.v1.json",
    "compact": "e1_contract_ablation.v1.json",
    "e3-source": "e3.ext_notes.v1.json",
    "e3-paired": "e3_contract_comparison.v1.json",
}


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(base: Path, value: str) -> Path:
    return (base / value).resolve()


def verified_manifest(path: Path, primary: dict, *, task: str) -> dict:
    """Check source identity, settings, dimensions, and recorded file checksums."""
    manifest = read(path)
    require(manifest.get(f"{task}_config_sha256") == config_sha256(primary),
            "manifest does not belong to the selected primary config; regenerate it")
    require(manifest.get("experiment_id") == primary["experiment_id"],
            "manifest experiment identity differs")
    for key in ("benchmark_config_sha256", "prompt_sha256"):
        require(manifest.get(key) == primary[key], "manifest " + key + " differs")
    artifact_keys = ("output_requests", "output_blueprint") if task == "e1" else ("output_requests",)
    for key in artifact_keys:
        require(digest(resolve(ROOT, manifest[key])) == manifest[key + "_sha256"],
                "manifest file checksum differs: " + key)
    if task == "e1":
        require(manifest.get("adapter_stage") == "slm_benchmark_e1_dataset",
                "not an E1 dataset manifest")
        for key, value in primary["expected_dataset"].items():
            require(manifest.get(key) == value, "E1 manifest count differs: " + key)
        require(manifest.get("construction_version") == primary["construction"]["version"],
                "E1 construction version differs")
    else:
        require(manifest.get("adapter_stage") == "slm_benchmark_e3_prepare",
                "not an E3 prepared dataset manifest")
        require(manifest.get("all_models_fit_context") is True,
                "E3 shared-context token audit has not passed")
        for key, value in primary["dataset"]["expected_counts"].items():
            require(manifest["counts"].get(key) == value, "E3 manifest count differs: " + key)
        require(manifest.get("context_policy") == primary["context_policy"],
                "E3 context policy differs")
        require(manifest.get("dataset_version") == primary["dataset"]["version"],
                "E3 dataset version differs")
        for key in ("notes_sha256", "labels_sha256", "source_audit_sha256"):
            require(manifest["source_files"].get(key) == primary["dataset"][key],
                    "E3 source provenance differs: " + key)
    return manifest


def propose(stage: str, source: Path, *, primary_path: Path | None = None) -> tuple[dict, dict]:
    """Return a bound copy and a non-patient audit; write no files."""
    source = source.resolve()
    original = read(source)
    require("local_binding" not in original, "bind from the released config, not an already-bound copy")
    result = copy.deepcopy(original)
    changes = {}

    def change(key: str, value) -> None:
        old = original
        new = result
        parts = key.split(".")
        for part in parts[:-1]:
            old, new = old[part], new[part]
        if old[parts[-1]] != value:
            changes[key] = {"before": old[parts[-1]], "after": value}
            new[parts[-1]] = value

    if stage == "e3-source":
        require(primary_path is None, "--primary-config applies only to follow-up stages")
        from cpg2.slm_benchmark import audit_ext_notes_dataset
        dataset = original["dataset"]
        audit_path = resolve(source.parent, dataset["source_audit"])
        saved = read(audit_path)
        observed = audit_ext_notes_dataset(dataset["root"])
        require(resolve(ROOT, saved["dataset_root"]) == resolve(ROOT, dataset["root"]),
                "source audit points to a different source directory")
        saved["dataset_root"] = observed["dataset_root"]
        require(saved == observed, "source audit differs from a new local audit")
        require(saved.get("restricted_text_or_identifiers_emitted") is False,
                "source audit is not aggregate-only")
        for filename, key in (("notes.csv", "notes_sha256"), ("labels.csv", "labels_sha256")):
            require(saved["files"][filename]["sha256"] == dataset[key],
                    "source bytes differ from the paper dataset: " + filename)
        for key, audit_key in (("requests", "label_rows"), ("notes", "notes"),
                               ("admissions", "admissions"), ("subjects", "subjects")):
            require(saved["counts"][audit_key] == dataset["expected_counts"][key],
                    "source audit count differs: " + key)
        change("dataset.source_audit_sha256", digest(audit_path))
    else:
        task = "e3" if stage == "e3-paired" else "e1"
        primary_key = f"primary_{task}_config"
        declared_primary = resolve(source.parent, original[primary_key])
        primary_path = (primary_path or declared_primary).resolve()
        require(primary_path.parent == source.parent,
                "primary config must remain in the same directory to preserve relative paths")
        declared_settings = read(declared_primary)
        selected_settings = read(primary_path)
        declared_settings.pop("local_binding", None)
        selected_settings.pop("local_binding", None)
        if task == "e3":
            # Rebinding a locally regenerated source audit is the sole allowed
            # primary-config difference; it must not change the clinical task.
            declared_settings["dataset"].pop("source_audit_sha256", None)
            selected_settings["dataset"].pop("source_audit_sha256", None)
        require(declared_settings == selected_settings,
                "selected primary config changes scientific settings, not just local audit binding")
        if task == "e1":
            from cpg2.slm_e1 import _load_e1_config
            _, primary = _load_e1_config(primary_path)
        else:
            from cpg2.slm_e3 import _load_config
            _, primary, *_ = _load_config(primary_path)
        manifest_path = resolve(source.parent, original["dataset_manifest"])
        verified_manifest(manifest_path, primary, task=task)
        change(primary_key, primary_path.name)
        change(primary_key + "_sha256", config_sha256(primary))
        change("dataset_manifest_sha256", digest(manifest_path))
        if stage == "e1-s2":
            results_path = resolve(source.parent, original["primary_results"])
            aggregate = read(results_path)
            require(aggregate.get("adapter_stage") == "slm_benchmark_e1_aggregate",
                    "not an E1 aggregate")
            require(aggregate.get("e1_config_sha256") == config_sha256(primary),
                    "E1 aggregate belongs to another configuration")
            require(aggregate.get("dataset_manifest_sha256") == digest(manifest_path),
                    "E1 aggregate belongs to another dataset manifest")
            require(aggregate.get("all_models_technically_completed") is True,
                    "primary E1 panel has not technically completed")
            require((aggregate.get("declared_models"), aggregate.get("requests_per_model")) == (8, 387),
                    "E1 aggregate dimensions differ")
            change("primary_results_sha256", digest(results_path))
    audit = {
        "stage": stage,
        "source_config_sha256": config_sha256(original),
        "changes": changes,
        "purpose": "Independent local rerun binding; does not change or relabel paper-run outputs.",
    }
    result["local_binding"] = audit
    return result, audit


def write_new(path: Path, payload: dict) -> None:
    """Publish a 0600 file without replacement, including under a write race."""
    if path.exists():
        require(read(path) == payload, "existing local config differs; choose a fresh name")
        require(path.stat().st_mode & 0o077 == 0, "existing local config permissions must be 0600")
        return
    descriptor, name = tempfile.mkstemp(prefix=".binding-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=CONFIGS)
    parser.add_argument("--config", type=Path, help="Released config; defaults to the named stage")
    parser.add_argument("--primary-config", type=Path, help="Verified local primary config, especially after e3-source binding")
    parser.add_argument("--output", type=Path, help="New sibling config; default: <source stem>.local.json")
    parser.add_argument("--write", action="store_true", help="Create the verified sibling copy; default only previews the changes")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT.resolve(), "run from the release repository root")
    source = (args.config or ROOT / "configs/slm_benchmark" / CONFIGS[args.stage]).resolve()
    require(source.is_relative_to(ROOT.resolve()), "config must be inside the release repository")
    output = (args.output or source.with_name(source.stem + ".local.json")).resolve()
    require(output.parent == source.parent and output != source,
            "output must be a separate sibling config so relative paths remain valid")
    require(output.name.endswith(".local.json"), "output must end in .local.json to remain Git-ignored")
    payload, audit = propose(args.stage, source, primary_path=args.primary_config)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.write:
        write_new(output, payload)
        print(f"Verified local config: {output.relative_to(ROOT)}")
    else:
        print("Preview only; use --write after reviewing the proposed bindings.")


if __name__ == "__main__":
    main()
