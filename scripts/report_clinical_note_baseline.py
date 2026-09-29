#!/usr/bin/env python3
"""Recompute the paper's label-only clinical-note baseline, without inference.

The frozen request file is read locally. Only expected labels are retained for
scoring; clinical text and row/cluster identifiers are never included in output.
This isolates the note baseline from the older E4 cross-task analysis pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from cpg2.ext_notes_benchmark import validate_ext_notes_labels
from cpg2.mimic.csvio import config_sha256
from cpg2.slm_e3 import _baseline_metrics, _hierarchical_correct


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/slm_benchmark/e3_contract_comparison.v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/local_restricted/slm_benchmark_v2/e3_contract_comparison_v1/baselines/clinical_note_majority.json"
FIXED_PREDICTION = {"detection": "yes", "encounter": "yes", "negation": "no"}
HEADS = {
    "detection": "detection",
    "encounter_gold_detection_yes": "encounter",
    "negation_gold_detection_yes": "negation",
}


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError("invalid JSON in baseline input") from None
    if not isinstance(value, dict):
        raise ValueError("baseline input must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("input path must be a nonempty string")
    return (base / value).resolve()


def build_report(config_path: Path, *, repository_root: Path = ROOT) -> dict:
    """Verify frozen inputs and retain every candidate in the joint denominator."""
    config_path, repository_root = config_path.resolve(), repository_root.resolve()
    config = _read_object(config_path)
    if config.get("freeze_status") != "FROZEN":
        raise ValueError("paired clinical-note configuration must be frozen")
    primary_path = _resolve(config_path.parent, config["primary_e3_config"])
    primary = _read_object(primary_path)
    if config_sha256(primary) != config["primary_e3_config_sha256"]:
        raise ValueError("primary E3 configuration hash differs")
    manifest_path = _resolve(config_path.parent, config["dataset_manifest"])
    if _sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash differs")
    manifest = _read_object(manifest_path)
    dataset = primary["dataset"]
    if (manifest.get("adapter_stage") != "slm_benchmark_e3_prepare"
            or manifest.get("experiment_id") != primary["experiment_id"]
            or manifest.get("e3_config_sha256") != config_sha256(primary)
            or manifest.get("dataset") != dataset["name"]
            or manifest.get("dataset_version") != dataset["version"]):
        raise ValueError("dataset manifest identity or primary configuration differs")
    source_root = _resolve(repository_root, dataset["root"])
    source_hashes = {}
    for name in ("notes", "labels"):
        observed = _sha256(_resolve(source_root, dataset[f"{name}_file"]))
        if (observed != dataset[f"{name}_sha256"]
                or observed != manifest["source_files"][f"{name}_sha256"]):
            raise ValueError(f"source {name} hash differs")
        source_hashes[f"{name}_sha256"] = observed
    audit_path = _resolve(primary_path.parent, dataset["source_audit"])
    source_hashes["source_audit_sha256"] = _sha256(audit_path)
    if (source_hashes["source_audit_sha256"] != dataset["source_audit_sha256"]
            or source_hashes != manifest["source_files"]):
        raise ValueError("source audit hash differs")
    requests_path = _resolve(repository_root, manifest["output_requests"])
    requests_hash = _sha256(requests_path)
    if requests_hash != manifest["output_requests_sha256"]:
        raise ValueError("request file hash differs")

    labels, request_ids = [], set()
    clusters = {key: set() for key in ("note", "admission", "subject")}
    with requests_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError("invalid JSON in request file") from None
            if not isinstance(row, dict) or not isinstance(row.get("expected"), dict):
                raise ValueError("request lacks expected labels")
            expected = row["expected"]
            if (set(expected) != {"detection", "encounter", "negation"}
                    or validate_ext_notes_labels(expected, allow_abstention=False)):
                raise ValueError("request contains invalid expected labels")
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id or request_id in request_ids:
                raise ValueError("request identifiers must be unique nonempty strings")
            request_ids.add(request_id)
            for name in clusters:
                value = row.get(f"{name}_cluster")
                if not isinstance(value, str) or not value:
                    raise ValueError("request has invalid cluster metadata")
                clusters[name].add(value)
            labels.append({"expected": dict(expected)})
    counts = {"requests": len(labels), **{f"{name}s": len(values) for name, values in clusters.items()}}
    if (not labels or any(counts[key] != value for key, value in dataset["expected_counts"].items())
            or any(manifest["counts"].get(key) != value for key, value in counts.items())
            or counts != config["expected_counts"]):
        raise ValueError("candidate or cluster counts differ from frozen declaration")
    observed_counts = {}
    for head, field in HEADS.items():
        eligible = labels if field == "detection" else [row for row in labels if row["expected"]["detection"] == "yes"]
        observed_counts[head] = dict(Counter(row["expected"][field] for row in eligible))
    if observed_counts != manifest["label_counts"]:
        raise ValueError("expected-label marginals differ from frozen manifest")
    for head, counts_for_head in observed_counts.items():
        largest = max(counts_for_head.values(), default=0)
        modes = [label for label, count in counts_for_head.items() if count == largest]
        if modes != [FIXED_PREDICTION[HEADS[head]]]:
            raise ValueError("fixed paper baseline is not the unique per-head majority")
    baseline = _baseline_metrics(labels)["majority_class"]
    parsed = {**FIXED_PREDICTION, "logical_valid": True, "abstained": False}
    correct = sum(_hierarchical_correct(row["expected"], parsed) for row in labels)
    return {
        "schema_version": "1.0",
        "analysis": "clinical_note_fixed_majority_label_baseline",
        "counts": counts,
        "predictions": dict(FIXED_PREDICTION),
        "hierarchical_joint": {"correct": correct, "n": len(labels), "exact_match": correct / len(labels)},
        "per_head": baseline,
        "label_counts": observed_counts,
        "definition": "Detection must match; encounter and negation must also match for reference detection-positive candidates. All candidates are retained. Downstream head metrics use reference detection-positive candidates and fixed class sets.",
        "interpretation": "A label-distribution reference, not a fitted model, held-out evaluation, clinical system, or model-selection candidate. No model responses are used.",
        "provenance": {
            "paired_config_sha256": config_sha256(config),
            "primary_e3_config_sha256": config_sha256(primary),
            "dataset_manifest_sha256": config["dataset_manifest_sha256"],
            "requests_sha256": requests_hash,
            "source_files": source_hashes,
        },
        "privacy": "Aggregate counts and metrics only; no clinical text, request identifiers, cluster identifiers, or model outputs.",
    }


def write_report(output: Path, report: dict, *, repository_root: Path = ROOT) -> None:
    output = output.resolve()
    restricted = (repository_root / "outputs/local_restricted").resolve()
    if restricted not in output.parents:
        raise ValueError("baseline output must remain under outputs/local_restricted")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise ValueError("existing baseline differs; choose a new output filename")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.config)
    write_report(args.output, report)
    joint = report["hierarchical_joint"]
    print(f"Verified {joint['n']} candidates; majority-label joint agreement {joint['exact_match']:.6f}. Aggregate saved locally.")


if __name__ == "__main__":
    main()
