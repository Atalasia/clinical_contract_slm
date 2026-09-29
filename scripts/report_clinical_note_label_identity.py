#!/usr/bin/env python3
"""Compare saved clinical-note label triples without inference or model loading.

Compare whole-output fence-normalized unconstrained responses with strictly
parsed constrained responses. Every frozen request remains in the denominator.
Only aggregate counts, rates, and provenance digests are written; clinical text,
raw responses, request identifiers, and cluster identifiers remain local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpg2.ext_notes_benchmark import validate_ext_notes_labels
from cpg2.mimic.csvio import config_sha256
from cpg2 import slm_e3_contract as paired


HEADS = ("detection", "encounter", "negation")
CLUSTERS = ("note_cluster", "admission_cluster", "subject_cluster")
SOURCE_NAMES = ("slm_e3_contract.py", "slm_e3.py", "ext_notes_benchmark.py",
                "slm_e0.py", "slm_benchmark.py", "mimic/csvio.py")
DEFAULT_CONFIG = ROOT / "configs/slm_benchmark/e3_contract_comparison.v1.local.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path, *, jsonl: bool = False):
    try:
        text = path.read_text(encoding="utf-8")
        value = [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid JSON in a local comparison artifact") from None
    require(isinstance(value, list if jsonl else dict), "unexpected comparison artifact type")
    return value


def resolve(base: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value.strip()), "artifact path must be a nonempty string")
    return (base / value).resolve()


def index_rows(rows: list[dict]) -> dict[str, dict]:
    require(isinstance(rows, list) and bool(rows), "response/request set must be nonempty")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "response/request rows must be objects")
        key = row.get("request_id")
        require(isinstance(key, str) and bool(key) and key not in result,
                "request identifiers must be unique nonempty strings")
        result[key] = row
    return result


def summarize_rows(unconstrained: list[dict], constrained: list[dict]) -> dict:
    """Reparse raw outputs, pair by ID, and never mistake two failures for labels."""
    left, right = index_rows(unconstrained), index_rows(constrained)
    require(set(left) == set(right), "paired response request identities differ")
    identical = identical_with_abstention = identical_logically_valid = 0
    both_parseable = both_schema_valid = both_valid = 0
    pair_status = Counter()
    head_counts = Counter({head: 0 for head in HEADS})
    diagnostics = {condition: {"failure_counts": Counter(), "valid_label_triples": 0,
                              "abstention_triples": 0, "fences_removed": 0}
                   for condition in paired.CONDITIONS}
    for key in left:
        a, b = left[key], right[key]
        require(all(a.get(field) == b.get(field) for field in paired.METADATA_FIELDS),
                "paired response metadata differ")
        predictions = []
        for condition, row in (("unconstrained", a), ("constrained", b)):
            require(isinstance(row.get("runtime_failed"), bool), "runtime flag must be Boolean")
            evaluation = paired.evaluate_output(row.get("raw_output"), row["expected"],
                                                runtime_failed=row["runtime_failed"])
            view = evaluation["fence_normalized"] if condition == "unconstrained" else evaluation
            parsed = view["parsed"]
            predictions.append(parsed)
            cell = diagnostics[condition]
            cell["valid_label_triples"] += parsed["logical_valid"]
            cell["abstention_triples"] += bool(parsed["logical_valid"] and parsed["abstained"])
            cell["fences_removed"] += bool(condition == "unconstrained" and evaluation["fence_removed"])
            if parsed["failure"]:
                cell["failure_counts"][parsed["failure"]] += 1
        p, q = predictions
        both_parseable += bool(p["parse_valid"] and q["parse_valid"])
        schema_valid = p["schema_valid"] and q["schema_valid"]
        both_schema_valid += schema_valid
        valid = p["logical_valid"] and q["logical_valid"]
        both_valid += valid
        pair_status[("both_valid" if valid else "unconstrained_invalid_only" if not p["logical_valid"] and q["logical_valid"]
                     else "constrained_invalid_only" if p["logical_valid"] else "both_invalid")] += 1
        # Label identity is not successful record validation: an inconsistent
        # hierarchy can still contain the same three schema-permitted labels.
        if schema_valid:
            equal = [p[head] == q[head] for head in HEADS]
            identical += all(equal)
            identical_logically_valid += bool(all(equal) and valid)
            identical_with_abstention += bool(all(equal) and p["abstained"])
            for head, same in zip(HEADS, equal, strict=True):
                head_counts[head] += same
    n = len(left)
    for cell in diagnostics.values():
        cell["failure_counts"] = {failure: cell["failure_counts"][failure] for failure in
                                  ("runtime_failure", "parse_failure", "schema_failure", "logical_failure")}
    return {
        "n_requests": n,
        "triple_identity": {"identical": identical, "n": n, "proportion": identical / n},
        "identical_triples_with_abstention": identical_with_abstention,
        "both_parseable": both_parseable,
        "both_schema_valid_label_triples": both_schema_valid,
        "both_valid_label_triples": both_valid,
        "identity_among_both_valid": {"identical": identical_logically_valid, "n": both_valid,
                                       "proportion": identical_logically_valid / both_valid if both_valid else None},
        "pair_status_counts": {name: pair_status[name] for name in
                               ("both_valid", "unconstrained_invalid_only", "constrained_invalid_only", "both_invalid")},
        "per_head_identity": {head: {"identical": head_counts[head], "n": n,
                                     "proportion": head_counts[head] / n} for head in HEADS},
        "conditions": diagnostics,
        "definition": "Exact equality of detection, encounter, and negation labels between conditions; not agreement with reference labels or clinical correctness.",
        "handling": {"unconstrained": "Existing whole-output Markdown-fence removal, then native parser.",
                     "constrained": "Native parser on the original raw response; no fence removal.",
                     "denominator": "All frozen requests, including invalid responses. Both triples must be schema-valid to count as identical, including for per-head counts. Logical consistency is not required for label identity; parse/schema failures cannot match.",
                     "logical_diagnostics": "both_valid_label_triples, identity_among_both_valid, pair_status_counts, and per-condition valid_label_triples/abstention_triples retain the stricter logical-validity requirement. The secondary identity numerator counts only identical, logically valid pairs.",
                     "native_labels": "Schema-permitted abstain and not_applicable values are compared literally. unsure remains a native negation label, not abstention. Matching abstentions count as identity, not accuracy.",
                     "normalization": "No extraction from prose, label repair, field deletion, generation, or retries."},
    }


def build_report(config_path: Path, *, manifest_id: str = "mg4_it_v1",
                 run_root: Path | None = None, repository_root: Path = ROOT) -> dict:
    """Validate an intact saved run; do not require its inference environment."""
    config_path, repository_root = config_path.resolve(), repository_root.resolve()
    config = read(config_path)
    require(config.get("schema_version") == "1.0" and config.get("freeze_status") == "FROZEN",
            "paired configuration must be frozen version 1.0")
    require(config.get("conditions") == list(paired.CONDITIONS), "unexpected paired conditions")
    require(type(config.get("checkpoint_batch_size")) is int and config["checkpoint_batch_size"] > 0,
            "invalid checkpoint batch size")
    primary_path = resolve(config_path.parent, config["primary_e3_config"])
    primary = read(primary_path)
    require(config_sha256(primary) == config["primary_e3_config_sha256"], "primary configuration hash differs")
    require(primary.get("freeze_status") == "FROZEN" and manifest_id in primary["panel"],
            "selected model is not in the frozen primary panel")
    manifest_path = resolve(config_path.parent, config["dataset_manifest"])
    require(digest(manifest_path) == config["dataset_manifest_sha256"], "dataset manifest hash differs")
    manifest = read(manifest_path)
    dataset = primary["dataset"]
    require(manifest.get("adapter_stage") == "slm_benchmark_e3_prepare"
            and manifest.get("experiment_id") == primary["experiment_id"]
            and manifest.get("e3_config_sha256") == config_sha256(primary)
            and manifest.get("dataset") == dataset["name"]
            and manifest.get("dataset_version") == dataset["version"], "dataset manifest identity differs")
    require(manifest.get("source_files") == {key: dataset[key] for key in
                                            ("notes_sha256", "labels_sha256", "source_audit_sha256")},
            "dataset source provenance differs")
    require(manifest.get("benchmark_config_sha256") == primary["benchmark_config_sha256"]
            and manifest.get("prompt_sha256") == primary["prompt_sha256"], "dataset prompt/benchmark provenance differs")
    requests_path = resolve(repository_root, manifest["output_requests"])
    requests_sha = digest(requests_path)
    require(requests_sha == manifest["output_requests_sha256"], "request file hash differs")
    requests = read(requests_path, jsonl=True)
    expected = index_rows(requests)
    notes, admissions = {}, {}
    for row in requests:
        require(isinstance(row.get("expected"), dict)
                and not validate_ext_notes_labels(row["expected"], allow_abstention=False), "invalid expected labels")
        require(all(isinstance(row.get(key), str) and row[key] for key in CLUSTERS), "invalid cluster metadata")
        note, admission, subject = (row[key] for key in CLUSTERS)
        require(notes.setdefault(note, (admission, subject)) == (admission, subject)
                and admissions.setdefault(admission, subject) == subject, "inconsistent cluster hierarchy")
    counts = {"requests": len(requests), **{name: len({row[key] for row in requests}) for name, key in
                                         zip(("notes", "admissions", "subjects"), CLUSTERS, strict=True)}}
    require(counts == config["expected_counts"] == dataset["expected_counts"]
            and all(manifest["counts"].get(key) == value for key, value in counts.items()), "frozen request/cluster counts differ")
    benchmark_path = resolve(primary_path.parent, primary["benchmark_config"])
    benchmark = read(benchmark_path)
    require(config_sha256(benchmark) == primary["benchmark_config_sha256"], "benchmark hash differs")
    selected = [(path, model) for value in benchmark["model_manifests"]
                for path in [resolve(benchmark_path.parent, value)] for model in [read(path)]
                if model.get("manifest_id") == manifest_id]
    require(len(selected) == 1, "model manifest must have one exact match")
    model_path, model = selected[0]
    prompt_path = resolve(primary_path.parent, primary["prompt"])
    prompt = read(prompt_path)
    require(config_sha256(prompt) == primary["prompt_sha256"], "prompt hash differs")
    override = config["wrapper_overrides"].get(manifest_id)
    wrapper_path = resolve(config_path.parent, override["wrapper"]) if override else resolve(model_path.parent, model["model"]["wrapper_spec"])
    wrapper_sha = digest(wrapper_path)
    require(wrapper_sha == (override["wrapper_sha256"] if override else model["verification"]["wrapper_spec_sha256"]), "wrapper hash differs")
    root = run_root.resolve() if run_root else resolve(config_path.parent, config["output_root"])
    preflight_path = root / "preflight.json"
    preflight = read(preflight_path)
    require(preflight.get("experiment_config_sha256") == config_sha256(config), "preflight configuration hash differs")
    require(preflight.get("all_models_fit_context") is True, "saved preflight did not pass context checks")
    observed = preflight.get("models", {}).get(manifest_id)
    require(isinstance(observed, dict), "preflight lacks the selected model")
    provenance = observed.get("provenance")
    require(isinstance(provenance, dict), "preflight lacks generation provenance")
    source_hashes = provenance.get("inference_source_sha256")
    require(isinstance(source_hashes, dict) and set(source_hashes) == set(SOURCE_NAMES)
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes.values()),
            "recorded generation-source hashes are incomplete or malformed")
    expected_provenance = {
        "experiment_id": config["experiment_id"], "experiment_config_sha256": config_sha256(config),
        "dataset_manifest_sha256": digest(manifest_path), "requests_sha256": requests_sha,
        "manifest_id": manifest_id, "manifest_sha256": config_sha256(model),
        "model_revision": model["model"]["revision"],
        "previously_verified_artifact_sha256": model["verification"]["artifact_sha256"],
        "wrapper_sha256": wrapper_sha, "prompt_sha256": config_sha256(prompt),
        "schema_sha256": config_sha256(prompt["output_schema"]), "runtime_versions": config["runtime_versions"],
        "inference_source_sha256": source_hashes, "structured_backend": config["structured_backend"],
        "generation": primary["generation"], "decoder_override": "constrained_decoding enabled only in constrained cell",
        "checkpoint_batch_size": config["checkpoint_batch_size"], "engine_overrides": config["engine_overrides"],
    }
    require(provenance == expected_provenance, "saved generation provenance differs from the supplied frozen configuration")
    require(observed.get("counts") == counts and observed.get("over_context") == 0
            and observed.get("complete_object_no_prefill") is True, "saved preflight counts/status differ")
    cells, artifact_hashes = {}, {}
    rendered = None
    for condition in paired.CONDITIONS:
        folder = root / "results" / manifest_id / condition
        audit_path, responses_path = folder / "audit.json", folder / "responses.jsonl"
        audit = read(audit_path)
        response_sha = digest(responses_path)
        require(audit.get("provenance") == provenance and audit.get("condition") == condition
                and audit.get("technical_completion") is True and audit.get("n") == len(requests), "cell audit provenance/completion differs")
        require(audit.get("responses_sha256") == response_sha, "response checksum differs")
        indexed = index_rows(read(responses_path, jsonl=True))
        require(set(indexed) == set(expected), "response coverage differs from frozen requests")
        rows = [indexed[key] for key in expected]
        if rendered is None:
            rendered = [{"request": request, **{key: row.get(key) for key in
                        ("prompt_sha256", "prompt_token_ids_sha256", "prompt_tokens_preflight")}}
                        for request, row in zip(requests, rows, strict=True)]
            for item in rendered:
                require(all(isinstance(item[key], str) and re.fullmatch(r"[0-9a-f]{64}", item[key]) for key in
                            ("prompt_sha256", "prompt_token_ids_sha256")), "invalid saved prompt fingerprint")
                require(type(item["prompt_tokens_preflight"]) is int and item["prompt_tokens_preflight"] > 0,
                        "invalid saved prompt token length")
            fingerprint = config_sha256([[item["request"]["request_id"], item["prompt_sha256"],
                                          item["prompt_token_ids_sha256"], item["prompt_tokens_preflight"]] for item in rendered])
            require(fingerprint == observed.get("rendered_prompt_fingerprint"), "saved preflight prompt fingerprint differs")
            lengths = [item["prompt_tokens_preflight"] for item in rendered]
            budget = primary["generation"]["max_output_tokens"]
            require(observed.get("output_token_budget") == budget
                    and observed.get("prompt_tokens_min") == min(lengths)
                    and observed.get("prompt_tokens_max") == max(lengths)
                    and observed.get("prompt_tokens_mean") == sum(lengths) / len(lengths)
                    and max(lengths) + budget <= primary["context_policy"]["max_context_tokens"],
                    "saved preflight lengths or context budget differ")
        paired._check_rows(rows, rendered, provenance, condition)
        specs = paired._batch_specs(rendered, config["checkpoint_batch_size"])
        require(audit.get("batches") == len(specs) and isinstance(audit.get("batch_sha256"), list)
                and len(audit["batch_sha256"]) == len(specs), "cell batch count differs")
        names = {f"batch_{index:05d}.json" for index, _ in specs}
        require({path.name for path in (folder / "batches").glob("batch_*.json")} == names, "missing or unexpected checkpoint batches")
        batch_hashes = []
        for index, items in specs:
            path = folder / "batches" / f"batch_{index:05d}.json"
            batch_sha = digest(path)
            require(batch_sha == audit["batch_sha256"][index], "checkpoint file checksum differs")
            batch = read(path)
            require(batch.get("provenance") == provenance and batch.get("condition") == condition
                    and batch.get("batch_index") == index and batch.get("technical_completion") is True,
                    "checkpoint provenance/completion differs")
            require(batch.get("rows_sha256") == config_sha256(batch.get("rows")), "checkpoint rows checksum differs")
            batch_rows = index_rows(batch["rows"])
            require(set(batch_rows) == {item["request"]["request_id"] for item in items}, "checkpoint request coverage differs")
            require(all(batch_rows[key] == indexed[key] for key in batch_rows), "checkpoint rows differ from saved responses")
            batch_hashes.append(batch_sha)
        cells[condition] = rows
        artifact_hashes[condition] = {"audit_sha256": digest(audit_path), "responses_sha256": response_sha,
                                      "batch_sha256": batch_hashes}
    result = summarize_rows(cells["unconstrained"], cells["constrained"])
    result.update(schema_version="1.0", analysis="clinical_note_cross_condition_label_identity",
                  manifest_id=manifest_id, counts=counts)
    result["provenance"] = {
        "paired_config_sha256": config_sha256(config), "primary_config_sha256": config_sha256(primary),
        "dataset_manifest_sha256": digest(manifest_path), "requests_sha256": requests_sha,
        "preflight_sha256": digest(preflight_path), "saved_generation_provenance": provenance,
        "artifacts": artifact_hashes,
        "current_analysis_source_sha256": {"scripts/" + Path(__file__).name: digest(Path(__file__)),
                                            **{"src/cpg2/" + name: digest(ROOT / "src/cpg2" / name) for name in
                                               ("slm_e3_contract.py", "slm_e3.py", "ext_notes_benchmark.py", "mimic/csvio.py")}},
        "verification_boundary": "Saved source/runtime/model hashes are recorded provenance, not a new audit of archived implementations, installed inference packages, or model weights. Request bytes, configuration bindings, prompt fingerprints, completed-cell/checkpoint checksums, metadata and raw-output evaluations are verified; prompts are not re-rendered. Original source CSV files are not required or read.",
    }
    result["privacy"] = "Aggregate counts, rates and digests only; no clinical text, raw outputs or request/cluster identifiers."
    return result


def write_report(output: Path, report: dict, *, repository_root: Path = ROOT) -> None:
    output = output.resolve()
    require((repository_root / "outputs/local_restricted").resolve() in output.parents,
            "comparison output must remain under outputs/local_restricted")
    paired._write_matching(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", type=Path, default=ROOT,
                        help="Workspace for repository-relative prepared-request paths; use the original workspace for archived runs")
    parser.add_argument("--run-root", type=Path, help="Completed run root; defaults to the supplied config's output_root")
    parser.add_argument("--manifest-id", default="mg4_it_v1")
    parser.add_argument("--output", type=Path, help="New restricted aggregate JSON; defaults to run-root/diagnostics/clinical_note_label_identity_<model>.json")
    args = parser.parse_args()
    report = build_report(args.config, manifest_id=args.manifest_id, run_root=args.run_root,
                          repository_root=args.repository_root)
    root = args.run_root.resolve() if args.run_root else resolve(args.config.resolve().parent, read(args.config)["output_root"])
    output = args.output or root / "diagnostics" / f"clinical_note_label_identity_{args.manifest_id}.json"
    write_report(output, report, repository_root=args.repository_root)
    value = report["triple_identity"]
    print(f"Verified {value['n']} paired requests; identical label triples {value['identical']}/{value['n']} ({value['proportion']:.6f}). Aggregate saved locally.")


if __name__ == "__main__":
    main()
