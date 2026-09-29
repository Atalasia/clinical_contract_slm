"""Post hoc diagnostics of retained contract-ablation outputs; no inference.

The strict endpoint and frozen validators are unchanged. The sole secondary
text transformation removes one exact enclosing Markdown fence from a complete
JSON object, then invokes the same full/compact evaluator.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .mimic.csvio import config_sha256, read_jsonl
from .slm_compact_contract import evaluate_contract_output
from .slm_contract_analysis import CONDITIONS, REFERENCE_STATES, _diagnostics, _validate_panel


MODELS = ("q15_pt_v1", "q15_it_v1", "q3_pt_v1", "q3_it_v1", "g4_it_v1",
          "mg4_it_v1", "mg15_it_v1", "q35_ref_v1")
CATEGORIES = ("runtime_failure", "parse_failure", "schema_failure", *(
    f"schema_valid__state_{correct}__abstain_{abstain}__logical_{logical}"
    for correct in ("correct", "wrong")
    for abstain in ("false", "true")
    for logical in ("valid", "invalid")
))
_FENCE = re.compile(r"\A```(?:json)?\r?\n(?P<body>.*?)\r?\n```\Z", re.DOTALL)


def normalize_enclosing_json_fence(raw: str | bytes | None) -> tuple[str | bytes | None, bool]:
    """Remove only a complete, lone, line-delimited ```json or ``` fence.

    Outer whitespace is allowed; tags are case-sensitive. The body must parse
    as one complete JSON object. Prose, nested/multiple fences, non-object JSON,
    unmatched fences, and truncated JSON remain byte-for-byte unchanged. No
    fields, JSON bytes inside the fence, or validator rules are repaired.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw, False
    elif isinstance(raw, str):
        text = raw
    else:
        return raw, False
    match = _FENCE.fullmatch(text.strip())
    if match is None or "```" in match["body"]:
        return raw, False
    body = match["body"]
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return raw, False
    return (body, True) if isinstance(payload, dict) else (raw, False)


def _score(row: dict[str, Any], raw: str | bytes | None) -> dict[str, Any]:
    evaluation = evaluate_contract_output(
        raw, contract=row["condition"].split("_", 1)[0],
        expected_criterion_id=row["criterion_id"], runtime_failed=row["runtime_failed"],
    )
    parsed = evaluation["parsed"]
    return {
        **evaluation,
        "state_correct": bool(parsed["logical_valid"] and not parsed["abstained"]
                              and parsed["state"] == row["expected_state"]),
        "state_only_correct": evaluation["state_only_state"] == row["expected_state"],
    }


def _category(row: dict[str, Any], evaluation: dict[str, Any]) -> str:
    parsed = evaluation["parsed"]
    if row["runtime_failed"] or parsed["failure"] == "runtime_failure":
        return "runtime_failure"
    if not parsed["parse_valid"]:
        return "parse_failure"
    if not parsed["schema_valid"]:
        return "schema_failure"
    # parsed.state is deliberately None for abstention. Use the separately
    # retained generated state, which survives abstention and schema rejection.
    assert evaluation["state_only_state"] is not None
    correct = "correct" if evaluation["state_only_correct"] else "wrong"
    abstain = str(parsed["abstained"]).lower()
    logical = "valid" if parsed["logical_valid"] else "invalid"
    return f"schema_valid__state_{correct}__abstain_{abstain}__logical_{logical}"


def _rates(counts: dict[str, int], denominators: dict[str, int]) -> dict[str, Any]:
    rates = {state: counts[state] / denominators[state] for state in REFERENCE_STATES}
    return {"count": sum(counts.values()), "by_reference_state": {
        state: {"count": counts[state], "denominator": denominators[state], "rate": rates[state]}
        for state in REFERENCE_STATES
    }, "micro_rate": sum(counts.values()) / sum(denominators.values()),
        "state_macro_rate": sum(rates.values()) / len(REFERENCE_STATES)}


def summarize_cell(rows: list[dict[str, Any]], *, normalize_fences: bool) -> dict[str, Any]:
    denominators = dict(Counter(row["expected_state"] for row in rows))
    if set(denominators) != set(REFERENCE_STATES):
        raise ValueError("each cell requires both reference-state strata")
    partitions = {name: Counter() for name in CATEGORIES}
    metrics = {name: Counter() for name in (
        "full_record_state_agreement", "state_only_agreement", "state_only_eligible",
        "state_only_correct_with_schema_failure", "state_only_wrong_with_schema_failure",
        "schema_failure_without_recognized_matching_id_state", "fence_removed",
        "parse_valid", "schema_valid", "logical_valid", "abstained", "valid_abstention",
    )}
    for row in rows:
        raw, removed = normalize_enclosing_json_fence(row["raw_output"]) if normalize_fences else (row["raw_output"], False)
        evaluated = _score(row, raw)
        state, parsed = row["expected_state"], evaluated["parsed"]
        category = _category(row, evaluated)
        partitions[category][state] += 1
        eligible = evaluated["state_only_state"] is not None
        observations = {
            "full_record_state_agreement": evaluated["state_correct"],
            "state_only_agreement": evaluated["state_only_correct"],
            "state_only_eligible": eligible,
            "state_only_correct_with_schema_failure": category == "schema_failure" and evaluated["state_only_correct"],
            "state_only_wrong_with_schema_failure": category == "schema_failure" and eligible and not evaluated["state_only_correct"],
            "schema_failure_without_recognized_matching_id_state": category == "schema_failure" and not eligible,
            "fence_removed": removed,
            "parse_valid": parsed["parse_valid"], "schema_valid": parsed["schema_valid"],
            "logical_valid": parsed["logical_valid"], "abstained": parsed["abstained"],
            "valid_abstention": parsed["abstained"] and parsed["logical_valid"],
        }
        for name, observed in observations.items():
            metrics[name][state] += bool(observed)
    for state in REFERENCE_STATES:
        assert sum(counts[state] for counts in partitions.values()) == denominators[state]
    result = {
        "n_requests": len(rows), "reference_state_counts": denominators,
        "partition": {key: _rates(value, denominators) for key, value in partitions.items()},
        "metrics": {key: _rates(value, denominators) for key, value in metrics.items()},
    }
    # This conditional diagnostic is explicitly distinct from all-request rates.
    result["state_only_conditional_accuracy"] = {
        state: {"correct": metrics["state_only_agreement"][state],
                "eligible": metrics["state_only_eligible"][state],
                "rate": metrics["state_only_agreement"][state] / metrics["state_only_eligible"][state]
                if metrics["state_only_eligible"][state] else None}
        for state in REFERENCE_STATES
    }
    return result


def summarize_sensitivity(panel: dict, saved_aggregate: dict) -> dict:
    """Check matched inputs and strict results before producing any diagnostics."""
    models, request_ids, aligned = _validate_panel(panel, [])
    if set(models) != set(saved_aggregate["models"]):
        raise ValueError("model panel differs from saved aggregate")
    reference_rows = aligned[models[0], CONDITIONS[0]]
    data = {"model_count": len(models), "conditions": list(CONDITIONS),
            "requests_per_cell": len(request_ids),
            "vignette_count": len({row["case_id"] for row in reference_rows}),
            "reference_state_counts": dict(Counter(row["expected_state"] for row in reference_rows))}
    if data != saved_aggregate["data"]:
        raise ValueError("panel denominators differ from saved aggregate")
    result = {
        "schema_version": "1.0", "analysis": "posthoc_contract_fence_and_state_decomposition",
        "data": data, "strict_endpoint_unchanged": True, "inference_performed": False,
        "normalization": "Remove one whole-response, line-delimited, lowercase json-tagged or unlabelled triple-backtick fence only when its body is a complete JSON object; allow outer whitespace. No prose extraction, missing-fence completion, multiple/nested blocks, field repair, or change to full/compact validation.",
        "interpretation": "Secondary post hoc sensitivity only; not a replacement for the frozen strict primary endpoint, not causal attribution, and not new model inference. Point estimates only; no new confidence intervals.",
        "decomposition": "Mutually exclusive runtime/parse/schema failure, followed by schema-valid generated-state correctness crossed with abstain flag and logical validity. Correctness uses retained state_only_state, including abstentions whose parsed.state is None. Every category has the original reference-stratum denominator; categories exhaust each stratum and their macro rates sum to one.",
        "state_only_definition": "Diagnostic only: complete JSON object, exact matching criterion ID and recognized generated state; ignores other schema fields, logical validity, and abstention. All-request rates assign zero to ineligible rows. Conditional eligible-object accuracy is separately labelled. Schema failures with recognized states are separately counted.",
        "models": {},
    }
    for model in models:
        result["models"][model] = {"conditions": {}}
        for condition in CONDITIONS:
            rows = aligned[model, condition]
            for row in rows:
                evaluated = _score(row, row["raw_output"])
                if any(row.get(key) != value for key, value in evaluated.items()):
                    raise ValueError(f"{model}/{condition}: stored strict evaluation or scoring differs")
            old = saved_aggregate["models"][model]["conditions"][condition]
            if any(old.get(key) != value for key, value in _diagnostics(rows).items()):
                raise ValueError(f"{model}/{condition}: strict diagnostics differ from saved aggregate")
            strict = summarize_cell(rows, normalize_fences=False)
            for endpoint, metric, rate in (
                ("state_macro_agreement", "full_record_state_agreement", "state_macro_rate"),
                ("strict_micro_agreement", "full_record_state_agreement", "micro_rate"),
                ("state_only_macro_agreement", "state_only_agreement", "state_macro_rate"),
                ("state_only_micro_agreement", "state_only_agreement", "micro_rate"),
            ):
                if not math.isclose(strict["metrics"][metric][rate], old[endpoint]["estimate"], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{model}/{condition}: strict endpoint differs from saved aggregate")
            normalized = summarize_cell(rows, normalize_fences=True)
            result["models"][model]["conditions"][condition] = {"strict": strict, "fence_normalized": normalized}
    result["verification"] = {"stored_row_evaluations_match": True, "saved_aggregate_diagnostics_match": True,
                              "saved_aggregate_point_estimates_match": True,
                              "complete_matched_panel": True}
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def analyze_existing_outputs(
    input_root: Path, *, repository_root: Path, config_path: Path | None = None,
) -> dict:
    """Read a complete frozen panel and verify its exact local configuration.

    ``config_path`` can select a separately bound replication configuration;
    this does not relax any saved-output hash or source-provenance checks.
    """
    input_root, repository_root = input_root.resolve(), repository_root.resolve()
    aggregate_path = input_root / "aggregate/results.json"
    aggregate = _read(aggregate_path)
    config_path = (
        Path(config_path).resolve() if config_path is not None
        else repository_root / "configs/slm_benchmark/e1_contract_ablation.v1.json"
    )
    config = _read(config_path)
    config_hash = config_sha256(config)
    if aggregate.get("experiment_config_sha256") != config_hash:
        raise ValueError("saved aggregate configuration hash differs")
    analysis_path = repository_root / "src/cpg2/slm_contract_analysis.py"
    if aggregate.get("analysis_source_sha256") != _sha256(analysis_path):
        raise ValueError("frozen aggregate analysis source changed")
    if set(aggregate["models"]) != set(MODELS):
        raise ValueError("expected the complete frozen eight-model panel")
    dataset_path = (config_path.parent / config["dataset_manifest"]).resolve()
    if _sha256(dataset_path) != config["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest changed")
    dataset = _read(dataset_path)
    blueprint_path = repository_root / dataset["output_blueprint"]
    requests_path = repository_root / dataset["output_requests"]
    if (_sha256(blueprint_path) != dataset["output_blueprint_sha256"]
            or _sha256(requests_path) != dataset["output_requests_sha256"]):
        raise ValueError("frozen blueprint or requests changed")
    expected = {request_id: {"case_id": case["case_id"], "domain": case["domain"],
                             "criterion_id": criterion, "expected_state": case["expected_states"][criterion]}
                for case in _read(blueprint_path)["cases"] for criterion, request_id in case["request_ids"].items()}
    if (len(expected) != config["expected_requests_per_cell"]
            or len({row["case_id"] for row in expected.values()}) != config["expected_cases"]):
        raise ValueError("frozen blueprint denominators differ")
    preflight_path = input_root / "preflight.json"
    preflight = _read(preflight_path)
    if preflight.get("experiment_config_sha256") != config_hash or preflight.get("all_models_fit_context") is not True:
        raise ValueError("preflight configuration/completion differs")
    if set(preflight["models"]) != set(MODELS):
        raise ValueError("preflight panel differs")
    panel, cells, model_provenance = {}, {}, {}
    for model in MODELS:
        panel[model], cells[model] = {}, {}
        provenance = preflight["models"][model]["provenance"]
        for key, expected_hash in (("experiment_config_sha256", config_hash),
                                   ("dataset_manifest_sha256", _sha256(dataset_path)),
                                   ("blueprint_sha256", dataset["output_blueprint_sha256"]),
                                   ("requests_sha256", dataset["output_requests_sha256"])):
            if provenance.get(key) != expected_hash:
                raise ValueError(f"{model}: provenance {key} differs")
        for name, digest in provenance["inference_source_sha256"].items():
            if Path(name).name != name or _sha256(repository_root / "src/cpg2" / name) != digest:
                raise ValueError(f"{model}: frozen inference source changed")
        model_provenance[model] = {key: provenance[key] for key in (
            "manifest_sha256", "model_id", "model_revision", "previously_verified_artifact_sha256",
            "wrapper_sha256", "prompt_sha256", "inference_source_sha256", "runtime_versions")}
        for condition in CONDITIONS:
            folder = input_root / "results" / model / condition
            responses_path, audit_path = folder / "responses.jsonl", folder / "audit.json"
            if not responses_path.is_file() or not audit_path.is_file():
                raise ValueError(f"missing cell artifacts: {model}/{condition}")
            audit = _read(audit_path)
            if (audit.get("provenance") != provenance or audit.get("condition") != condition
                    or audit.get("technical_completion") is not True
                    or audit.get("responses_sha256") != _sha256(responses_path)):
                raise ValueError(f"cell audit/provenance/checksum differs: {model}/{condition}")
            rows = list(read_jsonl(responses_path))
            ids = [row["request_id"] for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(expected):
                raise ValueError(f"duplicate/missing/unexpected responses: {model}/{condition}")
            if audit.get("n") != len(rows):
                raise ValueError(f"audit denominator differs: {model}/{condition}")
            for row in rows:
                if (row.get("manifest_id") != model or row.get("condition") != condition
                        or any(row.get(key) != value for key, value in expected[row["request_id"]].items())):
                    raise ValueError(f"model/condition/blueprint mismatch: {model}/{condition}")
            panel[model][condition] = rows
            cells[model][condition] = {"responses_sha256": _sha256(responses_path), "audit_sha256": _sha256(audit_path)}
    result = summarize_sensitivity(panel, aggregate)
    result["provenance"] = {
        "input_root": str(input_root), "aggregate_sha256": _sha256(aggregate_path),
        "preflight_sha256": _sha256(preflight_path), "experiment_config_sha256": config_hash,
        "dataset_manifest_sha256": _sha256(dataset_path),
        "blueprint_sha256": dataset["output_blueprint_sha256"],
        "requests_sha256": dataset["output_requests_sha256"], "cells": cells,
        "models": model_provenance,
        "diagnostic_source_sha256": {"slm_contract_sensitivity.py": _sha256(Path(__file__)),
                                     "analyze_contract_sensitivity.py": _sha256(repository_root / "scripts/analyze_contract_sensitivity.py")},
    }
    return result


def render_report(result: dict) -> str:
    lines = ["# Post hoc contract sensitivity diagnostics", "", result["interpretation"], "",
             result["normalization"], "", result["decomposition"], "", result["state_only_definition"], "",
             f"{result['data']['model_count']} models × four conditions × {result['data']['requests_per_cell']} requests; "
             f"{result['data']['vignette_count']} vignettes. Reference strata: "
             + ", ".join(f"{key}={value}" for key, value in result["data"]["reference_state_counts"].items()) + ".",
             "Strict row evaluations, diagnostics and endpoint point estimates match the stored results. Original outputs remain unchanged.",
             "", "## Strict → fence-normalized metrics", "",
             "All agreements are proportions over original reference-state denominators. Full/U and Full/C are full contract unconstrained/constrained; Compact/U and Compact/C are compact contract unconstrained/constrained.", "",
             "| Model | Condition | Fences removed | Parse-valid | Schema-valid | Full-record macro | Full-record correct count | State-only macro |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model, model_result in result["models"].items():
        for condition, cell in model_result["conditions"].items():
            strict, norm = [cell[mode]["metrics"] for mode in ("strict", "fence_normalized")]
            pairs = []
            for metric, key in (("parse_valid", "count"), ("schema_valid", "count"),
                                ("full_record_state_agreement", "state_macro_rate"),
                                ("full_record_state_agreement", "count"),
                                ("state_only_agreement", "state_macro_rate")):
                values = [str(side[metric][key]) if key == "count" else f"{side[metric][key]:.6f}" for side in (strict, norm)]
                pairs.append(" → ".join(values))
            lines.append(f"| {model} | {condition} | {norm['fence_removed']['count']} | " + " | ".join(pairs) + " |")
    lines += ["", "## Exhaustive decomposition by reference state", "",
              "Each cell below gives count / stratum denominator. Correct/wrong refers to the generated state even when abstain=true; logical validity is a separate axis. Zero-count categories are retained in JSON and omitted here only when zero in both modes and both strata."]
    for model, model_result in result["models"].items():
        for condition, cell in model_result["conditions"].items():
            lines += ["", f"### {model} / {condition}", "",
                      "| Category | Strict present | Strict absent | Normalized present | Normalized absent |",
                      "| --- | --- | --- | --- | --- |"]
            for category in CATEGORIES:
                strata = [cell[mode]["partition"][category]["by_reference_state"][state]
                          for mode in ("strict", "fence_normalized") for state in REFERENCE_STATES]
                if any(value["count"] for value in strata):
                    lines.append("| " + category + " | " + " | ".join(f"{value['count']} / {value['denominator']}" for value in strata) + " |")
    lines += ["", "## Input verification", "", "All 32 source response files are hashed, checked against their cell audits, and aligned with the frozen blueprint; preflight provenance and frozen inference source hashes are verified. The JSON report retains input and analysis-code SHA-256 hashes. Neither report includes response text or request identifiers."]
    return "\n".join(lines) + "\n"
