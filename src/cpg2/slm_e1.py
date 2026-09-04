from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .executor import execute_partial
from .io import write_json
from .mimic.csvio import config_sha256, read_jsonl, restricted_jsonl_writer, write_jsonl_row
from .registry import CriterionRegistry
from .schemas import CriterionAssessment
from .slm_benchmark import validate_benchmark_config
from .slm_e0 import (
    _NvmlSampler,
    _format_messages,
    _load_renderer,
    _load_vllm,
    _multimodal_limits,
    _sampling_params,
    parse_typed_e0_output,
)
from .states import CriterionState
from .target_spec import legacy_feature_name, load_targets
from .tree import load_tree


SCHEMA_VERSION = "1.0"
TASK = "typed_state"
RECOVERABLE_STATES = frozenset(
    {
        CriterionState.PRESENT_CURRENT.value,
        CriterionState.ABSENT_EXPLICIT.value,
        CriterionState.HISTORICAL.value,
        CriterionState.FAMILY_HISTORY_OR_OTHER_EXPERIENCER.value,
        CriterionState.UNCERTAIN.value,
        CriterionState.NOT_DOCUMENTED.value,
    }
)
STRICT_OUTPUT_KEYS = frozenset(
    {
        "criterion_id",
        "state",
        "value",
        "unit",
        "evidence",
        "reason_code",
        "confidence",
        "abstain",
    }
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, *values: str) -> str:
    joined = "\x1f".join(values)
    return prefix + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def _write_restricted_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    write_json(destination, payload)
    destination.chmod(0o600)


def _parse_feature_value(value: str) -> tuple[tuple[str, ...], int, bool]:
    text = value.strip()
    if not text:
        return (), 0, False
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text
    nested = False
    leaves: list[str] = []

    def visit(item: Any, depth: int) -> None:
        nonlocal nested
        if isinstance(item, Mapping):
            raise ValueError("feature mappings are ambiguous and unsupported")
        if isinstance(item, (list, tuple, set)):
            if depth > 0:
                nested = True
            values = sorted(item, key=repr) if isinstance(item, set) else item
            for child in values:
                visit(child, depth + 1)
            return
        normalized = str(item).strip()
        if normalized:
            leaves.append(normalized)

    visit(parsed, 0)
    return tuple(dict.fromkeys(leaves)), len(leaves), nested


def parse_feature_cell_recursive(value: str) -> tuple[str, ...]:
    """Flatten nested legacy feature containers without changing the v1 parser."""

    values, _leaf_count, _nested = _parse_feature_value(value)
    return values


def _load_e1_config(path: str | Path, *, require_authorized: bool = True) -> tuple[Path, dict[str, Any]]:
    source = Path(path)
    config = _read_object(source, "E1 config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E1 config schema")
    if config.get("freeze_status") != "FROZEN":
        raise ValueError("E1 config is not frozen")
    authorization = config.get("authorization", {})
    if require_authorized and (
        authorization.get("inference") is not True
        or "E1" not in authorization.get("authorized_experiments", [])
    ):
        raise ValueError("E1 inference is not authorized")
    base = source.parent
    benchmark_path = _resolve(base, config.get("benchmark_config"), "benchmark_config")
    benchmark = _read_object(benchmark_path, "benchmark config")
    if config_sha256(benchmark) != config.get("benchmark_config_sha256"):
        raise ValueError("E1 benchmark config hash mismatch")
    prompt_path = _resolve(base, config.get("prompt"), "prompt")
    prompt = _read_object(prompt_path, "typed-state prompt")
    if config_sha256(prompt) != config.get("prompt_sha256"):
        raise ValueError("E1 prompt hash mismatch")
    if prompt.get("freeze_status") != "FROZEN" or prompt.get("task") != TASK:
        raise ValueError("E1 prompt is not the frozen typed-state contract")
    return source, config


def _load_s2_config(
    path: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path]:
    source = Path(path)
    config = _read_object(source, "E1-S2 config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E1-S2 config schema")
    if config.get("freeze_status") != "FROZEN":
        raise ValueError("E1-S2 config is not frozen")
    authorization = config.get("authorization", {})
    if (
        authorization.get("inference") is not True
        or "E1-S2" not in authorization.get("authorized_experiments", [])
    ):
        raise ValueError("E1-S2 inference is not authorized")
    base = source.parent
    primary_path = _resolve(base, config.get("primary_e1_config"), "primary_e1_config")
    _loaded_path, primary = _load_e1_config(primary_path)
    if config_sha256(primary) != config.get("primary_e1_config_sha256"):
        raise ValueError("E1-S2 primary E1 config hash mismatch")
    dataset_path = _resolve(base, config.get("dataset_manifest"), "dataset_manifest")
    if _sha256_file(dataset_path) != config.get("dataset_manifest_sha256"):
        raise ValueError("E1-S2 dataset manifest hash mismatch")
    primary_results = _resolve(base, config.get("primary_results"), "primary_results")
    if _sha256_file(primary_results) != config.get("primary_results_sha256"):
        raise ValueError("E1-S2 primary results hash mismatch")
    intervention = config.get("intervention", {})
    if (
        intervention.get("semantic_prompt_changed") is not False
        or intervention.get("model_revisions_changed") is not False
        or intervention.get("dtype_or_sampling_changed") is not False
        or intervention.get("output_token_budget_changed") is not False
    ):
        raise ValueError("E1-S2 must change only the declared format layer")
    return source, config, primary_path, primary, dataset_path


def prepare_e1_dataset(
    config_path: str | Path,
    *,
    output_requests: str | Path,
    output_blueprint: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    source, config = _load_e1_config(config_path, require_authorized=False)
    base = source.parent
    construction = config["construction"]["version"]
    prompt_hash = config["prompt_sha256"]
    requests: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    source_rows = 0
    state_counts: Counter[str] = Counter()
    domain_request_counts: Counter[str] = Counter()
    domain_case_counts: Counter[str] = Counter()
    criterion_counts: Counter[str] = Counter()
    rows_without_target: Counter[str] = Counter()
    nested_cells = 0
    duplicate_leaves = 0
    conflicts = 0
    unmatched_feature_occurrences: Counter[str] = Counter()
    artifact_records: list[dict[str, Any]] = []

    for domain_spec in config["domains"]:
        domain = domain_spec["domain"]
        paths: dict[str, Path] = {}
        for field in ("vignettes", "tree", "registry", "target_spec"):
            path = _resolve(base, domain_spec[field], f"{domain}.{field}")
            observed_hash = _sha256_file(path)
            if observed_hash != domain_spec[f"{field}_sha256"]:
                raise ValueError(f"{domain}: {field} hash mismatch")
            paths[field] = path
            artifact_records.append(
                {"domain": domain, "role": field, "path": str(path), "sha256": observed_hash}
            )

        registry = CriterionRegistry.load(paths["registry"])
        _target_config, targets = load_targets(paths["target_spec"], registry=registry)
        targets_by_id = {target.criterion_id: target for target in targets}
        raw_tree = _read_object(paths["tree"], f"{domain} tree")
        feature_by_criterion = {
            node["criterion_id"]: legacy_feature_name(node)
            for node in raw_tree["nodes"]
            if node.get("node_type") == "criterion"
            and node.get("criterion_id") in targets_by_id
        }
        if set(feature_by_criterion) != set(targets_by_id):
            missing = sorted(set(targets_by_id) - set(feature_by_criterion))
            raise ValueError(f"{domain}: target criteria absent from tree: {missing}")
        selected_features = set(feature_by_criterion.values())

        with paths["vignettes"].open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                domain_spec["positive_features_column"],
                domain_spec["negative_features_column"],
                domain_spec["text_column"],
            }
            missing_columns = sorted(required - set(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(f"{paths['vignettes']}: missing columns {missing_columns}")
            for row_index, row in enumerate(reader):
                source_rows += 1
                positive, positive_leaves, positive_nested = _parse_feature_value(
                    row[domain_spec["positive_features_column"]]
                )
                negative, negative_leaves, negative_nested = _parse_feature_value(
                    row[domain_spec["negative_features_column"]]
                )
                nested_cells += int(positive_nested) + int(negative_nested)
                duplicate_leaves += positive_leaves - len(positive)
                duplicate_leaves += negative_leaves - len(negative)
                positive_set, negative_set = set(positive), set(negative)
                overlap = positive_set & negative_set
                if overlap:
                    conflicts += len(overlap)
                    raise ValueError(f"{domain} row {row_index}: positive/negative conflict")
                for feature in (positive_set | negative_set) - selected_features:
                    unmatched_feature_occurrences[domain] += 1
                text = row[domain_spec["text_column"]]
                if not text:
                    raise ValueError(f"{domain} row {row_index}: empty vignette text")
                case_id = _stable_id(
                    "e1_case_",
                    domain,
                    domain_spec["vignettes_sha256"],
                    str(row_index),
                    construction,
                )
                expected_states: dict[str, str] = {}
                request_ids: dict[str, str] = {}
                for criterion_id, feature_name in sorted(feature_by_criterion.items()):
                    if feature_name in positive_set:
                        expected = CriterionState.PRESENT_CURRENT.value
                    elif feature_name in negative_set:
                        expected = CriterionState.ABSENT_EXPLICIT.value
                    else:
                        continue
                    target = targets_by_id[criterion_id]
                    request_id = _stable_id(
                        "e1_request_", case_id, criterion_id, prompt_hash, construction
                    )
                    requests.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "request_id": request_id,
                            "case_id": case_id,
                            "domain": domain,
                            "criterion_id": criterion_id,
                            "criterion_text": target.criterion_text,
                            "evidence_text": text,
                            "construction_version": construction,
                        }
                    )
                    expected_states[criterion_id] = expected
                    request_ids[criterion_id] = request_id
                    state_counts[expected] += 1
                    domain_request_counts[domain] += 1
                    criterion_counts[f"{domain}:{criterion_id}"] += 1
                if expected_states:
                    cases.append(
                        {
                            "case_id": case_id,
                            "domain": domain,
                            "tree": str(paths["tree"]),
                            "expected_states": expected_states,
                            "request_ids": request_ids,
                        }
                    )
                    domain_case_counts[domain] += 1
                else:
                    rows_without_target[domain] += 1

    request_ids = [row["request_id"] for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("E1 request IDs are not unique")
    case_ids = [row["case_id"] for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("E1 case IDs are not unique")

    observed = {
        "source_rows": source_rows,
        "cases": len(cases),
        "requests": len(requests),
        "state_counts": dict(sorted(state_counts.items())),
        "domain_request_counts": dict(sorted(domain_request_counts.items())),
        "nested_feature_cells": nested_cells,
        "positive_negative_conflicts": conflicts,
    }
    if observed != config["expected_dataset"]:
        raise ValueError(f"E1 dataset counts differ from frozen expectation: {observed}")

    with restricted_jsonl_writer(output_requests) as output:
        for request in requests:
            write_jsonl_row(output, request)
    blueprint = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "construction_version": construction,
        "claim_boundary": config["claim_boundary"],
        "cases": cases,
    }
    _write_restricted_json(output_blueprint, blueprint)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e1_dataset",
        "experiment_id": config["experiment_id"],
        "e1_config": str(source),
        "e1_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config["benchmark_config_sha256"],
        "prompt_sha256": prompt_hash,
        "construction_version": construction,
        **observed,
        "domain_case_counts": dict(sorted(domain_case_counts.items())),
        "criterion_request_counts": dict(sorted(criterion_counts.items())),
        "rows_without_labeled_target": dict(sorted(rows_without_target.items())),
        "nested_duplicate_leaves_removed": duplicate_leaves,
        "unmatched_feature_occurrences_outside_selected_targets": dict(
            sorted(unmatched_feature_occurrences.items())
        ),
        "source_artifacts": artifact_records,
        "output_requests": str(output_requests),
        "output_requests_sha256": _sha256_file(output_requests),
        "output_blueprint": str(output_blueprint),
        "output_blueprint_sha256": _sha256_file(output_blueprint),
        "historical_v1_request_count": 317,
        "requests_added_by_recursive_fix": len(requests) - 317,
    }
    write_json(output_manifest, manifest)
    return manifest


@dataclass(frozen=True)
class _E1ModelContext:
    e1_config_path: Path
    e1_config: dict[str, Any]
    benchmark_path: Path
    benchmark: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    wrapper_path: Path
    wrapper: dict[str, Any]
    prompt_path: Path
    prompt: dict[str, Any]
    dataset_manifest_path: Path
    dataset_manifest: dict[str, Any]
    requests_path: Path
    blueprint_path: Path


def _load_model_context(
    config_path: str | Path,
    *,
    manifest_id: str,
    dataset_manifest_path: str | Path,
) -> _E1ModelContext:
    source, config = _load_e1_config(config_path)
    base = source.parent
    benchmark_path = _resolve(base, config["benchmark_config"], "benchmark_config")
    preflight = validate_benchmark_config(benchmark_path)
    if not preflight["inference_ready"]:
        raise ValueError("base benchmark is not inference-ready")
    benchmark = _read_object(benchmark_path, "benchmark config")
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    if manifest_id not in declared:
        raise ValueError(f"manifest is not in the declared E1 panel: {manifest_id}")
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    for value in benchmark["model_manifests"]:
        candidate = _resolve(benchmark_path.parent, value, "model manifest")
        loaded = _read_object(candidate, "model manifest")
        if loaded.get("manifest_id") == manifest_id:
            manifest_path, manifest = candidate, loaded
            break
    if manifest_path is None or manifest is None:
        raise ValueError(f"unknown manifest ID: {manifest_id}")
    wrapper_path = _resolve(manifest_path.parent, manifest["model"]["wrapper_spec"], "wrapper")
    if _sha256_file(wrapper_path) != manifest["verification"]["wrapper_spec_sha256"]:
        raise ValueError("frozen wrapper hash mismatch")
    wrapper = _read_object(wrapper_path, "wrapper")
    prompt_path = _resolve(base, config["prompt"], "prompt")
    prompt = _read_object(prompt_path, "prompt")
    dataset_path = Path(dataset_manifest_path)
    dataset = _read_object(dataset_path, "E1 dataset manifest")
    if dataset.get("e1_config_sha256") != config_sha256(config):
        raise ValueError("E1 dataset was not built from the frozen E1 config")
    if dataset.get("prompt_sha256") != config["prompt_sha256"]:
        raise ValueError("E1 dataset prompt hash mismatch")
    requests_path = Path(dataset["output_requests"])
    blueprint_path = Path(dataset["output_blueprint"])
    if _sha256_file(requests_path) != dataset["output_requests_sha256"]:
        raise ValueError("E1 requests hash mismatch")
    if _sha256_file(blueprint_path) != dataset["output_blueprint_sha256"]:
        raise ValueError("E1 blueprint hash mismatch")
    return _E1ModelContext(
        e1_config_path=source,
        e1_config=config,
        benchmark_path=benchmark_path,
        benchmark=benchmark,
        manifest_path=manifest_path,
        manifest=manifest,
        wrapper_path=wrapper_path,
        wrapper=wrapper,
        prompt_path=prompt_path,
        prompt=prompt,
        dataset_manifest_path=dataset_path,
        dataset_manifest=dataset,
        requests_path=requests_path,
        blueprint_path=blueprint_path,
    )


def _apply_s2_wrapper(
    context: _E1ModelContext,
    *,
    sensitivity_path: Path,
    sensitivity: Mapping[str, Any],
) -> _E1ModelContext:
    override = sensitivity.get("wrapper_overrides", {}).get(
        context.manifest["manifest_id"]
    )
    if override is None:
        return context
    wrapper_path = _resolve(
        sensitivity_path.parent,
        override.get("wrapper"),
        f"wrapper_overrides.{context.manifest['manifest_id']}.wrapper",
    )
    if _sha256_file(wrapper_path) != override.get("wrapper_sha256"):
        raise ValueError(f"E1-S2 wrapper hash mismatch: {wrapper_path}")
    wrapper = _read_object(wrapper_path, "E1-S2 wrapper")
    if (
        wrapper.get("freeze_status") != "FROZEN"
        or wrapper.get("sensitivity_only") is not True
        or wrapper.get("semantic_prompt_changed") is not False
        or wrapper.get("assistant_prefill")
    ):
        raise ValueError("invalid E1-S2 wrapper override")
    return replace(
        context,
        wrapper_path=wrapper_path,
        wrapper=wrapper,
    )


def _render_e1_requests(context: _E1ModelContext) -> tuple[list[dict[str, Any]], Any]:
    renderer, tokenizer = _load_renderer(context)
    schema_json = json.dumps(
        context.prompt["output_schema"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    mode = context.wrapper["mode"]
    runtime = context.manifest["runtime"]
    rendered_requests: list[dict[str, Any]] = []
    for request in read_jsonl(context.requests_path):
        user = context.prompt["user_template"].format(
            criterion_id=request["criterion_id"],
            criterion_text=request["criterion_text"],
            text=request["evidence_text"],
            output_schema_json=schema_json,
        )
        if mode == "plain_completion":
            rendered = context.wrapper["template"].format(
                system=context.prompt["system"], user=user
            )
        else:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": context.wrapper["add_generation_prompt"],
            }
            if context.wrapper.get("continue_final_message") is True:
                kwargs["continue_final_message"] = True
            if runtime["thinking_mode"] == "disabled":
                kwargs["enable_thinking"] = False
            rendered = renderer.apply_chat_template(
                _format_messages(context.wrapper, context.prompt["system"], user), **kwargs
            )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        output_budget = runtime["max_output_tokens"][TASK]
        rendered_requests.append(
            {
                "request": request,
                "rendered_prompt": rendered,
                "prompt_tokens_preflight": len(token_ids),
                "max_output_tokens": output_budget,
                "over_context": len(token_ids) + output_budget > runtime["max_context_tokens"],
                "response_prefix": context.wrapper.get("assistant_prefill", ""),
            }
        )
    return rendered_requests, tokenizer


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def audit_e1_tokens(
    config_path: str | Path,
    *,
    dataset_manifest_path: str | Path,
    output: str | Path,
    sensitivity_config_path: str | Path | None = None,
) -> dict[str, Any]:
    source, config = _load_e1_config(config_path)
    sensitivity_path: Path | None = None
    sensitivity: dict[str, Any] | None = None
    if sensitivity_config_path is not None:
        (
            sensitivity_path,
            sensitivity,
            frozen_primary_path,
            _frozen_primary,
            frozen_dataset_path,
        ) = _load_s2_config(sensitivity_config_path)
        if frozen_primary_path.resolve() != source.resolve():
            raise ValueError("E1-S2 token audit primary config path mismatch")
        if frozen_dataset_path.resolve() != Path(dataset_manifest_path).resolve():
            raise ValueError("E1-S2 token audit dataset path mismatch")
    benchmark_path = _resolve(source.parent, config["benchmark_config"], "benchmark")
    benchmark = _read_object(benchmark_path, "benchmark")
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    rows: list[dict[str, Any]] = []
    for manifest_id in declared:
        context = _load_model_context(
            config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path
        )
        if sensitivity_path is not None and sensitivity is not None:
            context = _apply_s2_wrapper(
                context,
                sensitivity_path=sensitivity_path,
                sensitivity=sensitivity,
            )
        rendered, _tokenizer = _render_e1_requests(context)
        lengths = [row["prompt_tokens_preflight"] for row in rendered]
        rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": context.manifest["model"]["model_id"],
                "wrapper_sha256": _sha256_file(context.wrapper_path),
                "requests": len(rendered),
                "minimum_prompt_tokens": min(lengths),
                "mean_prompt_tokens": fmean(lengths),
                "p95_prompt_tokens": _percentile(lengths, 0.95),
                "maximum_prompt_tokens": max(lengths),
                "max_output_tokens": context.manifest["runtime"]["max_output_tokens"][TASK],
                "max_context_tokens": context.manifest["runtime"]["max_context_tokens"],
                "over_context_requests": sum(row["over_context"] for row in rendered),
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": (
            "slm_benchmark_e1_s2_token_audit"
            if sensitivity is not None
            else "slm_benchmark_e1_token_audit"
        ),
        "experiment_id": (
            sensitivity["experiment_id"] if sensitivity is not None else config["experiment_id"]
        ),
        "e1_config_sha256": config_sha256(config),
        "sensitivity_config_sha256": (
            config_sha256(sensitivity) if sensitivity is not None else None
        ),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "all_models_fit_context": all(row["over_context_requests"] == 0 for row in rows),
        "models": rows,
    }
    write_json(output, result)
    return result


def _observed_label(parsed: Mapping[str, Any], runtime_failed: bool) -> str:
    if runtime_failed:
        return "INFERENCE_FAILURE"
    if parsed.get("abstained"):
        return "MODEL_ABSTENTION"
    if not parsed.get("parse_valid"):
        return "PARSE_FAILURE"
    if not parsed.get("schema_valid"):
        return "SCHEMA_FAILURE"
    if not parsed.get("logical_valid"):
        return "LOGICAL_FAILURE"
    return str(parsed.get("state"))


def _state_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    references = sorted({str(row["expected_state"]) for row in rows})
    per_state: dict[str, dict[str, Any]] = {}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        confusion[str(row["expected_state"])][str(row["observed_label"])] += 1
    for state in references:
        selected = [row for row in rows if row["expected_state"] == state]
        correct = sum(bool(row["state_correct"]) for row in selected)
        per_state[state] = {
            "n": len(selected),
            "correct": correct,
            "agreement": correct / len(selected) if selected else None,
        }
    agreements = [value["agreement"] for value in per_state.values() if value["agreement"] is not None]
    valid_predictions = [
        row for row in rows if row["parsed"].get("logical_valid") and not row["parsed"].get("abstained")
    ]
    correct = sum(bool(row["state_correct"]) for row in rows)
    return {
        "n": count,
        "runtime_completed": sum(not row["runtime_failed"] for row in rows),
        "parse_valid": sum(bool(row["parsed"].get("parse_valid")) for row in rows),
        "schema_valid": sum(bool(row["parsed"].get("schema_valid")) for row in rows),
        "logical_valid": sum(bool(row["parsed"].get("logical_valid")) for row in rows),
        "abstained": sum(bool(row["parsed"].get("abstained")) for row in rows),
        "state_correct_unconditional": correct,
        "state_agreement_unconditional": correct / count if count else None,
        "state_agreement_conditional_on_valid_prediction": (
            sum(bool(row["state_correct"]) for row in valid_predictions) / len(valid_predictions)
            if valid_predictions
            else None
        ),
        "macro_state_agreement": fmean(agreements) if agreements else None,
        "per_reference_state": per_state,
        "confusion": {
            expected: dict(sorted(observed.items()))
            for expected, observed in sorted(confusion.items())
        },
        "failure_counts": dict(
            sorted(Counter(str(row["observed_label"]) for row in rows if not row["state_correct"]).items())
        ),
    }


def _assessment_for_observed(record: Mapping[str, Any]) -> CriterionAssessment:
    parsed = record["parsed"]
    if parsed.get("logical_valid") and not parsed.get("abstained") and parsed.get("state"):
        state = CriterionState(parsed["state"])
    elif record["runtime_failed"]:
        state = CriterionState.INFERENCE_FAILURE
    elif parsed.get("abstained"):
        state = CriterionState.MODEL_ABSTENTION
    else:
        state = CriterionState.PARSE_FAILURE
    return CriterionAssessment(
        criterion_id=record["criterion_id"],
        state=state,
        source_modalities=("synthetic_vignette",),
        resolution_method="slm_e1_strict_parser",
        is_pre_cutoff=True,
        abstained=state == CriterionState.MODEL_ABSTENTION,
        parse_failure=state == CriterionState.PARSE_FAILURE,
        inference_failure=state == CriterionState.INFERENCE_FAILURE,
    )


def _tree_summary(records: Sequence[Mapping[str, Any]], blueprint: Mapping[str, Any]) -> dict[str, Any]:
    by_request = {row["request_id"]: row for row in records}
    rows: list[dict[str, Any]] = []
    for case in blueprint["cases"]:
        expected_assessments: dict[str, CriterionAssessment] = {}
        observed_assessments: dict[str, CriterionAssessment] = {}
        for criterion_id, expected_state in case["expected_states"].items():
            expected_assessments[criterion_id] = CriterionAssessment(
                criterion_id=criterion_id,
                state=CriterionState(expected_state),
                source_modalities=("synthetic_vignette",),
                resolution_method="recursive_explicit_feature_labels_v2",
                is_pre_cutoff=True,
            )
            observed_assessments[criterion_id] = _assessment_for_observed(
                by_request[case["request_ids"][criterion_id]]
            )
        tree = load_tree(case["tree"])
        expected_actions = set(execute_partial(tree, expected_assessments).compatible_action_ids)
        observed_actions = set(execute_partial(tree, observed_assessments).compatible_action_ids)
        union = expected_actions | observed_actions
        rows.append(
            {
                "domain": case["domain"],
                "exact": expected_actions == observed_actions,
                "expected_subset": expected_actions <= observed_actions,
                "observed_subset": observed_actions <= expected_actions,
                "jaccard": len(expected_actions & observed_actions) / len(union) if union else 1.0,
                "expected_size": len(expected_actions),
                "observed_size": len(observed_actions),
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "exact_action_set_match": fmean(bool(row["exact"]) for row in selected),
            "expected_action_set_subset_of_observed": fmean(bool(row["expected_subset"]) for row in selected),
            "observed_action_set_subset_of_expected": fmean(bool(row["observed_subset"]) for row in selected),
            "mean_action_set_jaccard": fmean(float(row["jaccard"]) for row in selected),
            "expected_mean_action_set_size": fmean(int(row["expected_size"]) for row in selected),
            "observed_mean_action_set_size": fmean(int(row["observed_size"]) for row in selected),
        }

    domains = sorted({str(row["domain"]) for row in rows})
    return {
        "overall": summarize(rows),
        "by_domain": {
            domain: summarize([row for row in rows if row["domain"] == domain])
            for domain in domains
        },
        "interpretation": "Compatible partial-action-set agreement; not terminal-action accuracy.",
    }


def _structured_sampling_params(context: _E1ModelContext):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    runtime = context.manifest["runtime"]
    return SamplingParams(
        temperature=runtime["temperature"],
        top_p=1,
        top_k=0,
        max_tokens=runtime["max_output_tokens"][TASK],
        seed=runtime["seed"],
        structured_outputs=StructuredOutputsParams(
            json=context.prompt["output_schema"],
            disable_additional_properties=True,
        ),
    )


def run_e1_model(
    config_path: str | Path,
    *,
    manifest_id: str,
    dataset_manifest_path: str | Path,
    token_audit_path: str | Path,
    output_responses: str | Path,
    output_audit: str | Path,
    sensitivity_config_path: str | Path | None = None,
) -> dict[str, Any]:
    context = _load_model_context(
        config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path
    )
    sensitivity_path: Path | None = None
    sensitivity: dict[str, Any] | None = None
    if sensitivity_config_path is not None:
        (
            sensitivity_path,
            sensitivity,
            frozen_primary_path,
            _frozen_primary,
            frozen_dataset_path,
        ) = _load_s2_config(sensitivity_config_path)
        if frozen_primary_path.resolve() != context.e1_config_path.resolve():
            raise ValueError("E1-S2 primary config path mismatch")
        if frozen_dataset_path.resolve() != Path(dataset_manifest_path).resolve():
            raise ValueError("E1-S2 dataset path mismatch")
        context = _apply_s2_wrapper(
            context,
            sensitivity_path=sensitivity_path,
            sensitivity=sensitivity,
        )
    experiment_id = (
        sensitivity["experiment_id"] if sensitivity is not None else context.e1_config["experiment_id"]
    )
    adapter_stage = (
        "slm_benchmark_e1_s2_model" if sensitivity is not None else "slm_benchmark_e1_model"
    )
    token_audit = _read_object(Path(token_audit_path), "E1 token audit")
    if token_audit.get("e1_config_sha256") != config_sha256(context.e1_config):
        raise ValueError("E1 token audit is stale")
    if token_audit.get("dataset_manifest_sha256") != _sha256_file(dataset_manifest_path):
        raise ValueError("E1 token audit dataset hash is stale")
    if sensitivity is not None and token_audit.get(
        "sensitivity_config_sha256"
    ) != config_sha256(sensitivity):
        raise ValueError("E1-S2 token audit sensitivity hash is stale")
    if token_audit.get("all_models_fit_context") is not True:
        raise ValueError("E1 token audit has over-context requests")
    rendered, _tokenizer = _render_e1_requests(context)
    if any(row["over_context"] for row in rendered):
        raise ValueError("E1 contains an over-context request")
    blueprint = _read_object(context.blueprint_path, "E1 blueprint")
    expected_by_request = {
        request_id: expected
        for case in blueprint["cases"]
        for criterion_id, request_id in case["request_ids"].items()
        for expected in [case["expected_states"][criterion_id]]
    }
    sampler = _NvmlSampler()
    sampler.start()
    llm = None
    try:
        load_started = time.perf_counter()
        llm = _load_vllm(context)
        load_seconds = time.perf_counter() - load_started
        sampler.mark_static()
        inference_started = time.perf_counter()
        sampling_params = (
            _structured_sampling_params(context)
            if sensitivity is not None
            else _sampling_params(
                context, context.manifest["runtime"]["max_output_tokens"][TASK]
            )
        )
        generated = llm.generate(
            [row["rendered_prompt"] for row in rendered],
            sampling_params,
            use_tqdm=False,
        )
        inference_seconds = time.perf_counter() - inference_started
        records: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        with restricted_jsonl_writer(output_responses) as output:
            for rendered_request, result in zip(rendered, generated):
                request = rendered_request["request"]
                choice = result.outputs[0] if result.outputs else None
                runtime_failed = choice is None
                continuation = None if choice is None else choice.text
                raw_output = (
                    None
                    if continuation is None
                    else rendered_request["response_prefix"] + continuation
                )
                parsed = parse_typed_e0_output(
                    raw_output,
                    expected_criterion_id=request["criterion_id"],
                    runtime_failed=runtime_failed,
                ).as_dict()
                expected = expected_by_request[request["request_id"]]
                correct = bool(
                    parsed["logical_valid"]
                    and not parsed["abstained"]
                    and parsed["state"] == expected
                )
                prompt_tokens = len(getattr(result, "prompt_token_ids", []) or [])
                output_tokens = 0 if choice is None else len(choice.token_ids)
                total_prompt_tokens += prompt_tokens
                total_output_tokens += output_tokens
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request["request_id"],
                    "case_id": request["case_id"],
                    "domain": request["domain"],
                    "criterion_id": request["criterion_id"],
                    "expected_state": expected,
                    "prompt_sha256": hashlib.sha256(
                        rendered_request["rendered_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "raw_output": raw_output,
                    "generated_continuation": continuation if rendered_request["response_prefix"] else None,
                    "response_prefix": rendered_request["response_prefix"],
                    "runtime_failed": runtime_failed,
                    "finish_reason": "missing_output" if choice is None else str(choice.finish_reason),
                    "parsed": parsed,
                    "observed_label": _observed_label(parsed, runtime_failed),
                    "state_correct": correct,
                }
                records.append(record)
                write_jsonl_row(output, record)
        overall = _state_summary(records)
        domains = sorted({row["domain"] for row in records})
        criteria = sorted({(row["domain"], row["criterion_id"]) for row in records})
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": adapter_stage,
            "experiment_id": experiment_id,
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "artifact_sha256": context.manifest["verification"]["artifact_sha256"],
            "e1_config_sha256": config_sha256(context.e1_config),
            "sensitivity_config_sha256": (
                config_sha256(sensitivity) if sensitivity is not None else None
            ),
            "primary_results_sha256": (
                sensitivity.get("primary_results_sha256")
                if sensitivity is not None
                else None
            ),
            "benchmark_config_sha256": config_sha256(context.benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(context.dataset_manifest_path),
            "requests_sha256": context.dataset_manifest["output_requests_sha256"],
            "blueprint_sha256": context.dataset_manifest["output_blueprint_sha256"],
            "runtime_version": context.manifest["verification"]["runtime_version"],
            "generation": {
                "temperature": context.manifest["runtime"]["temperature"],
                "seed": context.manifest["runtime"]["seed"],
                "constrained_decoding": sensitivity is not None,
                "structured_output_schema_sha256": (
                    config_sha256(context.prompt["output_schema"])
                    if sensitivity is not None
                    else None
                ),
                "thinking_mode": context.manifest["runtime"]["thinking_mode"],
                "multimodal_limits": _multimodal_limits(context.manifest),
            },
            "technical_completion": overall["runtime_completed"] == len(rendered),
            "metrics": {
                "overall": overall,
                "by_domain": {
                    domain: _state_summary([row for row in records if row["domain"] == domain])
                    for domain in domains
                },
                "by_criterion": {
                    f"{domain}:{criterion}": _state_summary(
                        [
                            row
                            for row in records
                            if row["domain"] == domain and row["criterion_id"] == criterion
                        ]
                    )
                    for domain, criterion in criteria
                },
                "tree": _tree_summary(records, blueprint),
            },
            "throughput": {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "requests_per_second": len(rendered) / inference_seconds if inference_seconds else None,
                "prompt_tokens": total_prompt_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens_per_second": (
                    (total_prompt_tokens + total_output_tokens) / inference_seconds
                    if inference_seconds
                    else None
                ),
            },
            "gpu_memory": None,
            "output_responses": str(output_responses),
        }
    except Exception as exc:
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": adapter_stage,
            "experiment_id": experiment_id,
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "e1_config_sha256": config_sha256(context.e1_config),
            "sensitivity_config_sha256": (
                config_sha256(sensitivity) if sensitivity is not None else None
            ),
            "technical_completion": False,
            "runtime_failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "gpu_memory": None,
            "output_responses": str(output_responses),
        }
    finally:
        sampler.stop()
        if llm is not None:
            del llm
        gc.collect()
    audit["gpu_memory"] = sampler.report()
    write_json(output_audit, audit)
    return audit


def _mcnemar_exact(a_only: int, b_only: int) -> float | None:
    discordant = a_only + b_only
    if discordant == 0:
        return None
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _bootstrap_difference(
    a: Mapping[str, Mapping[str, Any]],
    b: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_case: dict[str, list[str]] = defaultdict(list)
    for request_id, row in a.items():
        by_case[str(row["case_id"])].append(request_id)
    case_ids = sorted(by_case)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(replicates):
        numerator = 0
        denominator = 0
        for case_id in rng.choices(case_ids, k=len(case_ids)):
            ids = by_case[case_id]
            numerator += sum(
                int(bool(b[request_id]["state_correct"]))
                - int(bool(a[request_id]["state_correct"]))
                for request_id in ids
            )
            denominator += len(ids)
        differences.append(numerator / denominator)
    differences.sort()
    lower = differences[math.floor(0.025 * (replicates - 1))]
    upper = differences[math.ceil(0.975 * (replicates - 1))]
    return lower, upper


def _posthoc_output_diagnostics(
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    finish_reasons: Counter[str] = Counter()
    observed_labels: Counter[str] = Counter()
    failure_mechanisms: Counter[str] = Counter()
    recoverable = 0
    recoverable_correct = 0
    for row in responses:
        finish_reasons[str(row.get("finish_reason"))] += 1
        observed_labels[str(row.get("observed_label"))] += 1
        raw = row.get("raw_output")
        try:
            value = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            failure_mechanisms["not_valid_json"] += 1
            continue
        if not isinstance(value, dict):
            failure_mechanisms["json_not_object"] += 1
            continue
        keys = frozenset(value)
        if {"type", "properties"} <= keys and not STRICT_OUTPUT_KEYS <= keys:
            failure_mechanisms["schema_echo"] += 1
        if STRICT_OUTPUT_KEYS - keys:
            failure_mechanisms["missing_required_keys"] += 1
        if keys - STRICT_OUTPUT_KEYS:
            failure_mechanisms["additional_keys"] += 1
        if value.get("evidence") != []:
            failure_mechanisms["nonempty_or_invalid_evidence"] += 1
        if value.get("value") is not None:
            failure_mechanisms["non_null_value"] += 1
        if value.get("unit") is not None:
            failure_mechanisms["non_null_unit"] += 1
        reason_code = value.get("reason_code")
        if isinstance(reason_code, str) and reason_code in RECOVERABLE_STATES:
            failure_mechanisms["state_label_used_as_reason_code"] += 1
        state = value.get("state")
        if (
            value.get("criterion_id") == row.get("criterion_id")
            and isinstance(state, str)
            and state in RECOVERABLE_STATES
        ):
            recoverable += 1
            recoverable_correct += int(state == row.get("expected_state"))
    strict_correct = sum(bool(row.get("state_correct")) for row in responses)
    return {
        "status": "POST_HOC_ERROR_DIAGNOSTIC_NOT_PRIMARY_METRIC",
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "observed_label_counts": dict(sorted(observed_labels.items())),
        "failure_mechanism_counts_nonexclusive": dict(sorted(failure_mechanisms.items())),
        "recoverable_state_predictions": recoverable,
        "recoverable_state_correct": recoverable_correct,
        "recoverable_state_agreement_unconditional": (
            recoverable_correct / len(responses) if responses else None
        ),
        "recoverable_state_agreement_conditional": (
            recoverable_correct / recoverable if recoverable else None
        ),
        "additional_correct_states_hidden_by_contract_failures": max(
            0, recoverable_correct - strict_correct
        ),
        "interpretation": "Reads only a matching criterion_id and recognized state from parseable JSON, ignoring the remaining frozen contract. It is diagnostic and does not replace strict E1 agreement.",
    }


def aggregate_e1_results(
    config_path: str | Path,
    *,
    dataset_manifest_path: str | Path,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source, config = _load_e1_config(config_path)
    benchmark_path = _resolve(source.parent, config["benchmark_config"], "benchmark")
    benchmark = _read_object(benchmark_path, "benchmark")
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    response_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for manifest_id in declared:
        context = _load_model_context(
            config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path
        )
        audit_path = root / manifest_id / "audit.json"
        audit = _read_object(audit_path, "E1 model audit")
        expected_hashes = {
            "e1_config_sha256": config_sha256(config),
            "benchmark_config_sha256": config_sha256(benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "requests_sha256": context.dataset_manifest["output_requests_sha256"],
            "blueprint_sha256": context.dataset_manifest["output_blueprint_sha256"],
        }
        if audit.get("manifest_id") != manifest_id:
            raise ValueError(f"{audit_path}: manifest ID mismatch")
        for field, expected in expected_hashes.items():
            if audit.get(field) != expected:
                raise ValueError(f"{audit_path}: stale {field}")
        response_path = Path(audit["output_responses"])
        responses = {row["request_id"]: row for row in read_jsonl(response_path)}
        if len(responses) != context.dataset_manifest["requests"]:
            raise ValueError(f"{response_path}: response count mismatch")
        response_by_model[manifest_id] = responses
        overall = audit["metrics"]["overall"]
        rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": audit["model_id"],
                "panel_role": context.manifest["factors"]["panel_role"],
                "family": context.manifest["factors"]["family"],
                "size_class": context.manifest["factors"]["size_class"],
                "instruction_tuned": context.manifest["factors"]["instruction_tuned"],
                "medical_adapted": context.manifest["factors"]["medical_adapted"],
                "technical_completion": audit["technical_completion"],
                "metrics": audit["metrics"],
                "throughput": audit["throughput"],
                "gpu_memory": audit["gpu_memory"],
                "posthoc_output_diagnostics": _posthoc_output_diagnostics(
                    list(responses.values())
                ),
                "audit_path": str(audit_path),
                "state_agreement_unconditional": overall["state_agreement_unconditional"],
                "macro_state_agreement": overall["macro_state_agreement"],
            }
        )

    request_sets = {frozenset(value) for value in response_by_model.values()}
    if len(request_sets) != 1:
        raise ValueError("E1 models do not share identical request IDs")
    contrast_results: list[dict[str, Any]] = []
    stats = config["statistics"]
    for contrast_index, contrast in enumerate(benchmark["contrasts"]):
        a_id, b_id = contrast["models"]
        a, b = response_by_model[a_id], response_by_model[b_id]
        request_ids = sorted(a)
        a_correct = sum(bool(a[r]["state_correct"]) for r in request_ids)
        b_correct = sum(bool(b[r]["state_correct"]) for r in request_ids)
        a_only = sum(bool(a[r]["state_correct"]) and not bool(b[r]["state_correct"]) for r in request_ids)
        b_only = sum(bool(b[r]["state_correct"]) and not bool(a[r]["state_correct"]) for r in request_ids)
        lower, upper = _bootstrap_difference(
            a,
            b,
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]) + contrast_index,
        )
        contrast_results.append(
            {
                "contrast_id": contrast["contrast_id"],
                "claim_type": contrast["claim_type"],
                "factor": contrast["factor"],
                "model_a": a_id,
                "model_b": b_id,
                "direction": "model_b minus model_a",
                "n": len(request_ids),
                "model_a_agreement": a_correct / len(request_ids),
                "model_b_agreement": b_correct / len(request_ids),
                "paired_agreement_difference": (b_correct - a_correct) / len(request_ids),
                "cluster_bootstrap_95_ci": [lower, upper],
                "discordant_model_a_only_correct": a_only,
                "discordant_model_b_only_correct": b_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(a_only, b_only),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e1_aggregate",
        "experiment_id": config["experiment_id"],
        "purpose": "controlled synthetic typed-state grounding with explicit facts supplied",
        "e1_config": str(source),
        "e1_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config_sha256(benchmark),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "declared_models": len(declared),
        "requests_per_model": next(iter(rows))["metrics"]["overall"]["n"],
        "all_models_technically_completed": all(row["technical_completion"] for row in rows),
        "models": rows,
        "paired_contrasts": contrast_results,
        "prior_exposure": config["prior_exposure"],
        "claim_boundary": config["claim_boundary"],
        "panel_change_policy": benchmark["model_change_policy"],
        "authorization_boundary": "E3, E4, model downloads, and full P2/E5 were not run or authorized.",
    }
    write_json(output, result)
    return result


def aggregate_e1_s2_results(
    sensitivity_config_path: str | Path,
    *,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    (
        sensitivity_path,
        sensitivity,
        primary_config_path,
        primary_config,
        dataset_manifest_path,
    ) = _load_s2_config(sensitivity_config_path)
    primary_results_path = _resolve(
        sensitivity_path.parent,
        sensitivity["primary_results"],
        "primary_results",
    )
    primary_results = _read_object(primary_results_path, "primary E1 results")
    benchmark_path = _resolve(
        primary_config_path.parent,
        primary_config["benchmark_config"],
        "benchmark",
    )
    benchmark = _read_object(benchmark_path, "benchmark")
    declared = [
        *benchmark["panel"]["core_pilot"],
        *benchmark["panel"]["contemporary_references"],
    ]
    primary_rows = {row["manifest_id"]: row for row in primary_results["models"]}
    root = Path(results_root)
    model_rows: list[dict[str, Any]] = []
    within_model: list[dict[str, Any]] = []
    s2_responses_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    stats = sensitivity["statistics"]

    for model_index, manifest_id in enumerate(declared):
        context = _load_model_context(
            primary_config_path,
            manifest_id=manifest_id,
            dataset_manifest_path=dataset_manifest_path,
        )
        context = _apply_s2_wrapper(
            context,
            sensitivity_path=sensitivity_path,
            sensitivity=sensitivity,
        )
        audit_path = root / manifest_id / "audit.json"
        audit = _read_object(audit_path, "E1-S2 model audit")
        expected_hashes = {
            "e1_config_sha256": config_sha256(primary_config),
            "sensitivity_config_sha256": config_sha256(sensitivity),
            "primary_results_sha256": sensitivity["primary_results_sha256"],
            "benchmark_config_sha256": config_sha256(benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "requests_sha256": context.dataset_manifest["output_requests_sha256"],
            "blueprint_sha256": context.dataset_manifest["output_blueprint_sha256"],
        }
        if audit.get("manifest_id") != manifest_id:
            raise ValueError(f"{audit_path}: manifest ID mismatch")
        if audit.get("generation", {}).get("constrained_decoding") is not True:
            raise ValueError(f"{audit_path}: constrained decoding not recorded")
        for field, expected in expected_hashes.items():
            if audit.get(field) != expected:
                raise ValueError(f"{audit_path}: stale {field}")
        s2_response_path = Path(audit["output_responses"])
        s2_responses = {
            row["request_id"]: row for row in read_jsonl(s2_response_path)
        }
        primary_audit_path = Path(primary_rows[manifest_id]["audit_path"])
        primary_audit = _read_object(primary_audit_path, "primary E1 model audit")
        primary_responses = {
            row["request_id"]: row
            for row in read_jsonl(primary_audit["output_responses"])
        }
        if set(s2_responses) != set(primary_responses):
            raise ValueError(f"{manifest_id}: primary/S2 request IDs differ")
        s2_responses_by_model[manifest_id] = s2_responses
        request_ids = sorted(s2_responses)
        primary_correct = sum(
            bool(primary_responses[request_id]["state_correct"])
            for request_id in request_ids
        )
        s2_correct = sum(
            bool(s2_responses[request_id]["state_correct"])
            for request_id in request_ids
        )
        primary_only = sum(
            bool(primary_responses[request_id]["state_correct"])
            and not bool(s2_responses[request_id]["state_correct"])
            for request_id in request_ids
        )
        s2_only = sum(
            bool(s2_responses[request_id]["state_correct"])
            and not bool(primary_responses[request_id]["state_correct"])
            for request_id in request_ids
        )
        lower, upper = _bootstrap_difference(
            primary_responses,
            s2_responses,
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]) + model_index,
        )
        primary_overall = primary_rows[manifest_id]["metrics"]["overall"]
        s2_overall = audit["metrics"]["overall"]
        within_model.append(
            {
                "manifest_id": manifest_id,
                "direction": "E1-S2 minus primary E1",
                "n": len(request_ids),
                "primary_strict_agreement": primary_correct / len(request_ids),
                "s2_strict_agreement": s2_correct / len(request_ids),
                "paired_agreement_difference": (
                    s2_correct - primary_correct
                )
                / len(request_ids),
                "cluster_bootstrap_95_ci": [lower, upper],
                "primary_only_correct": primary_only,
                "s2_only_correct": s2_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(
                    primary_only, s2_only
                ),
                "primary_schema_valid": primary_overall["schema_valid"],
                "s2_schema_valid": s2_overall["schema_valid"],
                "primary_logical_valid": primary_overall["logical_valid"],
                "s2_logical_valid": s2_overall["logical_valid"],
            }
        )
        model_rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": audit["model_id"],
                "panel_role": context.manifest["factors"]["panel_role"],
                "family": context.manifest["factors"]["family"],
                "size_class": context.manifest["factors"]["size_class"],
                "instruction_tuned": context.manifest["factors"][
                    "instruction_tuned"
                ],
                "medical_adapted": context.manifest["factors"]["medical_adapted"],
                "technical_completion": audit["technical_completion"],
                "metrics": audit["metrics"],
                "throughput": audit["throughput"],
                "gpu_memory": audit["gpu_memory"],
                "output_diagnostics": _posthoc_output_diagnostics(
                    list(s2_responses.values())
                ),
                "wrapper_sha256": audit["wrapper_sha256"],
                "audit_path": str(audit_path),
            }
        )

    contrast_results: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(benchmark["contrasts"]):
        a_id, b_id = contrast["models"]
        a, b = s2_responses_by_model[a_id], s2_responses_by_model[b_id]
        request_ids = sorted(a)
        a_correct = sum(bool(a[r]["state_correct"]) for r in request_ids)
        b_correct = sum(bool(b[r]["state_correct"]) for r in request_ids)
        a_only = sum(
            bool(a[r]["state_correct"]) and not bool(b[r]["state_correct"])
            for r in request_ids
        )
        b_only = sum(
            bool(b[r]["state_correct"]) and not bool(a[r]["state_correct"])
            for r in request_ids
        )
        lower, upper = _bootstrap_difference(
            a,
            b,
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]) + 100 + contrast_index,
        )
        contrast_results.append(
            {
                "contrast_id": contrast["contrast_id"],
                "claim_type": contrast["claim_type"],
                "factor": contrast["factor"],
                "model_a": a_id,
                "model_b": b_id,
                "direction": "model_b minus model_a under E1-S2",
                "n": len(request_ids),
                "model_a_agreement": a_correct / len(request_ids),
                "model_b_agreement": b_correct / len(request_ids),
                "paired_agreement_difference": (
                    b_correct - a_correct
                )
                / len(request_ids),
                "cluster_bootstrap_95_ci": [lower, upper],
                "discordant_model_a_only_correct": a_only,
                "discordant_model_b_only_correct": b_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(a_only, b_only),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e1_s2_aggregate",
        "experiment_id": sensitivity["experiment_id"],
        "status": sensitivity["status"],
        "purpose": "exploratory isolation of schema-constrained generation on fixed E1 requests",
        "sensitivity_config": str(sensitivity_path),
        "sensitivity_config_sha256": config_sha256(sensitivity),
        "primary_e1_config_sha256": config_sha256(primary_config),
        "primary_results": str(primary_results_path),
        "primary_results_sha256": _sha256_file(primary_results_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "declared_models": len(declared),
        "requests_per_model": context.dataset_manifest["requests"],
        "all_models_technically_completed": all(
            row["technical_completion"] for row in model_rows
        ),
        "intervention": sensitivity["intervention"],
        "models": model_rows,
        "within_model_effects": within_model,
        "s2_paired_model_contrasts": contrast_results,
        "claim_boundary": sensitivity["claim_boundary"],
        "panel_decision": {
            "retained_manifest_ids": declared,
            "excluded_manifest_ids": [],
            "reason": "The sensitivity evaluates format enforcement and does not redefine the approved panel.",
        },
        "authorization_boundary": "E3, E4, model downloads, and full P2/E5 were not run or authorized.",
    }
    write_json(output, result)
    return result
