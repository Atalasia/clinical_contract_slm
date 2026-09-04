from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import re
import sys
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .ext_notes_benchmark import validate_ext_notes_labels
from .states import CriterionState


BENCHMARK_SCHEMA_VERSION = "2.0"
COMPONENT_SCHEMA_VERSION = "1.0"
MODEL_FREEZE_STATES = frozenset({"PROPOSED", "IDENTITY_VERIFIED", "FROZEN"})
DOCUMENT_FREEZE_STATES = frozenset({"DRAFT", "FROZEN"})
TASKS = frozenset({"typed_state", "ext_notes_context"})
PANEL_ROLES = frozenset({"core_pilot", "contemporary_reference"})
SIZE_CLASSES = frozenset({"small", "medium"})
TYPED_GATE_STATES = frozenset(
    {
        CriterionState.PRESENT_CURRENT.value,
        CriterionState.ABSENT_EXPLICIT.value,
        CriterionState.HISTORICAL.value,
        CriterionState.FAMILY_HISTORY_OR_OTHER_EXPERIENCER.value,
        CriterionState.UNCERTAIN.value,
        CriterionState.NOT_DOCUMENTED.value,
    }
)
REQUIRED_GATE_CATEGORIES = frozenset(
    {
        "present",
        "explicit_absence",
        "historical",
        "other_experiencer",
        "uncertain",
        "conflicting",
        "not_documented",
        "composite_partial",
        "distractor",
        "detection",
        "encounter",
        "negation",
        "schema",
    }
)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
PROCESSOR_FILES = frozenset(
    {
        "image_processor_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    }
)
WRAPPER_MODES = frozenset(
    {
        "plain_completion",
        "official_chat_template",
        "official_processor_chat_template",
    }
)


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _component_report() -> dict[str, list[str]]:
    return {"errors": [], "blockers": [], "warnings": []}


def validate_model_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str = "model manifest",
    check_local_paths: bool = True,
) -> dict[str, Any]:
    report = _component_report()
    errors, blockers, warnings = (
        report["errors"],
        report["blockers"],
        report["warnings"],
    )
    if manifest.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {COMPONENT_SCHEMA_VERSION}")
    manifest_id = manifest.get("manifest_id")
    if not _nonempty_string(manifest_id):
        errors.append(f"{label}: manifest_id must be a non-empty string")
        manifest_id = "<invalid>"
    freeze_status = manifest.get("freeze_status")
    if freeze_status not in MODEL_FREEZE_STATES:
        errors.append(f"{label}: invalid freeze_status")
    elif freeze_status != "FROZEN":
        blockers.append(f"{manifest_id}: model manifest is not FROZEN")

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        errors.append(f"{label}: model must be an object")
        model = {}
    model_id = model.get("model_id")
    if not _nonempty_string(model_id):
        errors.append(f"{label}: model.model_id must be a non-empty string")
    revision = model.get("revision")
    if revision is None:
        blockers.append(f"{manifest_id}: immutable model revision is not verified")
    elif not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        errors.append(f"{label}: model.revision must be a 40-character commit hash or null")
    snapshot = model.get("local_snapshot")
    if snapshot is None:
        blockers.append(f"{manifest_id}: local snapshot is not available")
    elif not _nonempty_string(snapshot):
        errors.append(f"{label}: model.local_snapshot must be a path or null")
    elif check_local_paths and not Path(snapshot).is_dir():
        blockers.append(f"{manifest_id}: local snapshot path is not present")

    license_spec = model.get("license")
    if not isinstance(license_spec, Mapping):
        errors.append(f"{label}: model.license must be an object")
    else:
        for field in ("id", "source", "verified_on"):
            if not _nonempty_string(license_spec.get(field)):
                blockers.append(f"{manifest_id}: license {field} is not verified")

    parameters = model.get("parameters")
    if not isinstance(parameters, Mapping):
        errors.append(f"{label}: model.parameters must be an object")
    else:
        language_b = parameters.get("language_model_b")
        if (
            isinstance(language_b, bool)
            or not isinstance(language_b, (int, float))
            or language_b <= 0
            or language_b > 5
        ):
            errors.append(f"{label}: language_model_b must be numeric in (0, 5]")
        for field in ("total_b", "active_b", "non_embedding_b"):
            value = parameters.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                errors.append(f"{label}: parameters.{field} must be positive or null")
        if not _nonempty_string(parameters.get("source")):
            blockers.append(f"{manifest_id}: parameter source is not verified")

    for field in ("architecture", "modality", "wrapper", "wrapper_spec"):
        if not _nonempty_string(model.get(field)):
            errors.append(f"{label}: model.{field} must be a non-empty string")
    for field in ("processor_required",):
        if not isinstance(model.get(field), bool):
            errors.append(f"{label}: model.{field} must be Boolean")

    factors = manifest.get("factors")
    if not isinstance(factors, Mapping):
        errors.append(f"{label}: factors must be an object")
        factors = {}
    if factors.get("panel_role") not in PANEL_ROLES:
        errors.append(f"{label}: invalid factors.panel_role")
    if factors.get("size_class") not in SIZE_CLASSES:
        errors.append(f"{label}: invalid factors.size_class")
    if not _nonempty_string(factors.get("family")):
        errors.append(f"{label}: factors.family must be a non-empty string")
    for field in ("instruction_tuned", "medical_adapted", "release_reference"):
        if not isinstance(factors.get(field), bool):
            errors.append(f"{label}: factors.{field} must be Boolean")
    if factors.get("panel_role") == "core_pilot" and factors.get("release_reference"):
        errors.append(f"{label}: core_pilot models cannot be release references")
    if (
        factors.get("panel_role") == "contemporary_reference"
        and not factors.get("release_reference")
    ):
        errors.append(f"{label}: contemporary references require release_reference=true")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append(f"{label}: runtime must be an object")
        runtime = {}
    for field in ("engine", "primary_dtype", "thinking_mode", "vision_encoder"):
        if not _nonempty_string(runtime.get(field)):
            errors.append(f"{label}: runtime.{field} must be a non-empty string")
    if runtime.get("temperature") != 0:
        errors.append(f"{label}: runtime.temperature must be 0")
    if runtime.get("seed") != 2026:
        errors.append(f"{label}: runtime.seed must be 2026")
    if runtime.get("offline_only") is not True:
        errors.append(f"{label}: runtime.offline_only must be true")
    budgets = runtime.get("max_output_tokens")
    if not isinstance(budgets, Mapping) or set(budgets) != TASKS:
        errors.append(f"{label}: runtime.max_output_tokens must cover both tasks")
    elif any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in budgets.values()
    ):
        errors.append(f"{label}: runtime output-token budgets must be positive integers")

    verification = manifest.get("verification")
    if not isinstance(verification, Mapping):
        errors.append(f"{label}: verification must be an object")
    else:
        for field in (
            "artifact_sha256",
            "tokenizer_sha256",
            "wrapper_spec_sha256",
        ):
            value = verification.get(field)
            if value is None:
                blockers.append(f"{manifest_id}: {field} is not frozen")
            elif not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                errors.append(f"{label}: verification.{field} must be SHA-256 or null")
        template_hash = verification.get("chat_template_sha256")
        if model.get("wrapper") != "plain_completion_v1" and template_hash is None:
            blockers.append(f"{manifest_id}: chat_template_sha256 is not frozen")
        elif template_hash is not None and (
            not isinstance(template_hash, str) or not _SHA256_RE.fullmatch(template_hash)
        ):
            errors.append(
                f"{label}: verification.chat_template_sha256 must be SHA-256 or null"
            )
        processor_hash = verification.get("processor_sha256")
        if model.get("processor_required") and processor_hash is None:
            blockers.append(f"{manifest_id}: processor_sha256 is not frozen")
        elif processor_hash is not None and (
            not isinstance(processor_hash, str) or not _SHA256_RE.fullmatch(processor_hash)
        ):
            errors.append(f"{label}: verification.processor_sha256 must be SHA-256 or null")
        if not _nonempty_string(verification.get("runtime_version")):
            blockers.append(f"{manifest_id}: runtime version is not frozen")

    if freeze_status == "FROZEN" and blockers:
        errors.append(f"{label}: FROZEN manifest still has readiness blockers")
    return {
        **report,
        "manifest_id": manifest_id,
        "model_id": model_id,
        "freeze_status": freeze_status,
        "panel_role": factors.get("panel_role"),
        "family": factors.get("family"),
        "size_class": factors.get("size_class"),
        "instruction_tuned": factors.get("instruction_tuned"),
        "medical_adapted": factors.get("medical_adapted"),
        "release_reference": factors.get("release_reference"),
        "local_snapshot_present": bool(snapshot and Path(snapshot).is_dir()),
    }


def validate_prompt_spec(
    spec: Mapping[str, Any],
    *,
    label: str = "prompt spec",
) -> dict[str, Any]:
    report = _component_report()
    errors, blockers = report["errors"], report["blockers"]
    if spec.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {COMPONENT_SCHEMA_VERSION}")
    prompt_id = spec.get("prompt_id")
    if not _nonempty_string(prompt_id):
        errors.append(f"{label}: prompt_id must be a non-empty string")
        prompt_id = "<invalid>"
    task = spec.get("task")
    if task not in TASKS:
        errors.append(f"{label}: unsupported task")
    freeze_status = spec.get("freeze_status")
    if freeze_status not in DOCUMENT_FREEZE_STATES:
        errors.append(f"{label}: invalid freeze_status")
    elif freeze_status != "FROZEN":
        blockers.append(f"{prompt_id}: prompt is not FROZEN")
    for field in ("system", "user_template"):
        if not _nonempty_string(spec.get(field)):
            errors.append(f"{label}: {field} must be a non-empty string")
    if _nonempty_string(spec.get("user_template")) and "{output_schema_json}" not in spec[
        "user_template"
    ]:
        errors.append(f"{label}: user_template must render output_schema_json")
    output_schema = spec.get("output_schema")
    if not isinstance(output_schema, Mapping):
        errors.append(f"{label}: output_schema must be an object")
    else:
        properties = output_schema.get("properties")
        required = output_schema.get("required")
        if (
            output_schema.get("type") != "object"
            or output_schema.get("additionalProperties") is not False
            or not isinstance(properties, Mapping)
            or not isinstance(required, list)
            or set(required) != set(properties or {})
        ):
            errors.append(f"{label}: output_schema must require exactly its declared properties")
        elif task == "typed_state":
            expected_keys = {
                "criterion_id",
                "state",
                "value",
                "unit",
                "evidence",
                "reason_code",
                "confidence",
                "abstain",
            }
            if set(properties) != expected_keys:
                errors.append(f"{label}: typed_state output keys do not match the frozen contract")
            evidence = properties.get("evidence")
            if not isinstance(evidence, Mapping) or evidence.get("maxItems") != 0:
                errors.append(f"{label}: typed_state evidence must be a frozen empty array")
        elif task == "ext_notes_context" and set(properties) != {
            "detection",
            "encounter",
            "negation",
        }:
            errors.append(f"{label}: E3 output keys do not match the frozen contract")
    if task == "ext_notes_context" and spec.get("label_definitions_verified") is not True:
        blockers.append(f"{prompt_id}: native E3 label definitions require dataset verification")
    return {**report, "prompt_id": prompt_id, "task": task, "freeze_status": freeze_status}


def validate_wrapper_spec(
    spec: Mapping[str, Any],
    *,
    label: str = "wrapper spec",
) -> dict[str, Any]:
    report = _component_report()
    errors, blockers = report["errors"], report["blockers"]
    if spec.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {COMPONENT_SCHEMA_VERSION}")
    wrapper_id = spec.get("wrapper_id")
    if not _nonempty_string(wrapper_id):
        errors.append(f"{label}: wrapper_id must be a non-empty string")
        wrapper_id = "<invalid>"
    freeze_status = spec.get("freeze_status")
    if freeze_status not in DOCUMENT_FREEZE_STATES:
        errors.append(f"{label}: invalid freeze_status")
    elif freeze_status != "FROZEN":
        blockers.append(f"{wrapper_id}: wrapper is not FROZEN")
    mode = spec.get("mode")
    if mode not in WRAPPER_MODES:
        errors.append(f"{label}: invalid wrapper mode")
    repair_cycle = spec.get("repair_cycle")
    if isinstance(repair_cycle, bool) or not isinstance(repair_cycle, int) or repair_cycle < 0:
        errors.append(f"{label}: repair_cycle must be a non-negative integer")
    if mode == "plain_completion":
        template = spec.get("template")
        if not _nonempty_string(template) or not all(
            placeholder in template for placeholder in ("{system}", "{user}")
        ):
            errors.append(f"{label}: plain template must contain system and user placeholders")
        if spec.get("add_generation_prompt") is not False:
            errors.append(f"{label}: plain completion cannot add a chat generation prompt")
    elif mode in {"official_chat_template", "official_processor_chat_template"}:
        messages = spec.get("messages")
        valid_messages = (
            isinstance(messages, list)
            and len(messages) == 2
            and all(isinstance(message, Mapping) for message in messages)
        )
        if not valid_messages or (
            [message.get("role") for message in messages] != ["system", "user"]
            or messages[0].get("content") != "{system}"
            or messages[1].get("content") != "{user}"
        ):
            errors.append(f"{label}: chat wrapper must contain the frozen system/user messages")
        assistant_prefill = spec.get("assistant_prefill")
        has_prefill = isinstance(assistant_prefill, str) and bool(assistant_prefill)
        if has_prefill:
            if spec.get("add_generation_prompt") is not False:
                errors.append(f"{label}: assistant prefill cannot add a generation prompt")
            if spec.get("continue_final_message") is not True:
                errors.append(f"{label}: assistant prefill must continue the final message")
        elif spec.get("add_generation_prompt") is not True:
            errors.append(f"{label}: official chat wrapper must add the generation prompt")
        if mode == "official_processor_chat_template" and spec.get("text_only_input") is not True:
            errors.append(f"{label}: processor wrapper must require text-only input")
    return {
        **report,
        "wrapper_id": wrapper_id,
        "mode": mode,
        "freeze_status": freeze_status,
        "repair_cycle": repair_cycle,
    }


def validate_gate_suite(
    suite: Mapping[str, Any],
    *,
    label: str = "gate suite",
) -> dict[str, Any]:
    report = _component_report()
    errors, blockers = report["errors"], report["blockers"]
    if suite.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {COMPONENT_SCHEMA_VERSION}")
    suite_id = suite.get("suite_id")
    if not _nonempty_string(suite_id):
        errors.append(f"{label}: suite_id must be a non-empty string")
        suite_id = "<invalid>"
    purpose = suite.get("purpose")
    if purpose not in {"development", "compatibility_gate"}:
        errors.append(f"{label}: invalid purpose")
    freeze_status = suite.get("freeze_status")
    if freeze_status not in DOCUMENT_FREEZE_STATES:
        errors.append(f"{label}: invalid freeze_status")
    elif freeze_status != "FROZEN":
        blockers.append(f"{suite_id}: gate suite is not FROZEN")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(f"{label}: cases must be a non-empty list")
        cases = []
    if purpose == "compatibility_gate" and not 48 <= len(cases) <= 80:
        errors.append(f"{label}: compatibility gate must contain 48-80 cases")
    seen: set[str] = set()
    task_counts: Counter[str] = Counter()
    categories: set[str] = set()
    for index, case in enumerate(cases):
        case_label = f"{label}.cases[{index}]"
        if not isinstance(case, Mapping):
            errors.append(f"{case_label}: case must be an object")
            continue
        case_id = case.get("case_id")
        if not _nonempty_string(case_id):
            errors.append(f"{case_label}: case_id must be a non-empty string")
        elif case_id in seen:
            errors.append(f"{case_label}: duplicate case_id {case_id}")
        else:
            seen.add(case_id)
        task = case.get("task")
        if task not in TASKS:
            errors.append(f"{case_label}: unsupported task")
            continue
        task_counts[task] += 1
        category = case.get("category")
        if not _nonempty_string(category):
            errors.append(f"{case_label}: category must be a non-empty string")
        else:
            categories.add(category)
        if case.get("synthetic") is not True:
            errors.append(f"{case_label}: all compatibility cases must be synthetic")
        inputs = case.get("input")
        expected = case.get("expected")
        if not isinstance(inputs, Mapping) or not isinstance(expected, Mapping):
            errors.append(f"{case_label}: input and expected must be objects")
            continue
        if task == "typed_state":
            for field in ("criterion_id", "criterion_text", "text"):
                if not _nonempty_string(inputs.get(field)):
                    errors.append(f"{case_label}: input.{field} is required")
            if set(expected) != {"state"} or expected.get("state") not in TYPED_GATE_STATES:
                errors.append(f"{case_label}: invalid expected typed state")
        else:
            for field in ("note", "target_mention", "candidate_concept", "semantic_type"):
                if not _nonempty_string(inputs.get(field)):
                    errors.append(f"{case_label}: input.{field} is required")
            validation_error = validate_ext_notes_labels(
                expected, allow_abstention=False
            )
            if validation_error:
                errors.append(f"{case_label}: {validation_error}")
    if purpose == "compatibility_gate":
        missing = sorted(REQUIRED_GATE_CATEGORIES - categories)
        if missing:
            errors.append(f"{label}: missing required categories: {missing}")
        if task_counts["typed_state"] < 24 or task_counts["ext_notes_context"] < 16:
            errors.append(f"{label}: gate requires >=24 typed and >=16 E3 cases")
    return {
        **report,
        "suite_id": suite_id,
        "purpose": purpose,
        "freeze_status": freeze_status,
        "case_count": len(cases),
        "task_counts": dict(sorted(task_counts.items())),
        "categories": sorted(categories),
    }


def _resolve(config_dir: Path, value: Any, label: str) -> Path:
    if not _nonempty_string(value):
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def validate_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Validate the E0 contract and report structural errors separately from blockers."""

    source = Path(path)
    config = _read_object(source, "benchmark config")
    config_dir = source.parent
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if config.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        errors.append(f"benchmark schema_version must be {BENCHMARK_SCHEMA_VERSION}")
    experiment_id = config.get("experiment_id")
    if not _nonempty_string(experiment_id):
        errors.append("experiment_id must be a non-empty string")
    if config.get("freeze_status") not in DOCUMENT_FREEZE_STATES:
        errors.append("benchmark freeze_status must be DRAFT or FROZEN")
    elif config.get("freeze_status") != "FROZEN":
        blockers.append("benchmark config is not FROZEN")
    design_path = _resolve(config_dir, config.get("design_document"), "design_document")
    if not design_path.is_file():
        errors.append(f"design document does not exist: {design_path}")
    declaration_path = _resolve(
        config_dir, config.get("declaration_review"), "declaration_review"
    )
    if not declaration_path.is_file():
        errors.append(f"declaration review does not exist: {declaration_path}")

    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        errors.append("authorization must be an object")
    elif authorization.get("inference") is not True:
        blockers.append("new model inference is not authorized")
    else:
        authorized = authorization.get("authorized_experiments")
        if authorized != ["E0"]:
            errors.append("Stage A authorization must be limited to E0")
        if authorization.get("model_downloads") is not False:
            errors.append("E0 authorization must not permit model downloads")
        if authorization.get("full_p2") is not False:
            errors.append("full P2 must remain separately unauthorized")

    e0_protocol = config.get("e0_protocol")
    if not isinstance(e0_protocol, Mapping):
        errors.append("e0_protocol must be an object")
        e0_protocol = {}
    repairs = e0_protocol.get("development_wrapper_repairs_max")
    if isinstance(repairs, bool) or not isinstance(repairs, int) or not 0 <= repairs <= 2:
        errors.append("e0_protocol development repair budget must be an integer in [0, 2]")
    for field, expected in (
        ("semantic_prompt_repairs_allowed", False),
        ("constrained_decoding", False),
        ("trust_remote_code", False),
        ("multimodal_text_only", True),
        ("qwen3_5_enable_thinking", False),
    ):
        if e0_protocol.get(field) is not expected:
            errors.append(f"e0_protocol.{field} must be {str(expected).lower()}")
    gpu_utilization = e0_protocol.get("gpu_memory_utilization")
    if (
        isinstance(gpu_utilization, bool)
        or not isinstance(gpu_utilization, (int, float))
        or not 0 < gpu_utilization < 1
    ):
        errors.append("e0_protocol.gpu_memory_utilization must be numeric in (0, 1)")
    max_num_seqs = e0_protocol.get("max_num_seqs")
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs <= 0:
        errors.append("e0_protocol.max_num_seqs must be a positive integer")

    model_reports: list[dict[str, Any]] = []
    wrapper_reports: list[dict[str, Any]] = []
    manifest_ids: set[str] = set()
    for index, value in enumerate(config.get("model_manifests", [])):
        manifest_path = _resolve(config_dir, value, f"model_manifests[{index}]")
        try:
            manifest = _read_object(manifest_path, "model manifest")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        report = validate_model_manifest(manifest, label=str(manifest_path))
        if report["manifest_id"] in manifest_ids:
            errors.append(f"duplicate manifest_id: {report['manifest_id']}")
        manifest_ids.add(report["manifest_id"])
        errors.extend(report["errors"])
        blockers.extend(report["blockers"])
        warnings.extend(report["warnings"])
        report["path"] = str(manifest_path)
        model_reports.append(report)
        model = manifest.get("model")
        verification = manifest.get("verification")
        if isinstance(model, Mapping):
            try:
                wrapper_path = _resolve(
                    manifest_path.parent,
                    model.get("wrapper_spec"),
                    f"{manifest_path}.model.wrapper_spec",
                )
                wrapper = _read_object(wrapper_path, "wrapper spec")
            except ValueError as exc:
                errors.append(str(exc))
            else:
                wrapper_report = validate_wrapper_spec(wrapper, label=str(wrapper_path))
                wrapper_report["path"] = str(wrapper_path)
                wrapper_report["manifest_id"] = report["manifest_id"]
                errors.extend(wrapper_report["errors"])
                blockers.extend(wrapper_report["blockers"])
                warnings.extend(wrapper_report["warnings"])
                expected_model_wrapper = {
                    "plain_completion": "plain_completion_v1",
                    "official_chat_template": "official_chat_template",
                    "official_processor_chat_template": "official_processor_chat_template",
                }.get(wrapper_report["mode"])
                if expected_model_wrapper and model.get("wrapper") != expected_model_wrapper:
                    errors.append(
                        f"{manifest_path}: model.wrapper conflicts with {wrapper_path}"
                    )
                if isinstance(verification, Mapping):
                    expected_hash = verification.get("wrapper_spec_sha256")
                    actual_hash = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
                    if expected_hash != actual_hash:
                        errors.append(
                            f"{manifest_path}: wrapper_spec_sha256 does not match {wrapper_path}"
                        )
                wrapper_reports.append(wrapper_report)

    panel = config.get("panel")
    if not isinstance(panel, Mapping):
        errors.append("panel must be an object")
        panel = {}
    core = panel.get("core_pilot")
    references = panel.get("contemporary_references")
    if not isinstance(core, list) or len(core) != 6 or len(set(core)) != 6:
        errors.append("panel.core_pilot must contain six unique manifest IDs")
        core = []
    if not isinstance(references, list) or len(references) != 2 or len(set(references)) != 2:
        errors.append("panel.contemporary_references must contain two unique manifest IDs")
        references = []
    unknown_panel_ids = sorted((set(core) | set(references)) - manifest_ids)
    if unknown_panel_ids:
        errors.append(f"panel references unknown manifests: {unknown_panel_ids}")
    unassigned = sorted(manifest_ids - set(core) - set(references))
    if unassigned:
        warnings.append(f"manifests are not assigned to the active E0 panel: {unassigned}")
    reports_by_id = {report["manifest_id"]: report for report in model_reports}
    for manifest_id in core:
        report = reports_by_id.get(manifest_id)
        if report and report["panel_role"] != "core_pilot":
            errors.append(f"{manifest_id}: core panel assignment conflicts with manifest role")
    for manifest_id in references:
        report = reports_by_id.get(manifest_id)
        if report and report["panel_role"] != "contemporary_reference":
            errors.append(
                f"{manifest_id}: contemporary-reference assignment conflicts with manifest role"
            )

    contrast_reports: list[dict[str, Any]] = []
    contrasts = config.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        errors.append("contrasts must be a non-empty list")
        contrasts = []
    contrast_ids: set[str] = set()
    for index, contrast in enumerate(contrasts):
        contrast_label = f"contrasts[{index}]"
        if not isinstance(contrast, Mapping):
            errors.append(f"{contrast_label} must be an object")
            continue
        contrast_id = contrast.get("contrast_id")
        if not _nonempty_string(contrast_id):
            errors.append(f"{contrast_label}.contrast_id must be a non-empty string")
            continue
        if contrast_id in contrast_ids:
            errors.append(f"duplicate contrast_id: {contrast_id}")
        contrast_ids.add(contrast_id)
        models = contrast.get("models")
        if (
            not isinstance(models, list)
            or len(models) != 2
            or len(set(models)) != 2
            or not all(_nonempty_string(model) for model in models)
        ):
            errors.append(f"{contrast_label}.models must contain two unique manifest IDs")
            models = []
        unknown = sorted(set(models) - manifest_ids)
        if unknown:
            errors.append(f"{contrast_label} references unknown manifests: {unknown}")
        claim_type = contrast.get("claim_type")
        if claim_type not in {"matched_mechanistic", "comparative_not_causal", "descriptive"}:
            errors.append(f"{contrast_label}.claim_type is invalid")
        if not _nonempty_string(contrast.get("factor")):
            errors.append(f"{contrast_label}.factor must be a non-empty string")
        if not _nonempty_string(contrast.get("interpretation")):
            errors.append(f"{contrast_label}.interpretation must be a non-empty string")
        if any(model in references for model in models) and claim_type != "descriptive":
            errors.append(
                f"{contrast_label}: contrasts containing contemporary references must be descriptive"
            )
        contrast_reports.append(
            {
                "contrast_id": contrast_id,
                "models": models,
                "claim_type": claim_type,
                "factor": contrast.get("factor"),
            }
        )

    prompt_reports: list[dict[str, Any]] = []
    prompts = config.get("prompts")
    if not isinstance(prompts, Mapping) or set(prompts) != TASKS:
        errors.append("prompts must map exactly the two benchmark tasks")
        prompts = {}
    for task, value in prompts.items():
        prompt_path = _resolve(config_dir, value, f"prompts.{task}")
        try:
            prompt = _read_object(prompt_path, "prompt spec")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        report = validate_prompt_spec(prompt, label=str(prompt_path))
        if report["task"] != task:
            errors.append(f"prompt task mismatch: expected {task}")
        errors.extend(report["errors"])
        blockers.extend(report["blockers"])
        warnings.extend(report["warnings"])
        report["path"] = str(prompt_path)
        prompt_reports.append(report)

    gate_reports: list[dict[str, Any]] = []
    gate_case_ids: dict[str, set[str]] = {}
    gate_suites = config.get("gate_suites")
    if not isinstance(gate_suites, Mapping) or set(gate_suites) != {
        "development",
        "compatibility_gate",
    }:
        errors.append("gate_suites must define development and compatibility_gate")
        gate_suites = {}
    for purpose, value in gate_suites.items():
        gate_path = _resolve(config_dir, value, f"gate_suites.{purpose}")
        try:
            suite = _read_object(gate_path, "gate suite")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        report = validate_gate_suite(suite, label=str(gate_path))
        if report["purpose"] != purpose:
            errors.append(f"gate-suite purpose mismatch: expected {purpose}")
        errors.extend(report["errors"])
        blockers.extend(report["blockers"])
        warnings.extend(report["warnings"])
        report["path"] = str(gate_path)
        gate_reports.append(report)
        gate_case_ids[purpose] = {
            case.get("case_id")
            for case in suite.get("cases", [])
            if isinstance(case, Mapping) and _nonempty_string(case.get("case_id"))
        }

    rerun_ids = e0_protocol.get("deterministic_rerun_case_ids")
    if (
        not isinstance(rerun_ids, list)
        or not rerun_ids
        or len(set(rerun_ids)) != len(rerun_ids)
        or not all(_nonempty_string(case_id) for case_id in rerun_ids)
    ):
        errors.append("e0_protocol.deterministic_rerun_case_ids must be unique case IDs")
    else:
        unknown_reruns = sorted(set(rerun_ids) - gate_case_ids.get("compatibility_gate", set()))
        if unknown_reruns:
            errors.append(f"deterministic rerun IDs are not compatibility cases: {unknown_reruns}")

    structurally_valid = not errors
    inference_ready = structurally_valid and not blockers
    if config.get("freeze_status") == "FROZEN" and not inference_ready:
        errors.append("FROZEN benchmark still has inference-readiness blockers")
        structurally_valid = False
        inference_ready = False
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_preflight",
        "experiment_id": experiment_id,
        "benchmark_config": str(source),
        "structurally_valid": structurally_valid,
        "inference_ready": inference_ready,
        "inference_authorized": bool(
            isinstance(authorization, Mapping) and authorization.get("inference") is True
        ),
        "errors": errors,
        "blockers": blockers,
        "warnings": warnings,
        "panel": {
            "core_pilot": core,
            "contemporary_references": references,
        },
        "contrasts": contrast_reports,
        "models": model_reports,
        "wrappers": wrapper_reports,
        "prompts": prompt_reports,
        "gate_suites": gate_reports,
    }


def _sha256_file(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    cache[resolved] = value
    return value


def _composite_fingerprint(entries: list[dict[str, Any]]) -> str | None:
    if not entries:
        return None
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _snapshot_entries(snapshot: Path, cache: dict[Path, str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path, cache),
            }
        )
    return entries


def _embedded_chat_template_entry(
    snapshot: Path,
) -> dict[str, Any] | None:
    tokenizer_config = snapshot / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return None
    try:
        value = json.loads(tokenizer_config.read_text(encoding="utf-8")).get(
            "chat_template"
        )
    except (OSError, json.JSONDecodeError):
        return None
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "path": "tokenizer_config.json#chat_template",
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def inventory_benchmark_artifacts(path: str | Path) -> dict[str, Any]:
    """Fingerprint declared local snapshots and wrapper specs without loading models."""

    source = Path(path)
    config = _read_object(source, "benchmark config")
    config_dir = source.parent
    model_values = config.get("model_manifests")
    if not isinstance(model_values, list) or not model_values:
        raise ValueError("benchmark model_manifests must be a non-empty list")
    cache: dict[Path, str] = {}
    models: list[dict[str, Any]] = []
    for index, value in enumerate(model_values):
        manifest_path = _resolve(config_dir, value, f"model_manifests[{index}]")
        manifest = _read_object(manifest_path, "model manifest")
        model = manifest.get("model")
        if not isinstance(model, Mapping):
            raise ValueError(f"{manifest_path}: model must be an object")
        snapshot_value = model.get("local_snapshot")
        if not _nonempty_string(snapshot_value):
            raise ValueError(f"{manifest_path}: local snapshot is not declared")
        snapshot = Path(snapshot_value)
        if not snapshot.is_dir():
            raise ValueError(f"{manifest_path}: local snapshot is absent: {snapshot}")
        wrapper_path = _resolve(
            manifest_path.parent, model.get("wrapper_spec"), "model.wrapper_spec"
        )
        if not wrapper_path.is_file():
            raise ValueError(f"{manifest_path}: wrapper spec is absent: {wrapper_path}")
        entries = _snapshot_entries(snapshot, cache)
        tokenizer_entries = [
            entry for entry in entries if Path(entry["path"]).name in TOKENIZER_FILES
        ]
        processor_entries = [
            entry for entry in entries if Path(entry["path"]).name in PROCESSOR_FILES
        ]
        template_entries = [
            entry
            for entry in entries
            if Path(entry["path"]).name.startswith("chat_template")
        ]
        if not template_entries:
            embedded = _embedded_chat_template_entry(snapshot)
            if embedded is not None:
                template_entries.append(embedded)
        wrapper_entry = {
            "path": wrapper_path.name,
            "size_bytes": wrapper_path.stat().st_size,
            "sha256": _sha256_file(wrapper_path, cache),
        }
        models.append(
            {
                "manifest_id": manifest.get("manifest_id"),
                "model_id": model.get("model_id"),
                "revision": model.get("revision"),
                "manifest_path": str(manifest_path),
                "snapshot": str(snapshot),
                "file_count": len(entries),
                "total_bytes": sum(entry["size_bytes"] for entry in entries),
                "artifact_sha256": _composite_fingerprint(entries),
                "tokenizer_sha256": _composite_fingerprint(tokenizer_entries),
                "processor_sha256": _composite_fingerprint(processor_entries),
                "chat_template_sha256": (
                    None
                    if model.get("wrapper") == "plain_completion_v1"
                    else _composite_fingerprint(template_entries)
                ),
                "wrapper_spec_sha256": wrapper_entry["sha256"],
                "files": entries,
            }
        )
    packages: dict[str, str | None] = {}
    for name in ("vllm", "transformers", "torch", "accelerate", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime_version = ";".join(f"{name}={value}" for name, value in packages.items())
    return {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_artifact_inventory",
        "experiment_id": config.get("experiment_id"),
        "benchmark_config": str(source),
        "hash_definition": (
            "artifact_sha256 is SHA-256 over sorted relative-path, size, and file-SHA256 "
            "records for every snapshot file"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "runtime_version": runtime_version,
        "unique_physical_files_hashed": len(cache),
        "models": models,
    }


def audit_ext_notes_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Audit MIMIC-III-Ext-Notes without returning restricted text or identifiers."""

    root = Path(dataset_root)
    labels_path = root / "labels.csv"
    notes_path = root / "notes.csv"
    for path in (labels_path, notes_path):
        if not path.is_file():
            raise ValueError(f"required dataset file is absent: {path}")

    cache: dict[Path, str] = {}
    note_lengths: dict[str, int] = {}
    note_texts: dict[str, str] = {}
    subject_ids: set[str] = set()
    admission_ids: set[str] = set()
    with notes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"row_id", "subject_id", "hadm_id", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("notes.csv does not contain the required fields")
        for row in reader:
            row_id = row["row_id"]
            text = row["text"]
            note_lengths[row_id] = len(text)
            note_texts[row_id] = text
            if row["subject_id"]:
                subject_ids.add(row["subject_id"])
            if row["hadm_id"]:
                admission_ids.add(row["hadm_id"])

    label_rows = 0
    label_note_ids: set[str] = set()
    detection_counts: Counter[str] = Counter()
    encounter_counts: Counter[str] = Counter()
    negation_counts: Counter[str] = Counter()
    semantic_type_counts: Counter[str] = Counter()
    label_crosstab: Counter[tuple[str, str, str]] = Counter()
    invalid_offsets = 0
    trigger_offset_mismatches = 0
    orphan_labels = 0
    documented_hierarchy_violations = 0
    detection_yes_missing_downstream = 0
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "row_id",
            "trigger_word",
            "start",
            "end",
            "detection",
            "encounter",
            "negation",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("labels.csv does not contain the required fields")
        semantic_field = "semtypes" if "semtypes" in reader.fieldnames else "semantic_type"
        for row in reader:
            label_rows += 1
            row_id = row["row_id"]
            label_note_ids.add(row_id)
            detection = row["detection"]
            encounter = row["encounter"]
            negation = row["negation"]
            detection_counts[detection] += 1
            encounter_counts[encounter] += 1
            negation_counts[negation] += 1
            label_crosstab[(detection, encounter, negation)] += 1
            if semantic_field in row:
                semantic_type_counts[row.get(semantic_field) or "<blank>"] += 1
            if detection == "no" and (encounter != "-" or negation != "-"):
                documented_hierarchy_violations += 1
            if detection == "yes" and (encounter == "-" or negation == "-"):
                detection_yes_missing_downstream += 1
            text = note_texts.get(row_id)
            if text is None:
                orphan_labels += 1
                continue
            try:
                start, end = int(row["start"]), int(row["end"])
            except (TypeError, ValueError):
                invalid_offsets += 1
                continue
            if start < 0 or end <= start or end > note_lengths[row_id]:
                invalid_offsets += 1
            elif text[start:end] != row["trigger_word"]:
                trigger_offset_mismatches += 1

    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path, cache),
        }
        for path in (labels_path, notes_path)
    }
    return {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "adapter_stage": "mimic_iii_ext_notes_dataset_audit",
        "dataset": "MIMIC-III-Ext-Notes",
        "dataset_version": "1.0.0",
        "dataset_root": str(root),
        "files": files,
        "counts": {
            "notes": len(note_lengths),
            "label_rows": label_rows,
            "labeled_notes": len(label_note_ids),
            "subjects": len(subject_ids),
            "admissions": len(admission_ids),
            "detection": dict(sorted(detection_counts.items())),
            "encounter": dict(sorted(encounter_counts.items())),
            "negation": dict(sorted(negation_counts.items())),
            "semantic_types": dict(sorted(semantic_type_counts.items())),
            "label_crosstab": {
                "|".join(key): value for key, value in sorted(label_crosstab.items())
            },
        },
        "integrity": {
            "orphan_label_rows": orphan_labels,
            "invalid_offsets": invalid_offsets,
            "trigger_offset_mismatches": trigger_offset_mismatches,
            "detection_no_non_dash_downstream_rows": documented_hierarchy_violations,
            "detection_yes_dash_downstream_rows": detection_yes_missing_downstream,
        },
        "evaluation_policy": {
            "raw_labels_preserved": True,
            "downstream_metrics_condition": "gold detection=yes",
            "joint_metric_ignores_downstream_when_gold_detection_no": True,
            "rows_rewritten": 0,
        },
        "restricted_text_or_identifiers_emitted": False,
    }
