from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .ext_notes_benchmark import parse_ext_notes_output
from .io import write_json
from .mimic.csvio import (
    config_sha256,
    read_jsonl,
    restricted_jsonl_writer,
    write_jsonl_row,
)
from .slm_e0 import _NvmlSampler, _format_messages, _multimodal_limits


SCHEMA_VERSION = "1.0"
TASK = "ext_notes_context"
FAILURE_LABEL = "__failure__"


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


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _stable_id(namespace: str, value: str, dataset_hash: str) -> str:
    digest = hashlib.sha256(
        f"{dataset_hash}\0{namespace}\0{value}".encode("utf-8")
    ).hexdigest()
    return f"e3_{namespace}_{digest[:24]}"


@dataclass(frozen=True)
class _E3Context:
    config_path: Path
    config: dict[str, Any]
    benchmark_path: Path
    benchmark: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    wrapper_path: Path
    wrapper: dict[str, Any]
    prompt_path: Path
    prompt: dict[str, Any]
    dataset_manifest_path: Path | None = None
    dataset_manifest: dict[str, Any] | None = None
    requests_path: Path | None = None


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    source = Path(path)
    config = _read_object(source, "E3 config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E3 config schema")
    if config.get("freeze_status") != "FROZEN":
        raise ValueError("E3 config is not frozen")
    authorization = config.get("authorization", {})
    if (
        authorization.get("inference") is not True
        or "E3" not in authorization.get("authorized_experiments", [])
    ):
        raise ValueError("E3 inference is not authorized")
    base = source.parent
    benchmark_path = _resolve(base, config.get("benchmark_config"), "benchmark_config")
    benchmark = _read_object(benchmark_path, "benchmark config")
    if config_sha256(benchmark) != config.get("benchmark_config_sha256"):
        raise ValueError("E3 benchmark config hash mismatch")
    declared = [
        *benchmark["panel"]["core_pilot"],
        *benchmark["panel"]["contemporary_references"],
    ]
    if config.get("panel") != declared:
        raise ValueError("E3 panel differs from the approved benchmark panel")
    prompt_path = _resolve(base, config.get("prompt"), "prompt")
    prompt = _read_object(prompt_path, "E3 prompt")
    if config_sha256(prompt) != config.get("prompt_sha256"):
        raise ValueError("E3 prompt hash mismatch")
    if prompt.get("freeze_status") != "FROZEN" or prompt.get("task") != TASK:
        raise ValueError("E3 prompt is not the frozen native-label contract")
    dataset = config.get("dataset", {})
    root = Path(dataset.get("root", ""))
    notes_path = root / dataset.get("notes_file", "")
    labels_path = root / dataset.get("labels_file", "")
    if _sha256_file(notes_path) != dataset.get("notes_sha256"):
        raise ValueError("E3 notes.csv hash mismatch")
    if _sha256_file(labels_path) != dataset.get("labels_sha256"):
        raise ValueError("E3 labels.csv hash mismatch")
    source_audit_path = _resolve(base, dataset.get("source_audit"), "source_audit")
    if _sha256_file(source_audit_path) != dataset.get("source_audit_sha256"):
        raise ValueError("E3 source audit hash mismatch")
    source_audit = _read_object(source_audit_path, "E3 source audit")
    if source_audit.get("restricted_text_or_identifiers_emitted") is not False:
        raise ValueError("E3 source audit is not aggregate-only")
    expected = dataset.get("expected_counts", {})
    source_counts = source_audit.get("counts", {})
    for config_key, audit_key in (
        ("requests", "label_rows"),
        ("notes", "notes"),
        ("admissions", "admissions"),
        ("subjects", "subjects"),
    ):
        if expected.get(config_key) != source_counts.get(audit_key):
            raise ValueError(f"E3 expected {config_key} count mismatch")
    return source, config, benchmark_path, benchmark, prompt_path, prompt


def _manifest_by_id(
    benchmark_path: Path, benchmark: Mapping[str, Any]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for index, value in enumerate(benchmark["model_manifests"]):
        path = _resolve(
            benchmark_path.parent, value, f"model_manifests[{index}]"
        )
        manifest = _read_object(path, "model manifest")
        result[manifest["manifest_id"]] = (path, manifest)
    return result


def _load_model_context(
    config_path: str | Path,
    *,
    manifest_id: str,
    dataset_manifest_path: str | Path | None = None,
) -> _E3Context:
    source, config, benchmark_path, benchmark, prompt_path, prompt = _load_config(
        config_path
    )
    manifests = _manifest_by_id(benchmark_path, benchmark)
    if manifest_id not in config["panel"] or manifest_id not in manifests:
        raise ValueError(f"unknown E3 manifest_id: {manifest_id}")
    manifest_path, manifest = manifests[manifest_id]
    wrapper_path = _resolve(
        manifest_path.parent, manifest["model"]["wrapper_spec"], "wrapper_spec"
    )
    if _sha256_file(wrapper_path) != manifest["verification"]["wrapper_spec_sha256"]:
        raise ValueError("E3 frozen wrapper hash mismatch")
    wrapper = _read_object(wrapper_path, "wrapper")
    runtime = manifest["runtime"]
    policy = config["context_policy"]
    generation = config["generation"]
    if (
        runtime["max_context_tokens"] != policy["max_context_tokens"]
        or runtime["max_output_tokens"][TASK] != policy["output_reserve_tokens"]
        or runtime["max_output_tokens"][TASK] != generation["max_output_tokens"]
        or runtime["primary_dtype"] != generation["dtype"]
        or runtime["temperature"] != generation["temperature"]
        or runtime["seed"] != generation["seed"]
    ):
        raise ValueError(f"{manifest_id}: runtime differs from the frozen E3 contract")
    dataset_manifest = None
    requests_path = None
    resolved_dataset_manifest_path = None
    if dataset_manifest_path is not None:
        resolved_dataset_manifest_path = Path(dataset_manifest_path)
        dataset_manifest = _read_object(
            resolved_dataset_manifest_path, "E3 dataset manifest"
        )
        if dataset_manifest.get("e3_config_sha256") != config_sha256(config):
            raise ValueError("E3 dataset manifest was not built from the frozen config")
        requests_path = Path(dataset_manifest["output_requests"])
        if _sha256_file(requests_path) != dataset_manifest["output_requests_sha256"]:
            raise ValueError("E3 request file hash mismatch")
    return _E3Context(
        config_path=source,
        config=config,
        benchmark_path=benchmark_path,
        benchmark=benchmark,
        manifest_path=manifest_path,
        manifest=manifest,
        wrapper_path=wrapper_path,
        wrapper=wrapper,
        prompt_path=prompt_path,
        prompt=prompt,
        dataset_manifest_path=resolved_dataset_manifest_path,
        dataset_manifest=dataset_manifest,
        requests_path=requests_path,
    )


def _load_renderer(context: _E3Context):
    from transformers import AutoProcessor, AutoTokenizer

    snapshot = context.manifest["model"]["local_snapshot"]
    mode = context.wrapper["mode"]
    if mode in {"plain_completion", "official_chat_template"}:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        return tokenizer, tokenizer
    if mode == "official_processor_chat_template":
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("official processor does not expose a tokenizer")
        return processor, tokenizer
    raise ValueError(f"unsupported E3 wrapper mode: {mode}")


def _render_prompt(
    context: _E3Context,
    renderer: Any,
    tokenizer: Any,
    request: Mapping[str, Any],
) -> tuple[str, int]:
    schema_json = json.dumps(
        context.prompt["output_schema"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    user = context.prompt["user_template"].format(
        note=request["note_context"],
        target_mention=request["target_mention"],
        candidate_concept=request["candidate_concept"],
        semantic_type=request["semantic_type_prompt"],
        output_schema_json=schema_json,
    )
    if context.wrapper["mode"] == "plain_completion":
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
        if context.manifest["runtime"]["thinking_mode"] == "disabled":
            kwargs["enable_thinking"] = False
        rendered = renderer.apply_chat_template(
            _format_messages(
                context.wrapper, context.prompt["system"], user
            ),
            **kwargs,
        )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, len(tokens)


def _window_bounds(note_length: int, start: int, end: int, maximum: int) -> tuple[int, int]:
    if note_length <= maximum:
        return 0, note_length
    mention_length = end - start
    if mention_length > maximum:
        raise ValueError("target mention exceeds the shared E3 character window")
    remaining = maximum - mention_length
    lower = max(0, start - remaining // 2)
    upper = min(note_length, end + (remaining - (start - lower)))
    if upper - lower < maximum:
        lower = max(0, upper - maximum)
    return lower, upper


def _tagged_slice(
    note: str, start: int, end: int, lower: int, upper: int, tags: Sequence[str]
) -> str:
    local_start = start - lower
    local_end = end - lower
    selected = note[lower:upper]
    return (
        selected[:local_start]
        + tags[0]
        + selected[local_start:local_end]
        + tags[1]
        + selected[local_end:]
    )


def _source_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    root = Path(dataset["root"])
    notes_path = root / dataset["notes_file"]
    labels_path = root / dataset["labels_file"]
    notes: dict[str, dict[str, str]] = {}
    with notes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            notes[row["row_id"]] = row
    tags = config["request_contract"]["target_tags"]
    mapping = config["request_contract"]["semantic_type_mapping"]
    maximum = int(config["context_policy"]["window_max_characters"])
    dataset_hash = dataset["labels_sha256"]
    candidates: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for ordinal, row in enumerate(reader):
            note_row = notes.get(row["row_id"])
            if note_row is None:
                raise ValueError(f"E3 label row {ordinal} has no source note")
            note = note_row["text"]
            if tags[0] in note or tags[1] in note:
                raise ValueError("source note already contains a frozen target tag")
            start, end = int(row["start"]), int(row["end"])
            if start < 0 or end <= start or end > len(note):
                raise ValueError(f"E3 label row {ordinal} has invalid offsets")
            if note[start:end] != row["trigger_word"]:
                raise ValueError(f"E3 label row {ordinal} trigger does not match note")
            semantic_raw = row.get("semtypes") or ""
            if semantic_raw not in mapping:
                raise ValueError(f"E3 label row {ordinal} has unknown semantic type")
            semantic_group = semantic_raw or config["request_contract"][
                "null_semantic_type_group"
            ]
            detection = row["detection"]
            if detection not in {"yes", "no"}:
                raise ValueError(f"E3 label row {ordinal} has invalid detection label")
            if detection == "yes":
                encounter = row["encounter"]
                negation = row["negation"]
                if encounter not in {"yes", "no"} or negation not in {
                    "yes",
                    "no",
                    "unsure",
                }:
                    raise ValueError(
                        f"E3 label row {ordinal} lacks eligible downstream labels"
                    )
            else:
                encounter = "not_applicable"
                negation = "not_applicable"
            lower, upper = _window_bounds(len(note), start, end, maximum)
            identity = json.dumps(
                {
                    "row_id": row["row_id"],
                    "ordinal": ordinal,
                    "start": start,
                    "end": end,
                    "concept": row["concept"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            request_id = _stable_id("request", identity, dataset_hash)
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request_id,
                    "source_ordinal": ordinal,
                    "note_cluster": _stable_id(
                        "note", row["row_id"], dataset_hash
                    ),
                    "admission_cluster": _stable_id(
                        "admission", note_row["hadm_id"], dataset_hash
                    ),
                    "subject_cluster": _stable_id(
                        "subject", note_row["subject_id"], dataset_hash
                    ),
                    "semantic_group": semantic_group,
                    "semantic_type_prompt": mapping[semantic_raw],
                    "target_mention": row["trigger_word"],
                    "candidate_concept": row["concept"],
                    "expected": {
                        "detection": detection,
                        "encounter": encounter,
                        "negation": negation,
                    },
                    "raw_source_labels": {
                        "detection": row["detection"],
                        "encounter": row["encounter"],
                        "negation": row["negation"],
                    },
                    "full_note": _tagged_slice(
                        note, start, end, 0, len(note), tags
                    ),
                    "window_note": _tagged_slice(
                        note, start, end, lower, upper, tags
                    ),
                    "original_note_characters": len(note),
                    "window_start": lower,
                    "window_end": upper,
                }
            )
    if len(candidates) != config["dataset"]["expected_counts"]["requests"]:
        raise ValueError("E3 prepared request count differs from the frozen expectation")
    return candidates


def _prompt_fingerprint(rows: Sequence[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256()
    for request_id, prompt_hash, token_count in rows:
        digest.update(request_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(prompt_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(token_count).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_e3_dataset(
    config_path: str | Path,
    *,
    output_requests: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    source, config, benchmark_path, benchmark, prompt_path, prompt = _load_config(
        config_path
    )
    candidates = _source_candidates(config)
    token_data: dict[str, dict[str, dict[str, Any]]] = {}
    all_full_fit = {candidate["request_id"]: True for candidate in candidates}
    manifests = _manifest_by_id(benchmark_path, benchmark)
    for manifest_id in config["panel"]:
        context = _load_model_context(source, manifest_id=manifest_id)
        renderer, tokenizer = _load_renderer(context)
        rows: dict[str, dict[str, Any]] = {}
        max_context = context.manifest["runtime"]["max_context_tokens"]
        reserve = context.manifest["runtime"]["max_output_tokens"][TASK]
        for candidate in candidates:
            full_request = {**candidate, "note_context": candidate["full_note"]}
            window_request = {**candidate, "note_context": candidate["window_note"]}
            full_prompt, full_tokens = _render_prompt(
                context, renderer, tokenizer, full_request
            )
            window_prompt, window_tokens = _render_prompt(
                context, renderer, tokenizer, window_request
            )
            full_fit = full_tokens + reserve <= max_context
            window_fit = window_tokens + reserve <= max_context
            all_full_fit[candidate["request_id"]] &= full_fit
            rows[candidate["request_id"]] = {
                "full_tokens": full_tokens,
                "window_tokens": window_tokens,
                "full_prompt_sha256": hashlib.sha256(
                    full_prompt.encode("utf-8")
                ).hexdigest(),
                "window_prompt_sha256": hashlib.sha256(
                    window_prompt.encode("utf-8")
                ).hexdigest(),
                "window_fit": window_fit,
            }
        token_data[manifest_id] = rows
        del renderer, tokenizer
        gc.collect()

    finalized: list[dict[str, Any]] = []
    for candidate in candidates:
        request_id = candidate["request_id"]
        mode = "full_note" if all_full_fit[request_id] else "shared_span_window"
        if mode == "shared_span_window" and not all(
            token_data[manifest_id][request_id]["window_fit"]
            for manifest_id in config["panel"]
        ):
            raise ValueError(
                f"{request_id}: frozen shared window exceeds at least one model context"
            )
        finalized.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"full_note", "window_note"}
            }
            | {
                "context_mode": mode,
                "note_context": (
                    candidate["full_note"]
                    if mode == "full_note"
                    else candidate["window_note"]
                ),
                "context_characters": len(
                    candidate["full_note"]
                    if mode == "full_note"
                    else candidate["window_note"]
                ),
            }
        )

    with restricted_jsonl_writer(output_requests) as output:
        for request in finalized:
            write_jsonl_row(output, request)
    response_reserve = config["context_policy"]["output_reserve_tokens"]
    token_rows: list[dict[str, Any]] = []
    for manifest_id in config["panel"]:
        manifest_path, manifest = manifests[manifest_id]
        wrapper_path = _resolve(
            manifest_path.parent,
            manifest["model"]["wrapper_spec"],
            "wrapper_spec",
        )
        selected: list[int] = []
        fingerprint_rows: list[tuple[str, str, int]] = []
        for request in finalized:
            data = token_data[manifest_id][request["request_id"]]
            prefix = "full" if request["context_mode"] == "full_note" else "window"
            selected.append(data[f"{prefix}_tokens"])
            fingerprint_rows.append(
                (
                    request["request_id"],
                    data[f"{prefix}_prompt_sha256"],
                    data[f"{prefix}_tokens"],
                )
            )
        token_rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": manifest["model"]["model_id"],
                "manifest_sha256": config_sha256(manifest),
                "wrapper_sha256": _sha256_file(wrapper_path),
                "requests": len(selected),
                "minimum_prompt_tokens": min(selected),
                "mean_prompt_tokens": fmean(selected),
                "p95_prompt_tokens": _percentile(selected, 0.95),
                "maximum_prompt_tokens": max(selected),
                "max_output_tokens": response_reserve,
                "max_context_tokens": manifest["runtime"]["max_context_tokens"],
                "over_context_requests": sum(
                    value + response_reserve
                    > manifest["runtime"]["max_context_tokens"]
                    for value in selected
                ),
                "rendered_prompt_fingerprint": _prompt_fingerprint(
                    fingerprint_rows
                ),
            }
        )

    label_counts = {
        "detection": dict(
            sorted(Counter(row["expected"]["detection"] for row in finalized).items())
        ),
        "encounter_gold_detection_yes": dict(
            sorted(
                Counter(
                    row["expected"]["encounter"]
                    for row in finalized
                    if row["expected"]["detection"] == "yes"
                ).items()
            )
        ),
        "negation_gold_detection_yes": dict(
            sorted(
                Counter(
                    row["expected"]["negation"]
                    for row in finalized
                    if row["expected"]["detection"] == "yes"
                ).items()
            )
        ),
    }
    context_lengths = [row["context_characters"] for row in finalized]
    source_audit_path = _resolve(
        source.parent, config["dataset"]["source_audit"], "source_audit"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e3_prepare",
        "experiment_id": config["experiment_id"],
        "e3_config": str(source),
        "e3_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config_sha256(benchmark),
        "prompt_sha256": config_sha256(prompt),
        "dataset": config["dataset"]["name"],
        "dataset_version": config["dataset"]["version"],
        "source_files": {
            "notes_sha256": config["dataset"]["notes_sha256"],
            "labels_sha256": config["dataset"]["labels_sha256"],
            "source_audit_sha256": _sha256_file(source_audit_path),
        },
        "counts": {
            "requests": len(finalized),
            "notes": len({row["note_cluster"] for row in finalized}),
            "admissions": len({row["admission_cluster"] for row in finalized}),
            "subjects": len({row["subject_cluster"] for row in finalized}),
            "semantic_groups": dict(
                sorted(Counter(row["semantic_group"] for row in finalized).items())
            ),
            "context_modes": dict(
                sorted(Counter(row["context_mode"] for row in finalized).items())
            ),
            "source_detection_no_non_dash_downstream": sum(
                row["raw_source_labels"]["detection"] == "no"
                and (
                    row["raw_source_labels"]["encounter"] != "-"
                    or row["raw_source_labels"]["negation"] != "-"
                )
                for row in finalized
            ),
        },
        "label_counts": label_counts,
        "context_policy": config["context_policy"],
        "context_characters": {
            "minimum": min(context_lengths),
            "mean": fmean(context_lengths),
            "p95": _percentile(context_lengths, 0.95),
            "maximum": max(context_lengths),
        },
        "token_audit": token_rows,
        "all_models_fit_context": all(
            row["over_context_requests"] == 0 for row in token_rows
        ),
        "output_requests": str(output_requests),
        "output_requests_sha256": _sha256_file(output_requests),
        "restricted_text_or_source_identifiers_in_manifest": False,
        "claim_boundary": config["claim_boundary"],
        "authorization_boundary": config["authorization_boundary"],
    }
    write_json(output_manifest, manifest)
    return manifest


def _classification_metrics(
    gold: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(predictions):
        raise ValueError("classification gold/prediction lengths differ")
    per_class: dict[str, dict[str, Any]] = {}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, observed in zip(gold, predictions):
        confusion[expected][observed] += 1
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predictions))
        fp = sum(g != label and p == label for g, p in zip(gold, predictions))
        fn = sum(g == label and p != label for g, p in zip(gold, predictions))
        support = sum(g == label for g in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "predicted": sum(p == label for p in predictions),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    valid = sum(prediction in labels for prediction in predictions)
    correct = sum(g == p for g, p in zip(gold, predictions))
    return {
        "n": len(gold),
        "labels": list(labels),
        "correct": correct,
        "accuracy": correct / len(gold) if gold else None,
        "macro_f1": fmean(per_class[label]["f1"] for label in labels),
        "balanced_accuracy": fmean(
            per_class[label]["recall"] for label in labels
        ),
        "per_class": per_class,
        "confusion": {
            expected: dict(sorted(values.items()))
            for expected, values in sorted(confusion.items())
        },
        "valid_nonabstained_predictions": valid,
        "coverage": valid / len(gold) if gold else None,
        "selective_risk": (
            1
            - sum(
                g == p
                for g, p in zip(gold, predictions)
                if p in labels
            )
            / valid
            if valid
            else None
        ),
    }


def _conditional_metrics(
    gold: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    selected = [
        (expected, observed)
        for expected, observed in zip(gold, predictions)
        if observed in labels
    ]
    metrics = _classification_metrics(
        [row[0] for row in selected],
        [row[1] for row in selected],
        labels,
    )
    metrics["eligible_n"] = len(gold)
    metrics["coverage"] = len(selected) / len(gold) if gold else None
    return metrics


def _prediction_heads(parsed: Mapping[str, Any]) -> dict[str, str]:
    if not parsed.get("logical_valid"):
        return {
            "detection": FAILURE_LABEL,
            "encounter": FAILURE_LABEL,
            "negation": FAILURE_LABEL,
        }
    detection = str(parsed.get("detection"))
    detection_prediction = (
        detection if detection in {"yes", "no"} else FAILURE_LABEL
    )
    encounter = (
        str(parsed.get("encounter"))
        if detection == "yes" and parsed.get("encounter") in {"yes", "no"}
        else FAILURE_LABEL
    )
    negation = (
        str(parsed.get("negation"))
        if detection == "yes"
        and parsed.get("negation") in {"yes", "no", "unsure"}
        else FAILURE_LABEL
    )
    return {
        "detection": detection_prediction,
        "encounter": encounter,
        "negation": negation,
    }


def _hierarchical_correct(
    expected: Mapping[str, str], parsed: Mapping[str, Any]
) -> bool:
    if not parsed.get("logical_valid"):
        return False
    if expected["detection"] == "no":
        return parsed.get("detection") == "no"
    return (
        parsed.get("detection") == expected["detection"]
        and parsed.get("encounter") == expected["encounter"]
        and parsed.get("negation") == expected["negation"]
    )


def _score_e3_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detection_gold = [row["expected"]["detection"] for row in rows]
    detection_predictions = [row["predictions"]["detection"] for row in rows]
    downstream = [row for row in rows if row["expected"]["detection"] == "yes"]
    encounter_gold = [row["expected"]["encounter"] for row in downstream]
    encounter_predictions = [row["predictions"]["encounter"] for row in downstream]
    negation_gold = [row["expected"]["negation"] for row in downstream]
    negation_predictions = [row["predictions"]["negation"] for row in downstream]
    detection = _classification_metrics(
        detection_gold, detection_predictions, ["yes", "no"]
    )
    encounter = _classification_metrics(
        encounter_gold, encounter_predictions, ["yes", "no"]
    )
    negation = _classification_metrics(
        negation_gold, negation_predictions, ["yes", "no", "unsure"]
    )
    detection["conditional_on_valid_nonabstained"] = _conditional_metrics(
        detection_gold, detection_predictions, ["yes", "no"]
    )
    encounter["conditional_on_valid_nonabstained"] = _conditional_metrics(
        encounter_gold, encounter_predictions, ["yes", "no"]
    )
    negation["conditional_on_valid_nonabstained"] = _conditional_metrics(
        negation_gold, negation_predictions, ["yes", "no", "unsure"]
    )
    joint_correct = sum(bool(row["hierarchical_joint_correct"]) for row in rows)
    jointly_covered = [
        row
        for row in rows
        if row["parsed"].get("logical_valid")
        and not row["parsed"].get("abstained")
    ]
    return {
        "n": len(rows),
        "clusters": {
            "notes": len({row["note_cluster"] for row in rows}),
            "admissions": len({row["admission_cluster"] for row in rows}),
            "subjects": len({row["subject_cluster"] for row in rows}),
        },
        "runtime_completed": sum(not row["runtime_failed"] for row in rows),
        "parse_valid": sum(bool(row["parsed"].get("parse_valid")) for row in rows),
        "schema_valid": sum(bool(row["parsed"].get("schema_valid")) for row in rows),
        "logical_valid": sum(bool(row["parsed"].get("logical_valid")) for row in rows),
        "abstained": sum(bool(row["parsed"].get("abstained")) for row in rows),
        "failure_counts": dict(
            sorted(
                Counter(
                    row["parsed"].get("failure") or "none" for row in rows
                ).items()
            )
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(row["finish_reason"]) for row in rows).items())
        ),
        "detection": detection,
        "encounter_gold_detection_yes": encounter,
        "negation_gold_detection_yes": negation,
        "hierarchical_joint": {
            "correct": joint_correct,
            "exact_match": joint_correct / len(rows) if rows else None,
            "valid_nonabstained_predictions": len(jointly_covered),
            "coverage": len(jointly_covered) / len(rows) if rows else None,
            "selective_risk": (
                1
                - sum(bool(row["hierarchical_joint_correct"]) for row in jointly_covered)
                / len(jointly_covered)
                if jointly_covered
                else None
            ),
        },
        "risk_coverage_note": "The frozen contract has no confidence field; only the observed abstention operating point is estimable.",
    }


def _baseline_metrics(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detection_gold = [row["expected"]["detection"] for row in requests]
    eligible = [row for row in requests if row["expected"]["detection"] == "yes"]
    encounter_gold = [row["expected"]["encounter"] for row in eligible]
    negation_gold = [row["expected"]["negation"] for row in eligible]

    def majority(values: Sequence[str]) -> str:
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], value))[0]

    detection_majority = majority(detection_gold)
    encounter_majority = majority(encounter_gold)
    negation_majority = majority(negation_gold)
    return {
        "metamap_accept_all_detection": {
            "prediction": "yes",
            "metrics": _classification_metrics(
                detection_gold, ["yes"] * len(requests), ["yes", "no"]
            ),
            "limitation": "Candidate-conditioned precision-related baseline; not end-to-end MetaMap recall.",
        },
        "majority_class": {
            "detection": {
                "prediction": detection_majority,
                "metrics": _classification_metrics(
                    detection_gold,
                    [detection_majority] * len(requests),
                    ["yes", "no"],
                ),
            },
            "encounter_gold_detection_yes": {
                "prediction": encounter_majority,
                "metrics": _classification_metrics(
                    encounter_gold,
                    [encounter_majority] * len(eligible),
                    ["yes", "no"],
                ),
            },
            "negation_gold_detection_yes": {
                "prediction": negation_majority,
                "metrics": _classification_metrics(
                    negation_gold,
                    [negation_majority] * len(eligible),
                    ["yes", "no", "unsure"],
                ),
            },
        },
    }


def _load_vllm(context: _E3Context):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    import vllm
    from vllm import LLM

    expected_version = context.manifest["verification"]["runtime_version"]
    if f"vllm={vllm.__version__}" not in expected_version.split(";"):
        raise RuntimeError(
            f"frozen runtime does not match installed vLLM {vllm.__version__}"
        )
    generation = context.config["generation"]
    kwargs: dict[str, Any] = {
        "model": context.manifest["model"]["local_snapshot"],
        "tokenizer": context.manifest["model"]["local_snapshot"],
        "runner": "generate",
        "dtype": generation["dtype"],
        "max_model_len": context.manifest["runtime"]["max_context_tokens"],
        "gpu_memory_utilization": generation["gpu_memory_utilization"],
        "max_num_seqs": generation["max_num_seqs"],
        "trust_remote_code": generation["trust_remote_code"],
        "seed": generation["seed"],
        "disable_log_stats": True,
    }
    limits = _multimodal_limits(context.manifest)
    if limits is not None:
        kwargs["limit_mm_per_prompt"] = limits
    return LLM(**kwargs)


def _sampling_params(context: _E3Context):
    from vllm import SamplingParams

    generation = context.config["generation"]
    return SamplingParams(
        temperature=generation["temperature"],
        top_p=1,
        top_k=0,
        max_tokens=generation["max_output_tokens"],
        seed=generation["seed"],
    )


def _render_e3_requests(
    context: _E3Context,
) -> tuple[list[dict[str, Any]], Any]:
    if context.requests_path is None or context.dataset_manifest is None:
        raise ValueError("E3 rendering requires a prepared dataset manifest")
    renderer, tokenizer = _load_renderer(context)
    rendered: list[dict[str, Any]] = []
    fingerprint_rows: list[tuple[str, str, int]] = []
    reserve = context.config["generation"]["max_output_tokens"]
    max_context = context.config["context_policy"]["max_context_tokens"]
    for request in read_jsonl(context.requests_path):
        prompt, tokens = _render_prompt(context, renderer, tokenizer, request)
        if tokens + reserve > max_context:
            raise ValueError(f"{request['request_id']}: E3 request exceeds context")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        fingerprint_rows.append((request["request_id"], prompt_hash, tokens))
        rendered.append(
            {
                "request": request,
                "rendered_prompt": prompt,
                "prompt_tokens_preflight": tokens,
                "response_prefix": context.wrapper.get("assistant_prefill", ""),
            }
        )
    audit_rows = {
        row["manifest_id"]: row
        for row in context.dataset_manifest["token_audit"]
    }
    expected = audit_rows.get(context.manifest["manifest_id"])
    if expected is None:
        raise ValueError("E3 dataset manifest lacks this model token audit")
    lengths = [row["prompt_tokens_preflight"] for row in rendered]
    observed = {
        "requests": len(rendered),
        "minimum_prompt_tokens": min(lengths),
        "mean_prompt_tokens": fmean(lengths),
        "p95_prompt_tokens": _percentile(lengths, 0.95),
        "maximum_prompt_tokens": max(lengths),
        "rendered_prompt_fingerprint": _prompt_fingerprint(fingerprint_rows),
    }
    for field, value in observed.items():
        if expected.get(field) != value:
            raise ValueError(f"E3 rendered prompt audit mismatch: {field}")
    return rendered, tokenizer


def run_e3_model(
    config_path: str | Path,
    *,
    manifest_id: str,
    dataset_manifest_path: str | Path,
    output_responses: str | Path,
    output_audit: str | Path,
) -> dict[str, Any]:
    context = _load_model_context(
        config_path,
        manifest_id=manifest_id,
        dataset_manifest_path=dataset_manifest_path,
    )
    rendered, _tokenizer = _render_e3_requests(context)
    sampler = _NvmlSampler()
    sampler.start()
    llm = None
    try:
        load_started = time.perf_counter()
        llm = _load_vllm(context)
        load_seconds = time.perf_counter() - load_started
        sampler.mark_static()
        inference_started = time.perf_counter()
        generated = llm.generate(
            [row["rendered_prompt"] for row in rendered],
            _sampling_params(context),
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
                parsed_prediction = parse_ext_notes_output(
                    raw_output, runtime_failed=runtime_failed
                )
                parsed = parsed_prediction.as_dict()
                parsed["parse_valid"] = parsed_prediction.failure not in {
                    "parse_failure",
                    "runtime_failure",
                }
                predictions = _prediction_heads(parsed)
                prompt_tokens = len(getattr(result, "prompt_token_ids", []) or [])
                output_tokens = 0 if choice is None else len(choice.token_ids)
                total_prompt_tokens += prompt_tokens
                total_output_tokens += output_tokens
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request["request_id"],
                    "note_cluster": request["note_cluster"],
                    "admission_cluster": request["admission_cluster"],
                    "subject_cluster": request["subject_cluster"],
                    "semantic_group": request["semantic_group"],
                    "context_mode": request["context_mode"],
                    "expected": request["expected"],
                    "prompt_sha256": hashlib.sha256(
                        rendered_request["rendered_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "raw_output": raw_output,
                    "generated_continuation": (
                        continuation if rendered_request["response_prefix"] else None
                    ),
                    "response_prefix": rendered_request["response_prefix"],
                    "runtime_failed": runtime_failed,
                    "finish_reason": (
                        "missing_output" if choice is None else str(choice.finish_reason)
                    ),
                    "parsed": parsed,
                    "predictions": predictions,
                    "head_correct": {
                        "detection": predictions["detection"]
                        == request["expected"]["detection"],
                        "encounter": (
                            predictions["encounter"]
                            == request["expected"]["encounter"]
                            if request["expected"]["detection"] == "yes"
                            else None
                        ),
                        "negation": (
                            predictions["negation"]
                            == request["expected"]["negation"]
                            if request["expected"]["detection"] == "yes"
                            else None
                        ),
                    },
                    "hierarchical_joint_correct": _hierarchical_correct(
                        request["expected"], parsed
                    ),
                }
                records.append(record)
                write_jsonl_row(output, record)
        overall = _score_e3_rows(records)
        semantic_groups = sorted({row["semantic_group"] for row in records})
        context_modes = sorted({row["context_mode"] for row in records})
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": "slm_benchmark_e3_model",
            "experiment_id": context.config["experiment_id"],
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "artifact_sha256": context.manifest["verification"]["artifact_sha256"],
            "e3_config_sha256": config_sha256(context.config),
            "benchmark_config_sha256": config_sha256(context.benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "requests_sha256": context.dataset_manifest["output_requests_sha256"],
            "runtime_version": context.manifest["verification"]["runtime_version"],
            "generation": {
                "temperature": context.config["generation"]["temperature"],
                "seed": context.config["generation"]["seed"],
                "constrained_decoding": False,
                "max_output_tokens": context.config["generation"][
                    "max_output_tokens"
                ],
                "thinking_mode": context.manifest["runtime"]["thinking_mode"],
                "multimodal_limits": _multimodal_limits(context.manifest),
            },
            "technical_completion": overall["runtime_completed"] == len(rendered),
            "metrics": {
                "overall": overall,
                "by_semantic_group": {
                    group: _score_e3_rows(
                        [row for row in records if row["semantic_group"] == group]
                    )
                    for group in semantic_groups
                },
                "by_context_mode": {
                    mode: _score_e3_rows(
                        [row for row in records if row["context_mode"] == mode]
                    )
                    for mode in context_modes
                },
            },
            "throughput": {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "requests_per_second": (
                    len(rendered) / inference_seconds if inference_seconds else None
                ),
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
            "privacy": {
                "raw_note_text_in_audit": False,
                "source_identifiers_in_audit": False,
                "responses_restricted_local": True,
            },
            "claim_boundary": context.config["claim_boundary"],
            "authorization_boundary": context.config["authorization_boundary"],
        }
    except Exception as exc:
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": "slm_benchmark_e3_model",
            "experiment_id": context.config["experiment_id"],
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "e3_config_sha256": config_sha256(context.config),
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


def _primary_endpoints(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    detection_counts: Counter[tuple[str, str]] = Counter()
    encounter_counts: Counter[tuple[str, str]] = Counter()
    negation_counts: Counter[tuple[str, str]] = Counter()
    joint_correct = 0
    for row in rows:
        expected = row["expected"]
        predictions = row["predictions"]
        detection_counts[(expected["detection"], predictions["detection"])] += 1
        if expected["detection"] == "yes":
            encounter_counts[(expected["encounter"], predictions["encounter"])] += 1
            negation_counts[(expected["negation"], predictions["negation"])] += 1
        joint_correct += bool(row["hierarchical_joint_correct"])

    def summarize(
        counts: Mapping[tuple[str, str], int], labels: Sequence[str]
    ) -> tuple[float, float]:
        f1_values: list[float] = []
        recalls: list[float] = []
        for label in labels:
            tp = counts.get((label, label), 0)
            fp = sum(
                count
                for (gold, prediction), count in counts.items()
                if gold != label and prediction == label
            )
            support = sum(
                count for (gold, _prediction), count in counts.items() if gold == label
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            f1_values.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            recalls.append(recall)
        return fmean(f1_values), fmean(recalls)

    detection_macro_f1, detection_balanced = summarize(
        detection_counts, ["yes", "no"]
    )
    encounter_macro_f1, _encounter_balanced = summarize(
        encounter_counts, ["yes", "no"]
    )
    negation_macro_f1, _negation_balanced = summarize(
        negation_counts, ["yes", "no", "unsure"]
    )
    return {
        "detection_macro_f1": detection_macro_f1,
        "detection_balanced_accuracy": detection_balanced,
        "encounter_macro_f1": encounter_macro_f1,
        "negation_macro_f1": negation_macro_f1,
        "hierarchical_joint_exact_match": (
            joint_correct / len(rows) if rows else 0.0
        ),
    }


def _mcnemar_exact(a_only: int, b_only: int) -> float | None:
    discordant = a_only + b_only
    if discordant == 0:
        return None
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * tail)


def _bootstrap_endpoint_differences(
    model_a: Mapping[str, Mapping[str, Any]],
    model_b: Mapping[str, Mapping[str, Any]],
    *,
    cluster_field: str,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    request_ids = sorted(model_a)
    if set(request_ids) != set(model_b):
        raise ValueError("paired E3 response request IDs differ")
    clusters: dict[str, list[str]] = defaultdict(list)
    for request_id in request_ids:
        a_cluster = str(model_a[request_id][cluster_field])
        b_cluster = str(model_b[request_id][cluster_field])
        if a_cluster != b_cluster:
            raise ValueError("paired E3 response clusters differ")
        clusters[a_cluster].append(request_id)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_ids = [
            request_id for cluster_id in sampled for request_id in clusters[cluster_id]
        ]
        a_values = _primary_endpoints([model_a[value] for value in sampled_ids])
        b_values = _primary_endpoints([model_b[value] for value in sampled_ids])
        for endpoint in a_values:
            differences[endpoint].append(b_values[endpoint] - a_values[endpoint])
    intervals: dict[str, list[float]] = {}
    for endpoint, values in differences.items():
        ordered = sorted(values)
        lower_index = max(0, math.floor(0.025 * len(ordered)))
        upper_index = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
        intervals[endpoint] = [ordered[lower_index], ordered[upper_index]]
    return intervals


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (identifier, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[identifier] = running
    return adjusted


def aggregate_e3_results(
    config_path: str | Path,
    *,
    dataset_manifest_path: str | Path,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source, config, benchmark_path, benchmark, prompt_path, prompt = _load_config(
        config_path
    )
    dataset_manifest_file = Path(dataset_manifest_path)
    dataset_manifest = _read_object(dataset_manifest_file, "E3 dataset manifest")
    if dataset_manifest.get("e3_config_sha256") != config_sha256(config):
        raise ValueError("E3 dataset manifest config hash mismatch")
    requests_path = Path(dataset_manifest["output_requests"])
    if _sha256_file(requests_path) != dataset_manifest["output_requests_sha256"]:
        raise ValueError("E3 aggregate request hash mismatch")
    requests = list(read_jsonl(requests_path))
    request_by_id = {row["request_id"]: row for row in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("E3 prepared requests contain duplicate request IDs")
    manifests = _manifest_by_id(benchmark_path, benchmark)
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    responses_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    expected_hashes = {
        "e3_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config_sha256(benchmark),
        "prompt_sha256": config_sha256(prompt),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_file),
        "requests_sha256": dataset_manifest["output_requests_sha256"],
    }
    for manifest_id in config["panel"]:
        manifest_path, manifest = manifests[manifest_id]
        wrapper_path = _resolve(
            manifest_path.parent,
            manifest["model"]["wrapper_spec"],
            "wrapper_spec",
        )
        audit_path = root / manifest_id / "audit.json"
        audit = _read_object(audit_path, "E3 model audit")
        if audit.get("manifest_id") != manifest_id:
            raise ValueError(f"{audit_path}: manifest ID mismatch")
        for field, expected in expected_hashes.items():
            if audit.get(field) != expected:
                raise ValueError(f"{audit_path}: stale {field}")
        if audit.get("manifest_sha256") != config_sha256(manifest):
            raise ValueError(f"{audit_path}: stale manifest hash")
        if audit.get("wrapper_sha256") != _sha256_file(wrapper_path):
            raise ValueError(f"{audit_path}: stale wrapper hash")
        response_path = Path(audit["output_responses"])
        responses = {row["request_id"]: row for row in read_jsonl(response_path)}
        if set(responses) != set(request_by_id):
            raise ValueError(f"{manifest_id}: E3 response request IDs differ")
        for request_id, response in responses.items():
            request = request_by_id[request_id]
            for field in (
                "note_cluster",
                "admission_cluster",
                "subject_cluster",
                "semantic_group",
                "context_mode",
                "expected",
            ):
                if response.get(field) != request.get(field):
                    raise ValueError(f"{manifest_id}: response {field} mismatch")
        recomputed = _score_e3_rows(list(responses.values()))
        if recomputed != audit["metrics"]["overall"]:
            raise ValueError(f"{manifest_id}: stored E3 metrics are stale")
        responses_by_model[manifest_id] = responses
        rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": manifest["model"]["model_id"],
                "panel_role": manifest["factors"]["panel_role"],
                "family": manifest["factors"]["family"],
                "size_class": manifest["factors"]["size_class"],
                "instruction_tuned": manifest["factors"]["instruction_tuned"],
                "medical_adapted": manifest["factors"]["medical_adapted"],
                "technical_completion": audit["technical_completion"],
                "metrics": audit["metrics"],
                "throughput": audit["throughput"],
                "gpu_memory": audit["gpu_memory"],
                "audit_path": str(audit_path),
            }
        )

    stats = config["statistics"]
    contrasts: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(benchmark["contrasts"]):
        model_a_id, model_b_id = contrast["models"]
        model_a = responses_by_model[model_a_id]
        model_b = responses_by_model[model_b_id]
        a_rows = [model_a[request_id] for request_id in sorted(model_a)]
        b_rows = [model_b[request_id] for request_id in sorted(model_b)]
        a_values = _primary_endpoints(a_rows)
        b_values = _primary_endpoints(b_rows)
        subject_intervals = _bootstrap_endpoint_differences(
            model_a,
            model_b,
            cluster_field="subject_cluster",
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]) + 20 * contrast_index,
        )
        note_intervals = _bootstrap_endpoint_differences(
            model_a,
            model_b,
            cluster_field="note_cluster",
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]) + 20 * contrast_index + 1,
        )
        request_ids = sorted(model_a)
        a_only = sum(
            bool(model_a[value]["hierarchical_joint_correct"])
            and not bool(model_b[value]["hierarchical_joint_correct"])
            for value in request_ids
        )
        b_only = sum(
            bool(model_b[value]["hierarchical_joint_correct"])
            and not bool(model_a[value]["hierarchical_joint_correct"])
            for value in request_ids
        )
        exact_p = _mcnemar_exact(a_only, b_only)
        contrasts.append(
            {
                "contrast_id": contrast["contrast_id"],
                "claim_type": contrast["claim_type"],
                "factor": contrast["factor"],
                "model_a": model_a_id,
                "model_b": model_b_id,
                "direction": "model_b minus model_a",
                "n": len(request_ids),
                "endpoints": {
                    endpoint: {
                        "model_a": a_values[endpoint],
                        "model_b": b_values[endpoint],
                        "paired_difference": b_values[endpoint] - a_values[endpoint],
                        "subject_cluster_bootstrap_95_ci": subject_intervals[
                            endpoint
                        ],
                        "note_cluster_bootstrap_95_ci": note_intervals[endpoint],
                    }
                    for endpoint in a_values
                },
                "hierarchical_joint_mcnemar": {
                    "model_a_only_correct": a_only,
                    "model_b_only_correct": b_only,
                    "exact_two_sided_p": exact_p,
                    "exact_two_sided_p_display": (
                        "<5e-324" if exact_p == 0.0 else None
                    ),
                    "holm_adjusted_p": None,
                    "holm_adjusted_p_display": None,
                    "confirmatory": contrast["contrast_id"]
                    in stats["confirmatory_contrast_ids"],
                },
            }
        )
    confirmatory_p = {
        row["contrast_id"]: row["hierarchical_joint_mcnemar"][
            "exact_two_sided_p"
        ]
        for row in contrasts
        if row["hierarchical_joint_mcnemar"]["confirmatory"]
        and row["hierarchical_joint_mcnemar"]["exact_two_sided_p"] is not None
    }
    adjusted = _holm_adjust(confirmatory_p)
    for row in contrasts:
        if row["contrast_id"] in adjusted:
            adjusted_p = adjusted[row["contrast_id"]]
            row["hierarchical_joint_mcnemar"]["holm_adjusted_p"] = adjusted_p
            if adjusted_p == 0.0:
                row["hierarchical_joint_mcnemar"][
                    "holm_adjusted_p_display"
                ] = "<5e-324"

    baseline = _baseline_metrics(requests)
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e3_aggregate",
        "experiment_id": config["experiment_id"],
        "status": "PRIMARY_E3",
        "task_name": "MetaMap-candidate validation and contextual mention grounding",
        "e3_config": str(source),
        "e3_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config_sha256(benchmark),
        "prompt_sha256": config_sha256(prompt),
        "dataset_manifest": str(dataset_manifest_file),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_file),
        "dataset": config["dataset"]["name"],
        "dataset_version": config["dataset"]["version"],
        "counts": dataset_manifest["counts"],
        "label_counts": dataset_manifest["label_counts"],
        "context_policy": dataset_manifest["context_policy"],
        "declared_models": len(config["panel"]),
        "all_models_technically_completed": all(
            row["technical_completion"] for row in rows
        ),
        "baselines": baseline,
        "models": rows,
        "paired_contrasts": contrasts,
        "statistics": stats,
        "panel_decision": {
            "retained_manifest_ids": config["panel"],
            "excluded_manifest_ids": [],
            "reason": "E3 preserves the approved panel; technical completion and semantic performance are reported separately.",
        },
        "privacy": {
            "contains_raw_note_text": False,
            "contains_source_identifiers": False,
            "contains_individual_request_rows": False,
            "restricted_source_and_response_artifacts_remain_local": True,
        },
        "claim_boundary": config["claim_boundary"],
        "authorization_boundary": config["authorization_boundary"],
    }
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("note_context", "target_mention", "candidate_concept"):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"E3 aggregate leaked restricted field: {forbidden}")
    write_json(output, result)
    return result
