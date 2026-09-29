"""Matched unconstrained/schema-constrained inference on frozen E3 requests.

No inference occurs on import or during preflight. Run from the repository root
with ``python -m cpg2.slm_e3_contract run-panel``. Completed batches are immutable
and resumable; legacy datasets, configurations and results are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys
import tempfile
import time

from .ext_notes_benchmark import parse_ext_notes_output, validate_ext_notes_labels
from .mimic.csvio import config_sha256, read_jsonl
from .slm_e0 import _multimodal_limits
from .slm_e3 import (
    _hierarchical_correct, _load_model_context, _load_renderer, _prediction_heads,
    _read_object, _render_prompt, _resolve, _sha256_file,
)

DEFAULT_CONFIG = "configs/slm_benchmark/e3_contract_comparison.v1.json"
CONDITIONS = ("unconstrained", "constrained")
RESTRICTED_ROOT = Path(__file__).resolve().parents[2] / "outputs/local_restricted"
METADATA_FIELDS = ("expected", "note_cluster", "admission_cluster", "subject_cluster",
                   "semantic_group", "context_mode")
FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Experiment:
    source: Path
    config: dict
    primary: Path
    dataset: Path
    benchmark: dict
    panel: tuple[str, ...]


def _offline() -> None:
    # Set before loading transformers/vLLM; no patient data may leave this host.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                "VLLM_DO_NOT_TRACK", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def _write_new(path: Path, value, *, jsonl: bool = False) -> None:
    """Publish a complete 0600 artifact atomically, without replacing a file."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic and fails if destination already exists.
    finally:
        temporary.unlink(missing_ok=True)


def _write_matching(path: Path, value, *, jsonl: bool = False) -> None:
    if path.exists():
        previous = list(read_jsonl(path)) if jsonl else _read_object(path, "existing artifact")
        if previous != value:
            raise ValueError(f"existing artifact differs; preserve it and use a fresh output root: {path}")
        if path.stat().st_mode & 0o077:
            raise ValueError(f"artifact permissions must be 0600: {path}")
    else:
        _write_new(path, value, jsonl=jsonl)


@contextmanager
def _lock(root: Path, name: str):
    folder = root / ".locks"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(folder / f"{name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds the {name} lock") from exc
        yield
    finally:
        os.close(descriptor)


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("vllm", "transformers", "torch", "numpy", "xgrammar", "accelerate", "huggingface-hub")}


def _experiment(config_path=DEFAULT_CONFIG) -> Experiment:
    source = Path(config_path).resolve()
    config = _read_object(source, "paired E3 config")
    if config.get("schema_version") != "1.0" or config.get("freeze_status") != "FROZEN":
        raise ValueError("paired E3 configuration must be frozen version 1.0")
    if tuple(config["conditions"]) != CONDITIONS or config["structured_backend"] != "xgrammar":
        raise ValueError("unexpected paired conditions or structured backend")
    if type(config["checkpoint_batch_size"]) is not int or config["checkpoint_batch_size"] <= 0:
        raise ValueError("checkpoint_batch_size must be a positive integer")
    if config.get("engine_overrides") != {"enable_prefix_caching": False, "add_special_tokens": False}:
        raise ValueError("paired engine must disable prefix caching and implicit special-token insertion")
    stats = config["statistics"]
    if (stats["primary_endpoint"] != "strict_hierarchical_joint_exact_match"
            or stats["cluster"] != "subject_cluster" or stats["confidence_level"] != 0.95):
        raise ValueError("unexpected endpoint, cluster or confidence level")
    if _versions() != config["runtime_versions"]:
        raise ValueError("installed packages differ from pinned runtime_versions")
    primary = _resolve(source.parent, config["primary_e3_config"], "primary E3 config").resolve()
    primary_config = _read_object(primary, "primary E3 config")
    if config_sha256(primary_config) != config["primary_e3_config_sha256"]:
        raise ValueError("original E3 configuration changed")
    dataset = _resolve(source.parent, config["dataset_manifest"], "dataset manifest").resolve()
    if _sha256_file(dataset) != config["dataset_manifest_sha256"]:
        raise ValueError("original E3 dataset manifest changed")
    if config["expected_counts"] != primary_config["dataset"]["expected_counts"]:
        raise ValueError("paired expected counts differ from original E3")
    benchmark = _read_object(_resolve(primary.parent, primary_config["benchmark_config"], "benchmark"), "benchmark")
    panel = tuple(primary_config["panel"])
    contrast_ids = [item["contrast_id"] for item in benchmark["contrasts"]]
    core_ids = stats["core_contrast_ids"]
    if len(core_ids) != len(set(core_ids)) or not set(core_ids) <= set(contrast_ids):
        raise ValueError("unknown or duplicate core contrasts")
    return Experiment(source, config, primary, dataset, benchmark, panel)


def _output_root(experiment: Experiment, output_root=None) -> Path:
    root = (Path(output_root) if output_root else _resolve(
        experiment.source.parent, experiment.config["output_root"], "output root")).resolve()
    restricted = RESTRICTED_ROOT.resolve()
    legacy = experiment.dataset.parent.parent.resolve()
    if root == restricted or not root.is_relative_to(restricted):
        raise ValueError("output root must be a subdirectory of outputs/local_restricted")
    if root.is_relative_to(legacy) or legacy.is_relative_to(root):
        raise ValueError("paired outputs must be separate from the legacy E3 dataset/results")
    return root


def _context(experiment: Experiment, manifest_id: str):
    if manifest_id not in experiment.panel:
        raise ValueError("model is not in the frozen paired panel")
    if Path.cwd().resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("run this command from the repository root; legacy request paths are repository-relative")
    context = _load_model_context(experiment.primary, manifest_id=manifest_id,
                                  dataset_manifest_path=experiment.dataset)
    override = experiment.config["wrapper_overrides"].get(manifest_id)
    if override:
        path = _resolve(experiment.source.parent, override["wrapper"], "wrapper").resolve()
        if _sha256_file(path) != override["wrapper_sha256"]:
            raise ValueError("paired wrapper override hash mismatch")
        context = replace(context, wrapper_path=path, wrapper=_read_object(path, "paired wrapper"))
    if (context.wrapper.get("assistant_prefill") or context.wrapper.get("continue_final_message")
            or any(message["role"] == "assistant" for message in context.wrapper.get("messages", []))):
        raise ValueError("both paired conditions must generate a complete object without assistant prefill")
    runtime = context.manifest["runtime"]
    generation = context.config["generation"]
    if (runtime["quantization"] is not None or generation["dtype"] != "bfloat16"
            or generation["temperature"] != 0 or generation["seed"] != 2026
            or generation["max_output_tokens"] != 96
            or context.config["context_policy"]["max_context_tokens"] != 2048):
        raise ValueError("runtime differs from the matched BF16/T0/2048/96 protocol")
    return context


def _render_requests(context) -> list[dict]:
    # Do NOT compare new wrappers to legacy rendered-prompt fingerprints.
    # The request-byte hash is checked by _load_model_context; contexts stay fixed.
    _offline()
    renderer, tokenizer = _load_renderer(context)
    rows = []
    for request in read_jsonl(context.requests_path):
        if validate_ext_notes_labels(request["expected"], allow_abstention=False):
            raise ValueError("prepared E3 request has invalid native reference labels")
        prompt, tokens = _render_prompt(context, renderer, tokenizer, request)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokens + context.config["generation"]["max_output_tokens"] > context.config["context_policy"]["max_context_tokens"]:
            raise ValueError("a frozen note context exceeds the paired context budget; do not truncate it")
        rows.append({"request": request, "rendered_prompt": prompt,
                     "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                     "prompt_token_ids_sha256": config_sha256(token_ids),
                     "prompt_tokens_preflight": tokens})
    ids = [row["request"]["request_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate prepared E3 requests")
    return rows


def _counts(rendered: list[dict]) -> dict:
    return {"requests": len(rendered), **{
        key: len({row["request"][field] for row in rendered}) for key, field in
        (("notes", "note_cluster"), ("admissions", "admission_cluster"), ("subjects", "subject_cluster"))}}


def _provenance(experiment: Experiment, context) -> dict:
    source_dir = Path(__file__).parent
    names = ("slm_e3_contract.py", "slm_e3.py", "ext_notes_benchmark.py", "slm_e0.py",
             "slm_benchmark.py", "mimic/csvio.py")
    return {
        "experiment_id": experiment.config["experiment_id"],
        "experiment_config_sha256": config_sha256(experiment.config),
        "dataset_manifest_sha256": _sha256_file(experiment.dataset),
        "requests_sha256": context.dataset_manifest["output_requests_sha256"],
        "manifest_id": context.manifest["manifest_id"],
        "manifest_sha256": config_sha256(context.manifest),
        "model_revision": context.manifest["model"]["revision"],
        "previously_verified_artifact_sha256": context.manifest["verification"]["artifact_sha256"],
        "wrapper_sha256": _sha256_file(context.wrapper_path),
        "prompt_sha256": config_sha256(context.prompt),
        "schema_sha256": config_sha256(context.prompt["output_schema"]),
        "runtime_versions": _versions(),
        "inference_source_sha256": {name: _sha256_file(source_dir / name) for name in names},
        "structured_backend": experiment.config["structured_backend"],
        "generation": context.config["generation"],
        "decoder_override": "constrained_decoding enabled only in constrained cell",
        "checkpoint_batch_size": experiment.config["checkpoint_batch_size"],
        "engine_overrides": experiment.config["engine_overrides"],
    }


def _model_preflight(experiment: Experiment, context, rendered: list[dict]) -> dict:
    if _counts(rendered) != experiment.config["expected_counts"]:
        raise ValueError("prepared request/cluster counts differ from the frozen dataset")
    lengths = [row["prompt_tokens_preflight"] for row in rendered]
    return {
        "provenance": _provenance(experiment, context), "counts": _counts(rendered),
        "rendered_prompt_fingerprint": config_sha256([
            [row["request"]["request_id"], row["prompt_sha256"], row["prompt_token_ids_sha256"], row["prompt_tokens_preflight"]]
            for row in rendered]),
        "prompt_tokens_min": min(lengths), "prompt_tokens_mean": fmean(lengths),
        "prompt_tokens_max": max(lengths), "output_token_budget": 96,
        "over_context": 0, "complete_object_no_prefill": True,
    }


def preflight(config_path=DEFAULT_CONFIG, *, output_root=None) -> dict:
    _offline()
    experiment = _experiment(config_path)
    root = _output_root(experiment, output_root)
    with _lock(root, "preflight"):
        report = {"experiment_config_sha256": config_sha256(experiment.config),
                  "all_models_fit_context": True, "inference_executed": False,
                  "schema_grammar_and_sampling_api_checked": True, "models": {}}
        for manifest_id in experiment.panel:
            context = _context(experiment, manifest_id)
            if not report["models"]:
                import xgrammar
                xgrammar.Grammar.from_json_schema(json.dumps(context.prompt["output_schema"]))
                for condition in CONDITIONS:
                    _sampling_params(context, condition)
            rendered = _render_requests(context)
            report["models"][manifest_id] = _model_preflight(experiment, context, rendered)
            print(f"preflight {manifest_id}: {len(rendered)} unchanged requests fit 2048 tokens", flush=True)
        _write_matching(root / "preflight.json", report)
    return report


def evaluate_output(raw, expected: dict, *, runtime_failed: bool = False) -> dict:
    def score(text):
        prediction = parse_ext_notes_output(text, runtime_failed=runtime_failed)
        parsed = prediction.as_dict()
        parsed["parse_valid"] = prediction.failure not in {"parse_failure", "runtime_failure"}
        return {"parsed": parsed, "predictions": _prediction_heads(parsed),
                "hierarchical_joint_correct": _hierarchical_correct(expected, parsed)}

    strict = score(raw)
    match = FENCE.fullmatch(raw) if isinstance(raw, str) else None
    normalized = match.group("body") if match else raw
    return {**strict, "fence_normalized": score(normalized), "fence_removed": match is not None}


def _sampling_params(context, condition: str):
    _offline()
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if condition not in CONDITIONS:
        raise ValueError("unknown decoding condition")
    generation = context.config["generation"]
    kwargs = dict(temperature=generation["temperature"], top_p=1, top_k=0,
                  max_tokens=generation["max_output_tokens"], seed=generation["seed"])
    if condition == "constrained":
        kwargs["structured_outputs"] = StructuredOutputsParams(
            json=context.prompt["output_schema"], disable_additional_properties=True)
    return SamplingParams(**kwargs)


def _load_engine(context, backend: str):
    _offline()
    from vllm import LLM

    generation = context.config["generation"]
    kwargs = dict(model=context.manifest["model"]["local_snapshot"],
                  tokenizer=context.manifest["model"]["local_snapshot"], runner="generate",
                  dtype=generation["dtype"], max_model_len=2048,
                  gpu_memory_utilization=generation["gpu_memory_utilization"],
                  max_num_seqs=generation["max_num_seqs"],
                  trust_remote_code=generation["trust_remote_code"], seed=generation["seed"],
                  disable_log_stats=True, enable_prefix_caching=False,
                  structured_outputs_config={"backend": backend})
    limits = _multimodal_limits(context.manifest)
    if limits is not None:
        kwargs["limit_mm_per_prompt"] = limits
    engine = LLM(**kwargs)
    if engine.llm_engine.vllm_config.structured_outputs_config.backend != backend:
        raise RuntimeError("resolved structured-output backend differs from pinned backend")
    return engine


def _hardware() -> dict:
    import torch
    return {"gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda}


def _record(item: dict, result, *, manifest_id: str, condition: str) -> dict:
    request = item["request"]
    # Verify returned prompt ordering; never associate a different note's output.
    if getattr(result, "prompt", item["rendered_prompt"]) != item["rendered_prompt"]:
        raise RuntimeError("engine returned a different prompt or request order")
    actual_tokens = result.prompt_token_ids or []
    if (len(actual_tokens) != item["prompt_tokens_preflight"]
            or config_sha256(actual_tokens) != item["prompt_token_ids_sha256"]):
        raise RuntimeError("engine prompt token IDs differ from tokenizer preflight")
    choice = result.outputs[0] if result.outputs else None
    raw = None if choice is None else choice.text
    return {"request_id": request["request_id"],
            **{key: request[key] for key in METADATA_FIELDS},
            "manifest_id": manifest_id, "condition": condition,
            "prompt_sha256": item["prompt_sha256"],
            "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
            "prompt_tokens_preflight": item["prompt_tokens_preflight"],
            "prompt_tokens": len(actual_tokens),
            "output_tokens": len(choice.token_ids) if choice is not None else 0,
            "raw_output": raw, "runtime_failed": choice is None,
            "finish_reason": str(choice.finish_reason) if choice is not None else "missing_output",
            **evaluate_output(raw, request["expected"], runtime_failed=choice is None)}


def _check_rows(rows: list[dict], rendered: list[dict], provenance: dict, condition: str) -> None:
    ids = [row.get("request_id") for row in rows]
    expected_ids = [item["request"]["request_id"] for item in rendered]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError("duplicate/missing/unexpected or reordered response IDs")
    for row, item in zip(rows, rendered, strict=True):
        request = item["request"]
        if any(row.get(key) != request[key] for key in METADATA_FIELDS):
            raise ValueError("response metadata differs from frozen requests")
        if (row.get("manifest_id") != provenance["manifest_id"] or row.get("condition") != condition
                or row.get("prompt_sha256") != item["prompt_sha256"]
                or row.get("prompt_token_ids_sha256") != item["prompt_token_ids_sha256"]
                or row.get("prompt_tokens_preflight") != item["prompt_tokens_preflight"]):
            raise ValueError("response model/condition/prompt mismatch")
        if row.get("runtime_failed") is not False or not isinstance(row.get("raw_output"), str):
            raise ValueError("runtime-failed responses cannot be treated as completed batches")
        evaluation = evaluate_output(row["raw_output"], request["expected"])
        if any(row.get(key) != value for key, value in evaluation.items()):
            raise ValueError("stored validation or scores differ from raw output")
        for key in ("prompt_tokens", "output_tokens"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("invalid response token count")
        if row["prompt_tokens"] != item["prompt_tokens_preflight"]:
            raise ValueError("response prompt length differs from preflight")
        if not isinstance(row.get("finish_reason"), str):
            raise ValueError("missing response finish reason")


def _check_batch(path: Path, provenance: dict, condition: str, index: int,
                 rendered: list[dict]) -> dict | None:
    if not path.exists():
        return None
    if path.stat().st_mode & 0o077:
        raise ValueError("restricted batch permissions must be 0600")
    batch = _read_object(path, "paired E3 checkpoint")
    if (batch.get("provenance") != provenance or batch.get("condition") != condition
            or batch.get("batch_index") != index or batch.get("technical_completion") is not True):
        raise ValueError("stale or incomplete batch; preserve it and use a fresh output root")
    rows = batch["rows"]
    if batch.get("rows_sha256") != config_sha256(rows):
        raise ValueError("batch response checksum mismatch")
    _check_rows(rows, rendered, provenance, condition)
    return batch


def _batch_specs(rendered: list[dict], size: int):
    return [(index, rendered[start:start + size])
            for index, start in enumerate(range(0, len(rendered), size))]


def _read_batches(folder: Path, specs, provenance: dict, condition: str) -> list[dict | None]:
    expected_names = {f"batch_{index:05d}.json" for index, _ in specs}
    actual_names = {path.name for path in (folder / "batches").glob("batch_*.json")}
    if actual_names - expected_names:
        raise ValueError("unexpected checkpoint batches in cell")
    return [_check_batch(folder / "batches" / f"batch_{index:05d}.json",
                         provenance, condition, index, items) for index, items in specs]


def _finalize_cell(folder: Path, batches: list[dict], provenance: dict, condition: str) -> list[dict]:
    rows = [row for batch in batches for row in batch["rows"]]
    responses = folder / "responses.jsonl"
    _write_matching(responses, rows, jsonl=True)
    audit = {
        "provenance": provenance, "condition": condition, "technical_completion": True,
        "n": len(rows), "batches": len(batches), "responses_sha256": _sha256_file(responses),
        "batch_sha256": [_sha256_file(folder / "batches" / f"batch_{i:05d}.json") for i in range(len(batches))],
        "inference_seconds": sum(batch["inference_seconds"] for batch in batches),
        "hardware": [dict(values) for values in sorted({tuple(sorted(batch["hardware"].items())) for batch in batches})],
        "timing_interpretation": "Descriptive only; fixed order, compilation, caching and interrupted sessions are not controlled for speed comparisons.",
    }
    _write_matching(folder / "audit.json", audit)
    return rows


def run_model(config_path=DEFAULT_CONFIG, *, manifest_id: str, output_root=None) -> None:
    _offline()
    experiment = _experiment(config_path)
    root = _output_root(experiment, output_root)
    if manifest_id not in experiment.panel:
        raise ValueError("model is not in the frozen paired panel")
    with _lock(root, manifest_id):
        context = _context(experiment, manifest_id)
        rendered = _render_requests(context)
        expected_preflight = _model_preflight(experiment, context, rendered)
        report = _read_object(root / "preflight.json", "paired token preflight")
        if (report.get("all_models_fit_context") is not True
                or report.get("experiment_config_sha256") != config_sha256(experiment.config)
                or report.get("models", {}).get(manifest_id) != expected_preflight):
            raise ValueError("preflight missing, stale or not passing; run preflight first")
        provenance = expected_preflight["provenance"]
        specs = _batch_specs(rendered, experiment.config["checkpoint_batch_size"])
        cached = {condition: _read_batches(root / "results" / manifest_id / condition,
                                          specs, provenance, condition) for condition in CONDITIONS}
        # Refuse stale finalized cells before performing any new inference.
        for condition, batches in cached.items():
            folder = root / "results" / manifest_id / condition
            if all(batch is not None for batch in batches):
                _finalize_cell(folder, batches, provenance, condition)
            elif (folder / "audit.json").exists() or (folder / "responses.jsonl").exists():
                raise ValueError("finalized cell has missing checkpoint batches")
        if all(batch is not None for batches in cached.values() for batch in batches):
            print(f"{manifest_id}: both completed conditions verified; no engine loaded", flush=True)
            return
        engine = _load_engine(context, experiment.config["structured_backend"])
        hardware = _hardware()
        for condition in CONDITIONS:
            folder = root / "results" / manifest_id / condition
            params = _sampling_params(context, condition)
            for index, items in specs:
                if cached[condition][index] is not None:
                    continue
                started = time.perf_counter()
                generated = engine.generate([item["rendered_prompt"] for item in items], params,
                                            use_tqdm=False, tokenization_kwargs={"add_special_tokens": False})
                elapsed = time.perf_counter() - started
                if len(generated) != len(items):
                    raise RuntimeError("inference returned the wrong number of responses")
                rows = [_record(item, result, manifest_id=manifest_id, condition=condition)
                        for item, result in zip(items, generated, strict=True)]
                complete = not any(row["runtime_failed"] for row in rows)
                batch = {"provenance": provenance, "condition": condition, "batch_index": index,
                         "technical_completion": complete, "rows": rows,
                         "rows_sha256": config_sha256(rows), "hardware": hardware,
                         "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "inference_seconds": elapsed}
                if not complete:
                    failure_path = folder / "failed_attempts" / f"batch_{index:05d}.{time.time_ns()}.json"
                    _write_new(failure_path, batch)
                    raise RuntimeError("missing engine output; failed attempt preserved locally, rerun to retry this technical failure")
                path = folder / "batches" / f"batch_{index:05d}.json"
                _write_new(path, batch)
                cached[condition][index] = _check_batch(path, provenance, condition, index, items)
                print(f"{manifest_id}/{condition}: batch {index + 1}/{len(specs)} ({len(rows)} responses, {elapsed:.1f}s)", flush=True)
            _finalize_cell(folder, cached[condition], provenance, condition)


def aggregate(config_path=DEFAULT_CONFIG, *, output_root=None) -> dict:
    from .slm_e3_contract_analysis import summarize_paired_e3

    _offline()
    experiment = _experiment(config_path)
    root = _output_root(experiment, output_root)
    panel = {}
    preflight_report = _read_object(root / "preflight.json", "paired token preflight")
    for manifest_id in experiment.panel:
        context = _context(experiment, manifest_id)
        rendered = _render_requests(context)
        observed = _model_preflight(experiment, context, rendered)
        if preflight_report.get("models", {}).get(manifest_id) != observed:
            raise ValueError("aggregate preflight provenance mismatch")
        provenance = observed["provenance"]
        specs = _batch_specs(rendered, experiment.config["checkpoint_batch_size"])
        panel[manifest_id] = {}
        for condition in CONDITIONS:
            folder = root / "results" / manifest_id / condition
            batches = _read_batches(folder, specs, provenance, condition)
            if any(batch is None for batch in batches):
                raise ValueError(f"incomplete cell {manifest_id}/{condition}; aggregate requires the whole panel")
            panel[manifest_id][condition] = _finalize_cell(folder, batches, provenance, condition)
    stats = experiment.config["statistics"]
    contrasts = [item for item in experiment.benchmark["contrasts"] if item["contrast_id"] in stats["core_contrast_ids"]]
    result = summarize_paired_e3(panel, contrasts, bootstrap_replicates=stats["bootstrap_replicates"],
                                bootstrap_seed=stats["bootstrap_seed"])
    result["experiment_config_sha256"] = config_sha256(experiment.config)
    result["analysis_source_sha256"] = _sha256_file(Path(__file__).with_name("slm_e3_contract_analysis.py"))
    result["claim_boundary"] = experiment.config["claim_boundary"]
    path = root / "aggregate/results.json"
    _write_matching(path, result)
    print(f"paired clinical-note analysis saved: {path}", flush=True)
    return result


def run_panel(config_path=DEFAULT_CONFIG, *, output_root=None) -> None:
    _offline()
    experiment = _experiment(config_path)
    root = _output_root(experiment, output_root)
    with _lock(root, "panel"):
        preflight(experiment.source, output_root=root)
        for manifest_id in experiment.panel:
            command = [sys.executable, "-m", "cpg2.slm_e3_contract", "run-model", "--config", str(experiment.source),
                       "--output-root", str(root), "--manifest-id", manifest_id]
            log_path = root / "logs" / f"{manifest_id}.{time.time_ns()}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            print(f"starting {manifest_id}; engine output retained locally in {log_path}", flush=True)
            # Do not stream engine exceptions: they may contain clinical prompts.
            with os.fdopen(descriptor, "w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"model {manifest_id} failed; completed batches are preserved; inspect restricted log {log_path}")
            print(f"completed {manifest_id}: both conditions; moving to next model", flush=True)
        aggregate(experiment.source, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run-model", "run-panel", "aggregate"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", help="Separate directory under outputs/local_restricted")
    parser.add_argument("--manifest-id", help="Required for run-model")
    args = parser.parse_args()
    if args.command == "run-model":
        if not args.manifest_id:
            parser.error("run-model requires --manifest-id")
        run_model(args.config, manifest_id=args.manifest_id, output_root=args.output_root)
    elif args.manifest_id:
        parser.error("--manifest-id is only valid for run-model")
    elif args.command == "preflight":
        preflight(args.config, output_root=args.output_root)
    elif args.command == "aggregate":
        aggregate(args.config, output_root=args.output_root)
    else:
        run_panel(args.config, output_root=args.output_root)


if __name__ == "__main__":
    main()
