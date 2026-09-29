"""Fresh, matched 2x2 output-contract experiment; no changes to legacy runs.

Run from the repository root with ``python -m cpg2.slm_contract_ablation``.
Each model runs in its own subprocess so GPU allocations are released between
models. Completed cells can be resumed only with identical inputs and code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from .mimic.csvio import config_sha256, read_jsonl
from .slm_compact_contract import evaluate_contract_output
from .slm_e0 import _multimodal_limits, _sampling_params
from .slm_e1 import (
    _apply_s2_wrapper, _load_model_context, _read_object, _render_e1_requests,
    _resolve, _sha256_file, _structured_sampling_params,
)

DEFAULT_CONFIG = "configs/slm_benchmark/e1_contract_ablation.v1.json"
CONDITIONS = ("full_unconstrained", "full_constrained", "compact_unconstrained", "compact_constrained")
PROMPT_TOKENIZATION = {"input": "rendered_string", "add_special_tokens": True}


def _write_new(path: Path, data, *, jsonl: bool = False) -> None:
    """Exclusive writes: never replace a previous experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        if jsonl:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("vllm", "transformers", "torch", "numpy", "xgrammar", "accelerate", "huggingface-hub")}


def _experiment(config_path: str | Path):
    source = Path(config_path).resolve()
    config = _read_object(source, "contract ablation config")
    if config.get("freeze_status") != "FROZEN" or config.get("authorization", {}).get("inference") is not True:
        raise ValueError("contract ablation must be frozen and authorized")
    if tuple(config["conditions"]) != CONDITIONS or config["structured_backend"] != "xgrammar":
        raise ValueError("unexpected conditions or unpinned structured backend")
    primary = _resolve(source.parent, config["primary_e1_config"], "primary config")
    if config_sha256(_read_object(primary, "primary config")) != config["primary_e1_config_sha256"]:
        raise ValueError("primary E1 configuration changed")
    dataset = _resolve(source.parent, config["dataset_manifest"], "dataset manifest")
    if _sha256_file(dataset) != config["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest changed")
    compact_path = _resolve(source.parent, config["compact_prompt"], "compact prompt")
    compact = _read_object(compact_path, "compact prompt")
    if config_sha256(compact) != config["compact_prompt_sha256"]:
        raise ValueError("compact prompt hash mismatch")
    primary_config = _read_object(primary, "primary config")
    benchmark = _read_object(_resolve(primary.parent, primary_config["benchmark_config"], "benchmark"), "benchmark")
    panel = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    return source, config, primary, dataset, compact_path, compact, benchmark, panel


def _contexts(experiment, manifest_id: str):
    source, config, primary, dataset, compact_path, compact, _, _ = experiment
    full = _load_model_context(primary, manifest_id=manifest_id, dataset_manifest_path=dataset)
    full = _apply_s2_wrapper(full, sensitivity_path=source, sensitivity=config)
    if full.wrapper.get("assistant_prefill"):
        raise ValueError("all four cells must generate the complete JSON object")
    runtime = full.manifest["runtime"]
    if (runtime["primary_dtype"] != "bfloat16" or runtime["quantization"] is not None
            or runtime["temperature"] != 0 or runtime["seed"] != 2026
            or runtime["max_context_tokens"] != 2048 or runtime["max_output_tokens"]["typed_state"] != 192):
        raise ValueError("runtime differs from the matched BF16/T0/2048/192 protocol")
    return {"full": full, "compact": replace(full, prompt_path=compact_path, prompt=compact)}


def _provenance(experiment, contexts) -> dict:
    _, config, _, dataset, _, _, _, _ = experiment
    full = contexts["full"]
    source_dir = Path(__file__).parent
    code_files = ("slm_contract_ablation.py", "slm_compact_contract.py", "slm_e0.py", "slm_e1.py", "slm_benchmark.py")
    return {
        "experiment_id": config["experiment_id"],
        "experiment_config_sha256": config_sha256(config),
        "dataset_manifest_sha256": _sha256_file(dataset),
        "requests_sha256": full.dataset_manifest["output_requests_sha256"],
        "blueprint_sha256": full.dataset_manifest["output_blueprint_sha256"],
        "manifest_id": full.manifest["manifest_id"],
        "manifest_sha256": config_sha256(full.manifest),
        "model_id": full.manifest["model"]["model_id"],
        "model_revision": full.manifest["model"]["revision"],
        "previously_verified_artifact_sha256": full.manifest["verification"]["artifact_sha256"],
        "wrapper_sha256": _sha256_file(full.wrapper_path),
        "prompt_sha256": {k: config_sha256(v.prompt) for k, v in contexts.items()},
        "runtime_versions": _versions(),
        "inference_source_sha256": {name: _sha256_file(source_dir / name) for name in code_files},
        "structured_backend": config["structured_backend"],
        "runtime": full.manifest["runtime"],
        "engine_protocol": full.benchmark["e0_protocol"],
        "prompt_tokenization": PROMPT_TOKENIZATION,
    }


def _expected_rows(context) -> dict:
    blueprint = _read_object(context.blueprint_path, "blueprint")
    return {request_id: {"expected_state": case["expected_states"][criterion],
                         "case_id": case["case_id"], "domain": case["domain"], "criterion_id": criterion}
            for case in blueprint["cases"] for criterion, request_id in case["request_ids"].items()}


def _render_ablation_requests(context):
    """Count the exact strings sent to vLLM, preserving its historical default.

    E1's shared renderer counts with special tokens disabled. This experiment
    sends strings to ``LLM.generate`` without overriding tokenization, so vLLM
    adds special tokens even when a chat template already contains a BOS token.
    Recount here rather than changing either the prompts or generation call.
    """
    rendered, tokenizer = _render_e1_requests(context)
    for row in rendered:
        count = len(tokenizer.encode(row["rendered_prompt"], add_special_tokens=True))
        row["prompt_tokens_preflight"] = count
        row["over_context"] = count + row["max_output_tokens"] > context.manifest["runtime"]["max_context_tokens"]
    return rendered, tokenizer


def _prompt_accounting(rendered) -> dict:
    return {row["request"]["request_id"]: {
        "prompt_sha256": hashlib.sha256(row["rendered_prompt"].encode()).hexdigest(),
        "prompt_tokens_preflight": row["prompt_tokens_preflight"],
        "max_output_tokens": row["max_output_tokens"],
    } for row in rendered}


def _check_prompt_accounting(row: dict, expected: dict, runtime: dict) -> None:
    count = row.get("prompt_tokens")
    preflight_count = expected.get("prompt_tokens_preflight")
    budget = expected.get("max_output_tokens")
    if (type(count) is not int or type(preflight_count) is not int or preflight_count < 1
            or count != preflight_count
            or type(row.get("prompt_tokens_preflight")) is not int
            or row["prompt_tokens_preflight"] != preflight_count):
        raise ValueError(f"prompt token count differs from preflight: {row.get('request_id')}")
    if row.get("prompt_sha256") != expected.get("prompt_sha256"):
        raise ValueError(f"prompt hash differs from preflight: {row.get('request_id')}")
    if type(budget) is not int or budget != runtime["max_output_tokens"]["typed_state"]:
        raise ValueError("prompt output budget differs from frozen runtime")
    if count + budget > runtime["max_context_tokens"]:
        raise ValueError(f"prompt plus output budget exceeds context: {row.get('request_id')}")


def _preflight_model(root: Path, manifest_id: str, provenance: dict) -> dict:
    audit = _read_object(root / "preflight.json", "token preflight")
    model = audit.get("models", {}).get(manifest_id, {})
    if (audit.get("all_models_fit_context") is not True or model.get("provenance") != provenance
            or provenance.get("prompt_tokenization") != PROMPT_TOKENIZATION):
        raise ValueError("preflight is missing, stale or not passing; select a fresh output root")
    return model


def preflight(config_path=DEFAULT_CONFIG, *, output_root=None) -> dict:
    experiment = _experiment(config_path)
    source, config, _, _, _, _, _, panel = experiment
    root = Path(output_root) if output_root else _resolve(source.parent, config["output_root"], "output root")
    report = {"experiment_config_sha256": config_sha256(config), "models": {}, "all_models_fit_context": True}
    for manifest_id in panel:
        contexts = _contexts(experiment, manifest_id)
        expected = _expected_rows(contexts["full"])
        if len(expected) != config["expected_requests_per_cell"] or len({r["case_id"] for r in expected.values()}) != config["expected_cases"]:
            raise ValueError("unexpected request or vignette counts")
        cells = {}
        for contract, context in contexts.items():
            rendered, _ = _render_ablation_requests(context)
            if len(rendered) != len(expected) or {r["request"]["request_id"] for r in rendered} != set(expected):
                raise ValueError("requests do not match blueprint")
            if any(r["over_context"] for r in rendered):
                raise ValueError(f"over-context request for {manifest_id}/{contract}")
            lengths = [r["prompt_tokens_preflight"] for r in rendered]
            cells[contract] = {"requests": len(rendered), "prompt_tokens_min": min(lengths),
                               "prompt_tokens_mean": fmean(lengths), "prompt_tokens_max": max(lengths),
                               "output_token_budget": 192, "over_context": 0,
                               "prompt_accounting": _prompt_accounting(rendered)}
        report["models"][manifest_id] = {"provenance": _provenance(experiment, contexts), "contracts": cells}
        print(f"preflight {manifest_id}: full/compact fit 2048 tokens", flush=True)
    path = root / "preflight.json"
    if path.exists():
        if _read_object(path, "existing preflight") != report:
            raise ValueError("existing preflight differs; select a fresh output root")
    else:
        _write_new(path, report)
    return report


def _load_engine(context, backend: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    import vllm
    from vllm import LLM
    if f"vllm={vllm.__version__}" not in context.manifest["verification"]["runtime_version"].split(";"):
        raise RuntimeError("installed vLLM differs from frozen model runtime")
    runtime, protocol = context.manifest["runtime"], context.benchmark["e0_protocol"]
    kwargs = dict(model=context.manifest["model"]["local_snapshot"],
                  tokenizer=context.manifest["model"]["local_snapshot"], runner="generate",
                  dtype=runtime["primary_dtype"], max_model_len=runtime["max_context_tokens"],
                  gpu_memory_utilization=protocol["gpu_memory_utilization"], max_num_seqs=protocol["max_num_seqs"],
                  trust_remote_code=protocol["trust_remote_code"], seed=runtime["seed"],
                  structured_outputs_config={"backend": backend})
    limits = _multimodal_limits(context.manifest)
    if limits is not None:
        kwargs["limit_mm_per_prompt"] = limits
    engine = LLM(**kwargs)
    resolved = engine.llm_engine.vllm_config.structured_outputs_config.backend
    if resolved != backend:
        raise RuntimeError(f"structured backend mismatch: {resolved!r}")
    return engine


def _check_cell(folder: Path, provenance: dict, condition: str, expected: dict,
                *, prompt_accounting: dict) -> list[dict] | None:
    audit_path, responses_path = folder / "audit.json", folder / "responses.jsonl"
    if not audit_path.exists() and not responses_path.exists():
        return None
    if not audit_path.exists() or not responses_path.exists():
        raise ValueError(f"incomplete artifacts in {folder}; preserve them and select a fresh output root")
    audit = _read_object(audit_path, "cell audit")
    if audit.get("provenance") != provenance or audit.get("condition") != condition or audit.get("technical_completion") is not True:
        raise ValueError(f"stale or incomplete cell: {folder}")
    if audit.get("responses_sha256") != _sha256_file(responses_path):
        raise ValueError(f"response checksum mismatch: {folder}")
    rows = list(read_jsonl(responses_path))
    ids = [r["request_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"duplicate/missing/unexpected responses: {folder}")
    if set(prompt_accounting) != set(expected):
        raise ValueError(f"preflight requests do not match blueprint: {folder}")
    for row in rows:
        _check_prompt_accounting(row, prompt_accounting[row["request_id"]], provenance["runtime"])
        if any(row.get(key) != value for key, value in expected[row["request_id"]].items()):
            raise ValueError(f"response/blueprint mismatch: {folder}")
        if row.get("condition") != condition or row.get("manifest_id") != provenance["manifest_id"]:
            raise ValueError(f"response condition/model mismatch: {folder}")
        evaluation = evaluate_contract_output(row.get("raw_output"), contract=condition.split("_", 1)[0],
                                              expected_criterion_id=row["criterion_id"],
                                              runtime_failed=row.get("runtime_failed", True))
        if any(row.get(key) != value for key, value in evaluation.items()):
            raise ValueError(f"stored validation differs from raw response: {folder}")
        parsed = evaluation["parsed"]
        correct = bool(parsed["logical_valid"] and not parsed["abstained"] and parsed["state"] == row["expected_state"])
        if (row.get("state_correct") is not correct
                or row.get("state_only_correct") is not (evaluation["state_only_state"] == row["expected_state"])
                or row.get("runtime_failed") is not False):
            raise ValueError(f"stored scoring/completion mismatch: {folder}")
    return rows


def run_model(config_path=DEFAULT_CONFIG, *, manifest_id, output_root=None) -> None:
    experiment = _experiment(config_path)
    source, config, _, _, _, _, _, _ = experiment
    root = Path(output_root) if output_root else _resolve(source.parent, config["output_root"], "output root")
    contexts = _contexts(experiment, manifest_id)
    provenance = _provenance(experiment, contexts)
    model_preflight = _preflight_model(root, manifest_id, provenance)
    expected = _expected_rows(contexts["full"])
    rendered = {contract: _render_ablation_requests(context)[0] for contract, context in contexts.items()}
    if any(row["over_context"] for rows in rendered.values() for row in rows):
        raise ValueError("over-context requests at runtime")
    accounting = {contract: _prompt_accounting(rows) for contract, rows in rendered.items()}
    for contract, rows in rendered.items():
        if len(rows) != len(expected) or set(accounting[contract]) != set(expected):
            raise ValueError("requests do not match blueprint")
        if model_preflight.get("contracts", {}).get(contract, {}).get("prompt_accounting") != accounting[contract]:
            raise ValueError("rendered prompts or token counts differ from preflight; select a fresh output root")
    pending = [condition for condition in CONDITIONS
               if _check_cell(root / "results" / manifest_id / condition, provenance, condition, expected,
                              prompt_accounting=accounting[condition.split("_", 1)[0]]) is None]
    if not pending:
        print(f"{manifest_id}: all four cells already verified", flush=True)
        return
    engine = _load_engine(contexts["full"], config["structured_backend"])
    import torch
    hardware = {"gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda}
    for condition in pending:
        contract, decoder = condition.split("_", 1)
        context = contexts[contract]
        params = _structured_sampling_params(context) if decoder == "constrained" else _sampling_params(context, 192)
        started = time.perf_counter()
        generated = engine.generate([row["rendered_prompt"] for row in rendered[contract]], params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if len(generated) != len(rendered[contract]):
            raise RuntimeError("inference returned the wrong number of responses")
        records = []
        for item, result in zip(rendered[contract], generated, strict=True):
            request = item["request"]
            returned_prompt = getattr(result, "prompt", None)
            if returned_prompt is not None and returned_prompt != item["rendered_prompt"]:
                raise RuntimeError("inference returned a prompt different from the corresponding request")
            choice = result.outputs[0] if result.outputs else None
            raw = None if choice is None else choice.text
            evaluation = evaluate_contract_output(raw, contract=contract, expected_criterion_id=request["criterion_id"], runtime_failed=choice is None)
            parsed = evaluation["parsed"]
            gold = expected[request["request_id"]]["expected_state"]
            record = {
                "request_id": request["request_id"], **expected[request["request_id"]],
                "manifest_id": manifest_id, "condition": condition,
                "prompt_sha256": hashlib.sha256(item["rendered_prompt"].encode()).hexdigest(),
                "prompt_tokens": len(result.prompt_token_ids or []),
                "prompt_tokens_preflight": item["prompt_tokens_preflight"],
                "output_tokens": 0 if choice is None else len(choice.token_ids),
                "raw_output": raw, **evaluation,
                "runtime_failed": choice is None,
                "finish_reason": "missing_output" if choice is None else str(choice.finish_reason),
                "state_correct": bool(parsed["logical_valid"] and not parsed["abstained"] and parsed["state"] == gold),
                "state_only_correct": evaluation["state_only_state"] == gold,
            }
            _check_prompt_accounting(record, accounting[contract][request["request_id"]], context.manifest["runtime"])
            records.append(record)
        folder = root / "results" / manifest_id / condition
        responses = folder / "responses.jsonl"
        _write_new(responses, records, jsonl=True)
        _write_new(folder / "audit.json", {
            "provenance": provenance, "condition": condition, "n": len(records),
            "technical_completion": not any(r["runtime_failed"] for r in records),
            "resolved_structured_backend": config["structured_backend"] if decoder == "constrained" else None,
            "structured_schema_sha256": config_sha256(context.prompt["output_schema"]) if decoder == "constrained" else None,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "inference_seconds": elapsed, "responses_sha256": _sha256_file(responses),
            "timing_interpretation": "Diagnostic only; includes backend compilation/cache effects and is not a scale or deployment benchmark.",
        })
        print(f"completed {manifest_id}/{condition}: {len(records)} responses in {elapsed:.1f}s", flush=True)


def aggregate(config_path=DEFAULT_CONFIG, *, output_root=None) -> dict:
    from .slm_contract_analysis import summarize_ablation
    experiment = _experiment(config_path)
    source, config, _, _, _, _, benchmark, panel_ids = experiment
    root = Path(output_root) if output_root else _resolve(source.parent, config["output_root"], "output root")
    panel = {}
    for manifest_id in panel_ids:
        contexts = _contexts(experiment, manifest_id)
        provenance = _provenance(experiment, contexts)
        model_preflight = _preflight_model(root, manifest_id, provenance)
        expected = _expected_rows(contexts["full"])
        panel[manifest_id] = {}
        for condition in CONDITIONS:
            rows = _check_cell(root / "results" / manifest_id / condition, provenance, condition, expected,
                               prompt_accounting=model_preflight.get("contracts", {}).get(
                                   condition.split("_", 1)[0], {}).get("prompt_accounting", {}))
            if rows is None:
                raise ValueError(f"missing cell {manifest_id}/{condition}")
            panel[manifest_id][condition] = rows
    result = summarize_ablation(panel, benchmark["contrasts"],
                               bootstrap_replicates=config["statistics"]["bootstrap_replicates"],
                               bootstrap_seed=config["statistics"]["bootstrap_seed"])
    result["experiment_config_sha256"] = config_sha256(config)
    result["analysis_source_sha256"] = _sha256_file(Path(__file__).with_name("slm_contract_analysis.py"))
    path = root / "aggregate" / "results.json"
    if path.exists():
        if _read_object(path, "existing analysis") != result:
            raise ValueError("analysis output already exists with different contents")
    else:
        _write_new(path, result)
    print(f"analysis saved: {path}", flush=True)
    return result


def run_panel(config_path=DEFAULT_CONFIG, *, output_root=None) -> None:
    experiment = _experiment(config_path)
    source, config, _, _, _, _, _, panel = experiment
    root = (Path(output_root) if output_root else _resolve(source.parent, config["output_root"], "output root")).resolve()
    preflight(source, output_root=root)
    for manifest_id in panel:
        cmd = [sys.executable, "-m", "cpg2.slm_contract_ablation", "run-model", "--config", str(source),
               "--output-root", str(root), "--manifest-id", manifest_id]
        print(f"starting {manifest_id}", flush=True)
        log_path = root / "logs" / f"{manifest_id}.{time.time_ns()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as log:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f"model {manifest_id} failed; see {log_path}")
    aggregate(source, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run-model", "run-panel", "aggregate"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root")
    parser.add_argument("--manifest-id")
    args = parser.parse_args()
    kwargs = {"output_root": args.output_root}
    if args.command == "run-model":
        if not args.manifest_id:
            parser.error("run-model requires --manifest-id")
        run_model(args.config, manifest_id=args.manifest_id, **kwargs)
    elif args.command == "preflight":
        preflight(args.config, **kwargs)
    elif args.command == "aggregate":
        aggregate(args.config, **kwargs)
    else:
        run_panel(args.config, **kwargs)


if __name__ == "__main__":
    main()
