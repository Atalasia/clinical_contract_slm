from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ext_notes_benchmark import parse_ext_notes_output
from .io import write_json
from .mimic.csvio import config_sha256, restricted_jsonl_writer, write_jsonl_row
from .slm_benchmark import TASKS, validate_benchmark_config
from .states import CriterionState


TYPED_OUTPUT_KEYS = frozenset(
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
TYPED_STATES = frozenset(
    {
        CriterionState.PRESENT_CURRENT.value,
        CriterionState.ABSENT_EXPLICIT.value,
        CriterionState.HISTORICAL.value,
        CriterionState.FAMILY_HISTORY_OR_OTHER_EXPERIENCER.value,
        CriterionState.UNCERTAIN.value,
        CriterionState.NOT_DOCUMENTED.value,
    }
)
STATE_REASON_CODES = {
    CriterionState.PRESENT_CURRENT.value: "explicit_present",
    CriterionState.ABSENT_EXPLICIT.value: "explicit_negation",
    CriterionState.HISTORICAL.value: "historical",
    CriterionState.FAMILY_HISTORY_OR_OTHER_EXPERIENCER.value: "other_experiencer",
    CriterionState.UNCERTAIN.value: "ambiguous",
    CriterionState.NOT_DOCUMENTED.value: "not_documented",
}
REASON_CODES = frozenset({*STATE_REASON_CODES.values(), "model_abstention"})


@dataclass(frozen=True)
class TypedE0Prediction:
    state: str | None
    parse_valid: bool
    schema_valid: bool
    logical_valid: bool
    abstained: bool
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _typed_failure(kind: str, *, parse_valid: bool = False) -> TypedE0Prediction:
    return TypedE0Prediction(
        state=None,
        parse_valid=parse_valid,
        schema_valid=False,
        logical_valid=False,
        abstained=False,
        failure=kind,
    )


def parse_typed_e0_output(
    raw_output: str | bytes | None,
    *,
    expected_criterion_id: str,
    runtime_failed: bool = False,
) -> TypedE0Prediction:
    """Parse the frozen typed-state contract while separating syntax and logic."""

    if runtime_failed or raw_output is None:
        return _typed_failure("runtime_failure")
    try:
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8")
        payload = json.loads(raw_output)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return _typed_failure("parse_failure")
    if not isinstance(payload, dict) or set(payload) != TYPED_OUTPUT_KEYS:
        return _typed_failure("schema_failure", parse_valid=True)
    confidence = payload.get("confidence")
    state = payload.get("state")
    reason_code = payload.get("reason_code")
    schema_valid = (
        payload.get("criterion_id") == expected_criterion_id
        and isinstance(state, str)
        and state in TYPED_STATES
        and payload.get("value") is None
        and payload.get("unit") is None
        and payload.get("evidence") == []
        and isinstance(reason_code, str)
        and reason_code in REASON_CODES
        and isinstance(payload.get("abstain"), bool)
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and 0 <= confidence <= 1
    )
    if not schema_valid:
        return _typed_failure("schema_failure", parse_valid=True)
    abstained = payload["abstain"]
    if abstained:
        logical_valid = (
            state == CriterionState.NOT_DOCUMENTED.value
            and reason_code == "model_abstention"
            and confidence == 0
        )
        return TypedE0Prediction(
            state=None,
            parse_valid=True,
            schema_valid=True,
            logical_valid=logical_valid,
            abstained=True,
            failure=None if logical_valid else "logical_failure",
        )
    logical_valid = reason_code == STATE_REASON_CODES[state]
    return TypedE0Prediction(
        state=state,
        parse_valid=True,
        schema_valid=True,
        logical_valid=logical_valid,
        abstained=False,
        failure=None if logical_valid else "logical_failure",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _E0Context:
    benchmark_path: Path
    benchmark: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    wrapper_path: Path
    wrapper: dict[str, Any]
    gate_path: Path
    gate: dict[str, Any]
    prompts: dict[str, tuple[Path, dict[str, Any]]]


def _load_context(
    benchmark_path: str | Path,
    *,
    manifest_id: str,
    suite: str,
) -> _E0Context:
    source = Path(benchmark_path)
    report = validate_benchmark_config(source)
    if not report["inference_ready"]:
        raise ValueError(
            "benchmark is not inference-ready: "
            + "; ".join([*report["errors"], *report["blockers"]])
        )
    benchmark = _read_object(source, "benchmark config")
    authorization = benchmark["authorization"]
    if "E0" not in authorization.get("authorized_experiments", []):
        raise ValueError("E0 inference is not authorized")
    if suite not in {"development", "compatibility_gate"}:
        raise ValueError("suite must be development or compatibility_gate")

    base = source.parent
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    for index, value in enumerate(benchmark["model_manifests"]):
        candidate = _resolve(base, value, f"model_manifests[{index}]")
        loaded = _read_object(candidate, "model manifest")
        if loaded.get("manifest_id") == manifest_id:
            manifest_path, manifest = candidate, loaded
            break
    if manifest_path is None or manifest is None:
        raise ValueError(f"unknown manifest_id: {manifest_id}")

    wrapper_path = _resolve(
        manifest_path.parent,
        manifest["model"]["wrapper_spec"],
        "model.wrapper_spec",
    )
    wrapper = _read_object(wrapper_path, "wrapper spec")
    expected_wrapper_hash = manifest["verification"]["wrapper_spec_sha256"]
    if _sha256_file(wrapper_path) != expected_wrapper_hash:
        raise ValueError(f"wrapper hash does not match frozen manifest: {wrapper_path}")

    gate_path = _resolve(base, benchmark["gate_suites"][suite], f"gate_suites.{suite}")
    gate = _read_object(gate_path, "gate suite")
    if any(case.get("synthetic") is not True for case in gate.get("cases", [])):
        raise ValueError("E0 may run only fully synthetic gate cases")

    prompts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for task in TASKS:
        prompt_path = _resolve(base, benchmark["prompts"][task], f"prompts.{task}")
        prompts[task] = (prompt_path, _read_object(prompt_path, "prompt spec"))
    return _E0Context(
        benchmark_path=source,
        benchmark=benchmark,
        manifest_path=manifest_path,
        manifest=manifest,
        wrapper_path=wrapper_path,
        wrapper=wrapper,
        gate_path=gate_path,
        gate=gate,
        prompts=prompts,
    )


def _format_messages(
    wrapper: Mapping[str, Any], system: str, user: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in wrapper["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"].format(system=system, user=user),
            }
        )
    assistant_prefill = wrapper.get("assistant_prefill")
    if isinstance(assistant_prefill, str) and assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    return messages


def _load_renderer(context: _E0Context):
    from transformers import AutoProcessor, AutoTokenizer

    model = context.manifest["model"]
    snapshot = model["local_snapshot"]
    mode = context.wrapper["mode"]
    if mode == "plain_completion":
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        return tokenizer, tokenizer
    if mode == "official_chat_template":
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
    raise ValueError(f"unsupported wrapper mode: {mode}")


def _render_requests(
    context: _E0Context,
) -> tuple[list[dict[str, Any]], Any]:
    renderer, tokenizer = _load_renderer(context)
    runtime = context.manifest["runtime"]
    max_context = runtime["max_context_tokens"]
    mode = context.wrapper["mode"]
    requests: list[dict[str, Any]] = []
    for case in context.gate["cases"]:
        task = case["task"]
        _, prompt = context.prompts[task]
        variables = dict(case["input"])
        variables["output_schema_json"] = json.dumps(
            prompt["output_schema"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        user = prompt["user_template"].format(**variables)
        if mode == "plain_completion":
            rendered = context.wrapper["template"].format(
                system=prompt["system"], user=user
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
                _format_messages(context.wrapper, prompt["system"], user), **kwargs
            )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        output_budget = runtime["max_output_tokens"][task]
        if len(token_ids) + output_budget > max_context:
            raise ValueError(
                f"{case['case_id']}: rendered input plus output budget exceeds max context"
            )
        requests.append(
            {
                "case": case,
                "task": task,
                "rendered_prompt": rendered,
                "prompt_tokens_preflight": len(token_ids),
                "max_output_tokens": output_budget,
                "response_prefix": context.wrapper.get("assistant_prefill", ""),
            }
        )
    return requests, tokenizer


class _NvmlSampler:
    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self.baseline_bytes: int | None = None
        self.static_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.device_name: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._pynvml: Any = None

    def _used(self) -> int:
        return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.device_name = name.decode() if isinstance(name, bytes) else str(name)
            self.baseline_bytes = self._used()
            self.peak_bytes = self.baseline_bytes
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return

        def poll() -> None:
            while not self._stop.wait(0.05):
                try:
                    self.peak_bytes = max(self.peak_bytes or 0, self._used())
                except Exception:
                    return

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def mark_static(self) -> None:
        if self.available:
            self.static_bytes = self._used()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.available:
            try:
                self.peak_bytes = max(self.peak_bytes or 0, self._used())
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def report(self) -> dict[str, Any]:
        baseline = self.baseline_bytes
        return {
            "available": self.available,
            "device_name": self.device_name,
            "measurement_error": self.error,
            "device_used_baseline_bytes": baseline,
            "device_used_after_load_bytes": self.static_bytes,
            "device_used_peak_bytes": self.peak_bytes,
            "model_static_delta_bytes": (
                None
                if baseline is None or self.static_bytes is None
                else max(0, self.static_bytes - baseline)
            ),
            "model_peak_delta_bytes": (
                None
                if baseline is None or self.peak_bytes is None
                else max(0, self.peak_bytes - baseline)
            ),
            "definition": "NVML device-used memory relative to the pre-load baseline; includes runtime allocations and may include concurrent device users",
        }


def _multimodal_limits(manifest: Mapping[str, Any]) -> dict[str, int] | None:
    if manifest["model"]["modality"] != "multimodal":
        return None
    architecture = manifest["model"]["architecture"]
    if architecture == "Qwen3_5ForConditionalGeneration":
        return {"image": 0, "video": 0}
    return {"image": 0}


def _load_vllm(context: _E0Context):
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
    manifest = context.manifest
    runtime = manifest["runtime"]
    protocol = context.benchmark["e0_protocol"]
    kwargs: dict[str, Any] = {
        "model": manifest["model"]["local_snapshot"],
        "tokenizer": manifest["model"]["local_snapshot"],
        "runner": "generate",
        "dtype": runtime["primary_dtype"],
        "max_model_len": runtime["max_context_tokens"],
        "gpu_memory_utilization": protocol["gpu_memory_utilization"],
        "max_num_seqs": protocol["max_num_seqs"],
        "trust_remote_code": protocol["trust_remote_code"],
        "seed": runtime["seed"],
    }
    limits = _multimodal_limits(manifest)
    if limits is not None:
        kwargs["limit_mm_per_prompt"] = limits
    return LLM(**kwargs)


def _sampling_params(context: _E0Context, max_tokens: int):
    from vllm import SamplingParams

    runtime = context.manifest["runtime"]
    return SamplingParams(
        temperature=runtime["temperature"],
        top_p=1,
        top_k=0,
        max_tokens=max_tokens,
        seed=runtime["seed"],
    )


def _generate_by_task(llm: Any, context: _E0Context, requests: Sequence[Mapping[str, Any]]):
    results: list[Any] = [None] * len(requests)
    for task in sorted(TASKS):
        indexed = [
            (index, request)
            for index, request in enumerate(requests)
            if request["task"] == task
        ]
        if not indexed:
            continue
        generated = llm.generate(
            [request["rendered_prompt"] for _, request in indexed],
            _sampling_params(context, indexed[0][1]["max_output_tokens"]),
            use_tqdm=False,
        )
        for (index, _), result in zip(indexed, generated):
            results[index] = result
    return results


def _prediction_record(
    request: Mapping[str, Any], raw_output: str | None, *, runtime_failed: bool
) -> tuple[dict[str, Any], bool]:
    case = request["case"]
    expected = case["expected"]
    if request["task"] == "typed_state":
        parsed = parse_typed_e0_output(
            raw_output,
            expected_criterion_id=case["input"]["criterion_id"],
            runtime_failed=runtime_failed,
        )
        correct = parsed.logical_valid and parsed.state == expected["state"]
        return parsed.as_dict(), correct
    parsed_ext = parse_ext_notes_output(raw_output, runtime_failed=runtime_failed)
    parsed_record = parsed_ext.as_dict()
    parsed_record["parse_valid"] = parsed_ext.failure not in {
        "parse_failure",
        "runtime_failure",
    }
    correct = (
        parsed_ext.logical_valid
        and not parsed_ext.abstained
        and parsed_ext.detection == expected["detection"]
        and parsed_ext.encounter == expected["encounter"]
        and parsed_ext.negation == expected["negation"]
    )
    return parsed_record, correct


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        completed = sum(not row["runtime_failed"] for row in rows)
        parse_valid = sum(bool(row["parsed"].get("parse_valid")) for row in rows)
        schema_valid = sum(bool(row["parsed"].get("schema_valid")) for row in rows)
        logical_valid = sum(bool(row["parsed"].get("logical_valid")) for row in rows)
        abstained = sum(bool(row["parsed"].get("abstained")) for row in rows)
        correct = sum(bool(row["semantic_correct"]) for row in rows)
        valid_rows = [row for row in rows if row["parsed"].get("logical_valid")]
        conditional_correct = sum(bool(row["semantic_correct"]) for row in valid_rows)
        return {
            "cases": count,
            "runtime_completed": completed,
            "runtime_completion_rate": completed / count if count else None,
            "parse_valid": parse_valid,
            "parse_valid_rate": parse_valid / count if count else None,
            "schema_valid": schema_valid,
            "schema_valid_rate": schema_valid / count if count else None,
            "logical_valid": logical_valid,
            "logical_valid_rate": logical_valid / count if count else None,
            "abstained": abstained,
            "semantic_correct_unconditional": correct,
            "semantic_agreement_unconditional": correct / count if count else None,
            "semantic_correct_conditional_on_logical_valid": conditional_correct,
            "semantic_agreement_conditional_on_logical_valid": (
                conditional_correct / len(valid_rows) if valid_rows else None
            ),
            "failure_counts": dict(
                sorted(Counter(row["parsed"].get("failure") or "none" for row in rows).items())
            ),
            "finish_reason_counts": dict(
                sorted(Counter(str(row["finish_reason"]) for row in rows).items())
            ),
        }

    return {
        "overall": summarize(records),
        "by_task": {task: summarize(rows) for task, rows in sorted(grouped.items())},
    }


def run_e0_gate(
    benchmark_path: str | Path,
    *,
    manifest_id: str,
    suite: str,
    output_responses: str | Path,
    output_audit: str | Path,
) -> dict[str, Any]:
    """Run one frozen, fully synthetic E0 suite for one model in one process."""

    context = _load_context(benchmark_path, manifest_id=manifest_id, suite=suite)
    requests, _ = _render_requests(context)
    sampler = _NvmlSampler()
    sampler.start()
    load_started = time.perf_counter()
    llm = None
    try:
        llm = _load_vllm(context)
        load_seconds = time.perf_counter() - load_started
        sampler.mark_static()
        inference_started = time.perf_counter()
        generated = _generate_by_task(llm, context, requests)
        inference_seconds = time.perf_counter() - inference_started

        records: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        with restricted_jsonl_writer(output_responses) as output:
            for request, result in zip(requests, generated):
                choice = result.outputs[0] if result.outputs else None
                runtime_failed = choice is None
                generated_continuation = None if choice is None else choice.text
                raw_output = (
                    None
                    if generated_continuation is None
                    else request["response_prefix"] + generated_continuation
                )
                finish_reason = "missing_output" if choice is None else str(choice.finish_reason)
                parsed, correct = _prediction_record(
                    request, raw_output, runtime_failed=runtime_failed
                )
                prompt_tokens = len(getattr(result, "prompt_token_ids", []) or [])
                output_tokens = 0 if choice is None else len(choice.token_ids)
                total_prompt_tokens += prompt_tokens
                total_output_tokens += output_tokens
                record = {
                    "schema_version": "1.0",
                    "case_id": request["case"]["case_id"],
                    "task": request["task"],
                    "prompt_sha256": hashlib.sha256(
                        request["rendered_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "raw_output": raw_output,
                    "generated_continuation": (
                        generated_continuation if request["response_prefix"] else None
                    ),
                    "response_prefix": request["response_prefix"],
                    "runtime_failed": runtime_failed,
                    "finish_reason": finish_reason,
                    "parsed": parsed,
                    "semantic_correct": correct,
                }
                records.append(record)
                write_jsonl_row(output, record)

        rerun_ids = set(
            context.benchmark["e0_protocol"].get("deterministic_rerun_case_ids", [])
        )
        rerun_requests = [
            request for request in requests if request["case"]["case_id"] in rerun_ids
        ]
        rerun_results: list[dict[str, Any]] = []
        if rerun_requests:
            repeated = _generate_by_task(llm, context, rerun_requests)
            primary_by_id = {record["case_id"]: record for record in records}
            for request, result in zip(rerun_requests, repeated):
                choice = result.outputs[0] if result.outputs else None
                raw_output = (
                    None
                    if choice is None
                    else request["response_prefix"] + choice.text
                )
                case_id = request["case"]["case_id"]
                repeated_parsed, repeated_correct = _prediction_record(
                    request, raw_output, runtime_failed=choice is None
                )
                primary = primary_by_id[case_id]
                rerun_results.append(
                    {
                        "case_id": case_id,
                        "exact_raw_match": raw_output == primary["raw_output"],
                        "parsed_outcome_match": repeated_parsed == primary["parsed"],
                        "semantic_correct_match": repeated_correct
                        == primary["semantic_correct"],
                        "repeated_parsed": repeated_parsed,
                        "output_sha256": (
                            None
                            if raw_output is None
                            else hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
                        ),
                    }
                )
        aggregates = _aggregate_records(records)
        completed = aggregates["overall"]["runtime_completed"]
        throughput = {
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "requests_per_second": (
                len(requests) / inference_seconds if inference_seconds else None
            ),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens_per_second": (
                (total_prompt_tokens + total_output_tokens) / inference_seconds
                if inference_seconds
                else None
            ),
        }
        audit = {
            "schema_version": "1.0",
            "adapter_stage": "slm_benchmark_e0_gate",
            "experiment_id": context.benchmark["experiment_id"],
            "suite": suite,
            "purpose": "technical compatibility gate; not a model leaderboard",
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "artifact_sha256": context.manifest["verification"]["artifact_sha256"],
            "benchmark_config_sha256": config_sha256(context.benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "gate_sha256": config_sha256(context.gate),
            "prompt_sha256": {
                task: config_sha256(prompt) for task, (_, prompt) in context.prompts.items()
            },
            "runtime_version": context.manifest["verification"]["runtime_version"],
            "generation": {
                "temperature": context.manifest["runtime"]["temperature"],
                "seed": context.manifest["runtime"]["seed"],
                "constrained_decoding": False,
                "thinking_mode": context.manifest["runtime"]["thinking_mode"],
                "multimodal_limits": _multimodal_limits(context.manifest),
            },
            "technical_completion": completed == len(requests),
            "aggregates": aggregates,
            "deterministic_reruns": rerun_results,
            "deterministic_reruns_all_exact": (
                all(item["exact_raw_match"] for item in rerun_results)
                if rerun_results
                else None
            ),
            "throughput": throughput,
            "gpu_memory": None,
            "output_responses": str(output_responses),
        }
    except Exception as exc:
        audit = {
            "schema_version": "1.0",
            "adapter_stage": "slm_benchmark_e0_gate",
            "experiment_id": context.benchmark["experiment_id"],
            "suite": suite,
            "purpose": "technical compatibility gate; not a model leaderboard",
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
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


def aggregate_e0_results(
    benchmark_path: str | Path,
    *,
    gate_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Aggregate completed compatibility audits without treating E0 as a leaderboard."""

    source = Path(benchmark_path)
    benchmark = _read_object(source, "benchmark config")
    base = source.parent
    root = Path(gate_root)
    manifest_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for index, value in enumerate(benchmark["model_manifests"]):
        manifest_path = _resolve(base, value, f"model_manifests[{index}]")
        manifest = _read_object(manifest_path, "model manifest")
        manifest_by_id[manifest["manifest_id"]] = (manifest_path, manifest)

    declared = [
        *benchmark["panel"]["core_pilot"],
        *benchmark["panel"]["contemporary_references"],
    ]
    benchmark_hash = config_sha256(benchmark)
    gate_path = _resolve(
        base,
        benchmark["gate_suites"]["compatibility_gate"],
        "gate_suites.compatibility_gate",
    )
    expected_gate_hash = config_sha256(_read_object(gate_path, "compatibility gate"))
    expected_prompt_hashes = {
        task: config_sha256(
            _read_object(
                _resolve(base, benchmark["prompts"][task], f"prompts.{task}"),
                f"{task} prompt",
            )
        )
        for task in TASKS
    }
    rows: list[dict[str, Any]] = []
    gate_hashes: set[str] = set()
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    for manifest_id in declared:
        manifest_path, manifest = manifest_by_id[manifest_id]
        audit_path = root / manifest_id / "compatibility.audit.json"
        audit = _read_object(audit_path, "E0 compatibility audit")
        if audit.get("suite") != "compatibility_gate":
            raise ValueError(f"{audit_path}: not a compatibility-gate audit")
        if audit.get("manifest_id") != manifest_id:
            raise ValueError(f"{audit_path}: manifest ID mismatch")
        if audit.get("benchmark_config_sha256") != benchmark_hash:
            raise ValueError(f"{audit_path}: benchmark config hash is stale")
        if audit.get("manifest_sha256") != config_sha256(manifest):
            raise ValueError(f"{audit_path}: model manifest hash is stale")
        if audit.get("artifact_sha256") != manifest["verification"][
            "artifact_sha256"
        ]:
            raise ValueError(f"{audit_path}: model artifact hash mismatch")
        if audit.get("gate_sha256") != expected_gate_hash:
            raise ValueError(f"{audit_path}: compatibility gate hash is stale")
        if audit.get("prompt_sha256") != expected_prompt_hashes:
            raise ValueError(f"{audit_path}: prompt hash set is stale")
        gate_hashes.add(audit["gate_sha256"])
        for task, digest in audit["prompt_sha256"].items():
            prompt_hashes[task].add(digest)
        wrapper_path = _resolve(
            manifest_path.parent,
            manifest["model"]["wrapper_spec"],
            "model.wrapper_spec",
        )
        wrapper = _read_object(wrapper_path, "wrapper spec")
        if audit.get("wrapper_sha256") != _sha256_file(wrapper_path):
            raise ValueError(f"{audit_path}: wrapper hash is stale")
        task_results = {
            task: {
                key: audit["aggregates"]["by_task"][task][key]
                for key in (
                    "cases",
                    "runtime_completed",
                    "parse_valid",
                    "schema_valid",
                    "logical_valid",
                    "abstained",
                    "semantic_correct_unconditional",
                    "semantic_agreement_unconditional",
                    "semantic_agreement_conditional_on_logical_valid",
                )
            }
            for task in sorted(TASKS)
        }
        rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": manifest["model"]["model_id"],
                "panel_role": manifest["factors"]["panel_role"],
                "family": manifest["factors"]["family"],
                "size_class": manifest["factors"]["size_class"],
                "instruction_tuned": manifest["factors"]["instruction_tuned"],
                "medical_adapted": manifest["factors"]["medical_adapted"],
                "wrapper_id": wrapper["wrapper_id"],
                "wrapper_repair_cycle": wrapper["repair_cycle"],
                "technical_completion": audit["technical_completion"],
                "deterministic_reruns_all_exact": audit[
                    "deterministic_reruns_all_exact"
                ],
                "task_results": task_results,
                "load_seconds": audit["throughput"]["load_seconds"],
                "inference_seconds": audit["throughput"]["inference_seconds"],
                "requests_per_second": audit["throughput"]["requests_per_second"],
                "model_static_delta_bytes": audit["gpu_memory"][
                    "model_static_delta_bytes"
                ],
                "model_peak_delta_bytes": audit["gpu_memory"]["model_peak_delta_bytes"],
                "audit_path": str(audit_path),
            }
        )

    if len(gate_hashes) != 1:
        raise ValueError("compatibility audits do not share one frozen gate hash")
    nonshared_prompts = {
        task: sorted(values) for task, values in prompt_hashes.items() if len(values) != 1
    }
    if nonshared_prompts:
        raise ValueError(f"compatibility audits do not share prompt hashes: {nonshared_prompts}")

    total_requests = sum(
        task["cases"] for row in rows for task in row["task_results"].values()
    )
    completed_requests = sum(
        task["runtime_completed"]
        for row in rows
        for task in row["task_results"].values()
    )
    nondeterministic = [
        row["manifest_id"]
        for row in rows
        if row["deterministic_reruns_all_exact"] is not True
    ]
    repeat_path = root / "mg4_it_v1" / "compatibility.repeat.audit.json"
    repeat_evidence = None
    if repeat_path.is_file():
        repeat = _read_object(repeat_path, "E0 repeat audit")
        repeat_evidence = {
            "audit_path": str(repeat_path),
            "deterministic_reruns_all_exact": repeat.get(
                "deterministic_reruns_all_exact"
            ),
            "reruns": repeat.get("deterministic_reruns"),
        }

    result = {
        "schema_version": "1.0",
        "adapter_stage": "slm_benchmark_e0_aggregate",
        "experiment_id": benchmark["experiment_id"],
        "purpose": "technical compatibility and compute-planning audit; not a semantic model leaderboard",
        "benchmark_config": str(source),
        "benchmark_config_sha256": benchmark_hash,
        "gate_sha256": next(iter(gate_hashes)),
        "prompt_sha256": {
            task: next(iter(values)) for task, values in sorted(prompt_hashes.items())
        },
        "declared_models": len(declared),
        "total_requests": total_requests,
        "completed_requests": completed_requests,
        "all_models_technically_completed": all(
            row["technical_completion"] for row in rows
        ),
        "models_with_nonidentical_exact_rerun": nondeterministic,
        "determinism_follow_up": repeat_evidence,
        "models": rows,
        "wrapper_repairs": {
            "budget_per_model": benchmark["e0_protocol"][
                "development_wrapper_repairs_max"
            ],
            "gemma_family_repairs_used": 1,
            "repair": "fixed assistant JSON opening delimiter through the official processor template",
            "semantic_prompt_changed": False,
            "constrained_decoding_used": False,
            "cycle_0_audits_preserved": True,
        },
        "panel_decision": {
            "retained_manifest_ids": declared,
            "excluded_manifest_ids": [],
            "substituted_manifest_ids": [],
            "reason": "All declared models loaded and completed every frozen E0 request; low format or semantic agreement is not an exclusion criterion.",
        },
        "measurement_cautions": [
            "NVML deltas include vLLM runtime and KV-cache reservation under a fixed 0.85 GPU-memory-utilization target; they are not weight-only footprints.",
            "Requests-per-second includes model-dependent generated token counts and is intended for compute planning, not a normalized efficiency claim.",
            "E0 semantic agreement is a compatibility diagnostic on 48 synthetic cases and must not be presented as the held-out model ranking.",
        ],
        "authorization_boundary": "E1, E3, E4, model downloads, and full P2 were not run or authorized by this E0 aggregate.",
    }
    write_json(output, result)
    return result
