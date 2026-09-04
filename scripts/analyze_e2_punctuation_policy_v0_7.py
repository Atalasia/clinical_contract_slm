#!/usr/bin/env python3
"""Replay E2-A outputs under a punctuation-normalized binary policy.

This is an inference-free, versioned analysis of the frozen E2-A decision logs.  The
additional policy accepts a complete, case-insensitive ``Yes`` or ``No`` response with
at most one terminal period.  It does not extract substrings, repair Markdown, or
default an unrecognized response to ``No``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cpg2.historical_synthetic import normalize_terminal_label
from cpg2.slm_e2 import (  # noqa: E402
    OUTPUT_FAILURE,
    RUNTIME_FAILURE,
    STRICT_COMPLETED,
    TREE_FAILURE,
    advance_traversal,
    current_query,
    new_traversal,
    parse_binary_output,
    stratified_bootstrap_scores,
)


ARTIFACT_VERSION = "0.7"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
POLICY_ID = "complete_binary_optional_single_terminal_period"
_PUNCTUATION_PATTERN = re.compile(r"(yes|no)\.?", re.IGNORECASE)
_LEGACY_TOKEN_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
EXPECTED_MODEL_IDS = {
    "q15_pt_v1",
    "q15_it_v1",
    "q3_pt_v1",
    "q3_it_v1",
    "g4_it_v1",
    "mg4_it_v1",
    "mg15_it_v1",
    "q35_ref_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    return rows


def provenance_record(path: Path, e2_root: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(e2_root.resolve())),
        "sha256": sha256_file(path),
    }


def parse_punctuation_output(
    raw_output: str | None, *, runtime_failed: bool = False
) -> str | None:
    """Return a binary answer only for the complete optional-period contract."""

    if runtime_failed or not isinstance(raw_output, str):
        return None
    match = _PUNCTUATION_PATTERN.fullmatch(raw_output.strip())
    return match.group(1).casefold() if match else None


def classify_strict_failure_output(
    raw_output: str | None, *, runtime_failed: bool = False
) -> str:
    """Coarsely classify a strict failure without disclosing its raw text."""

    if runtime_failed:
        return "runtime_failure"
    if parse_punctuation_output(raw_output) is not None:
        return "binary_with_single_terminal_period"
    text = "" if raw_output is None else str(raw_output)
    if _LEGACY_TOKEN_PATTERN.search(text):
        return "other_text_with_standalone_binary_token"
    return "no_standalone_binary_token"


def _fail_traversal(
    traversal: dict[str, Any], status: str, *, query_id: str
) -> None:
    traversal["status"] = status
    traversal["failure_state_id"] = traversal["current_state_id"]
    traversal["failure_query_id"] = query_id


def replay_case(
    machine: Mapping[str, Any],
    source_case: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay one legacy-path log until the stricter punctuation policy terminates."""

    case_id = str(source_case["case_id"])
    domain = str(source_case["domain"])
    ordered = sorted(decision_rows, key=lambda row: int(row["round"]))
    rounds = [int(row["round"]) for row in ordered]
    if len(rounds) != len(set(rounds)):
        raise ValueError(f"{case_id}: duplicate decision round")

    traversal = new_traversal(machine)
    consumed = 0
    while traversal["status"] == "ACTIVE":
        if consumed >= len(ordered):
            raise ValueError(f"{case_id}: decision log ended before replay terminated")
        row = ordered[consumed]
        query = current_query(machine, traversal)
        current_state_id = str(traversal["current_state_id"])
        if row.get("case_id") != case_id or row.get("domain") != domain:
            raise ValueError(f"{case_id}: decision metadata mismatch")
        if row.get("query_id") != query["query_id"]:
            raise ValueError(
                f"{case_id}: expected query {query['query_id']}, got {row.get('query_id')}"
            )
        if row.get("state_id") != current_state_id:
            raise ValueError(f"{case_id}:{query['query_id']}: state ID mismatch")
        if row.get("criterion") != query["criterion"]:
            raise ValueError(f"{case_id}:{query['query_id']}: criterion mismatch")

        runtime_failed = bool(row.get("runtime_failed"))
        raw_output = row.get("raw_output")
        frozen_parse = parse_binary_output(raw_output, runtime_failed=runtime_failed)
        if row.get("parsed") != frozen_parse:
            raise ValueError(f"{case_id}:{query['query_id']}: frozen parse mismatch")

        answer = parse_punctuation_output(
            raw_output if isinstance(raw_output, str) else None,
            runtime_failed=runtime_failed,
        )
        traversal["decision_calls"] += int(answer is None)
        consumed += 1
        if runtime_failed:
            _fail_traversal(traversal, RUNTIME_FAILURE, query_id=str(query["query_id"]))
        elif answer is None:
            _fail_traversal(traversal, OUTPUT_FAILURE, query_id=str(query["query_id"]))
        else:
            if answer != frozen_parse["legacy_answer"]:
                raise ValueError(
                    f"{case_id}:{query['query_id']}: punctuation and legacy answers differ"
                )
            advance_traversal(machine, traversal, answer)

    if traversal["status"] == STRICT_COMPLETED and consumed != len(ordered):
        raise ValueError(f"{case_id}: replay completed before the legacy decision log")

    completed = traversal["status"] == STRICT_COMPLETED
    predicted = (
        normalize_terminal_label(str(traversal["predicted_action"]))
        if completed
        else None
    )
    expected = normalize_terminal_label(str(source_case["expected_action"]))
    return {
        "status": traversal["status"],
        "predicted_action": predicted,
        "terminal_correct": completed and predicted == expected,
        "decision_calls": int(traversal["decision_calls"]),
        "nodes_visited": len(traversal["visited_states"]),
        "failure_state_id": traversal["failure_state_id"],
        "failure_query_id": traversal["failure_query_id"],
        "consumed_decision_rows": consumed,
        "unconsumed_legacy_decision_rows": len(ordered) - consumed,
    }


def summarize_outcomes(
    source_by_case: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(source_by_case) != set(outcomes):
        raise ValueError("outcome case IDs differ from the source dataset")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for case_id, source in source_by_case.items():
        by_domain[str(source["domain"])].append(case_id)

    domain_rows: dict[str, dict[str, Any]] = {}
    for domain, case_ids in sorted(by_domain.items()):
        ordered_ids = sorted(case_ids)
        correct = sum(bool(outcomes[case_id]["terminal_correct"]) for case_id in ordered_ids)
        statuses = Counter(str(outcomes[case_id]["status"]) for case_id in ordered_ids)
        domain_rows[domain] = {
            "n": len(ordered_ids),
            "correct": correct,
            "accuracy": correct / len(ordered_ids),
            "completed_paths": statuses[STRICT_COMPLETED],
            "invalid_output_paths": statuses[OUTPUT_FAILURE],
            "runtime_failure_paths": statuses[RUNTIME_FAILURE],
            "tree_failure_paths": statuses[TREE_FAILURE],
        }

    all_ids = sorted(source_by_case)
    statuses = Counter(str(outcomes[case_id]["status"]) for case_id in all_ids)
    correct = sum(bool(outcomes[case_id]["terminal_correct"]) for case_id in all_ids)
    return {
        "n": len(all_ids),
        "terminal_correct": correct,
        "pooled_terminal_accuracy": correct / len(all_ids),
        "equal_domain_terminal_accuracy": fmean(
            row["accuracy"] for row in domain_rows.values()
        ),
        "completed_paths": statuses[STRICT_COMPLETED],
        "invalid_output_paths": statuses[OUTPUT_FAILURE],
        "runtime_failure_paths": statuses[RUNTIME_FAILURE],
        "tree_failure_paths": statuses[TREE_FAILURE],
        "by_domain": domain_rows,
    }


def percentile_interval(
    values: Sequence[float], *, confidence: float = 0.95
) -> list[float]:
    if not values:
        raise ValueError("cannot calculate an interval from no values")
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    return [
        ordered[math.floor(tail * (len(ordered) - 1))],
        ordered[math.ceil((1.0 - tail) * (len(ordered) - 1))],
    ]


def compare_policies(
    reference: Mapping[str, Mapping[str, Any]],
    alternative: Mapping[str, Mapping[str, Any]],
    reference_bootstrap: Sequence[float],
    alternative_bootstrap: Sequence[float],
    *,
    reference_score: float,
    alternative_score: float,
) -> dict[str, Any]:
    if set(reference) != set(alternative):
        raise ValueError("policy case sets differ")
    rescued = sum(
        not bool(reference[case_id]["terminal_correct"])
        and bool(alternative[case_id]["terminal_correct"])
        for case_id in reference
    )
    lost = sum(
        bool(reference[case_id]["terminal_correct"])
        and not bool(alternative[case_id]["terminal_correct"])
        for case_id in reference
    )
    differences = [
        alternative_value - reference_value
        for reference_value, alternative_value in zip(
            reference_bootstrap, alternative_bootstrap
        )
    ]
    return {
        "equal_domain_accuracy_difference": alternative_score - reference_score,
        "paired_domain_stratified_bootstrap_95_ci": percentile_interval(differences),
        "rescued_cases": rescued,
        "lost_cases": lost,
        "casewise_correctness_identical": all(
            bool(reference[case_id]["terminal_correct"])
            == bool(alternative[case_id]["terminal_correct"])
            for case_id in reference
        ),
    }


def _validate_stored_outcome(
    case_id: str,
    expected_action: str,
    policy: str,
    outcome: Mapping[str, Any],
) -> None:
    completed = outcome.get("status") == STRICT_COMPLETED
    predicted = outcome.get("predicted_action") if completed else None
    recomputed = completed and predicted == expected_action
    if bool(outcome.get("terminal_correct")) != recomputed:
        raise ValueError(f"{case_id}:{policy}: stored correctness mismatch")


def analyze(e2_root: Path) -> dict[str, Any]:
    e2_root = e2_root.resolve()
    aggregate_path = e2_root / "aggregate/results.json"
    manifest_path = e2_root / "dataset/manifest.json"
    blueprint_path = e2_root / "dataset/blueprint.json"
    source_cases_path = e2_root / "dataset/cases.jsonl"

    aggregate = read_object(aggregate_path, "E2 aggregate")
    manifest = read_object(manifest_path, "E2 dataset manifest")
    blueprint = read_object(blueprint_path, "E2 blueprint")
    if aggregate.get("dataset_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("aggregate dataset-manifest hash is stale")
    if manifest.get("output_blueprint_sha256") != sha256_file(blueprint_path):
        raise ValueError("dataset blueprint hash is stale")
    if manifest.get("output_cases_sha256") != sha256_file(source_cases_path):
        raise ValueError("dataset source-case hash is stale")
    for field in ("experiment_id", "e2_config_sha256", "benchmark_config_sha256"):
        if aggregate.get(field) != manifest.get(field):
            raise ValueError(f"aggregate and dataset disagree on {field}")

    source_rows = read_jsonl(source_cases_path, "E2 source cases")
    source_by_case = {str(row["case_id"]): row for row in source_rows}
    if len(source_by_case) != len(source_rows):
        raise ValueError("duplicate E2 source case ID")
    if len(source_by_case) != manifest.get("cases"):
        raise ValueError("E2 source case count disagrees with manifest")
    domain_by_case = {
        case_id: str(row["domain"]) for case_id, row in source_by_case.items()
    }

    aggregate_models = {
        str(row["manifest_id"]): row for row in aggregate.get("models", [])
    }
    if set(aggregate_models) != EXPECTED_MODEL_IDS:
        raise ValueError("E2 aggregate model panel differs from the frozen eight-model panel")
    legacy_sensitivity = {
        str(row["manifest_id"]): row
        for row in aggregate.get("within_model_legacy_sensitivity", [])
    }
    if set(legacy_sensitivity) != EXPECTED_MODEL_IDS:
        raise ValueError("E2 legacy sensitivity panel is incomplete")

    provenance_models: dict[str, Any] = {}
    stored_strict: dict[str, dict[str, Mapping[str, Any]]] = {}
    stored_legacy: dict[str, dict[str, Mapping[str, Any]]] = {}
    punctuation: dict[str, dict[str, Mapping[str, Any]]] = {}
    mechanism_counts: dict[str, Counter[str]] = {}
    model_ids: dict[str, str] = {}

    for manifest_id in sorted(EXPECTED_MODEL_IDS):
        result_dir = e2_root / "results" / manifest_id
        audit_path = result_dir / "audit.json"
        decision_path = result_dir / "decisions.jsonl"
        stored_cases_path = result_dir / "cases.jsonl"
        audit = read_object(audit_path, f"{manifest_id} audit")
        if audit.get("manifest_id") != manifest_id or audit.get("technical_completion") is not True:
            raise ValueError(f"{manifest_id}: invalid or incomplete audit")
        expected_hashes = {
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "blueprint_sha256": sha256_file(blueprint_path),
            "cases_source_sha256": sha256_file(source_cases_path),
            "e2_config_sha256": aggregate["e2_config_sha256"],
            "benchmark_config_sha256": aggregate["benchmark_config_sha256"],
        }
        for field, expected in expected_hashes.items():
            if audit.get(field) != expected:
                raise ValueError(f"{manifest_id}: stale {field}")
        if audit.get("output_decisions_sha256") != sha256_file(decision_path):
            raise ValueError(f"{manifest_id}: decision-result hash mismatch")
        if audit.get("output_cases_sha256") != sha256_file(stored_cases_path):
            raise ValueError(f"{manifest_id}: case-result hash mismatch")

        decisions = read_jsonl(decision_path, f"{manifest_id} decisions")
        if len(decisions) != audit.get("decision_count"):
            raise ValueError(f"{manifest_id}: decision row count mismatch")
        decision_ids = [str(row.get("decision_id")) for row in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError(f"{manifest_id}: duplicate decision ID")
        decisions_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        classes: Counter[str] = Counter()
        for row in decisions:
            case_id = str(row.get("case_id"))
            if case_id not in source_by_case:
                raise ValueError(f"{manifest_id}: decision refers to unknown case")
            parsed = parse_binary_output(
                row.get("raw_output") if isinstance(row.get("raw_output"), str) else None,
                runtime_failed=bool(row.get("runtime_failed")),
            )
            if row.get("parsed") != parsed:
                raise ValueError(f"{manifest_id}:{case_id}: decision parse mismatch")
            if row.get("strict_active_before") and not parsed["strict_valid"]:
                classes[
                    classify_strict_failure_output(
                        row.get("raw_output")
                        if isinstance(row.get("raw_output"), str)
                        else None,
                        runtime_failed=bool(row.get("runtime_failed")),
                    )
                ] += 1
            decisions_by_case[case_id].append(row)
        if set(decisions_by_case) != set(source_by_case):
            raise ValueError(f"{manifest_id}: decision case set is incomplete")

        stored_rows = read_jsonl(stored_cases_path, f"{manifest_id} stored cases")
        stored_by_case = {str(row["case_id"]): row for row in stored_rows}
        if len(stored_by_case) != len(stored_rows) or set(stored_by_case) != set(source_by_case):
            raise ValueError(f"{manifest_id}: stored case set mismatch")
        strict_rows: dict[str, Mapping[str, Any]] = {}
        legacy_rows: dict[str, Mapping[str, Any]] = {}
        replay_rows: dict[str, Mapping[str, Any]] = {}
        for case_id, source_case in source_by_case.items():
            stored = stored_by_case[case_id]
            expected = normalize_terminal_label(str(source_case["expected_action"]))
            if stored.get("domain") != source_case["domain"] or stored.get("expected_action") != expected:
                raise ValueError(f"{manifest_id}:{case_id}: stored metadata mismatch")
            _validate_stored_outcome(case_id, expected, "strict", stored["strict"])
            _validate_stored_outcome(case_id, expected, "legacy", stored["legacy"])
            if len(decisions_by_case[case_id]) != stored["legacy"]["decision_calls"]:
                raise ValueError(f"{manifest_id}:{case_id}: legacy decision count mismatch")
            strict_rows[case_id] = stored["strict"]
            legacy_rows[case_id] = stored["legacy"]
            replay_rows[case_id] = replay_case(
                blueprint["machines"][str(source_case["domain"])],
                source_case,
                decisions_by_case[case_id],
            )

        strict_summary = summarize_outcomes(source_by_case, strict_rows)
        legacy_summary = summarize_outcomes(source_by_case, legacy_rows)
        aggregate_model = aggregate_models[manifest_id]
        for policy_name, observed, stored_metrics in (
            ("strict", strict_summary, aggregate_model["strict"]),
            ("legacy", legacy_summary, aggregate_model["legacy_coercion_sensitivity"]),
        ):
            expected_score = stored_metrics["primary_domain_balanced_terminal_accuracy"]
            if not math.isclose(
                observed["equal_domain_terminal_accuracy"],
                expected_score,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{manifest_id}: {policy_name} aggregate score mismatch")

        stored_strict[manifest_id] = strict_rows
        stored_legacy[manifest_id] = legacy_rows
        punctuation[manifest_id] = replay_rows
        mechanism_counts[manifest_id] = classes
        model_ids[manifest_id] = str(audit["model_id"])
        provenance_models[manifest_id] = {
            "audit": provenance_record(audit_path, e2_root),
            "decisions": provenance_record(decision_path, e2_root),
            "cases": provenance_record(stored_cases_path, e2_root),
        }

    strict_correct = {
        model: {case_id: bool(row["terminal_correct"]) for case_id, row in rows.items()}
        for model, rows in stored_strict.items()
    }
    punctuation_correct = {
        model: {case_id: bool(row["terminal_correct"]) for case_id, row in rows.items()}
        for model, rows in punctuation.items()
    }
    legacy_correct = {
        model: {case_id: bool(row["terminal_correct"]) for case_id, row in rows.items()}
        for model, rows in stored_legacy.items()
    }
    strict_bootstrap = stratified_bootstrap_scores(
        strict_correct,
        domain_by_case,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    punctuation_bootstrap = stratified_bootstrap_scores(
        punctuation_correct,
        domain_by_case,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    legacy_bootstrap = stratified_bootstrap_scores(
        legacy_correct,
        domain_by_case,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    model_rows: list[dict[str, Any]] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for manifest_id in sorted(EXPECTED_MODEL_IDS):
        strict_summary = summarize_outcomes(source_by_case, stored_strict[manifest_id])
        punctuation_summary = summarize_outcomes(source_by_case, punctuation[manifest_id])
        legacy_summary = summarize_outcomes(source_by_case, stored_legacy[manifest_id])
        punctuation_vs_strict = compare_policies(
            stored_strict[manifest_id],
            punctuation[manifest_id],
            strict_bootstrap[manifest_id],
            punctuation_bootstrap[manifest_id],
            reference_score=strict_summary["equal_domain_terminal_accuracy"],
            alternative_score=punctuation_summary["equal_domain_terminal_accuracy"],
        )
        legacy_vs_strict = compare_policies(
            stored_strict[manifest_id],
            stored_legacy[manifest_id],
            strict_bootstrap[manifest_id],
            legacy_bootstrap[manifest_id],
            reference_score=strict_summary["equal_domain_terminal_accuracy"],
            alternative_score=legacy_summary["equal_domain_terminal_accuracy"],
        )
        frozen_legacy = legacy_sensitivity[manifest_id]
        if (
            legacy_vs_strict["rescued_cases"] != frozen_legacy["rescued_cases"]
            or legacy_vs_strict["lost_cases"] != frozen_legacy["lost_cases"]
            or any(
                not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
                for a, b in zip(
                    legacy_vs_strict["paired_domain_stratified_bootstrap_95_ci"],
                    frozen_legacy["paired_stratified_bootstrap_95_ci"],
                )
            )
        ):
            raise ValueError(f"{manifest_id}: reproduced legacy sensitivity mismatch")
        comparisons[manifest_id] = {
            "strict": strict_summary,
            "punctuation": punctuation_summary,
            "legacy": legacy_summary,
            "punctuation_vs_strict": punctuation_vs_strict,
            "legacy_vs_strict": legacy_vs_strict,
        }
        model_rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": model_ids[manifest_id],
                "strict": strict_summary,
                "punctuation_normalized": punctuation_summary,
                "legacy_coercion": legacy_summary,
                "punctuation_vs_strict": punctuation_vs_strict,
                "legacy_vs_strict": legacy_vs_strict,
                "strict_failure_output_class_counts": dict(
                    sorted(mechanism_counts[manifest_id].items())
                ),
            }
        )

    q15 = comparisons["q15_it_v1"]
    mg15 = comparisons["mg15_it_v1"]
    checks = {
        "q15_it_punctuation_score_rounds_to_0_612": round(
            q15["punctuation"]["equal_domain_terminal_accuracy"], 3
        )
        == 0.612,
        "q15_it_punctuation_rescued_lost_is_184_0": (
            q15["punctuation_vs_strict"]["rescued_cases"],
            q15["punctuation_vs_strict"]["lost_cases"],
        )
        == (184, 0),
        "q15_it_punctuation_matches_legacy_casewise": all(
            punctuation_correct["q15_it_v1"][case_id]
            == legacy_correct["q15_it_v1"][case_id]
            for case_id in source_by_case
        ),
        "mg15_it_punctuation_score_rounds_to_0_473": round(
            mg15["punctuation"]["equal_domain_terminal_accuracy"], 3
        )
        == 0.473,
        "mg15_it_punctuation_rescued_lost_is_0_0": (
            mg15["punctuation_vs_strict"]["rescued_cases"],
            mg15["punctuation_vs_strict"]["lost_cases"],
        )
        == (0, 0),
        "mg15_it_punctuation_matches_strict_casewise": all(
            punctuation_correct["mg15_it_v1"][case_id]
            == strict_correct["mg15_it_v1"][case_id]
            for case_id in source_by_case
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"manuscript regression checks failed: {failed}")

    return {
        "artifact": "E2-A punctuation-normalized evaluator sensitivity",
        "artifact_version": ARTIFACT_VERSION,
        "status": "COMPLETE_INFERENCE_FREE_REPLAY",
        "generation_script": "scripts/analyze_e2_punctuation_policy_v0_7.py",
        "generation_script_sha256": sha256_file(Path(__file__).resolve()),
        "policy": {
            "policy_id": POLICY_ID,
            "accepted_output": (
                "Complete case-insensitive Yes or No with at most one terminal period, "
                "after trimming surrounding whitespace."
            ),
            "rejected_repairs": [
                "substring extraction",
                "Markdown repair",
                "default-to-No coercion",
            ],
            "invalid_output_policy": "Stop the path and score the case incorrect.",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "vignette within domain",
            "method": (
                "Shared domain-stratified draws across models and policies; paired "
                "empirical-percentile 95% intervals for within-model policy differences."
            ),
        },
        "input_provenance": {
            "aggregate": provenance_record(aggregate_path, e2_root),
            "dataset_manifest": provenance_record(manifest_path, e2_root),
            "blueprint": provenance_record(blueprint_path, e2_root),
            "source_cases": provenance_record(source_cases_path, e2_root),
            "models": provenance_models,
            "e2_config_sha256": aggregate["e2_config_sha256"],
            "benchmark_config_sha256": aggregate["benchmark_config_sha256"],
        },
        "population": {
            "cases": len(source_by_case),
            "domain_case_counts": dict(sorted(Counter(domain_by_case.values()).items())),
            "models": len(model_rows),
        },
        "models": model_rows,
        "manuscript_regression_checks": checks,
        "privacy": {
            "contains_raw_output": False,
            "contains_case_or_decision_identifiers": False,
            "contains_vignette_text": False,
        },
    }


def write_output(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    destination.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2-root",
        type=Path,
        default=ROOT / "outputs/local_restricted/slm_benchmark_v2/e2a",
        help="E2-A artifact root containing aggregate/, dataset/, and results/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Derived JSON destination (default: E2 root derived/e2-punctuation-policy-v0.7.json)",
    )
    arguments = parser.parse_args()
    output = arguments.output or (
        arguments.e2_root / "derived/e2-punctuation-policy-v0.7.json"
    )
    payload = analyze(arguments.e2_root)
    write_output(output, payload)
    print(f"wrote {output}")
    for manifest_id in ("q15_it_v1", "mg15_it_v1"):
        row = next(item for item in payload["models"] if item["manifest_id"] == manifest_id)
        score = row["punctuation_normalized"]["equal_domain_terminal_accuracy"]
        change = row["punctuation_vs_strict"]
        print(
            f"{manifest_id}: punctuation={score:.6f}; "
            f"rescued/lost={change['rescued_cases']}/{change['lost_cases']}"
        )


if __name__ == "__main__":
    main()
