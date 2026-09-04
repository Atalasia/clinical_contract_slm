from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import math
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from .historical_synthetic import END_LABEL, normalize_terminal_label
from .io import write_json
from .mimic.csvio import config_sha256, read_jsonl, restricted_jsonl_writer, write_jsonl_row
from .slm_benchmark import validate_benchmark_config
from .slm_e0 import (
    _NvmlSampler,
    _format_messages,
    _load_renderer,
    _load_vllm,
    _multimodal_limits,
    _sampling_params,
)


SCHEMA_VERSION = "1.0"
TASK = "historical_tree_binary"
EXPERIMENT_LABEL = "E2-A"
STRICT_COMPLETED = "COMPLETED"
OUTPUT_FAILURE = "INVALID_OUTPUT"
RUNTIME_FAILURE = "RUNTIME_FAILURE"
TREE_FAILURE = "TREE_FAILURE"
_ANSWER_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


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
    payload = "\x1f".join(values).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:24]


def _write_restricted_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    write_json(destination, payload)
    destination.chmod(0o600)


def _percentile(values: Sequence[int | float], fraction: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _parse_feature_cell(value: Any) -> tuple[str, ...]:
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text
    leaves: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            raise ValueError("feature mappings are ambiguous and unsupported")
        if isinstance(item, (list, tuple, set)):
            values = sorted(item, key=repr) if isinstance(item, set) else item
            for child in values:
                visit(child)
            return
        normalized = str(item).strip()
        if normalized:
            leaves.append(normalized)

    visit(parsed)
    return tuple(dict.fromkeys(leaves))


def parse_binary_output(raw_output: str | None, *, runtime_failed: bool = False) -> dict[str, Any]:
    """Apply the frozen strict parser and the historical-coercion sensitivity parser."""

    if runtime_failed:
        return {
            "runtime_failed": True,
            "strict_valid": False,
            "strict_answer": None,
            "legacy_answer": None,
            "legacy_coerced": False,
            "failure": RUNTIME_FAILURE,
        }
    text = "" if raw_output is None else str(raw_output)
    stripped = text.strip()
    lowered = stripped.casefold()
    strict_valid = lowered in {"yes", "no"}
    match = _ANSWER_PATTERN.search(text)
    legacy_answer = match.group(1).casefold() if match else "no"
    return {
        "runtime_failed": False,
        "strict_valid": strict_valid,
        "strict_answer": lowered if strict_valid else None,
        "legacy_answer": legacy_answer,
        "legacy_coerced": not strict_valid,
        "failure": None if strict_valid else OUTPUT_FAILURE,
    }


def _target(kind: str, value: str) -> dict[str, str]:
    if kind not in {"state", "terminal"}:
        raise ValueError(f"unsupported target kind: {kind}")
    return {"kind": kind, "value": value}


def _linear_machine(domain: str, tree: Mapping[str, Any]) -> dict[str, Any]:
    raw_nodes = tree.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError(f"{domain}: linear tree has no nodes")
    names = [str(node.get("name", "")).strip() for node in raw_nodes]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{domain}: node names must be nonempty and unique")
    state_by_name = {name: f"{domain}.n{index:02d}" for index, name in enumerate(names, 1)}
    states: dict[str, dict[str, Any]] = {}
    terminals: set[str] = set()
    for index, node in enumerate(raw_nodes, 1):
        state_id = f"{domain}.n{index:02d}"
        criteria = node.get("criteria")
        minimum = node.get("min_criteria")
        if criteria:
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(f"{state_id}: criteria must be a nonempty list")
            if not isinstance(minimum, int) or not 1 <= minimum <= len(criteria):
                raise ValueError(f"{state_id}: invalid min_criteria")
            questions = [
                {
                    "query_id": f"{state_id}.c{criterion_index:02d}",
                    "criterion": str(criterion),
                    "reference_source": "criteria",
                }
                for criterion_index, criterion in enumerate(criteria, 1)
            ]
            minimum_yes = minimum
        else:
            if minimum not in {None, 0}:
                raise ValueError(f"{state_id}: min_criteria without criteria")
            questions = [
                {
                    "query_id": state_id,
                    "criterion": names[index - 1],
                    "reference_source": "positive_features",
                }
            ]
            minimum_yes = 1

        def resolve_transition(raw: Any, *, branch: str) -> dict[str, str]:
            value = str(raw).strip()
            if value in state_by_name:
                return _target("state", state_by_name[value])
            terminal = normalize_terminal_label(value)
            if branch == "no" and terminal != END_LABEL:
                raise ValueError(f"{state_id}: dangling No transition {value!r}")
            terminals.add(terminal)
            return _target("terminal", terminal)

        states[state_id] = {
            "state_id": state_id,
            "node_name": names[index - 1],
            "questions": questions,
            "minimum_yes": minimum_yes,
            "yes_target": resolve_transition(node.get("yes_action"), branch="yes"),
            "no_target": resolve_transition(node.get("no_node"), branch="no"),
        }
    return {
        "domain": domain,
        "tree_format": "linear_threshold",
        "root_state_id": f"{domain}.n01",
        "states": states,
        "terminal_labels": sorted(terminals),
    }


_PROSTATE_SECTION_SLUGS = {
    "symptoms": "symptoms",
    "actions for patients with unexplained symptoms of metastatic prostate cancer": "metastatic",
    "actions for patients with luts": "luts",
    "actions for patients with incidental elevated psa results": "incidental_psa",
    "nomograms": "nomograms",
}


def _sectioned_machine(domain: str, tree: Mapping[str, Any]) -> dict[str, Any]:
    sections_by_fold = {str(key).strip().casefold(): str(key) for key in tree}
    if set(sections_by_fold) != set(_PROSTATE_SECTION_SLUGS):
        raise ValueError(f"{domain}: unexpected sectioned-tree keys")
    state_lookup: dict[tuple[str, str], str] = {}
    ordered_sections: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    for folded, slug in _PROSTATE_SECTION_SLUGS.items():
        source_key = sections_by_fold[folded]
        section = tree.get(source_key)
        nodes = section.get("nodes") if isinstance(section, Mapping) else None
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"{domain}.{slug}: section has no nodes")
        names = [str(node.get("name", "")).strip() for node in nodes]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{domain}.{slug}: node names must be nonempty and unique")
        for index, name in enumerate(names, 1):
            state_lookup[(folded, name)] = f"{domain}.{slug}.n{index:02d}"
        ordered_sections.append((folded, slug, nodes))

    terminals: set[str] = set()
    states: dict[str, dict[str, Any]] = {}
    for folded, slug, nodes in ordered_sections:
        for index, node in enumerate(nodes, 1):
            state_id = f"{domain}.{slug}.n{index:02d}"

            def resolve_yes(raw: Any) -> dict[str, str]:
                value = str(raw).strip()
                target_section = value.casefold()
                if target_section in sections_by_fold:
                    source_section = tree[sections_by_fold[target_section]]["nodes"]
                    first_name = str(source_section[0]["name"]).strip()
                    return _target("state", state_lookup[(target_section, first_name)])
                terminal = normalize_terminal_label(value)
                terminals.add(terminal)
                return _target("terminal", terminal)

            def resolve_no(raw: Any) -> dict[str, str]:
                value = str(raw).strip()
                local = state_lookup.get((folded, value))
                if local is not None:
                    return _target("state", local)
                terminal = normalize_terminal_label(value)
                if terminal != END_LABEL:
                    raise ValueError(f"{state_id}: dangling No transition {value!r}")
                terminals.add(terminal)
                return _target("terminal", terminal)

            node_name = str(node["name"]).strip()
            states[state_id] = {
                "state_id": state_id,
                "node_name": node_name,
                "questions": [
                    {
                        "query_id": state_id,
                        "criterion": node_name,
                        "reference_source": "positive_features",
                    }
                ],
                "minimum_yes": 1,
                "yes_target": resolve_yes(node.get("yes_action")),
                "no_target": resolve_no(node.get("no_node")),
            }
    root_name = str(tree[sections_by_fold["symptoms"]]["nodes"][0]["name"]).strip()
    return {
        "domain": domain,
        "tree_format": "sectioned",
        "root_state_id": state_lookup[("symptoms", root_name)],
        "states": states,
        "terminal_labels": sorted(terminals),
    }


def build_historical_machine(
    domain: str, tree: Mapping[str, Any], tree_format: str
) -> dict[str, Any]:
    if tree_format == "linear_threshold":
        machine = _linear_machine(domain, tree)
    elif tree_format == "sectioned":
        machine = _sectioned_machine(domain, tree)
    else:
        raise ValueError(f"{domain}: unsupported tree format {tree_format}")
    query_ids = [
        question["query_id"]
        for state in machine["states"].values()
        for question in state["questions"]
    ]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{domain}: duplicate query IDs")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state_id: str) -> None:
        if state_id in visiting:
            raise ValueError(f"{domain}: cycle detected at {state_id}")
        if state_id in visited:
            return
        if state_id not in machine["states"]:
            raise ValueError(f"{domain}: dangling state target {state_id}")
        visiting.add(state_id)
        for branch in ("yes_target", "no_target"):
            target = machine["states"][state_id][branch]
            if target["kind"] == "state":
                visit(target["value"])
        visiting.remove(state_id)
        visited.add(state_id)

    visit(machine["root_state_id"])
    unreachable = set(machine["states"]) - visited
    for state_id in sorted(unreachable):
        visit(state_id)
    machine["unreachable_state_ids"] = sorted(unreachable)
    machine["query_definition_count"] = len(query_ids)
    machine["state_count"] = len(machine["states"])
    _execute_all_no(machine)
    return machine


def new_traversal(machine: Mapping[str, Any]) -> dict[str, Any]:
    root = str(machine["root_state_id"])
    return {
        "current_state_id": root,
        "question_index": 0,
        "yes_count": 0,
        "status": "ACTIVE",
        "predicted_action": None,
        "decision_calls": 0,
        "visited_states": [root],
        "branches": [],
        "failure_state_id": None,
        "failure_query_id": None,
    }


def current_query(machine: Mapping[str, Any], traversal: Mapping[str, Any]) -> dict[str, Any]:
    if traversal["status"] != "ACTIVE":
        raise ValueError("cannot request a query from an inactive traversal")
    state = machine["states"][traversal["current_state_id"]]
    return state["questions"][int(traversal["question_index"])]


def _fail_traversal(
    traversal: dict[str, Any], status: str, *, query_id: str | None = None
) -> None:
    traversal["status"] = status
    traversal["failure_state_id"] = traversal["current_state_id"]
    traversal["failure_query_id"] = query_id


def advance_traversal(
    machine: Mapping[str, Any], traversal: dict[str, Any], answer: str
) -> dict[str, Any] | None:
    """Apply one exact yes/no answer and return a completed node-branch record, if any."""

    if answer not in {"yes", "no"}:
        raise ValueError(f"invalid traversal answer: {answer}")
    question = current_query(machine, traversal)
    traversal["decision_calls"] += 1
    state = machine["states"][traversal["current_state_id"]]
    if answer == "yes":
        traversal["yes_count"] += 1
    branch: str | None = None
    if traversal["yes_count"] >= state["minimum_yes"]:
        branch = "yes"
    elif traversal["question_index"] + 1 >= len(state["questions"]):
        branch = "no"
    else:
        traversal["question_index"] += 1
        return None

    branch_record = {
        "state_id": state["state_id"],
        "node_name": state["node_name"],
        "branch": branch,
        "questions_consumed": int(traversal["question_index"]) + 1,
        "yes_count": int(traversal["yes_count"]),
    }
    traversal["branches"].append(branch_record)
    target = state[f"{branch}_target"]
    if target["kind"] == "terminal":
        traversal["status"] = STRICT_COMPLETED
        traversal["predicted_action"] = normalize_terminal_label(target["value"])
    elif target["kind"] == "state":
        target_id = target["value"]
        if target_id not in machine["states"]:
            _fail_traversal(traversal, TREE_FAILURE, query_id=question["query_id"])
        elif target_id in traversal["visited_states"]:
            _fail_traversal(traversal, TREE_FAILURE, query_id=question["query_id"])
        else:
            traversal["current_state_id"] = target_id
            traversal["question_index"] = 0
            traversal["yes_count"] = 0
            traversal["visited_states"].append(target_id)
    else:
        _fail_traversal(traversal, TREE_FAILURE, query_id=question["query_id"])
    return branch_record


def _execute_all_no(machine: Mapping[str, Any]) -> str:
    traversal = new_traversal(machine)
    maximum = sum(len(state["questions"]) for state in machine["states"].values()) + 1
    for _ in range(maximum):
        if traversal["status"] != "ACTIVE":
            break
        advance_traversal(machine, traversal, "no")
    if traversal["status"] != STRICT_COMPLETED:
        raise ValueError(f"{machine['domain']}: all-No traversal did not terminate")
    return str(traversal["predicted_action"])


def _reference_traversal(
    machine: Mapping[str, Any], *, positive: set[str], criteria: set[str]
) -> dict[str, Any]:
    traversal = new_traversal(machine)
    decisions: list[dict[str, Any]] = []
    maximum = sum(len(state["questions"]) for state in machine["states"].values()) + 1
    for _ in range(maximum):
        if traversal["status"] != "ACTIVE":
            break
        question = current_query(machine, traversal)
        source = positive if question["reference_source"] == "positive_features" else criteria
        answer = "yes" if question["criterion"] in source else "no"
        decisions.append(
            {
                "state_id": traversal["current_state_id"],
                "query_id": question["query_id"],
                "answer": answer,
            }
        )
        advance_traversal(machine, traversal, answer)
    if traversal["status"] != STRICT_COMPLETED:
        raise ValueError(f"{machine['domain']}: reference traversal did not terminate")
    return {
        "predicted_action": traversal["predicted_action"],
        "decisions": decisions,
        "branches": traversal["branches"],
        "decision_calls": traversal["decision_calls"],
        "visited_states": traversal["visited_states"],
    }


def _load_e2_config(
    path: str | Path, *, require_authorized: bool = True
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    source = Path(path)
    config = _read_object(source, "E2 config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E2 config schema")
    if config.get("freeze_status") != "FROZEN" or config.get("experiment_label") != EXPERIMENT_LABEL:
        raise ValueError("E2-A config is not frozen")
    authorization = config.get("authorization", {})
    if require_authorized and (
        authorization.get("inference") is not True
        or EXPERIMENT_LABEL not in authorization.get("authorized_experiments", [])
    ):
        raise ValueError("E2-A inference is not authorized")
    benchmark_path = _resolve(source.parent, config.get("benchmark_config"), "benchmark_config")
    benchmark = _read_object(benchmark_path, "benchmark config")
    if config_sha256(benchmark) != config.get("benchmark_config_sha256"):
        raise ValueError("E2-A benchmark config hash mismatch")
    prompt_path = _resolve(source.parent, config.get("prompt"), "prompt")
    prompt = _read_object(prompt_path, "E2 prompt")
    if config_sha256(prompt) != config.get("prompt_sha256"):
        raise ValueError("E2-A prompt hash mismatch")
    if prompt.get("freeze_status") != "FROZEN" or prompt.get("task") != TASK:
        raise ValueError("E2-A prompt is not the frozen historical binary contract")
    return source, config, benchmark_path, benchmark, prompt_path, prompt


def prepare_e2_dataset(
    config_path: str | Path,
    *,
    output_cases: str | Path,
    output_blueprint: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    source, config, _benchmark_path, _benchmark, _prompt_path, _prompt = _load_e2_config(
        config_path, require_authorized=False
    )
    machines: dict[str, dict[str, Any]] = {}
    source_artifacts: list[dict[str, Any]] = []
    domain_specs: dict[str, Mapping[str, Any]] = {}
    for domain_spec in config["domains"]:
        domain = str(domain_spec["domain"])
        if domain in machines:
            raise ValueError(f"duplicate E2 domain: {domain}")
        tree_path = _resolve(source.parent, domain_spec["historical_tree"], f"{domain}.tree")
        vignette_path = _resolve(source.parent, domain_spec["vignettes"], f"{domain}.vignettes")
        if _sha256_file(tree_path) != domain_spec["historical_tree_sha256"]:
            raise ValueError(f"{domain}: historical tree hash mismatch")
        if _sha256_file(vignette_path) != domain_spec["vignettes_sha256"]:
            raise ValueError(f"{domain}: vignette hash mismatch")
        tree = _read_object(tree_path, f"{domain} historical tree")
        machines[domain] = build_historical_machine(domain, tree, str(domain_spec["tree_format"]))
        if machines[domain]["unreachable_state_ids"]:
            raise ValueError(
                f"{domain}: unreachable frozen source states: {machines[domain]['unreachable_state_ids']}"
            )
        domain_specs[domain] = domain_spec
        source_artifacts.extend(
            [
                {"domain": domain, "role": "historical_tree", "path": str(tree_path), "sha256": _sha256_file(tree_path)},
                {"domain": domain, "role": "vignettes", "path": str(vignette_path), "sha256": _sha256_file(vignette_path)},
            ]
        )

    cases: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    domain_reference_queries: Counter[str] = Counter()
    reference_class_sets: dict[str, set[str]] = defaultdict(set)
    conflicts = 0
    count_field_mismatches = 0
    reference_terminal_matches = 0
    aggregate_membership_mismatches = 0
    unmatched_features: Counter[str] = Counter()
    unmatched_criteria: Counter[str] = Counter()

    for domain, domain_spec in domain_specs.items():
        machine = machines[domain]
        feature_names = {state["node_name"] for state in machine["states"].values()}
        aggregate_names = {
            question["criterion"]
            for state in machine["states"].values()
            for question in state["questions"]
            if question["reference_source"] == "criteria"
        }
        vignette_path = _resolve(source.parent, domain_spec["vignettes"], f"{domain}.vignettes")
        with vignette_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                domain_spec["positive_features_column"],
                domain_spec["negative_features_column"],
                domain_spec["action_column"],
                domain_spec["text_column"],
            }
            aggregate_column = domain_spec.get("aggregate_criteria_column")
            if aggregate_column:
                required.add(aggregate_column)
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{domain}: missing vignette columns: {missing}")
            for row_index, row in enumerate(reader):
                positive = set(_parse_feature_cell(row[domain_spec["positive_features_column"]]))
                negative = set(_parse_feature_cell(row[domain_spec["negative_features_column"]]))
                criteria = set(_parse_feature_cell(row.get(aggregate_column, ""))) if aggregate_column else set()
                conflicts += len(positive & negative)
                for field, observed in (("n_positive_features", positive), ("n_negative_features", negative)):
                    if field in row and str(row.get(field, "")).strip():
                        try:
                            count_field_mismatches += int(int(row[field]) != len(observed))
                        except ValueError as exc:
                            raise ValueError(f"{domain} row {row_index}: invalid {field}") from exc
                unmatched_features.update(sorted((positive | negative) - feature_names))
                unmatched_criteria.update(sorted(criteria - aggregate_names))
                for state in machine["states"].values():
                    if state["questions"][0]["reference_source"] != "criteria":
                        continue
                    threshold_met = sum(q["criterion"] in criteria for q in state["questions"]) >= state["minimum_yes"]
                    aggregate_membership_mismatches += int((state["node_name"] in positive) != threshold_met)
                text = str(row[domain_spec["text_column"]])
                if not text.strip():
                    raise ValueError(f"{domain} row {row_index}: blank vignette text")
                expected_action = normalize_terminal_label(str(row[domain_spec["action_column"]]).strip())
                reference = _reference_traversal(machine, positive=positive, criteria=criteria)
                reference_terminal_matches += int(reference["predicted_action"] == expected_action)
                domain_reference_queries[domain] += int(reference["decision_calls"])
                domain_counts[domain] += 1
                reference_class_sets[domain].add(expected_action)
                case_id = _stable_id(
                    "e2_case_", domain, str(row_index), hashlib.sha256(text.encode("utf-8")).hexdigest()
                )
                cases.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "case_id": case_id,
                        "domain": domain,
                        "source_row_index": row_index,
                        "text": text,
                        "expected_action": expected_action,
                        "positive_features": sorted(positive),
                        "negative_features": sorted(negative),
                        "aggregate_criteria": sorted(criteria),
                        "reference": reference,
                    }
                )

    observed = {
        "source_rows": len(cases),
        "cases": len(cases),
        "domain_case_counts": dict(sorted(domain_counts.items())),
        "query_definitions": sum(machine["query_definition_count"] for machine in machines.values()),
        "reference_route_queries": sum(domain_reference_queries.values()),
        "domain_reference_route_queries": dict(sorted(domain_reference_queries.items())),
        "reference_terminal_classes": {
            domain: len(labels) for domain, labels in sorted(reference_class_sets.items())
        },
        "positive_negative_conflicts": conflicts,
        "reference_terminal_matches": reference_terminal_matches,
    }
    for field, expected in config["expected_dataset"].items():
        if observed.get(field) != expected:
            raise ValueError(f"E2-A dataset invariant mismatch for {field}: {observed.get(field)!r} != {expected!r}")
    if count_field_mismatches:
        raise ValueError(f"E2-A feature count-field mismatches: {count_field_mismatches}")
    if aggregate_membership_mismatches:
        raise ValueError(f"E2-A aggregate membership mismatches: {aggregate_membership_mismatches}")
    if unmatched_features or unmatched_criteria:
        raise ValueError(
            f"E2-A unmatched feature labels: {dict(unmatched_features)}; criteria: {dict(unmatched_criteria)}"
        )

    cases_path = Path(output_cases).resolve()
    with restricted_jsonl_writer(cases_path) as output:
        for case in cases:
            write_jsonl_row(output, case)
    blueprint = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e2_prepare_blueprint",
        "experiment_id": config["experiment_id"],
        "construction_version": config["construction"]["version"],
        "machines": machines,
        "all_no_terminal_by_domain": {
            domain: _execute_all_no(machine) for domain, machine in sorted(machines.items())
        },
        "reference_terminal_labels_by_domain": {
            domain: sorted(labels) for domain, labels in sorted(reference_class_sets.items())
        },
    }
    blueprint_path = Path(output_blueprint).resolve()
    _write_restricted_json(blueprint_path, blueprint)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e2_prepare",
        "experiment_id": config["experiment_id"],
        "e2_config": str(source.resolve()),
        "e2_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config["benchmark_config_sha256"],
        "prompt_sha256": config["prompt_sha256"],
        "construction_version": config["construction"]["version"],
        **observed,
        "count_field_mismatches": count_field_mismatches,
        "aggregate_membership_mismatches": aggregate_membership_mismatches,
        "unmatched_feature_occurrences": dict(sorted(unmatched_features.items())),
        "unmatched_criteria_occurrences": dict(sorted(unmatched_criteria.items())),
        "exhaustive_token_audit_prompts_per_model": sum(
            domain_counts[domain] * machines[domain]["query_definition_count"] for domain in machines
        ),
        "source_artifacts": source_artifacts,
        "output_cases": str(cases_path),
        "output_cases_sha256": _sha256_file(cases_path),
        "output_blueprint": str(blueprint_path),
        "output_blueprint_sha256": _sha256_file(blueprint_path),
        "claim_boundary": config["claim_boundary"],
    }
    _write_restricted_json(output_manifest, manifest)
    return manifest


@dataclass(frozen=True)
class _E2ModelContext:
    e2_config_path: Path
    e2_config: dict[str, Any]
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
    cases_path: Path
    blueprint_path: Path


def _load_model_context(
    config_path: str | Path, *, manifest_id: str, dataset_manifest_path: str | Path
) -> _E2ModelContext:
    source, config, benchmark_path, benchmark, prompt_path, prompt = _load_e2_config(config_path)
    preflight = validate_benchmark_config(benchmark_path)
    if not preflight["inference_ready"]:
        raise ValueError("base benchmark is not inference-ready")
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    if manifest_id not in declared:
        raise ValueError(f"manifest is not in the declared E2-A panel: {manifest_id}")
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
    runtime = manifest["runtime"]
    inference = config["inference"]
    for field, expected in (
        ("primary_dtype", inference["primary_dtype"]),
        ("temperature", inference["temperature"]),
        ("seed", inference["seed"]),
    ):
        if runtime.get(field) != expected:
            raise ValueError(f"{manifest_id}: runtime {field} differs from E2-A freeze")
    override = config.get("wrapper_overrides", {}).get(manifest_id)
    if override is None:
        wrapper_path = _resolve(manifest_path.parent, manifest["model"]["wrapper_spec"], "wrapper")
        expected_wrapper_hash = manifest["verification"]["wrapper_spec_sha256"]
    else:
        wrapper_path = _resolve(source.parent, override["wrapper"], f"{manifest_id}.wrapper_override")
        expected_wrapper_hash = override["wrapper_sha256"]
    if _sha256_file(wrapper_path) != expected_wrapper_hash:
        raise ValueError(f"{manifest_id}: frozen E2-A wrapper hash mismatch")
    wrapper = _read_object(wrapper_path, "E2-A wrapper")
    if wrapper.get("freeze_status") != "FROZEN" or wrapper.get("assistant_prefill"):
        raise ValueError(f"{manifest_id}: E2-A wrapper must be frozen and have no response prefill")
    if override is not None and TASK not in wrapper.get("task_scope", []):
        raise ValueError(f"{manifest_id}: wrapper override is not scoped to E2-A")
    dataset_path = Path(dataset_manifest_path)
    dataset = _read_object(dataset_path, "E2-A dataset manifest")
    if dataset.get("e2_config_sha256") != config_sha256(config):
        raise ValueError("E2-A dataset was not built from the frozen config")
    if dataset.get("prompt_sha256") != config["prompt_sha256"]:
        raise ValueError("E2-A dataset prompt hash mismatch")
    cases_path = Path(dataset["output_cases"])
    blueprint_path = Path(dataset["output_blueprint"])
    if _sha256_file(cases_path) != dataset["output_cases_sha256"]:
        raise ValueError("E2-A cases hash mismatch")
    if _sha256_file(blueprint_path) != dataset["output_blueprint_sha256"]:
        raise ValueError("E2-A blueprint hash mismatch")
    return _E2ModelContext(
        e2_config_path=source,
        e2_config=config,
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
        cases_path=cases_path,
        blueprint_path=blueprint_path,
    )


def _render_prompt(
    context: _E2ModelContext,
    *,
    renderer: Any,
    tokenizer: Any,
    text: str,
    criterion: str,
) -> dict[str, Any]:
    question = context.prompt["question_template"].format(criterion=criterion)
    user = context.prompt["user_template"].format(text=text, question=question)
    if context.wrapper["mode"] == "plain_completion":
        rendered = context.wrapper["template"].format(system=context.prompt["system"], user=user)
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
            _format_messages(context.wrapper, context.prompt["system"], user), **kwargs
        )
    token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
    output_budget = int(context.e2_config["inference"]["max_output_tokens"])
    return {
        "question": question,
        "rendered_prompt": rendered,
        "prompt_tokens_preflight": token_count,
        "over_context": token_count + output_budget > context.manifest["runtime"]["max_context_tokens"],
    }


def audit_e2_tokens(
    config_path: str | Path, *, dataset_manifest_path: str | Path, output: str | Path
) -> dict[str, Any]:
    source, config, _benchmark_path, benchmark, _prompt_path, _prompt = _load_e2_config(config_path)
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    cases = list(read_jsonl(_read_object(Path(dataset_manifest_path), "E2-A dataset manifest")["output_cases"]))
    blueprint = _read_object(
        Path(_read_object(Path(dataset_manifest_path), "E2-A dataset manifest")["output_blueprint"]),
        "E2-A blueprint",
    )
    rows: list[dict[str, Any]] = []
    for manifest_id in declared:
        context = _load_model_context(config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path)
        renderer, tokenizer = _load_renderer(context)
        lengths: list[int] = []
        over_context = 0
        for case in cases:
            machine = blueprint["machines"][case["domain"]]
            for state in machine["states"].values():
                for question in state["questions"]:
                    rendered = _render_prompt(
                        context,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        text=case["text"],
                        criterion=question["criterion"],
                    )
                    lengths.append(rendered["prompt_tokens_preflight"])
                    over_context += int(rendered["over_context"])
        rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": context.manifest["model"]["model_id"],
                "wrapper_sha256": _sha256_file(context.wrapper_path),
                "potential_prompts": len(lengths),
                "minimum_prompt_tokens": min(lengths),
                "mean_prompt_tokens": fmean(lengths),
                "p95_prompt_tokens": _percentile(lengths, 0.95),
                "maximum_prompt_tokens": max(lengths),
                "max_output_tokens": config["inference"]["max_output_tokens"],
                "max_context_tokens": context.manifest["runtime"]["max_context_tokens"],
                "over_context_prompts": over_context,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e2_token_audit",
        "experiment_id": config["experiment_id"],
        "e2_config_sha256": config_sha256(config),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "exhaustive_case_by_domain_question_audit": True,
        "potential_prompts_per_model": int(_read_object(Path(dataset_manifest_path), "manifest")["exhaustive_token_audit_prompts_per_model"]),
        "all_models_fit_context": all(row["over_context_prompts"] == 0 for row in rows),
        "models": rows,
    }
    write_json(output, result)
    return result


def compute_terminal_metrics(
    expected: Sequence[str], predicted: Sequence[str | None], *, labels: Sequence[str] | None = None
) -> dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    normalized_expected = [normalize_terminal_label(value) for value in expected]
    normalized_predicted = [None if value is None else normalize_terminal_label(value) for value in predicted]
    fixed_labels = sorted(set(normalized_expected)) if labels is None else list(labels)
    if set(normalized_expected) - set(fixed_labels):
        raise ValueError("fixed labels omit a reference terminal")
    per_label: dict[str, dict[str, Any]] = {}
    for label in fixed_labels:
        tp = sum(e == label and p == label for e, p in zip(normalized_expected, normalized_predicted))
        fp = sum(e != label and p == label for e, p in zip(normalized_expected, normalized_predicted))
        fn = sum(e == label and p != label for e, p in zip(normalized_expected, normalized_predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": sum(e == label for e in normalized_expected),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(e == p for e, p in zip(normalized_expected, normalized_predicted))
    return {
        "n": len(expected),
        "correct": correct,
        "accuracy": correct / len(expected) if expected else None,
        "failure_n": sum(value is None for value in normalized_predicted),
        "label_count": len(fixed_labels),
        "macro_precision": fmean(row["precision"] for row in per_label.values()) if per_label else None,
        "macro_recall": fmean(row["recall"] for row in per_label.values()) if per_label else None,
        "macro_f1": fmean(row["f1"] for row in per_label.values()) if per_label else None,
        "per_reference_label": per_label,
    }


def _binary_action_metrics(expected: Sequence[str], predicted: Sequence[str | None]) -> dict[str, Any]:
    expected_binary = [normalize_terminal_label(value) != END_LABEL for value in expected]
    predicted_binary = [None if value is None else normalize_terminal_label(value) != END_LABEL for value in predicted]
    valid = [(e, p) for e, p in zip(expected_binary, predicted_binary) if p is not None]
    tp = sum(e and p for e, p in valid)
    fp = sum(not e and p for e, p in valid)
    fn = sum(e and not p for e, p in valid) + sum(e and p is None for e, p in zip(expected_binary, predicted_binary))
    tn = sum(not e and not p for e, p in valid)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(expected),
        "valid_prediction_n": len(valid),
        "coverage": len(valid) / len(expected) if expected else None,
        "failure_n": len(expected) - len(valid),
        "unconditional_accuracy": (tp + tn) / len(expected) if expected else None,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "positive_reference_n": sum(expected_binary),
        "negative_reference_n": sum(not value for value in expected_binary),
    }


def summarize_policy_cases(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    if policy not in {"strict", "legacy"}:
        raise ValueError("policy must be strict or legacy")
    domains = sorted({str(row["domain"]) for row in rows})
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in domains:
        selected = [row for row in rows if row["domain"] == domain]
        expected = [str(row["expected_action"]) for row in selected]
        predicted = [row[policy]["predicted_action"] for row in selected]
        labels = sorted(set(expected))
        terminal = compute_terminal_metrics(expected, predicted, labels=labels)
        terminal["binary_action_vs_end"] = _binary_action_metrics(expected, predicted)
        by_domain[domain] = terminal
    expected_all = [str(row["expected_action"]) for row in rows]
    predicted_all = [row[policy]["predicted_action"] for row in rows]
    overall = compute_terminal_metrics(expected_all, predicted_all)
    overall["binary_action_vs_end"] = _binary_action_metrics(expected_all, predicted_all)
    balanced = fmean(by_domain[domain]["accuracy"] for domain in domains) if domains else None
    return {
        "primary_domain_balanced_terminal_accuracy": balanced,
        "pooled_terminal": overall,
        "by_domain": by_domain,
    }


def _record_branch_divergence(
    run_state: dict[str, Any], policy: str, branch: Mapping[str, Any]
) -> None:
    field = f"{policy}_first_divergence_state_id"
    if run_state[field] is not None:
        return
    expected = run_state["reference_branch_by_state"].get(branch["state_id"])
    if expected is None:
        raise RuntimeError(
            f"{run_state['case']['case_id']}: reached non-reference state without a prior divergence"
        )
    if expected != branch["branch"]:
        run_state[field] = branch["state_id"]


def _policy_case_record(
    run_state: Mapping[str, Any], policy: str
) -> dict[str, Any]:
    traversal = run_state[policy]
    expected = str(run_state["case"]["expected_action"])
    completed = traversal["status"] == STRICT_COMPLETED
    predicted = traversal["predicted_action"] if completed else None
    first_divergence = run_state[f"{policy}_first_divergence_state_id"]
    if completed and first_divergence is None and predicted != expected:
        raise RuntimeError(
            f"{run_state['case']['case_id']}: terminal mismatch without a node-branch divergence"
        )
    return {
        "status": traversal["status"],
        "predicted_action": predicted,
        "terminal_correct": completed and predicted == expected,
        "path_exact": completed and first_divergence is None,
        "first_divergence_state_id": first_divergence,
        "decision_calls": traversal["decision_calls"],
        "nodes_visited": len(traversal["visited_states"]),
        "failure_state_id": traversal["failure_state_id"],
        "failure_query_id": traversal["failure_query_id"],
        "visited_states": traversal["visited_states"],
        "branches": traversal["branches"],
    }


def _path_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    *,
    invalid_decision_calls: int = 0,
    attempted_decision_calls: int | None = None,
) -> dict[str, Any]:
    selected = [row[policy] for row in rows]
    calls = [int(row["decision_calls"]) for row in selected]
    nodes = [int(row["nodes_visited"]) for row in selected]
    first_divergence = Counter(
        str(row["first_divergence_state_id"])
        for row in selected
        if row.get("first_divergence_state_id") is not None
    )
    statuses = Counter(str(row["status"]) for row in selected)
    attempted = sum(calls) if attempted_decision_calls is None else attempted_decision_calls
    return {
        "n": len(selected),
        "status_counts": dict(sorted(statuses.items())),
        "completed_n": statuses[STRICT_COMPLETED],
        "completion_rate": statuses[STRICT_COMPLETED] / len(selected) if selected else None,
        "invalid_output_case_n": statuses[OUTPUT_FAILURE],
        "invalid_output_case_rate": statuses[OUTPUT_FAILURE] / len(selected) if selected else None,
        "runtime_failure_case_n": statuses[RUNTIME_FAILURE],
        "runtime_failure_case_rate": statuses[RUNTIME_FAILURE] / len(selected) if selected else None,
        "path_exact_n": sum(bool(row["path_exact"]) for row in selected),
        "path_exact_rate": fmean(bool(row["path_exact"]) for row in selected) if selected else None,
        "invalid_decision_calls": invalid_decision_calls,
        "attempted_decision_calls": attempted,
        "invalid_decision_call_rate": invalid_decision_calls / attempted if attempted else None,
        "mean_decision_calls": fmean(calls) if calls else None,
        "median_decision_calls": median(calls) if calls else None,
        "p95_decision_calls": _percentile(calls, 0.95),
        "mean_nodes_visited": fmean(nodes) if nodes else None,
        "first_divergence_state_counts": dict(sorted(first_divergence.items())),
    }


def run_e2_model(
    config_path: str | Path,
    *,
    manifest_id: str,
    dataset_manifest_path: str | Path,
    token_audit_path: str | Path,
    output_decisions: str | Path,
    output_cases: str | Path,
    output_audit: str | Path,
) -> dict[str, Any]:
    context = _load_model_context(
        config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path
    )
    token_audit = _read_object(Path(token_audit_path), "E2-A token audit")
    if token_audit.get("e2_config_sha256") != config_sha256(context.e2_config):
        raise ValueError("E2-A token audit is stale")
    if token_audit.get("dataset_manifest_sha256") != _sha256_file(dataset_manifest_path):
        raise ValueError("E2-A token audit dataset hash is stale")
    if token_audit.get("all_models_fit_context") is not True:
        raise ValueError("E2-A token audit contains over-context prompts")
    token_model = next(
        (row for row in token_audit.get("models", []) if row.get("manifest_id") == manifest_id),
        None,
    )
    if token_model is None or token_model.get("wrapper_sha256") != _sha256_file(context.wrapper_path):
        raise ValueError("E2-A token audit does not match the selected model wrapper")

    cases = list(read_jsonl(context.cases_path))
    if len(cases) != context.dataset_manifest["cases"]:
        raise ValueError("E2-A case count mismatch")
    blueprint = _read_object(context.blueprint_path, "E2-A blueprint")
    runs: list[dict[str, Any]] = []
    for case in cases:
        reference_branch_by_state = {
            row["state_id"]: row["branch"] for row in case["reference"]["branches"]
        }
        if len(reference_branch_by_state) != len(case["reference"]["branches"]):
            raise ValueError(f"{case['case_id']}: repeated state in reference route")
        machine = blueprint["machines"][case["domain"]]
        runs.append(
            {
                "case": case,
                "machine": machine,
                "strict": new_traversal(machine),
                "legacy": new_traversal(machine),
                "reference_branch_by_state": reference_branch_by_state,
                "strict_first_divergence_state_id": None,
                "legacy_first_divergence_state_id": None,
            }
        )

    sampler = _NvmlSampler()
    sampler.start()
    llm = None
    strict_attempted_calls = 0
    strict_invalid_calls = 0
    legacy_coerced_calls = 0
    runtime_failure_calls = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    generated_decisions = 0
    rounds = 0
    inference_seconds = 0.0
    try:
        renderer, tokenizer = _load_renderer(context)
        load_started = time.perf_counter()
        llm = _load_vllm(context)
        load_seconds = time.perf_counter() - load_started
        sampler.mark_static()
        sampling_params = _sampling_params(
            context, int(context.e2_config["inference"]["max_output_tokens"])
        )
        maximum_rounds = max(
            sum(len(state["questions"]) for state in machine["states"].values())
            for machine in blueprint["machines"].values()
        ) + 1
        inference_started = time.perf_counter()
        with restricted_jsonl_writer(output_decisions) as decision_output:
            while True:
                active = [run for run in runs if run["legacy"]["status"] == "ACTIVE"]
                if not active:
                    break
                rounds += 1
                if rounds > maximum_rounds:
                    raise RuntimeError("E2-A adaptive traversal exceeded its acyclic round bound")
                rendered_batch: list[dict[str, Any]] = []
                for run in active:
                    legacy_query = current_query(run["machine"], run["legacy"])
                    strict_active = run["strict"]["status"] == "ACTIVE"
                    if strict_active:
                        strict_query = current_query(run["machine"], run["strict"])
                        if strict_query["query_id"] != legacy_query["query_id"]:
                            raise RuntimeError(
                                f"{run['case']['case_id']}: strict and legacy paths diverged before an invalid output"
                            )
                    rendered = _render_prompt(
                        context,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        text=run["case"]["text"],
                        criterion=legacy_query["criterion"],
                    )
                    if rendered["over_context"]:
                        raise RuntimeError(
                            f"{run['case']['case_id']}:{legacy_query['query_id']}: prompt exceeds context despite token audit"
                        )
                    rendered_batch.append(
                        {
                            "run": run,
                            "query": legacy_query,
                            "strict_active": strict_active,
                            **rendered,
                        }
                    )
                generated = llm.generate(
                    [row["rendered_prompt"] for row in rendered_batch],
                    sampling_params,
                    use_tqdm=False,
                )
                if len(generated) != len(rendered_batch):
                    raise RuntimeError("vLLM returned a different number of E2-A results than requests")
                for rendered, result in zip(rendered_batch, generated):
                    run = rendered["run"]
                    query = rendered["query"]
                    choice = result.outputs[0] if result.outputs else None
                    runtime_failed = choice is None
                    raw_output = None if choice is None else choice.text
                    parsed = parse_binary_output(raw_output, runtime_failed=runtime_failed)
                    strict_branch = None
                    legacy_branch = None
                    if rendered["strict_active"]:
                        strict_attempted_calls += 1
                        if runtime_failed:
                            run["strict"]["decision_calls"] += 1
                            _fail_traversal(run["strict"], RUNTIME_FAILURE, query_id=query["query_id"])
                            if run["strict_first_divergence_state_id"] is None:
                                run["strict_first_divergence_state_id"] = run["strict"]["failure_state_id"]
                        elif parsed["strict_valid"]:
                            strict_branch = advance_traversal(
                                run["machine"], run["strict"], parsed["strict_answer"]
                            )
                            if strict_branch is not None:
                                _record_branch_divergence(run, "strict", strict_branch)
                        else:
                            strict_invalid_calls += 1
                            run["strict"]["decision_calls"] += 1
                            _fail_traversal(run["strict"], OUTPUT_FAILURE, query_id=query["query_id"])
                            if run["strict_first_divergence_state_id"] is None:
                                run["strict_first_divergence_state_id"] = run["strict"]["failure_state_id"]
                    if runtime_failed:
                        runtime_failure_calls += 1
                        run["legacy"]["decision_calls"] += 1
                        _fail_traversal(run["legacy"], RUNTIME_FAILURE, query_id=query["query_id"])
                        if run["legacy_first_divergence_state_id"] is None:
                            run["legacy_first_divergence_state_id"] = run["legacy"]["failure_state_id"]
                    else:
                        legacy_coerced_calls += int(parsed["legacy_coerced"])
                        legacy_branch = advance_traversal(
                            run["machine"], run["legacy"], parsed["legacy_answer"]
                        )
                        if legacy_branch is not None:
                            _record_branch_divergence(run, "legacy", legacy_branch)
                    prompt_tokens = len(getattr(result, "prompt_token_ids", []) or [])
                    output_tokens = 0 if choice is None else len(choice.token_ids)
                    total_prompt_tokens += prompt_tokens
                    total_output_tokens += output_tokens
                    generated_decisions += 1
                    source = (
                        set(run["case"]["positive_features"])
                        if query["reference_source"] == "positive_features"
                        else set(run["case"]["aggregate_criteria"])
                    )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "decision_id": _stable_id(
                            "e2_decision_",
                            manifest_id,
                            run["case"]["case_id"],
                            str(rounds),
                            query["query_id"],
                        ),
                        "case_id": run["case"]["case_id"],
                        "domain": run["case"]["domain"],
                        "round": rounds,
                        "state_id": run["legacy"]["failure_state_id"] if runtime_failed else (
                            legacy_branch["state_id"] if legacy_branch is not None else run["legacy"]["current_state_id"]
                        ),
                        "query_id": query["query_id"],
                        "criterion": query["criterion"],
                        "question": rendered["question"],
                        "metadata_answer": "yes" if query["criterion"] in source else "no",
                        "strict_active_before": rendered["strict_active"],
                        "prompt_sha256": hashlib.sha256(
                            rendered["rendered_prompt"].encode("utf-8")
                        ).hexdigest(),
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "raw_output": raw_output,
                        "runtime_failed": runtime_failed,
                        "finish_reason": "missing_output" if choice is None else str(choice.finish_reason),
                        "parsed": parsed,
                        "strict_branch_completed": strict_branch,
                        "legacy_branch_completed": legacy_branch,
                    }
                    write_jsonl_row(decision_output, record)
        inference_seconds = time.perf_counter() - inference_started

        case_records: list[dict[str, Any]] = []
        with restricted_jsonl_writer(output_cases) as case_output:
            for run in runs:
                strict_record = _policy_case_record(run, "strict")
                legacy_record = _policy_case_record(run, "legacy")
                if strict_record["terminal_correct"] and not legacy_record["terminal_correct"]:
                    raise RuntimeError(
                        f"{run['case']['case_id']}: legacy sensitivity cannot lose a strict-correct case"
                    )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": run["case"]["case_id"],
                    "domain": run["case"]["domain"],
                    "expected_action": run["case"]["expected_action"],
                    "reference_decision_calls": run["case"]["reference"]["decision_calls"],
                    "reference_nodes_visited": len(run["case"]["reference"]["visited_states"]),
                    "strict": strict_record,
                    "legacy": legacy_record,
                }
                case_records.append(record)
                write_jsonl_row(case_output, record)

        strict_scores = summarize_policy_cases(case_records, "strict")
        legacy_scores = summarize_policy_cases(case_records, "legacy")
        strict_diag = _path_diagnostics(
            case_records,
            "strict",
            invalid_decision_calls=strict_invalid_calls,
            attempted_decision_calls=strict_attempted_calls,
        )
        legacy_diag = _path_diagnostics(case_records, "legacy")
        technical_completion = (
            runtime_failure_calls == 0
            and legacy_diag["completed_n"] == len(case_records)
            and all(row["legacy"]["status"] != TREE_FAILURE for row in case_records)
        )
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": "slm_benchmark_e2_model",
            "experiment_id": context.e2_config["experiment_id"],
            "experiment_label": EXPERIMENT_LABEL,
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "artifact_sha256": context.manifest["verification"]["artifact_sha256"],
            "e2_config_sha256": config_sha256(context.e2_config),
            "benchmark_config_sha256": config_sha256(context.benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(context.dataset_manifest_path),
            "cases_source_sha256": context.dataset_manifest["output_cases_sha256"],
            "blueprint_sha256": context.dataset_manifest["output_blueprint_sha256"],
            "runtime_version": context.manifest["verification"]["runtime_version"],
            "generation": {
                "temperature": context.manifest["runtime"]["temperature"],
                "seed": context.manifest["runtime"]["seed"],
                "max_output_tokens": context.e2_config["inference"]["max_output_tokens"],
                "constrained_decoding": False,
                "thinking_mode": context.manifest["runtime"]["thinking_mode"],
                "adaptive_batching": context.e2_config["inference"]["adaptive_batching"],
                "multimodal_limits": _multimodal_limits(context.manifest),
            },
            "technical_completion": technical_completion,
            "case_count": len(case_records),
            "decision_count": generated_decisions,
            "rounds": rounds,
            "metrics": {
                "strict": strict_scores,
                "legacy_coercion_sensitivity": legacy_scores,
                "strict_path_diagnostics": strict_diag,
                "legacy_path_diagnostics": legacy_diag,
                "legacy_minus_strict_domain_balanced_accuracy": (
                    legacy_scores["primary_domain_balanced_terminal_accuracy"]
                    - strict_scores["primary_domain_balanced_terminal_accuracy"]
                ),
                "strict_incorrect_rescued_by_legacy_n": sum(
                    not row["strict"]["terminal_correct"] and row["legacy"]["terminal_correct"]
                    for row in case_records
                ),
                "legacy_coerced_decision_calls": legacy_coerced_calls,
            },
            "throughput": {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "adaptive_rounds": rounds,
                "generated_decisions": generated_decisions,
                "decisions_per_second": generated_decisions / inference_seconds if inference_seconds else None,
                "cases_per_second": len(case_records) / inference_seconds if inference_seconds else None,
                "mean_generated_decisions_per_case": generated_decisions / len(case_records) if case_records else None,
                "prompt_tokens": total_prompt_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens_per_second": (
                    (total_prompt_tokens + total_output_tokens) / inference_seconds
                    if inference_seconds
                    else None
                ),
                "interpretation": "Case throughput is path-dependent; compare per-decision throughput and decision counts alongside cases per second.",
            },
            "gpu_memory": None,
            "output_decisions": str(Path(output_decisions).resolve()),
            "output_decisions_sha256": _sha256_file(output_decisions),
            "output_cases": str(Path(output_cases).resolve()),
            "output_cases_sha256": _sha256_file(output_cases),
            "claim_boundary": context.e2_config["claim_boundary"],
        }
    except Exception as exc:
        audit = {
            "schema_version": SCHEMA_VERSION,
            "adapter_stage": "slm_benchmark_e2_model",
            "experiment_id": context.e2_config["experiment_id"],
            "experiment_label": EXPERIMENT_LABEL,
            "manifest_id": manifest_id,
            "model_id": context.manifest["model"]["model_id"],
            "model_revision": context.manifest["model"]["revision"],
            "e2_config_sha256": config_sha256(context.e2_config),
            "technical_completion": False,
            "runtime_failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "gpu_memory": None,
            "output_decisions": str(Path(output_decisions).resolve()),
            "output_cases": str(Path(output_cases).resolve()),
        }
    finally:
        sampler.stop()
        if llm is not None:
            del llm
        gc.collect()
    audit["gpu_memory"] = sampler.report()
    write_json(output_audit, audit)
    return audit


def compute_e2_baselines(
    cases: Sequence[Mapping[str, Any]], blueprint: Mapping[str, Any]
) -> dict[str, Any]:
    domains = sorted({str(case["domain"]) for case in cases})
    majority_actions: dict[str, str] = {}
    majority_support: dict[str, int] = {}
    for domain in domains:
        counts = Counter(
            normalize_terminal_label(str(case["expected_action"]))
            for case in cases
            if case["domain"] == domain
        )
        maximum = max(counts.values())
        majority = min(label for label, count in counts.items() if count == maximum)
        majority_actions[domain] = majority
        majority_support[domain] = maximum
    all_no_actions = {
        domain: normalize_terminal_label(blueprint["all_no_terminal_by_domain"][domain])
        for domain in domains
    }

    def score(actions: Mapping[str, str]) -> dict[str, Any]:
        rows = [
            {
                "domain": case["domain"],
                "expected_action": normalize_terminal_label(str(case["expected_action"])),
                "strict": {"predicted_action": actions[str(case["domain"])]},
                "legacy": {"predicted_action": actions[str(case["domain"])]},
            }
            for case in cases
        ]
        return summarize_policy_cases(rows, "strict")

    return {
        "per_domain_majority_terminal": {
            "status": "IN_SAMPLE_DESCRIPTIVE_BASELINE",
            "terminal_by_domain": majority_actions,
            "support_by_domain": majority_support,
            "metrics": score(majority_actions),
        },
        "deterministic_all_no_traversal": {
            "terminal_by_domain": all_no_actions,
            "metrics": score(all_no_actions),
        },
        "metadata_reference_route": {
            "status": "PREPARATION_INTEGRITY_CHECK_NOT_A_PERFORMANCE_BASELINE",
            "terminal_matches": len(cases),
            "n": len(cases),
        },
    }


def _mcnemar_exact(a_only: int, b_only: int) -> float | None:
    discordant = a_only + b_only
    if discordant == 0:
        return None
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _holm_adjust(values: Sequence[float | None]) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    indexed = sorted((value, index) for index, value in enumerate(values) if value is not None)
    running = 0.0
    count = len(indexed)
    for rank, (value, index) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        output[index] = running
    return output


def _interval(values: Sequence[float], confidence: float = 0.95) -> list[float]:
    if not values:
        raise ValueError("cannot calculate an interval from no values")
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    lower = ordered[math.floor(tail * (len(ordered) - 1))]
    upper = ordered[math.ceil((1.0 - tail) * (len(ordered) - 1))]
    return [lower, upper]


def stratified_bootstrap_scores(
    correct_by_model: Mapping[str, Mapping[str, bool]],
    domain_by_case: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Shared-draw vignette bootstrap of the domain-balanced correctness endpoint."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    case_ids = set(domain_by_case)
    if not case_ids:
        raise ValueError("bootstrap case set is empty")
    for model, values in correct_by_model.items():
        if set(values) != case_ids:
            raise ValueError(f"{model}: bootstrap case set differs")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for case_id, domain in domain_by_case.items():
        by_domain[str(domain)].append(case_id)
    if len(by_domain) < 2:
        raise ValueError("stratified domain-balanced bootstrap requires multiple domains")
    for ids in by_domain.values():
        ids.sort()
    output = {model: [] for model in correct_by_model}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = {
            domain: [ids[rng.randrange(len(ids))] for _ in ids]
            for domain, ids in sorted(by_domain.items())
        }
        for model, correctness in correct_by_model.items():
            domain_accuracies = [
                fmean(bool(correctness[case_id]) for case_id in ids)
                for _domain, ids in sampled.items()
            ]
            output[model].append(fmean(domain_accuracies))
    return output


def _midranks_descending(scores: Mapping[str, float]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for model, score in scores.items():
        greater = sum(value > score for value in scores.values())
        equal = sum(value == score for value in scores.values())
        ranks[model] = 1.0 + greater + (equal - 1) / 2.0
    return ranks


def aggregate_e2_results(
    config_path: str | Path,
    *,
    dataset_manifest_path: str | Path,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source, config, _benchmark_path, benchmark, _prompt_path, _prompt = _load_e2_config(config_path)
    dataset_manifest = _read_object(Path(dataset_manifest_path), "E2-A dataset manifest")
    if dataset_manifest.get("e2_config_sha256") != config_sha256(config):
        raise ValueError("E2-A aggregate dataset is stale")
    dataset_cases = list(read_jsonl(dataset_manifest["output_cases"]))
    blueprint = _read_object(Path(dataset_manifest["output_blueprint"]), "E2-A blueprint")
    source_by_case = {case["case_id"]: case for case in dataset_cases}
    if len(source_by_case) != len(dataset_cases):
        raise ValueError("E2-A source case IDs are not unique")
    declared = [*benchmark["panel"]["core_pilot"], *benchmark["panel"]["contemporary_references"]]
    root = Path(results_root)
    model_case_rows: dict[str, dict[str, dict[str, Any]]] = {}
    model_rows: list[dict[str, Any]] = []
    for manifest_id in declared:
        context = _load_model_context(
            config_path, manifest_id=manifest_id, dataset_manifest_path=dataset_manifest_path
        )
        audit_path = root / manifest_id / "audit.json"
        audit = _read_object(audit_path, "E2-A model audit")
        expected_hashes = {
            "e2_config_sha256": config_sha256(config),
            "benchmark_config_sha256": config_sha256(benchmark),
            "manifest_sha256": config_sha256(context.manifest),
            "wrapper_sha256": _sha256_file(context.wrapper_path),
            "prompt_sha256": config_sha256(context.prompt),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "cases_source_sha256": dataset_manifest["output_cases_sha256"],
            "blueprint_sha256": dataset_manifest["output_blueprint_sha256"],
        }
        if audit.get("manifest_id") != manifest_id:
            raise ValueError(f"{audit_path}: manifest ID mismatch")
        for field, expected in expected_hashes.items():
            if audit.get(field) != expected:
                raise ValueError(f"{audit_path}: stale {field}")
        if audit.get("technical_completion") is not True:
            raise ValueError(f"{audit_path}: model inference is not technically complete")
        decision_path = Path(audit["output_decisions"])
        if _sha256_file(decision_path) != audit.get("output_decisions_sha256"):
            raise ValueError(f"{decision_path}: decision-result hash mismatch")
        decision_ids: set[str] = set()
        decision_count = 0
        for decision in read_jsonl(decision_path):
            decision_count += 1
            decision_id = str(decision.get("decision_id"))
            if decision_id in decision_ids:
                raise ValueError(f"{decision_path}: duplicate decision ID {decision_id}")
            decision_ids.add(decision_id)
            if decision.get("case_id") not in source_by_case:
                raise ValueError(f"{decision_path}: decision refers to an unknown case")
        if decision_count != audit.get("decision_count"):
            raise ValueError(f"{decision_path}: decision row count mismatch")
        case_path = Path(audit["output_cases"])
        if _sha256_file(case_path) != audit.get("output_cases_sha256"):
            raise ValueError(f"{case_path}: result hash mismatch")
        rows = {row["case_id"]: row for row in read_jsonl(case_path)}
        if set(rows) != set(source_by_case):
            raise ValueError(f"{manifest_id}: case result IDs differ from the frozen dataset")
        for case_id, row in rows.items():
            source_case = source_by_case[case_id]
            if (
                row.get("domain") != source_case["domain"]
                or row.get("expected_action") != source_case["expected_action"]
            ):
                raise ValueError(f"{manifest_id}:{case_id}: frozen case metadata mismatch")
            if row["strict"]["terminal_correct"] and not row["legacy"]["terminal_correct"]:
                raise ValueError(f"{manifest_id}:{case_id}: invalid strict-to-legacy loss")
        model_case_rows[manifest_id] = rows
        ordered_rows = [rows[case_id] for case_id in sorted(rows)]
        strict_metrics = summarize_policy_cases(ordered_rows, "strict")
        legacy_metrics = summarize_policy_cases(ordered_rows, "legacy")
        if strict_metrics != audit["metrics"]["strict"] or legacy_metrics != audit["metrics"]["legacy_coercion_sensitivity"]:
            raise ValueError(f"{audit_path}: stored metrics do not match case-level results")
        model_rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": audit["model_id"],
                "panel_role": context.manifest["factors"]["panel_role"],
                "family": context.manifest["factors"]["family"],
                "size_class": context.manifest["factors"]["size_class"],
                "instruction_tuned": context.manifest["factors"]["instruction_tuned"],
                "medical_adapted": context.manifest["factors"]["medical_adapted"],
                "technical_completion": bool(audit["technical_completion"]),
                "strict": strict_metrics,
                "legacy_coercion_sensitivity": legacy_metrics,
                "path_diagnostics": {
                    "strict": audit["metrics"]["strict_path_diagnostics"],
                    "legacy": audit["metrics"]["legacy_path_diagnostics"],
                },
                "throughput": audit["throughput"],
                "gpu_memory": audit["gpu_memory"],
                "audit_path": str(audit_path),
            }
        )

    domain_by_case = {case_id: str(case["domain"]) for case_id, case in source_by_case.items()}
    strict_correct = {
        model: {case_id: bool(row["strict"]["terminal_correct"]) for case_id, row in rows.items()}
        for model, rows in model_case_rows.items()
    }
    legacy_correct = {
        model: {case_id: bool(row["legacy"]["terminal_correct"]) for case_id, row in rows.items()}
        for model, rows in model_case_rows.items()
    }
    statistics = config["statistics"]
    replicates = int(statistics["bootstrap_replicates"])
    seed = int(statistics["bootstrap_seed"])
    strict_bootstrap = stratified_bootstrap_scores(
        strict_correct, domain_by_case, replicates=replicates, seed=seed
    )
    legacy_bootstrap = stratified_bootstrap_scores(
        legacy_correct, domain_by_case, replicates=replicates, seed=seed
    )
    point_scores = {
        row["manifest_id"]: float(row["strict"]["primary_domain_balanced_terminal_accuracy"])
        for row in model_rows
    }
    point_ranks = _midranks_descending(point_scores)
    rank_samples: dict[str, list[float]] = {model: [] for model in declared}
    top_counts: Counter[str] = Counter()
    for replicate in range(replicates):
        values = {model: strict_bootstrap[model][replicate] for model in declared}
        ranks = _midranks_descending(values)
        maximum = max(values.values())
        for model in declared:
            rank_samples[model].append(ranks[model])
            top_counts[model] += int(values[model] == maximum)
    by_model_row = {row["manifest_id"]: row for row in model_rows}
    within_model_sensitivity: list[dict[str, Any]] = []
    for model in declared:
        row = by_model_row[model]
        row["leaderboard"] = {
            "point_rank_midrank": point_ranks[model],
            "bootstrap_rank_95_interval": _interval(rank_samples[model]),
            "bootstrap_probability_in_top_tie_set": top_counts[model] / replicates,
            "strict_primary_score_95_interval": _interval(strict_bootstrap[model]),
        }
        differences = [
            legacy - strict
            for strict, legacy in zip(strict_bootstrap[model], legacy_bootstrap[model])
        ]
        strict_n = sum(strict_correct[model].values())
        legacy_n = sum(legacy_correct[model].values())
        strict_only = sum(
            strict_correct[model][case_id] and not legacy_correct[model][case_id]
            for case_id in source_by_case
        )
        legacy_only = sum(
            legacy_correct[model][case_id] and not strict_correct[model][case_id]
            for case_id in source_by_case
        )
        if strict_only:
            raise ValueError(f"{model}: legacy sensitivity lost strict-correct cases")
        within_model_sensitivity.append(
            {
                "manifest_id": model,
                "direction": "legacy coercion minus strict parser",
                "strict_correct": strict_n,
                "legacy_correct": legacy_n,
                "rescued_cases": legacy_only,
                "lost_cases": strict_only,
                "domain_balanced_accuracy_difference": (
                    row["legacy_coercion_sensitivity"]["primary_domain_balanced_terminal_accuracy"]
                    - row["strict"]["primary_domain_balanced_terminal_accuracy"]
                ),
                "paired_stratified_bootstrap_95_ci": _interval(differences),
            }
        )

    contrasts: list[dict[str, Any]] = []
    core_p_values: list[float | None] = []
    for contrast_index, contrast in enumerate(benchmark["contrasts"]):
        a_id, b_id = contrast["models"]
        differences = [
            b - a for a, b in zip(strict_bootstrap[a_id], strict_bootstrap[b_id])
        ]
        a_only = sum(
            strict_correct[a_id][case_id] and not strict_correct[b_id][case_id]
            for case_id in source_by_case
        )
        b_only = sum(
            strict_correct[b_id][case_id] and not strict_correct[a_id][case_id]
            for case_id in source_by_case
        )
        p_value = _mcnemar_exact(a_only, b_only)
        row = {
            "contrast_id": contrast["contrast_id"],
            "claim_type": contrast["claim_type"],
            "factor": contrast["factor"],
            "model_a": a_id,
            "model_b": b_id,
            "direction": "model_b minus model_a",
            "model_a_domain_balanced_accuracy": point_scores[a_id],
            "model_b_domain_balanced_accuracy": point_scores[b_id],
            "domain_balanced_accuracy_difference": point_scores[b_id] - point_scores[a_id],
            "paired_stratified_bootstrap_95_ci": _interval(differences),
            "pooled_discordant_model_a_only_correct": a_only,
            "pooled_discordant_model_b_only_correct": b_only,
            "pooled_mcnemar_exact_two_sided_p": p_value,
            "pooled_mcnemar_holm_p_core_five": None,
            "interpretation": contrast["interpretation"],
        }
        contrasts.append(row)
        if contrast_index < 5:
            core_p_values.append(p_value)
    adjusted = _holm_adjust(core_p_values)
    for index, value in enumerate(adjusted):
        contrasts[index]["pooled_mcnemar_holm_p_core_five"] = value

    model_rows.sort(
        key=lambda row: (
            -row["strict"]["primary_domain_balanced_terminal_accuracy"],
            row["manifest_id"],
        )
    )
    baselines = compute_e2_baselines(dataset_cases, blueprint)
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_stage": "slm_benchmark_e2_aggregate",
        "experiment_id": config["experiment_id"],
        "experiment_label": EXPERIMENT_LABEL,
        "purpose": "standardized historical forced-binary adaptive tree-traversal bridge",
        "e2_config": str(source),
        "e2_config_sha256": config_sha256(config),
        "benchmark_config_sha256": config_sha256(benchmark),
        "dataset_manifest": str(Path(dataset_manifest_path).resolve()),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "declared_models": len(declared),
        "cases_per_model": len(dataset_cases),
        "primary_endpoint": statistics["primary_endpoint"],
        "all_models_technically_completed": all(row["technical_completion"] for row in model_rows),
        "leaderboard_order": [row["manifest_id"] for row in model_rows],
        "models": model_rows,
        "within_model_legacy_sensitivity": within_model_sensitivity,
        "paired_contrasts": contrasts,
        "baselines": baselines,
        "bootstrap": {
            "method": "paired vignette resampling within each domain with shared draws across all models",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": statistics["confidence_level"],
        },
        "scope_status": config["scope_status"],
        "prior_exposure": config["prior_exposure"],
        "claim_boundary": config["claim_boundary"],
        "efficiency_boundary": "Adaptive paths differ by model; raw case throughput is not a standalone model-quality ranking.",
    }
    write_json(output, result)
    return result
