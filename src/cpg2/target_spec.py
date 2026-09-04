from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .registry import CriterionRegistry


@dataclass(frozen=True)
class Target:
    criterion_id: str
    criterion_text: str
    source_ids: frozenset[str]
    query_terms: tuple[str, ...]


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _read_object(value: dict[str, Any] | str | Path, name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result = dict(value)
    else:
        source = Path(value)
        try:
            result = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read {name} {source}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a JSON object")
    return result


def legacy_feature_name(node: Mapping[str, Any]) -> str:
    """Recover the source feature name used to construct the frozen E1 labels."""

    metadata = node.get("normalization_metadata")
    if isinstance(metadata, Mapping):
        legacy_name = metadata.get("legacy_name")
        if isinstance(legacy_name, str) and legacy_name:
            return legacy_name
    reference = node.get("source_reference")
    if isinstance(reference, str) and "/" in reference:
        section_reference = reference.split(":", maxsplit=1)[-1]
        if "/" in section_reference:
            return section_reference.split("/", maxsplit=1)[1]
    raise ValueError(f"criterion node {node.get('node_id')} lacks a legacy feature name")


def _registry_target_texts(
    registry: CriterionRegistry,
) -> tuple[dict[str, str], set[str]]:
    texts: dict[str, str] = {}
    aggregate_parents: set[str] = set()
    for definition in registry:
        texts[definition.criterion_id] = definition.criterion_text
        if definition.aggregation_rule is not None:
            aggregate_parents.add(definition.criterion_id)
            for component in definition.aggregation_rule["components"]:
                texts[component["criterion_id"]] = component["criterion_text"]
    return texts, aggregate_parents


def load_targets(
    value: dict[str, Any] | str | Path,
    *,
    registry: CriterionRegistry,
) -> tuple[dict[str, Any], list[Target]]:
    """Load the frozen target subset used by E1 request construction."""

    spec = _read_object(value, "target spec")
    if spec.get("schema_version") != "1.0":
        raise ValueError("target spec schema_version must be 1.0")
    definitions = list(registry)
    domains = {definition.domain for definition in definitions}
    if len(domains) != 1 or spec.get("domain") != next(iter(domains)):
        raise ValueError("target spec domain does not match the registry")
    retrieval = spec.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("target spec requires retrieval")
    _require_string(retrieval.get("version"), "retrieval.version")
    top_k = retrieval.get("top_k_per_note")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k_per_note must be a positive integer")

    texts, aggregate_parents = _registry_target_texts(registry)
    raw_targets = spec.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("target spec requires targets")
    targets: list[Target] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{index}] must be an object")
        criterion_id = _require_string(
            item.get("criterion_id"), f"targets[{index}].criterion_id"
        )
        if criterion_id in seen:
            raise ValueError(f"duplicate target {criterion_id}")
        seen.add(criterion_id)
        if criterion_id not in texts:
            raise ValueError(f"target {criterion_id!r} is not in the registry")
        if criterion_id in aggregate_parents:
            raise ValueError(f"targets must be components, not aggregate parent {criterion_id!r}")
        source_ids = item.get("source_ids")
        query_terms = item.get("query_terms")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not all(isinstance(source_id, str) and source_id for source_id in source_ids)
        ):
            raise ValueError(f"targets[{index}].source_ids must be non-empty strings")
        if (
            not isinstance(query_terms, list)
            or not query_terms
            or not all(isinstance(term, str) and term.strip() for term in query_terms)
        ):
            raise ValueError(f"targets[{index}].query_terms must be non-empty strings")
        targets.append(
            Target(
                criterion_id=criterion_id,
                criterion_text=_require_string(
                    item.get("criterion_definition", texts[criterion_id]),
                    f"targets[{index}].criterion_definition",
                ),
                source_ids=frozenset(source_ids),
                query_terms=tuple(query_terms),
            )
        )
    return spec, targets
