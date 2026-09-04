from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .states import AnchorEligibility, CriterionState


COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class CriterionDefinition:
    criterion_id: str
    domain: str
    node_ids: tuple[str, ...]
    criterion_text: str
    criterion_type: str
    resolution_mode: AnchorEligibility
    state_space: tuple[CriterionState, ...]
    temporal_rule: dict[str, Any]
    missing_policy: str
    source_reference: str
    structured_sources: tuple[dict[str, Any], ...] = ()
    note_search_terms: tuple[str, ...] = ()
    numeric_rule: dict[str, Any] | None = None
    anchor_eligible: bool = False
    counterfactual_types: tuple[str, ...] = ()
    merge_policy: str = "preserve_conflict"
    aggregation_rule: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionDefinition":
        required = {
            "criterion_id",
            "domain",
            "node_ids",
            "criterion_text",
            "criterion_type",
            "resolution_mode",
            "state_space",
            "temporal_rule",
            "missing_policy",
            "source_reference",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"criterion missing fields: {sorted(missing)}")
        if data["missing_policy"] != "unresolved":
            raise ValueError(f"{data['criterion_id']}: missing_policy must be unresolved")
        if not isinstance(data["node_ids"], list) or not data["node_ids"]:
            raise ValueError(f"{data['criterion_id']}: node_ids must be a non-empty list")
        temporal = data["temporal_rule"]
        if not isinstance(temporal, dict) or temporal.get("cutoff_relation") not in {
            "at_or_before",
            "history_at_or_before",
        } or "lookback_hours" not in temporal:
            raise ValueError(f"{data['criterion_id']}: invalid temporal_rule")
        state_space = tuple(CriterionState(value) for value in data["state_space"])
        if len(set(state_space)) != len(state_space):
            raise ValueError(f"{data['criterion_id']}: duplicate states")
        numeric = data.get("numeric_rule")
        if numeric is not None:
            if numeric.get("operator") not in COMPARATORS:
                raise ValueError(f"{data['criterion_id']}: invalid numeric operator")
            if "threshold" not in numeric or numeric["threshold"] is None:
                raise ValueError(f"{data['criterion_id']}: numeric threshold is not frozen")
            if isinstance(numeric["threshold"], bool) or not isinstance(numeric["threshold"], (int, float)):
                raise ValueError(f"{data['criterion_id']}: numeric threshold must be numeric")
            if not isinstance(numeric.get("canonical_unit"), str) or not numeric["canonical_unit"]:
                raise ValueError(f"{data['criterion_id']}: canonical_unit is required")
        aggregation = data.get("aggregation_rule")
        if aggregation is not None:
            if not isinstance(aggregation, dict) or aggregation.get("operator") != "at_least":
                raise ValueError(f"{data['criterion_id']}: invalid aggregation_rule")
            components = aggregation.get("components")
            minimum = aggregation.get("min_present")
            if not isinstance(components, list) or not components:
                raise ValueError(f"{data['criterion_id']}: aggregation components are required")
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or not 1 <= minimum <= len(components)
            ):
                raise ValueError(f"{data['criterion_id']}: invalid aggregation min_present")
            component_ids: set[str] = set()
            for index, component in enumerate(components):
                if not isinstance(component, dict) or set(component) != {"criterion_id", "criterion_text"}:
                    raise ValueError(
                        f"{data['criterion_id']}: component {index} requires criterion_id/criterion_text"
                    )
                if not all(isinstance(component[key], str) and component[key] for key in component):
                    raise ValueError(f"{data['criterion_id']}: component {index} fields are empty")
                if component["criterion_id"] in component_ids:
                    raise ValueError(f"{data['criterion_id']}: duplicate aggregation component")
                component_ids.add(component["criterion_id"])
        mode = AnchorEligibility(data["resolution_mode"])
        anchor_eligible = bool(data.get("anchor_eligible", False))
        if anchor_eligible and mode not in {
            AnchorEligibility.STRUCTURED_ANCHOR_ELIGIBLE,
            AnchorEligibility.DERIVED_DETERMINISTIC,
        }:
            raise ValueError(f"{data['criterion_id']}: anchor_eligible conflicts with resolution_mode")
        return cls(
            criterion_id=data["criterion_id"],
            domain=data["domain"],
            node_ids=tuple(data["node_ids"]),
            criterion_text=data["criterion_text"],
            criterion_type=data["criterion_type"],
            resolution_mode=mode,
            state_space=state_space,
            temporal_rule=dict(temporal),
            missing_policy=data["missing_policy"],
            source_reference=data["source_reference"],
            structured_sources=tuple(data.get("structured_sources", [])),
            note_search_terms=tuple(data.get("note_search_terms", [])),
            numeric_rule=dict(numeric) if numeric is not None else None,
            anchor_eligible=anchor_eligible,
            counterfactual_types=tuple(data.get("counterfactual_types", [])),
            merge_policy=data.get("merge_policy", "preserve_conflict"),
            aggregation_rule=dict(aggregation) if aggregation is not None else None,
        )


class CriterionRegistry:
    def __init__(self, *, version: str, criteria: list[CriterionDefinition]):
        if not version:
            raise ValueError("registry version is required")
        self.version = version
        self._criteria = {item.criterion_id: item for item in criteria}
        if len(self._criteria) != len(criteria):
            raise ValueError("duplicate criterion_id in registry")

    def __getitem__(self, criterion_id: str) -> CriterionDefinition:
        try:
            return self._criteria[criterion_id]
        except KeyError as exc:
            raise KeyError(f"criterion {criterion_id!r} is not registered") from exc

    def __iter__(self):
        return iter(self._criteria.values())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionRegistry":
        if not isinstance(data, dict) or not isinstance(data.get("criteria"), list):
            raise ValueError("registry must contain a criteria list")
        return cls(
            version=str(data.get("version", "")),
            criteria=[CriterionDefinition.from_dict(item) for item in data["criteria"]],
        )

    @classmethod
    def load(cls, path: str | Path) -> "CriterionRegistry":
        source = Path(path)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def validate_tree_coverage(self, tree: Any) -> list[str]:
        errors: list[str] = []
        for node in tree.criterion_nodes:
            if node.criterion_id not in self._criteria:
                errors.append(f"tree criterion missing from registry: {node.criterion_id}")
                continue
            definition = self._criteria[node.criterion_id]
            if node.node_id not in definition.node_ids:
                errors.append(f"{node.criterion_id}: registry does not list node {node.node_id}")
            if definition.domain != tree.domain:
                errors.append(f"{node.criterion_id}: registry domain does not match tree")
            node_states = set(node.state_branch_map or {})
            registry_states = set(definition.state_space)
            if node_states != registry_states:
                errors.append(f"{node.criterion_id}: registry and node state spaces differ")
        return sorted(errors)
