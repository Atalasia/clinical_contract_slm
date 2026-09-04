from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .states import CriterionState


class TreeValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TreeAction:
    action_id: str
    action_label: str


@dataclass(frozen=True)
class TreeNode:
    node_id: str
    node_type: str
    criterion_id: str | None = None
    question: str | None = None
    criterion_type: str | None = None
    temporal_rule: dict[str, Any] | None = None
    state_branch_map: dict[CriterionState, tuple[str, ...]] | None = None
    branches: dict[str, str] | None = None
    branch_actions: dict[str, tuple[TreeAction, ...]] | None = None
    terminal_action_id: str | None = None
    terminal_action_label: str | None = None
    gap_reason: str | None = None
    source_reference: str | None = None


@dataclass(frozen=True)
class DecisionTree:
    tree_id: str
    domain: str
    version: str
    root_id: str
    nodes: dict[str, TreeNode]

    @property
    def criterion_nodes(self) -> tuple[TreeNode, ...]:
        return tuple(node for node in self.nodes.values() if node.node_type == "criterion")


def _nonempty(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return False
    return True


def validate_tree(data: dict[str, Any]) -> dict[str, list[str]]:
    """Validate a normalized tree and return errors/warnings without coercion."""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return {"errors": ["tree must be a JSON object"], "warnings": []}
    for field in ("tree_id", "domain", "version", "root_id"):
        _nonempty(data.get(field), field, errors)
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return {"errors": errors + ["nodes must be a non-empty list"], "warnings": warnings}

    ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        node_id = node.get("node_id")
        if not _nonempty(node_id, f"nodes[{index}].node_id", errors):
            continue
        if node_id in by_id:
            errors.append(f"duplicate node_id: {node_id}")
            continue
        ids.append(node_id)
        by_id[node_id] = node

    root_id = data.get("root_id")
    if root_id not in by_id:
        errors.append("root_id does not reference an existing node")

    edges: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    criterion_ids: dict[str, list[str]] = {}
    required_states = {state.value for state in CriterionState}
    for node_id, node in by_id.items():
        node_type = node.get("node_type")
        if node_type not in {"criterion", "terminal", "path_end", "configuration_gap"}:
            errors.append(
                f"{node_id}: node_type must be criterion, terminal, path_end, or configuration_gap"
            )
            continue
        if not _nonempty(node.get("source_reference"), f"{node_id}.source_reference", errors):
            pass
        if node_type == "terminal":
            _nonempty(node.get("terminal_action_id"), f"{node_id}.terminal_action_id", errors)
            _nonempty(node.get("terminal_action_label"), f"{node_id}.terminal_action_label", errors)
            forbidden = {"criterion_id", "state_branch_map", "branches"} & node.keys()
            if forbidden:
                errors.append(f"{node_id}: terminal contains criterion fields {sorted(forbidden)}")
            continue
        if node_type == "path_end":
            forbidden = {"criterion_id", "state_branch_map", "branches", "terminal_action_id"} & node.keys()
            if forbidden:
                errors.append(f"{node_id}: path_end contains incompatible fields {sorted(forbidden)}")
            continue
        if node_type == "configuration_gap":
            _nonempty(node.get("gap_reason"), f"{node_id}.gap_reason", errors)
            forbidden = {"criterion_id", "state_branch_map", "branches", "terminal_action_id"} & node.keys()
            if forbidden:
                errors.append(
                    f"{node_id}: configuration_gap contains incompatible fields {sorted(forbidden)}"
                )
            continue

        criterion_id = node.get("criterion_id")
        if _nonempty(criterion_id, f"{node_id}.criterion_id", errors):
            criterion_ids.setdefault(criterion_id, []).append(node_id)
        _nonempty(node.get("question"), f"{node_id}.question", errors)
        _nonempty(node.get("criterion_type"), f"{node_id}.criterion_type", errors)
        temporal = node.get("temporal_rule")
        if not isinstance(temporal, dict) or temporal.get("cutoff_relation") not in {
            "at_or_before",
            "history_at_or_before",
        }:
            errors.append(f"{node_id}.temporal_rule requires a supported cutoff_relation")
        elif "lookback_hours" not in temporal:
            errors.append(f"{node_id}.temporal_rule requires lookback_hours (null is allowed)")
        elif temporal["lookback_hours"] is not None and (
            isinstance(temporal["lookback_hours"], bool)
            or not isinstance(temporal["lookback_hours"], (int, float))
            or temporal["lookback_hours"] <= 0
        ):
            errors.append(f"{node_id}.temporal_rule.lookback_hours must be positive or null")

        branches = node.get("branches")
        if not isinstance(branches, dict) or len(branches) < 2:
            errors.append(f"{node_id}.branches must define at least two branches")
            branches = {}
        else:
            for branch, target in branches.items():
                if not isinstance(branch, str) or not branch:
                    errors.append(f"{node_id}: branch names must be non-empty strings")
                if target not in by_id:
                    errors.append(f"{node_id}: branch {branch!r} references missing node {target!r}")
                else:
                    edges[node_id].add(target)

        branch_actions = node.get("branch_actions", {})
        if not isinstance(branch_actions, dict):
            errors.append(f"{node_id}.branch_actions must be an object when supplied")
        else:
            invalid_action_branches = set(branch_actions) - set(branches)
            if invalid_action_branches:
                errors.append(
                    f"{node_id}: branch_actions has unknown branches {sorted(invalid_action_branches)}"
                )
            for branch, actions in branch_actions.items():
                if not isinstance(actions, list) or not actions:
                    errors.append(f"{node_id}: branch_actions[{branch!r}] must be a non-empty list")
                    continue
                for index, action in enumerate(actions):
                    if not isinstance(action, dict):
                        errors.append(f"{node_id}: branch action {branch}[{index}] must be an object")
                        continue
                    if set(action) != {"action_id", "action_label"}:
                        errors.append(
                            f"{node_id}: branch action {branch}[{index}] requires only action_id/action_label"
                        )
                        continue
                    _nonempty(action.get("action_id"), f"{node_id}.{branch}[{index}].action_id", errors)
                    _nonempty(
                        action.get("action_label"), f"{node_id}.{branch}[{index}].action_label", errors
                    )

        state_map = node.get("state_branch_map")
        if not isinstance(state_map, dict):
            errors.append(f"{node_id}.state_branch_map must be an object")
            continue
        unknown_states = set(state_map) - required_states
        missing_states = required_states - set(state_map)
        if unknown_states:
            errors.append(f"{node_id}: unknown states in state_branch_map: {sorted(unknown_states)}")
        if missing_states:
            errors.append(f"{node_id}: state_branch_map is missing states: {sorted(missing_states)}")
        for state, admissible in state_map.items():
            if not isinstance(admissible, list) or not admissible:
                errors.append(f"{node_id}: state {state} must map to a non-empty branch list")
                continue
            if len(set(admissible)) != len(admissible):
                errors.append(f"{node_id}: state {state} contains duplicate branches")
            invalid = set(admissible) - set(branches)
            if invalid:
                errors.append(f"{node_id}: state {state} maps to unknown branches {sorted(invalid)}")

    for criterion_id, node_ids in criterion_ids.items():
        if len(node_ids) > 1:
            warnings.append(
                f"criterion_id {criterion_id!r} is reused by nodes {sorted(node_ids)}; "
                "node-specific semantics remain authoritative"
            )

    if root_id in by_id:
        reachable: set[str] = set()
        pending = [root_id]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(sorted(edges.get(current, ()), reverse=True))
        unreachable = sorted(set(by_id) - reachable)
        for node_id in unreachable:
            errors.append(f"unreachable node: {node_id}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str, stack: list[str]) -> None:
            if node_id in visiting:
                cycle_start = stack.index(node_id)
                errors.append("cycle detected: " + " -> ".join(stack[cycle_start:] + [node_id]))
                return
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in sorted(edges.get(node_id, ())):
                visit(target, stack + [target])
            visiting.remove(node_id)
            visited.add(node_id)

        visit(root_id, [root_id])

    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def tree_from_dict(data: dict[str, Any]) -> DecisionTree:
    report = validate_tree(data)
    if report["errors"]:
        raise TreeValidationError("Invalid normalized tree: " + "; ".join(report["errors"]))
    nodes: dict[str, TreeNode] = {}
    for raw in data["nodes"]:
        if raw["node_type"] == "terminal":
            node = TreeNode(
                node_id=raw["node_id"],
                node_type="terminal",
                terminal_action_id=raw["terminal_action_id"],
                terminal_action_label=raw["terminal_action_label"],
                source_reference=raw["source_reference"],
            )
        elif raw["node_type"] == "path_end":
            node = TreeNode(
                node_id=raw["node_id"],
                node_type="path_end",
                source_reference=raw["source_reference"],
            )
        elif raw["node_type"] == "configuration_gap":
            node = TreeNode(
                node_id=raw["node_id"],
                node_type="configuration_gap",
                gap_reason=raw["gap_reason"],
                source_reference=raw["source_reference"],
            )
        else:
            node = TreeNode(
                node_id=raw["node_id"],
                node_type="criterion",
                criterion_id=raw["criterion_id"],
                question=raw["question"],
                criterion_type=raw["criterion_type"],
                temporal_rule=dict(raw["temporal_rule"]),
                state_branch_map={
                    CriterionState(state): tuple(branches)
                    for state, branches in raw["state_branch_map"].items()
                },
                branches=dict(raw["branches"]),
                branch_actions={
                    branch: tuple(
                        TreeAction(action["action_id"], action["action_label"])
                        for action in actions
                    )
                    for branch, actions in raw.get("branch_actions", {}).items()
                },
                source_reference=raw["source_reference"],
            )
        nodes[node.node_id] = node
    return DecisionTree(
        tree_id=data["tree_id"],
        domain=data["domain"],
        version=data["version"],
        root_id=data["root_id"],
        nodes=nodes,
    )


def load_tree(path: str | Path) -> DecisionTree:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TreeValidationError(f"Could not load tree {source}: {exc}") from exc
    return tree_from_dict(data)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized:
        return normalized
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def audit_legacy_tree(path: str | Path) -> dict[str, Any]:
    """Audit old CPGPrompt files without guessing missing machine-readable edges."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(source), "format": "unreadable", "errors": [str(exc)], "warnings": []}
    if not isinstance(data, dict):
        return {"path": str(source), "format": "unknown", "errors": ["root must be an object"], "warnings": []}
    if isinstance(data.get("nodes"), list):
        sections = {source.stem: data["nodes"]}
    else:
        sections = {
            str(name).strip(): section["nodes"]
            for name, section in data.items()
            if isinstance(section, dict) and isinstance(section.get("nodes"), list)
        }
    errors: list[str] = []
    warnings: list[str] = []
    reports: dict[str, Any] = {}
    for section, nodes in sections.items():
        labels = [_slug(str(node.get("name", ""))) for node in nodes if isinstance(node, dict)]
        by_label = {label: node for label, node in zip(labels, nodes)}
        if len(by_label) != len(labels):
            errors.append(f"{section}: duplicate normalized node labels")
        edges: dict[str, str] = {}
        explicit_terminals = 0
        for label, node in by_label.items():
            name = str(node.get("name", label))
            if not isinstance(node.get("yes_action"), str) or not node["yes_action"].strip():
                errors.append(f"{section}/{name}: yes_action is missing")
            no_node = node.get("no_node")
            if no_node is None:
                warnings.append(f"{section}/{name}: no_node is absent; terminal behavior is not explicit")
            elif str(no_node).strip().casefold() == "end of decision tree":
                explicit_terminals += 1
            else:
                target = _slug(str(no_node))
                if target not in by_label:
                    errors.append(f"{section}/{name}: dangling no_node {no_node!r}")
                else:
                    edges[label] = target
            action = str(node.get("yes_action", ""))
            if re.search(r"\b(?:proceed|continue) to (?:the )?node\b", action, re.IGNORECASE):
                warnings.append(
                    f"{section}/{name}: action text describes continuation but no yes_node field exists"
                )
            criteria = node.get("criteria")
            minimum = node.get("min_criteria")
            if criteria is not None:
                if not isinstance(criteria, list) or not criteria:
                    errors.append(f"{section}/{name}: criteria must be a non-empty list")
                elif not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= minimum <= len(criteria):
                    errors.append(f"{section}/{name}: min_criteria is invalid")
                warnings.append(
                    f"{section}/{name}: legacy aggregate criteria require explicit normalized criterion semantics"
                )
            warnings.append(f"{section}/{name}: criterion type, temporal rule, and source passage are not encoded")

        reachable: set[str] = set()
        current = labels[0] if labels else None
        while current in by_label and current not in reachable:
            reachable.add(current)
            current = edges.get(current)
        unreachable = sorted(set(labels) - reachable)
        for label in unreachable:
            errors.append(f"{section}: unreachable under encoded edges: {label}")
        reports[section] = {
            "node_count": len(labels),
            "machine_reachable_count": len(reachable),
            "unreachable_node_ids": unreachable,
            "explicit_end_count": explicit_terminals,
        }
    if not sections:
        errors.append("no legacy CPGPrompt sections recognized")
    return {
        "path": str(source),
        "format": "legacy_cpgprompt",
        "sections": reports,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "executable_as_normalized": False,
    }
