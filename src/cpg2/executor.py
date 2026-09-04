from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .schemas import CriterionAssessment
from .states import CriterionState
from .tree import DecisionTree, TreeNode


@dataclass(frozen=True)
class ExecutionFailure:
    node_id: str
    reason: str


@dataclass(frozen=True)
class ExecutionResult:
    patient_key: str
    encounter_key: str
    domain: str
    pipeline_id: str
    tree_version: str
    decision_cutoff: str
    reachable_path_ids: tuple[str, ...]
    reachable_paths: tuple[tuple[str, ...], ...]
    reachable_leaf_ids: tuple[str, ...]
    compatible_action_ids: tuple[str, ...]
    compatible_action_sequences: tuple[tuple[str, ...], ...]
    resolved_node_ids: tuple[str, ...]
    unresolved_node_ids: tuple[str, ...]
    decision_critical_unresolved_node_ids: tuple[str, ...]
    first_unresolved_depth: int | None
    resolved_path_depth: int
    action_set_size: int
    action_ambiguity: float
    unique_action: bool
    execution_failures: tuple[ExecutionFailure, ...]
    evidence_provenance: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reachable_path_ids"] = list(self.reachable_path_ids)
        data["reachable_paths"] = [list(path) for path in self.reachable_paths]
        data["reachable_leaf_ids"] = list(self.reachable_leaf_ids)
        data["compatible_action_ids"] = list(self.compatible_action_ids)
        data["compatible_action_sequences"] = [
            list(sequence) for sequence in self.compatible_action_sequences
        ]
        data["resolved_node_ids"] = list(self.resolved_node_ids)
        data["unresolved_node_ids"] = list(self.unresolved_node_ids)
        data["decision_critical_unresolved_node_ids"] = list(
            self.decision_critical_unresolved_node_ids
        )
        data["execution_failures"] = [asdict(item) for item in self.execution_failures]
        data["evidence_provenance"] = {
            key: list(value) for key, value in sorted(self.evidence_provenance.items())
        }
        return data


@dataclass
class _Traversal:
    paths: list[tuple[str, ...]]
    leaves: set[str]
    actions: set[str]
    action_sequences: set[tuple[str, ...]]
    resolved: set[str]
    unresolved: set[str]
    unresolved_depths: list[int]
    resolved_depths: list[int]
    failures: list[ExecutionFailure]
    evidence: dict[str, tuple[str, ...]]


def _path_id(path: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(path).encode("utf-8")).hexdigest()[:16]
    return f"path_{digest}"


def _admissible_branches(
    node: TreeNode,
    assessments: Mapping[str, CriterionAssessment],
    override: str | None,
) -> tuple[str, ...]:
    assert node.criterion_id is not None
    assert node.state_branch_map is not None
    assert node.branches is not None
    if override is not None:
        if override not in node.branches:
            return ()
        return (override,)
    assessment = assessments.get(node.criterion_id)
    state = assessment.state if assessment is not None else CriterionState.NOT_DOCUMENTED
    return tuple(sorted(node.state_branch_map.get(state, ())))


def _traverse(
    tree: DecisionTree,
    assessments: Mapping[str, CriterionAssessment],
    overrides: Mapping[str, str],
) -> _Traversal:
    result = _Traversal([], set(), set(), set(), set(), set(), [], [], [], {})
    frontier: list[tuple[str, tuple[str, ...], tuple[str, ...], int]] = [
        (tree.root_id, (), (), 0)
    ]
    step_limit = max(1, len(tree.nodes) * 2)
    while frontier:
        node_id, prior_path, prior_actions, depth = frontier.pop()
        path = prior_path + (node_id,)
        if len(path) > step_limit or node_id in prior_path:
            result.failures.append(ExecutionFailure(node_id, "cycle_or_step_limit"))
            continue
        node = tree.nodes.get(node_id)
        if node is None:
            result.failures.append(ExecutionFailure(node_id, "dangling_edge"))
            continue
        if node.node_type == "terminal":
            if not node.terminal_action_id:
                result.failures.append(ExecutionFailure(node_id, "terminal_missing_action"))
                continue
            result.paths.append(path)
            result.leaves.add(node_id)
            sequence = prior_actions + (node.terminal_action_id,)
            result.action_sequences.add(sequence)
            result.actions.update(sequence)
            continue
        if node.node_type == "path_end":
            if not prior_actions:
                result.failures.append(ExecutionFailure(node_id, "path_end_without_encoded_action"))
                continue
            result.paths.append(path)
            result.leaves.add(node_id)
            result.action_sequences.add(prior_actions)
            result.actions.update(prior_actions)
            continue
        if node.node_type == "configuration_gap":
            result.failures.append(
                ExecutionFailure(node_id, f"configuration_gap:{node.gap_reason or 'unspecified'}")
            )
            continue

        branches = _admissible_branches(node, assessments, overrides.get(node_id))
        if not branches:
            result.failures.append(ExecutionFailure(node_id, "no_admissible_branch"))
            continue
        if len(branches) == 1:
            result.resolved.add(node_id)
            result.resolved_depths.append(depth)
            assessment = assessments.get(node.criterion_id or "")
            if assessment is not None:
                result.evidence[node_id] = assessment.evidence_ids
        else:
            result.unresolved.add(node_id)
            result.unresolved_depths.append(depth)
        assert node.branches is not None
        # Reverse push order so traversal output follows sorted branch names.
        for branch in reversed(branches):
            target = node.branches.get(branch)
            if target is None:
                result.failures.append(ExecutionFailure(node_id, f"branch_missing_target:{branch}"))
                continue
            action_ids = tuple(
                action.action_id for action in (node.branch_actions or {}).get(branch, ())
            )
            frontier.append((target, path, prior_actions + action_ids, depth + 1))
    result.paths.sort()
    return result


def execute_partial(
    tree: DecisionTree,
    assessments: Mapping[str, CriterionAssessment],
    *,
    patient_key: str = "unknown",
    encounter_key: str = "unknown",
    pipeline_id: str = "unknown",
    decision_cutoff: str = "unknown",
    compute_decision_critical: bool = True,
) -> ExecutionResult:
    """Enumerate every tree action compatible with typed criterion assessments.

    Set compute_decision_critical false for a reachability-only pass when the
    counterfactual branch traversals used to identify critical unresolved nodes are
    not needed.
    """

    for key, value in assessments.items():
        if key != value.criterion_id:
            raise ValueError(f"assessment key {key!r} does not match criterion_id {value.criterion_id!r}")
    base = _traverse(tree, assessments, {})
    critical: set[str] = set()
    if compute_decision_critical:
        for node_id in sorted(base.unresolved):
            node = tree.nodes[node_id]
            assert node.branches is not None
            branch_action_sets = {
                tuple(sorted(_traverse(tree, assessments, {node_id: branch}).action_sequences))
                for branch in sorted(node.branches)
            }
            if len(branch_action_sets) > 1:
                critical.add(node_id)

    sequences = tuple(sorted(base.action_sequences))
    action_ids: list[str] = []
    for sequence in sequences:
        if len(sequence) == 1:
            action_ids.append(sequence[0])
        else:
            digest = hashlib.sha256("\x1f".join(sequence).encode("utf-8")).hexdigest()[:16]
            action_ids.append(f"action_sequence_{digest}")
    actions = tuple(sorted(action_ids))
    paths = tuple(base.paths)
    first_unresolved = min(base.unresolved_depths) if base.unresolved_depths else None
    # Number of resolved criterion decisions before the first unresolved decision.
    if first_unresolved is None:
        resolved_depth = max(base.resolved_depths, default=-1) + 1
    else:
        resolved_depth = first_unresolved
    failures = tuple(
        sorted(set(base.failures), key=lambda item: (item.node_id, item.reason))
    )
    return ExecutionResult(
        patient_key=patient_key,
        encounter_key=encounter_key,
        domain=tree.domain,
        pipeline_id=pipeline_id,
        tree_version=tree.version,
        decision_cutoff=decision_cutoff,
        reachable_path_ids=tuple(_path_id(path) for path in paths),
        reachable_paths=paths,
        reachable_leaf_ids=tuple(sorted(base.leaves)),
        compatible_action_ids=actions,
        compatible_action_sequences=sequences,
        resolved_node_ids=tuple(sorted(base.resolved)),
        unresolved_node_ids=tuple(sorted(base.unresolved)),
        decision_critical_unresolved_node_ids=tuple(sorted(critical)),
        first_unresolved_depth=first_unresolved,
        resolved_path_depth=resolved_depth,
        action_set_size=len(actions),
        action_ambiguity=math.log2(max(1, len(actions))),
        unique_action=len(actions) == 1 and not failures,
        execution_failures=failures,
        evidence_provenance=dict(sorted(base.evidence.items())),
    )
