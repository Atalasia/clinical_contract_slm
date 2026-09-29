"""Paired, subject-clustered analysis of the native E3 decoding experiment.

All request-level material stays in memory. The returned summary contains only
aggregate counts, metrics, model names, and declared contrast identifiers.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .ext_notes_benchmark import validate_ext_notes_labels
from .slm_e3 import _hierarchical_correct, _prediction_heads, _score_e3_rows


CONDITIONS = ("unconstrained", "constrained")
CLUSTER_FIELDS = ("note_cluster", "admission_cluster", "subject_cluster")
VIEWS = ("strict", "fence_normalized")
_FAILURES = {None, "runtime_failure", "parse_failure", "schema_failure", "logical_failure"}


def _scoring_row(row: Mapping[str, Any], view: str) -> dict[str, Any]:
    """Select one evaluation without allowing it to replace pairing metadata."""
    evaluation = row if view == "strict" else row.get("fence_normalized")
    if not isinstance(evaluation, Mapping):
        raise ValueError("every response requires a fence_normalized evaluation")
    for key in ("parsed", "predictions", "hierarchical_joint_correct"):
        if key not in evaluation:
            raise ValueError(f"response evaluation is missing {key}")
    parsed = evaluation["parsed"]
    if not isinstance(parsed, Mapping):
        raise ValueError("response parsed evaluation must be an object")
    for key in ("parse_valid", "schema_valid", "logical_valid", "abstained"):
        if not isinstance(parsed.get(key), bool):
            raise ValueError(f"parsed {key} must be boolean")
    if parsed.get("failure") not in _FAILURES:
        raise ValueError("unknown response failure category")
    if parsed["logical_valid"] and not parsed["schema_valid"]:
        raise ValueError("logical validity requires schema validity")
    if parsed["schema_valid"] and not parsed["parse_valid"]:
        raise ValueError("schema validity requires parse validity")
    if row["runtime_failed"] and parsed["parse_valid"]:
        raise ValueError("runtime failures cannot be parse valid")
    if parsed["logical_valid"] and validate_ext_notes_labels(
        {key: parsed.get(key) for key in ("detection", "encounter", "negation")},
        allow_abstention=True,
    ):
        raise ValueError("logically valid response has invalid native labels")
    if evaluation["predictions"] != _prediction_heads(parsed):
        raise ValueError("stored response head predictions disagree with parsed labels")
    correct = evaluation["hierarchical_joint_correct"]
    if not isinstance(correct, bool) or correct != _hierarchical_correct(row["expected"], parsed):
        raise ValueError("stored hierarchical correctness disagrees with parsed labels")
    return {**row, **{key: evaluation[key] for key in
                    ("parsed", "predictions", "hierarchical_joint_correct")}}


def _align_panel(panel: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]):
    if not isinstance(panel, Mapping) or not panel:
        raise ValueError("paired panel must contain at least one model")
    if any(not isinstance(model, str) or not model for model in panel):
        raise ValueError("model identifiers must be nonempty strings")
    models = sorted(panel)
    indexed: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for model in models:
        if not isinstance(panel[model], Mapping) or set(panel[model]) != set(CONDITIONS):
            raise ValueError("every model must have exactly both decoding conditions")
        indexed[model] = {}
        for condition in CONDITIONS:
            rows = panel[model][condition]
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
                raise ValueError("each paired response cell must be a nonempty sequence")
            by_id: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("response rows must be objects")
                request_id = row.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("response request IDs must be nonempty strings")
                if request_id in by_id:
                    raise ValueError("duplicate response request IDs in a cell")
                for field in CLUSTER_FIELDS:
                    if not isinstance(row.get(field), str) or not row[field]:
                        raise ValueError(f"response is missing a valid {field}")
                expected = row.get("expected")
                if not isinstance(expected, Mapping) or validate_ext_notes_labels(
                    expected, allow_abstention=False
                ):
                    raise ValueError("response expected labels violate the native E3 contract")
                if not isinstance(row.get("runtime_failed"), bool):
                    raise ValueError("response runtime_failed must be boolean")
                if not isinstance(row.get("finish_reason"), str):
                    raise ValueError("response finish_reason must be a string")
                prompt_hash = row.get("prompt_sha256")
                if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash):
                    raise ValueError("response is missing a valid prompt_sha256")
                # Check both scoring views before any score or comparison is emitted.
                for view in VIEWS:
                    _scoring_row(row, view)
                by_id[request_id] = row
            indexed[model][condition] = by_id

    reference = indexed[models[0]]["unconstrained"]
    request_ids = sorted(reference)
    for model in models:
        for condition in CONDITIONS:
            cell = indexed[model][condition]
            if set(cell) != set(reference):
                raise ValueError("paired response request identities differ across cells")
            for request_id in request_ids:
                row, original = cell[request_id], reference[request_id]
                if row["expected"] != original["expected"]:
                    raise ValueError("paired response expected labels differ across cells")
                if any(row[field] != original[field] for field in CLUSTER_FIELDS):
                    raise ValueError("paired response clusters differ across cells")
                if row["prompt_sha256"] != indexed[model]["unconstrained"][request_id]["prompt_sha256"]:
                    raise ValueError("rendered prompt hashes differ between decoding conditions")

    # A note belongs to one admission and an admission to one subject. Reject a
    # malformed hierarchy instead of silently selecting a different clustering.
    notes: dict[str, tuple[str, str]] = {}
    admissions: dict[str, str] = {}
    for row in reference.values():
        note, admission, subject = (row[field] for field in CLUSTER_FIELDS)
        if notes.setdefault(note, (admission, subject)) != (admission, subject):
            raise ValueError("note clusters do not have a consistent admission/subject")
        if admissions.setdefault(admission, subject) != subject:
            raise ValueError("admission clusters do not have a consistent subject")
    aligned = {model: {condition: [indexed[model][condition][key] for key in request_ids]
                       for condition in CONDITIONS} for model in models}
    return models, aligned


def _validate_contrasts(contrasts: Sequence[Mapping[str, Any]], models: Sequence[str]):
    if not isinstance(contrasts, Sequence) or isinstance(contrasts, (str, bytes)):
        raise ValueError("contrasts must be a sequence of declared model pairs")
    result = []
    seen: set[str] = set()
    for contrast in contrasts:
        if not isinstance(contrast, Mapping):
            raise ValueError("contrast declarations must be objects")
        identifier, pair = contrast.get("contrast_id"), contrast.get("models")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("contrast IDs must be unique nonempty strings")
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes))
                or len(pair) != 2 or pair[0] == pair[1]
                or any(model not in models for model in pair)):
            raise ValueError("contrast models must be two distinct models in the complete panel")
        seen.add(identifier)
        result.append({"contrast_id": identifier, "models": list(pair)})
    return result


def _interval(values: np.ndarray) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("nonfinite paired bootstrap values")
    return [float(value) for value in np.quantile(values, [0.025, 0.975], method="linear")]


def _difference(estimate: float, samples: np.ndarray) -> dict[str, Any]:
    return {"difference": float(estimate), "subject_cluster_ci95": _interval(samples)}


def summarize_paired_e3(
    panel: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    contrasts: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Summarize complete matched cells, with no request-level output.

    ``panel[manifest_id][condition]`` is a sequence of native E3 response rows.
    Every row also contains ``fence_normalized`` with ``parsed``, ``predictions``
    and ``hierarchical_joint_correct``. Contrasts follow benchmark declarations:
    ``{"contrast_id": ..., "models": [model_a, model_b]}``. Model contrasts use
    B minus A; decoding effects use constrained minus unconstrained. The
    interaction is (B-A under constrained) minus (B-A under unconstrained).

    Resampling subjects with replacement retains all requests in each sampled
    subject. Each replicate estimates total correct / total requests, NOT an
    equal-weight average of subject accuracies. The same subject draws are used
    for every model, condition, scoring view, and contrast. Percentile 95% CIs
    are unadjusted descriptive intervals; no independent-candidate tests or
    claims of significant rank reversal are made.
    """
    if (not isinstance(bootstrap_replicates, int) or isinstance(bootstrap_replicates, bool)
            or bootstrap_replicates < 1):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if (not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool)
            or bootstrap_seed < 0):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    models, aligned = _align_panel(panel)
    declared_contrasts = _validate_contrasts(contrasts, models)
    reference = aligned[models[0]]["unconstrained"]
    n = len(reference)
    subject_ids = sorted({row["subject_cluster"] for row in reference})
    subject_index = {subject: index for index, subject in enumerate(subject_ids)}
    row_subjects = np.array([subject_index[row["subject_cluster"]] for row in reference])
    subject_totals = np.bincount(row_subjects, minlength=len(subject_ids))
    rng = np.random.default_rng(bootstrap_seed)
    # Counts are equivalent to drawing n_subjects subjects with replacement.
    weights = rng.multinomial(
        len(subject_ids), np.full(len(subject_ids), 1 / len(subject_ids)),
        size=bootstrap_replicates,
    )
    bootstrap_denominators = weights @ subject_totals
    cell_keys = [(model, condition) for model in models for condition in CONDITIONS]
    cell_index = {key: index for index, key in enumerate(cell_keys)}
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "primary_endpoint": "strict_hierarchical_joint_exact_match",
        "n_requests": n,
        "n_models": len(models),
        "conditions": list(CONDITIONS),
        "clusters": {
            "notes": len({row["note_cluster"] for row in reference}),
            "admissions": len({row["admission_cluster"] for row in reference}),
            "subjects": len(subject_ids),
        },
        "statistics": {
            "cluster_unit": "subject",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_rng": "numpy.random.default_rng (PCG64)",
            "confidence_level": 0.95,
            "interval_method": "percentile, linear quantiles",
            "paired_draws_shared_across_all_cells_and_scoring_views": True,
            "estimand": "request-weighted hierarchical joint agreement",
            "decoding_difference_direction": "constrained minus unconstrained",
            "model_contrast_direction": "second declared model minus first declared model",
            "interaction_direction": "constrained model contrast minus unconstrained model contrast",
            "multiplicity": "unadjusted descriptive confidence intervals; no hypothesis tests",
            "rankings": "descriptive observed scores; ties retained; no rank-certainty claim",
            "small_cluster_warning": (
                "Fewer than two subjects: resampling cannot estimate between-subject uncertainty."
                if len(subject_ids) < 2 else None
            ),
        },
        "scope": (
            "Candidate-conditioned native E3 label agreement; a second-task decoding comparison, "
            "not end-to-end extraction or identical-criterion synthetic-to-EHR transfer."
        ),
    }
    for view in VIEWS:
        scoring = {model: {condition: [_scoring_row(row, view) for row in aligned[model][condition]]
                           for condition in CONDITIONS} for model in models}
        successes = np.column_stack([
            np.bincount(row_subjects, weights=[row["hierarchical_joint_correct"]
                                             for row in scoring[model][condition]],
                        minlength=len(subject_ids))
            for model, condition in cell_keys
        ])
        bootstrap_scores = (weights @ successes) / bootstrap_denominators[:, None]
        estimates = successes.sum(axis=0) / n
        cells = {model: {} for model in models}
        within_model = {}
        for model, condition in cell_keys:
            index = cell_index[(model, condition)]
            metrics = _score_e3_rows(scoring[model][condition])
            metrics["hierarchical_joint"]["subject_cluster_ci95"] = _interval(bootstrap_scores[:, index])
            cells[model][condition] = metrics
        for model in models:
            u, c = (cell_index[(model, condition)] for condition in CONDITIONS)
            within_model[model] = {
                "unconstrained_agreement": float(estimates[u]),
                "constrained_agreement": float(estimates[c]),
                **_difference(estimates[c] - estimates[u], bootstrap_scores[:, c] - bootstrap_scores[:, u]),
            }
        comparisons = []
        for contrast in declared_contrasts:
            model_a, model_b = contrast["models"]
            au, ac = (cell_index[(model_a, condition)] for condition in CONDITIONS)
            bu, bc = (cell_index[(model_b, condition)] for condition in CONDITIONS)
            unconstrained_difference = estimates[bu] - estimates[au]
            constrained_difference = estimates[bc] - estimates[ac]
            comparisons.append({
                **contrast,
                "unconstrained": _difference(
                    unconstrained_difference, bootstrap_scores[:, bu] - bootstrap_scores[:, au]),
                "constrained": _difference(
                    constrained_difference, bootstrap_scores[:, bc] - bootstrap_scores[:, ac]),
                "decoding_interaction": _difference(
                    constrained_difference - unconstrained_difference,
                    (bootstrap_scores[:, bc] - bootstrap_scores[:, ac])
                    - (bootstrap_scores[:, bu] - bootstrap_scores[:, au])),
                "observed_sign_reversal": bool(unconstrained_difference * constrained_difference < 0),
                "observed_tie_in_either_condition": bool(
                    unconstrained_difference == 0 or constrained_difference == 0),
            })
        rankings, winners = {}, {}
        for condition in CONDITIONS:
            ordered = sorted(models, key=lambda model: (
                -cells[model][condition]["hierarchical_joint"]["correct"], model))
            ranking = []
            previous_correct, rank = None, 0
            for position, model in enumerate(ordered, 1):
                joint = cells[model][condition]["hierarchical_joint"]
                if joint["correct"] != previous_correct:
                    rank = position
                ranking.append({"manifest_id": model, "rank": rank,
                                "correct": joint["correct"], "agreement": joint["exact_match"]})
                previous_correct = joint["correct"]
            rankings[condition] = ranking
            winners[condition] = [row["manifest_id"] for row in ranking if row["rank"] == 1]
        result[view] = {
            "role": "primary" if view == "strict" else "secondary sensitivity",
            "cells": cells,
            "within_model_decoding_effects": within_model,
            "model_contrasts": comparisons,
            "rankings": rankings,
            "winners": winners,
        }
    return result
