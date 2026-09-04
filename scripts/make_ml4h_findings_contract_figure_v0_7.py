#!/usr/bin/env python3
"""Render the state-macro ML4H Findings contrast figure from frozen outputs."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cpg3-matplotlib-cache")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
E1_RESULTS = Path(
    os.environ.get(
        "CPG2_E1_RESULTS",
        ROOT / "outputs/local_restricted/slm_benchmark_v2/e1/aggregate/results.json",
    )
)
CONSTRAINED_RESULTS = Path(
    os.environ.get(
        "CPG2_E1_S2_RESULTS",
        ROOT
        / "outputs/local_restricted/slm_benchmark_v2/e1_s2/aggregate/results.json",
    )
)
E1_RESPONSES = Path(
    os.environ.get(
        "CPG2_E1_RESPONSES",
        ROOT / "outputs/local_restricted/slm_benchmark_v2/e1/results",
    )
)
CONSTRAINED_RESPONSES = Path(
    os.environ.get(
        "CPG2_E1_S2_RESPONSES",
        ROOT / "outputs/local_restricted/slm_benchmark_v2/e1_s2/results",
    )
)
OUTPUT_DIR = Path(
    os.environ.get("CPG2_FIGURE_OUTPUT_DIR", ROOT / "paper_artifacts/figures")
)
DERIVED_OUTPUT = Path(
    os.environ.get(
        "CPG2_DERIVED_OUTPUT",
        ROOT
        / "paper_artifacts/derived/ml4h-findings-contract-analysis-v0.7.json",
    )
)
FIGURE_STEM = "ml4h-findings-output-contract-contrasts-v0.7"
REFERENCE_STATES = ("PRESENT_CURRENT", "ABSENT_EXPLICIT")
RECOVERABLE_STATES = frozenset(
    {
        "PRESENT_CURRENT",
        "ABSENT_EXPLICIT",
        "HISTORICAL",
        "FAMILY_HISTORY_OR_OTHER_EXPERIENCER",
        "UNCERTAIN",
        "NOT_DOCUMENTED",
    }
)
BOOTSTRAP_REPLICATES = 10_000
PRIMARY_BOOTSTRAP_SEED = 2026
CONSTRAINED_BOOTSTRAP_SEED = 12026
INTERACTION_BOOTSTRAP_SEED = 2226

CONTRASTS = [
    ("qwen_instruction_small", "Qwen2.5 1.5B: Instruct − base"),
    ("qwen_instruction_medium", "Qwen2.5 3B: Instruct − base"),
    ("qwen_size_pretrained", "Qwen2.5 base: 3B − 1.5B"),
    ("qwen_size_instruction", "Qwen2.5 Instruct: 3B − 1.5B"),
    ("gemma_medical_adaptation_it", "4B Instruct: MedGemma − Gemma 3"),
]

REFERENCE_CONTRASTS = [
    "medgemma_release_reference",
    "contemporary_general_reference",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_path(path: Path, environment_name: str) -> str:
    """Avoid embedding a caller's absolute local path in the derived artifact."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return f"${{{environment_name}}}"


def load_responses(root: Path, manifest_id: str) -> dict[str, dict]:
    path = root / manifest_id / "responses.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_request = {row["request_id"]: row for row in rows}
    if len(by_request) != len(rows):
        raise RuntimeError(f"Duplicate request IDs in {path}")
    return by_request


def state_macro_agreement(
    responses: dict[str, dict], request_ids: list[str]
) -> float:
    state_scores = []
    for state in REFERENCE_STATES:
        selected = [
            request_id
            for request_id in request_ids
            if responses[request_id]["expected_state"] == state
        ]
        if not selected:
            raise RuntimeError(f"Bootstrap sample omitted reference state {state}")
        state_scores.append(
            sum(bool(responses[request_id]["state_correct"]) for request_id in selected)
            / len(selected)
        )
    return fmean(state_scores)


def state_macro_difference(
    model_a: dict[str, dict], model_b: dict[str, dict], request_ids: list[str]
) -> float:
    return state_macro_agreement(model_b, request_ids) - state_macro_agreement(
        model_a, request_ids
    )


def recoverable_state_correct(row: dict) -> bool:
    raw_output = row.get("raw_output")
    if not isinstance(raw_output, str):
        return False
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    state = payload.get("state")
    return bool(
        payload.get("criterion_id") == row.get("criterion_id")
        and isinstance(state, str)
        and state in RECOVERABLE_STATES
        and state == row.get("expected_state")
    )


def recoverable_state_micro_agreement(responses: dict[str, dict]) -> float:
    return fmean(recoverable_state_correct(row) for row in responses.values())


def recoverable_state_macro_agreement(
    responses: dict[str, dict], request_ids: list[str]
) -> float:
    state_scores = []
    for state in REFERENCE_STATES:
        selected = [
            request_id
            for request_id in request_ids
            if responses[request_id]["expected_state"] == state
        ]
        if not selected:
            raise RuntimeError(f"Sample omitted reference state {state}")
        state_scores.append(
            fmean(
                recoverable_state_correct(responses[request_id])
                for request_id in selected
            )
        )
    return fmean(state_scores)


def strict_micro_agreement(responses: dict[str, dict]) -> float:
    return fmean(bool(row["state_correct"]) for row in responses.values())


def percentile_interval(
    ordered_values: list[float], *, alpha: float
) -> tuple[float, float]:
    last = len(ordered_values) - 1
    lower = ordered_values[math.floor((alpha / 2) * last)]
    upper = ordered_values[math.ceil((1 - alpha / 2) * last)]
    return lower, upper


def inverse_empirical_percentile_interval(
    values: list[float], *, alpha: float
) -> tuple[float, float]:
    lower, upper = np.quantile(
        values, [alpha / 2, 1 - alpha / 2], method="inverted_cdf"
    )
    return float(lower), float(upper)


def bootstrap_macro_difference(
    model_a: dict[str, dict],
    model_b: dict[str, dict],
    *,
    seed: int,
) -> tuple[float, float]:
    request_ids = sorted(model_a)
    if set(request_ids) != set(model_b):
        raise RuntimeError("Paired model response request IDs differ")
    by_case: dict[str, list[str]] = defaultdict(list)
    for request_id in request_ids:
        row_a = model_a[request_id]
        row_b = model_b[request_id]
        if (
            row_a["case_id"] != row_b["case_id"]
            or row_a["expected_state"] != row_b["expected_state"]
        ):
            raise RuntimeError("Paired model response metadata differ")
        by_case[str(row_a["case_id"])].append(request_id)

    case_ids = sorted(by_case)
    rng = random.Random(seed)
    differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            request_id
            for case_id in rng.choices(case_ids, k=len(case_ids))
            for request_id in by_case[case_id]
        ]
        differences.append(state_macro_difference(model_a, model_b, sampled_ids))
    differences.sort()
    return percentile_interval(differences, alpha=0.05)


def bootstrap_macro_interaction(
    primary_a: dict[str, dict],
    primary_b: dict[str, dict],
    constrained_a: dict[str, dict],
    constrained_b: dict[str, dict],
    *,
    seed: int,
) -> dict[str, object]:
    request_ids = sorted(primary_a)
    response_sets = (primary_b, constrained_a, constrained_b)
    if any(set(request_ids) != set(responses) for responses in response_sets):
        raise RuntimeError("Interaction response request IDs differ")

    by_case: dict[str, list[str]] = defaultdict(list)
    for request_id in request_ids:
        reference = primary_a[request_id]
        metadata = (reference["case_id"], reference["expected_state"])
        for responses in response_sets:
            row = responses[request_id]
            if (row["case_id"], row["expected_state"]) != metadata:
                raise RuntimeError("Interaction response metadata differ")
        by_case[str(reference["case_id"])].append(request_id)

    def interaction(sampled_ids: list[str]) -> float:
        return state_macro_difference(
            constrained_a, constrained_b, sampled_ids
        ) - state_macro_difference(primary_a, primary_b, sampled_ids)

    point = interaction(request_ids)
    case_ids = sorted(by_case)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [
            request_id
            for case_id in rng.choices(case_ids, k=len(case_ids))
            for request_id in by_case[case_id]
        ]
        draws.append(interaction(sampled_ids))
    nominal = inverse_empirical_percentile_interval(draws, alpha=0.05)
    bonferroni = inverse_empirical_percentile_interval(
        draws, alpha=0.05 / len(CONTRASTS)
    )
    return {
        "difference_of_differences": point,
        "cluster_bootstrap_95_ci": list(nominal),
        "bonferroni_adjusted_99_ci": list(bonferroni),
        "bonferroni_component_confidence_level": 0.99,
        "bonferroni_familywise_confidence_level": 0.95,
        "bootstrap_seed": seed,
    }


def macro_contrasts(
    aggregate_path: Path,
    response_root: Path,
    *,
    aggregate_key: str,
    seed_base: int,
) -> dict[str, dict]:
    aggregate = load_json(aggregate_path)
    declared = {row["contrast_id"]: row for row in aggregate[aggregate_key]}
    macro_by_model = {
        row["manifest_id"]: row["metrics"]["overall"]["macro_state_agreement"]
        for row in aggregate["models"]
    }
    needed_models = {
        model_id
        for contrast_id, _ in CONTRASTS
        for model_id in (
            declared[contrast_id]["model_a"],
            declared[contrast_id]["model_b"],
        )
    }
    responses = {
        model_id: load_responses(response_root, model_id)
        for model_id in needed_models
    }

    results = {}
    for contrast_index, (contrast_id, _) in enumerate(CONTRASTS):
        declaration = declared[contrast_id]
        model_a = declaration["model_a"]
        model_b = declaration["model_b"]
        request_ids = sorted(responses[model_a])
        observed_a = state_macro_agreement(responses[model_a], request_ids)
        observed_b = state_macro_agreement(responses[model_b], request_ids)
        if not np.isclose(observed_a, macro_by_model[model_a]) or not np.isclose(
            observed_b, macro_by_model[model_b]
        ):
            raise RuntimeError("Response-level state-macro result disagrees with aggregate")
        lower, upper = bootstrap_macro_difference(
            responses[model_a],
            responses[model_b],
            seed=seed_base + contrast_index,
        )
        results[contrast_id] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_a_agreement": observed_a,
            "model_b_agreement": observed_b,
            "paired_agreement_difference": observed_b - observed_a,
            "cluster_bootstrap_95_ci": [lower, upper],
            "bootstrap_seed": seed_base + contrast_index,
        }
    return results


def condition_data(
    aggregate_path: Path,
    response_root: Path,
    *,
    aggregate_key: str,
    diagnostic_key: str,
) -> tuple[dict, dict[str, dict], dict[str, dict], dict[str, dict]]:
    aggregate = load_json(aggregate_path)
    models = {row["manifest_id"]: row for row in aggregate["models"]}
    declarations = {
        row["contrast_id"]: row for row in aggregate[aggregate_key]
    }
    responses = {
        manifest_id: load_responses(response_root, manifest_id)
        for manifest_id in models
    }

    request_sets = {tuple(sorted(rows)) for rows in responses.values()}
    if len(request_sets) != 1:
        raise RuntimeError(f"Model request sets differ under {aggregate_path}")

    for manifest_id, model in models.items():
        rows = responses[manifest_id]
        observed_macro = state_macro_agreement(rows, sorted(rows))
        observed_micro = strict_micro_agreement(rows)
        observed_recoverable_micro = recoverable_state_micro_agreement(rows)
        stored_overall = model["metrics"]["overall"]
        stored_diagnostic = model[diagnostic_key]
        if not np.isclose(
            observed_macro, stored_overall["macro_state_agreement"]
        ):
            raise RuntimeError(
                f"State-macro result disagrees with aggregate for {manifest_id}"
            )
        if not np.isclose(
            observed_micro, stored_overall["state_agreement_unconditional"]
        ):
            raise RuntimeError(
                f"Strict-micro result disagrees with aggregate for {manifest_id}"
            )
        if not np.isclose(
            observed_recoverable_micro,
            stored_diagnostic["recoverable_state_agreement_unconditional"],
        ):
            raise RuntimeError(
                f"Recoverable-state result disagrees with aggregate for {manifest_id}"
            )

    return aggregate, models, declarations, responses


def endpoint_values(responses: dict[str, dict]) -> dict[str, float]:
    request_ids = sorted(responses)
    return {
        "state_macro_agreement": state_macro_agreement(responses, request_ids),
        "strict_micro_agreement": strict_micro_agreement(responses),
        "recoverable_state_macro_agreement_posthoc": (
            recoverable_state_macro_agreement(responses, request_ids)
        ),
        "recoverable_state_micro_agreement_posthoc": (
            recoverable_state_micro_agreement(responses)
        ),
    }


def point_contrast(
    declaration: dict,
    models: dict[str, dict],
    responses: dict[str, dict],
) -> dict[str, object]:
    model_a = declaration["model_a"]
    model_b = declaration["model_b"]
    request_ids = sorted(responses[model_a])
    macro_a = state_macro_agreement(responses[model_a], request_ids)
    macro_b = state_macro_agreement(responses[model_b], request_ids)
    recoverable_macro_a = recoverable_state_macro_agreement(
        responses[model_a], request_ids
    )
    recoverable_macro_b = recoverable_state_macro_agreement(
        responses[model_b], request_ids
    )
    recoverable_micro_a = recoverable_state_micro_agreement(responses[model_a])
    recoverable_micro_b = recoverable_state_micro_agreement(responses[model_b])
    return {
        "model_a": model_a,
        "model_a_model_id": models[model_a]["model_id"],
        "model_b": model_b,
        "model_b_model_id": models[model_b]["model_id"],
        "orientation": "model_b_minus_model_a",
        "model_a_state_macro_agreement": macro_a,
        "model_b_state_macro_agreement": macro_b,
        "state_macro_difference": macro_b - macro_a,
        "model_a_recoverable_state_macro_agreement_posthoc": (
            recoverable_macro_a
        ),
        "model_b_recoverable_state_macro_agreement_posthoc": (
            recoverable_macro_b
        ),
        "recoverable_state_macro_difference_posthoc": (
            recoverable_macro_b - recoverable_macro_a
        ),
        "model_a_recoverable_state_micro_agreement_posthoc": (
            recoverable_micro_a
        ),
        "model_b_recoverable_state_micro_agreement_posthoc": (
            recoverable_micro_b
        ),
        "recoverable_state_micro_difference_posthoc": (
            recoverable_micro_b - recoverable_micro_a
        ),
    }


def build_analysis_payload(
    primary_contrasts: dict[str, dict],
    constrained_contrasts: dict[str, dict],
) -> dict[str, object]:
    (
        primary_aggregate,
        primary_models,
        primary_declarations,
        primary_responses,
    ) = condition_data(
        E1_RESULTS,
        E1_RESPONSES,
        aggregate_key="paired_contrasts",
        diagnostic_key="posthoc_output_diagnostics",
    )
    (
        constrained_aggregate,
        constrained_models,
        constrained_declarations,
        constrained_responses,
    ) = condition_data(
        CONSTRAINED_RESULTS,
        CONSTRAINED_RESPONSES,
        aggregate_key="s2_paired_model_contrasts",
        diagnostic_key="output_diagnostics",
    )

    if set(primary_models) != set(constrained_models):
        raise RuntimeError("Model panels differ between output-contract conditions")
    if set(primary_declarations) != set(constrained_declarations):
        raise RuntimeError("Contrast declarations differ between conditions")

    model_rows = []
    for manifest_id in primary_models:
        if (
            primary_models[manifest_id]["model_id"]
            != constrained_models[manifest_id]["model_id"]
        ):
            raise RuntimeError(f"Model ID differs for {manifest_id}")
        model_rows.append(
            {
                "manifest_id": manifest_id,
                "model_id": primary_models[manifest_id]["model_id"],
                "unconstrained": endpoint_values(primary_responses[manifest_id]),
                "json_schema_constrained": endpoint_values(
                    constrained_responses[manifest_id],
                ),
            }
        )

    core_rows = []
    for contrast_index, (contrast_id, label) in enumerate(CONTRASTS):
        primary_declaration = primary_declarations[contrast_id]
        constrained_declaration = constrained_declarations[contrast_id]
        if (
            primary_declaration["model_a"],
            primary_declaration["model_b"],
        ) != (
            constrained_declaration["model_a"],
            constrained_declaration["model_b"],
        ):
            raise RuntimeError(f"Contrast models differ for {contrast_id}")
        model_a = primary_declaration["model_a"]
        model_b = primary_declaration["model_b"]
        interaction = bootstrap_macro_interaction(
            primary_responses[model_a],
            primary_responses[model_b],
            constrained_responses[model_a],
            constrained_responses[model_b],
            seed=INTERACTION_BOOTSTRAP_SEED + contrast_index,
        )
        unconstrained_point = point_contrast(
            primary_declaration, primary_models, primary_responses
        )
        constrained_point = point_contrast(
            constrained_declaration, constrained_models, constrained_responses
        )
        if not np.isclose(
            interaction["difference_of_differences"],
            constrained_point["state_macro_difference"]
            - unconstrained_point["state_macro_difference"],
        ):
            raise RuntimeError(f"Interaction point estimate differs for {contrast_id}")
        unconstrained_point.update(
            {
                "state_macro_cluster_bootstrap_95_ci": primary_contrasts[
                    contrast_id
                ]["cluster_bootstrap_95_ci"],
                "state_macro_bootstrap_seed": primary_contrasts[contrast_id][
                    "bootstrap_seed"
                ],
            }
        )
        constrained_point.update(
            {
                "state_macro_cluster_bootstrap_95_ci": constrained_contrasts[
                    contrast_id
                ]["cluster_bootstrap_95_ci"],
                "state_macro_bootstrap_seed": constrained_contrasts[contrast_id][
                    "bootstrap_seed"
                ],
            }
        )
        core_rows.append(
            {
                "contrast_id": contrast_id,
                "label": label,
                "unconstrained": unconstrained_point,
                "json_schema_constrained": constrained_point,
                "contract_by_model_interaction": interaction,
                "state_macro_selection_reversal": bool(
                    np.sign(unconstrained_point["state_macro_difference"])
                    != np.sign(constrained_point["state_macro_difference"])
                ),
                "recoverable_state_macro_direction_reversal_posthoc": bool(
                    np.sign(
                        unconstrained_point[
                            "recoverable_state_macro_difference_posthoc"
                        ]
                    )
                    != np.sign(
                        constrained_point[
                            "recoverable_state_macro_difference_posthoc"
                        ]
                    )
                ),
            }
        )

    descriptive_rows = []
    for contrast_id in REFERENCE_CONTRASTS:
        primary_declaration = primary_declarations[contrast_id]
        constrained_declaration = constrained_declarations[contrast_id]
        if (
            primary_declaration["model_a"],
            primary_declaration["model_b"],
        ) != (
            constrained_declaration["model_a"],
            constrained_declaration["model_b"],
        ):
            raise RuntimeError(f"Contrast models differ for {contrast_id}")
        unconstrained_point = point_contrast(
            primary_declaration, primary_models, primary_responses
        )
        constrained_point = point_contrast(
            constrained_declaration, constrained_models, constrained_responses
        )
        descriptive_rows.append(
            {
                "contrast_id": contrast_id,
                "claim_type": primary_declaration.get("claim_type"),
                "factor": primary_declaration.get("factor"),
                "unconstrained": unconstrained_point,
                "json_schema_constrained": constrained_point,
                "state_macro_selection_reversal": bool(
                    np.sign(unconstrained_point["state_macro_difference"])
                    != np.sign(constrained_point["state_macro_difference"])
                ),
            }
        )

    request_ids = sorted(next(iter(primary_responses.values())))
    case_count = len(
        {
            row["case_id"]
            for row in next(iter(primary_responses.values())).values()
        }
    )
    return {
        "artifact": "ML4H Findings output-contract analysis",
        "artifact_version": "0.7",
        "generation_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "units": "proportion; multiply by 100 for percentage points",
        "endpoint_hierarchy": {
            "model_selection_endpoint": {
                "name": "state_macro_agreement",
                "definition": (
                    "Unweighted mean of agreement for PRESENT_CURRENT and "
                    "ABSENT_EXPLICIT; contract-invalid responses are incorrect."
                ),
            },
            "operational_endpoint": {
                "name": "strict_micro_agreement",
                "definition": (
                    "Request-level state agreement under the complete frozen "
                    "output contract; contract-invalid responses are incorrect."
                ),
            },
            "posthoc_diagnostic": {
                "names": [
                    "recoverable_state_macro_agreement_posthoc",
                    "recoverable_state_micro_agreement_posthoc",
                ],
                "definition": (
                    "Agreement after reading only a matching criterion_id and "
                    "recognized state from parseable JSON; other contract fields "
                    "and abstention are ignored."
                ),
                "macro_definition": (
                    "Unweighted mean of this recoverable-state agreement for "
                    "PRESENT_CURRENT and ABSENT_EXPLICIT."
                ),
                "micro_definition": "Request-weighted recoverable-state agreement.",
                "status": "post_hoc_error_diagnostic_not_primary_metric",
            },
        },
        "data": {
            "requests": len(request_ids),
            "cases": case_count,
            "reference_states": list(REFERENCE_STATES),
            "unconstrained_aggregate": display_path(E1_RESULTS, "CPG2_E1_RESULTS"),
            "json_schema_constrained_aggregate": display_path(
                CONSTRAINED_RESULTS, "CPG2_E1_S2_RESULTS"
            ),
            "unconstrained_dataset_manifest_sha256": primary_aggregate[
                "dataset_manifest_sha256"
            ],
            "json_schema_constrained_dataset_manifest_sha256": (
                constrained_aggregate["dataset_manifest_sha256"]
            ),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "synthetic_vignette_case",
            "condition_specific_seed_rule": (
                "2026 + contrast_index for unconstrained; "
                "12026 + contrast_index for JSON-schema constrained"
            ),
            "interaction_seed_rule": "2226 + contrast_index",
            "interaction_nominal_confidence_level": 0.95,
            "interaction_bonferroni_component_confidence_level": 0.99,
            "interaction_bonferroni_familywise_confidence_level": 0.95,
            "interaction_family_size": len(CONTRASTS),
        },
        "models": model_rows,
        "prespecified_core_contrasts": core_rows,
        "descriptive_release_reference_contrasts": descriptive_rows,
    }


def values(rows: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = np.array(
        [rows[contrast_id]["paired_agreement_difference"] * 100 for contrast_id, _ in CONTRASTS]
    )
    low = np.array(
        [rows[contrast_id]["cluster_bootstrap_95_ci"][0] * 100 for contrast_id, _ in CONTRASTS]
    )
    high = np.array(
        [rows[contrast_id]["cluster_bootstrap_95_ci"][1] * 100 for contrast_id, _ in CONTRASTS]
    )
    return point, low, high


def main() -> None:
    primary = macro_contrasts(
        E1_RESULTS,
        E1_RESPONSES,
        aggregate_key="paired_contrasts",
        seed_base=PRIMARY_BOOTSTRAP_SEED,
    )
    constrained = macro_contrasts(
        CONSTRAINED_RESULTS,
        CONSTRAINED_RESPONSES,
        aggregate_key="s2_paired_model_contrasts",
        seed_base=CONSTRAINED_BOOTSTRAP_SEED,
    )
    missing = {
        contrast_id
        for contrast_id, _ in CONTRASTS
        if contrast_id not in primary or contrast_id not in constrained
    }
    if missing:
        raise RuntimeError(f"Missing declared contrasts: {sorted(missing)}")

    p_point, p_low, p_high = values(primary)
    c_point, c_low, c_high = values(constrained)
    reversed_direction = np.sign(p_point) != np.sign(c_point)
    if reversed_direction.tolist() != [True, False, True, True, False]:
        raise RuntimeError("Frozen direction-reversal pattern changed")

    analysis_payload = build_analysis_payload(primary, constrained)
    DERIVED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DERIVED_OUTPUT.write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.25,
            "svg.hashsalt": "ml4h-findings-v0.7",
        }
    )
    # Match the 6.5-inch ML4H text block closely so the typeset figure is not
    # downscaled and its labels remain legible in the two-column layout.
    fig, ax = plt.subplots(figsize=(6.45, 3.1))
    fig.subplots_adjust(left=0.37, right=0.985, bottom=0.20, top=0.78)
    y = np.arange(len(CONTRASTS), dtype=float)
    offset = 0.14
    reversal_color = "#F6E7B0"

    for row, is_reversed in zip(y, reversed_direction, strict=True):
        if is_reversed:
            ax.axhspan(
                row - 0.43,
                row + 0.43,
                color=reversal_color,
                alpha=0.60,
                zorder=0,
            )

    ax.axvline(0, color="#666666", linewidth=0.9, linestyle="--", zorder=1)
    unconstrained_artist = ax.errorbar(
        p_point,
        y - offset,
        xerr=np.vstack((p_point - p_low, p_high - p_point)),
        fmt="o",
        color="#333333",
        ecolor="#333333",
        markersize=4.2,
        capsize=2.2,
        linewidth=1.05,
        label="Unconstrained",
        zorder=3,
    )
    constrained_artist = ax.errorbar(
        c_point,
        y + offset,
        xerr=np.vstack((c_point - c_low, c_high - c_point)),
        fmt="s",
        color="#1768AC",
        ecolor="#1768AC",
        markersize=4.0,
        capsize=2.2,
        linewidth=1.05,
        label="JSON-schema constrained",
        zorder=3,
    )

    ax.set_yticks(y, [label for _, label in CONTRASTS])
    ax.invert_yaxis()
    ax.set_xlim(-42, 87)
    ax.set_xticks([-40, -20, 0, 20, 40, 60, 80])
    ax.set_xlabel(
        "State-macro difference (first − second), percentage points"
    )
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.55, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    reversal_key = Patch(
        facecolor=reversal_color,
        edgecolor="#D9BC64",
        linewidth=0.6,
        alpha=0.75,
        label="Selection reversal",
    )
    fig.legend(
        handles=[unconstrained_artist, constrained_artist, reversal_key],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        ncol=3,
        columnspacing=1.35,
        handletextpad=0.55,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "make_ml4h_findings_contract_figure_v0_7.py",
        "Date": None,
    }
    fig.savefig(
        OUTPUT_DIR / f"{FIGURE_STEM}.pdf",
        metadata=metadata,
    )
    fig.savefig(
        OUTPUT_DIR / f"{FIGURE_STEM}.svg",
        metadata=metadata,
    )
    fig.savefig(
        OUTPUT_DIR / f"{FIGURE_STEM}.png",
        dpi=240,
        metadata={"Creator": "make_ml4h_findings_contract_figure_v0_7.py"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
