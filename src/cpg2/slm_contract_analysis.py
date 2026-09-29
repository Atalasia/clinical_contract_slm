"""Paired analysis for the frozen compact-contract, 2 x 2 follow-up.

Input validation deliberately requires a complete matched panel.  This module
does not read response files, repair predictions, select models, or verify run
provenance: the experiment runner performs those steps before calling it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


CONDITIONS = (
    "full_unconstrained",
    "full_constrained",
    "compact_unconstrained",
    "compact_constrained",
)
REFERENCE_STATES = ("PRESENT_CURRENT", "ABSENT_EXPLICIT")
ENDPOINTS = (
    "state_macro_agreement",
    "strict_micro_agreement",
    "state_only_macro_agreement",
    "state_only_micro_agreement",
)
_ALIGNMENT_FIELDS = ("case_id", "domain", "criterion_id", "expected_state")


def _validate_panel(
    panel: dict[str, dict[str, list[dict[str, Any]]]],
    contrasts: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[dict[str, Any]]]]:
    if not isinstance(panel, dict) or not panel:
        raise ValueError("analysis requires a nonempty model panel")
    if any(not isinstance(model, str) or not model for model in panel):
        raise ValueError("model IDs must be nonempty strings")
    models = sorted(panel)
    reference: dict[str, dict[str, Any]] | None = None
    aligned: dict[tuple[str, str], list[dict[str, Any]]] = {}
    request_ids: list[str] = []
    for model in models:
        cells = panel[model]
        if not isinstance(cells, dict) or set(cells) != set(CONDITIONS):
            raise ValueError(f"{model}: expected exactly four ablation conditions")
        for condition in CONDITIONS:
            rows = cells[condition]
            label = f"{model}/{condition}"
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{label}: response cell is empty or not a list")
            indexed: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{label}: response must be an object")
                for key in ("request_id", *_ALIGNMENT_FIELDS):
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError(f"{label}: {key} must be a nonempty string")
                request_id = row["request_id"]
                if request_id in indexed:
                    raise ValueError(f"{label}: duplicate request_id {request_id}")
                indexed[request_id] = row
                if row["expected_state"] not in REFERENCE_STATES:
                    raise ValueError(f"{label}: unexpected reference state")
                for key in ("state_correct", "state_only_correct", "runtime_failed"):
                    if not isinstance(row.get(key), bool):
                        raise ValueError(f"{label}: {key} must be Boolean")
                parsed = row.get("parsed")
                if not isinstance(parsed, dict):
                    raise ValueError(f"{label}: parsed must be an object")
                for key in ("parse_valid", "schema_valid", "logical_valid", "abstained"):
                    if not isinstance(parsed.get(key), bool):
                        raise ValueError(f"{label}: parsed.{key} must be Boolean")
                if "failure" not in parsed or not (
                    parsed["failure"] is None or isinstance(parsed["failure"], str)
                ):
                    raise ValueError(f"{label}: parsed.failure must be a string or null")
                if parsed["schema_valid"] and not parsed["parse_valid"]:
                    raise ValueError(f"{label}: schema validity requires parse validity")
                if parsed["logical_valid"] and not parsed["schema_valid"]:
                    raise ValueError(f"{label}: logical validity requires schema validity")
                if row["state_correct"] and not (
                    parsed["logical_valid"]
                    and not parsed["abstained"]
                    and row["state_only_correct"]
                ):
                    raise ValueError(f"{label}: full-record correctness flags contradict")
                if row["runtime_failed"] and (
                    parsed["parse_valid"] or row["state_only_correct"]
                ):
                    raise ValueError(f"{label}: runtime failure cannot be a parsed prediction")
                for key in ("output_tokens", "prompt_tokens"):
                    tokens = row.get(key)
                    if tokens is None and row["runtime_failed"]:
                        continue
                    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                        raise ValueError(f"{label}: {key} must be a nonnegative integer")
            if reference is None:
                reference = indexed
                request_ids = sorted(indexed)
                if {row["expected_state"] for row in rows} != set(REFERENCE_STATES):
                    raise ValueError("analysis requires both reference-state strata")
            elif set(indexed) != set(reference):
                raise ValueError(f"{label}: incomplete or mismatched request IDs")
            else:
                for request_id in request_ids:
                    for key in _ALIGNMENT_FIELDS:
                        if indexed[request_id][key] != reference[request_id][key]:
                            raise ValueError(f"{label}: paired metadata differ for {request_id}: {key}")
            aligned[model, condition] = [indexed[key] for key in request_ids]
    if not isinstance(contrasts, list):
        raise ValueError("contrasts must be a list")
    seen_contrasts: set[str] = set()
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ValueError("contrast must be an object")
        contrast_id = contrast.get("contrast_id")
        if not isinstance(contrast_id, str) or not contrast_id or contrast_id in seen_contrasts:
            raise ValueError("contrast IDs must be unique nonempty strings")
        seen_contrasts.add(contrast_id)
        pair = contrast.get("models")
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"{contrast_id}: contrast requires two distinct models")
        if any(not isinstance(model, str) or model not in panel for model in pair):
            raise ValueError(f"{contrast_id}: contrast model is absent from panel")
    return models, request_ids, aligned


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {"n_requests": total}
    for flag in ("parse_valid", "schema_valid", "logical_valid"):
        count = sum(row["parsed"][flag] for row in rows)
        result[flag] = {"count": count, "denominator": total, "rate": count / total}
    result["abstention_count"] = sum(row["parsed"]["abstained"] for row in rows)
    result["valid_abstention_count"] = sum(
        row["parsed"]["abstained"] and row["parsed"]["logical_valid"] for row in rows
    )
    result["runtime_failure_count"] = sum(row["runtime_failed"] for row in rows)
    length_count = sum(row.get("finish_reason") == "length" for row in rows)
    result["length_truncation"] = {
        "count": length_count,
        "denominator": total,
        "rate": length_count / total,
        "definition": "finish_reason equals length; stopping at the output-token limit, not an independent semantic-error judgment",
        "missing_finish_reason_count": sum(row.get("finish_reason") is None for row in rows),
    }
    result["failure_counts"] = dict(sorted(Counter(
        row["parsed"]["failure"] or "none" for row in rows
    ).items()))
    schema_count = result["schema_valid"]["count"]
    invalid_logic_count = sum(
        row["parsed"]["schema_valid"] and not row["parsed"]["logical_valid"] for row in rows
    )
    result["logical_invalid_among_schema_valid"] = {
        "count": invalid_logic_count,
        "denominator": schema_count,
        "rate": invalid_logic_count / schema_count if schema_count else None,
    }
    for key in ("output_tokens", "prompt_tokens"):
        tokens = [row[key] for row in rows if row.get(key) is not None]
        result[key] = {
            "total": sum(tokens),
            "mean": sum(tokens) / len(tokens) if tokens else None,
            "observed_requests": len(tokens),
            "missing_requests": total - len(tokens),
        }
    result["reference_state_counts"] = {
        state: sum(row["expected_state"] == state for row in rows)
        for state in REFERENCE_STATES
    }
    return result


def summarize_ablation(
    panel: dict[str, dict[str, list[dict[str, Any]]]],
    contrasts: list[dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260922,
) -> dict[str, Any]:
    """Summarize complete cells using one shared paired vignette bootstrap.

    Resampling uses cases (not requests) with replacement, retaining all
    requests within each drawn case.  A resample missing either reference state
    is excluded only from macro endpoints.  Micro endpoints retain every draw.
    All contrasts are second declared model minus first; interaction signs are
    explicit in the returned metadata.  Intervals are unadjusted descriptive
    percentile intervals, not claims of significance or causal effects.
    """
    import numpy as np

    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    models, request_ids, aligned = _validate_panel(panel, contrasts)
    cells = [(model, condition) for model in models for condition in CONDITIONS]
    cell_index = {cell: index for index, cell in enumerate(cells)}
    reference_rows = aligned[cells[0]]
    case_ids = sorted({row["case_id"] for row in reference_rows})
    case_index = {case_id: index for index, case_id in enumerate(case_ids)}
    gold_counts = np.zeros((len(case_ids), 2), dtype=np.float64)
    successes = np.zeros((len(case_ids), len(cells), 2, 2), dtype=np.float64)
    for row_index, reference in enumerate(reference_rows):
        case = case_index[reference["case_id"]]
        state = REFERENCE_STATES.index(reference["expected_state"])
        gold_counts[case, state] += 1
        for index, cell in enumerate(cells):
            row = aligned[cell][row_index]
            successes[case, index, 0, state] += row["state_correct"]
            successes[case, index, 1, state] += row["state_only_correct"]
    gold_total = gold_counts.sum(axis=0)
    success_total = successes.sum(axis=0)
    estimates = np.empty((len(cells), len(ENDPOINTS)), dtype=np.float64)
    estimates[:, 0] = (success_total[:, 0] / gold_total).mean(axis=-1)
    estimates[:, 1] = success_total[:, 0].sum(axis=-1) / gold_total.sum()
    estimates[:, 2] = (success_total[:, 1] / gold_total).mean(axis=-1)
    estimates[:, 3] = success_total[:, 1].sum(axis=-1) / gold_total.sum()

    replicates = np.full((bootstrap_replicates, len(cells), len(ENDPOINTS)), np.nan)
    rng = np.random.default_rng(bootstrap_seed)
    probabilities = np.full(len(case_ids), 1 / len(case_ids))
    missing_stratum_replicates = 0
    # Bounded temporary arrays; every endpoint/model/condition shares each draw.
    for start in range(0, bootstrap_replicates, 512):
        stop = min(start + 512, bootstrap_replicates)
        draws = rng.multinomial(len(case_ids), probabilities, size=stop - start)
        denominators = draws @ gold_counts
        numerators = (draws @ successes.reshape(len(case_ids), -1)).reshape(
            stop - start, len(cells), 2, 2
        )
        complete = (denominators > 0).all(axis=1)
        missing_stratum_replicates += int((~complete).sum())
        for outcome, macro_index, micro_index in ((0, 0, 1), (1, 2, 3)):
            block = replicates[start:stop, :, macro_index]
            block[complete] = (
                numerators[complete, :, outcome, :] / denominators[complete, None, :]
            ).mean(axis=-1)
            replicates[start:stop, :, micro_index] = (
                numerators[:, :, outcome, :].sum(axis=-1) / denominators.sum(axis=-1)[:, None]
            )

    def describe(estimate: float, samples: Any) -> dict[str, Any]:
        finite = samples[np.isfinite(samples)]
        lower, upper = np.quantile(finite, [0.025, 0.975], method="linear") if finite.size else (None, None)
        return {
            "estimate": float(estimate),
            "ci_low": None if lower is None else float(lower),
            "ci_high": None if upper is None else float(upper),
            "confidence": 0.95,
            "replicates_used": int(finite.size),
            "replicates_excluded": bootstrap_replicates - int(finite.size),
        }

    def linear_summary(terms: list[tuple[tuple[str, str], int]]) -> dict[str, Any]:
        point = sum(weight * estimates[cell_index[cell]] for cell, weight in terms)
        sampled = sum(weight * replicates[:, cell_index[cell]] for cell, weight in terms)
        return {endpoint: describe(point[index], sampled[:, index]) for index, endpoint in enumerate(ENDPOINTS)}

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "paired_compact_contract_factorial_followup",
        "endpoint_definitions": {
            "state_macro_agreement": "Primary model-selection endpoint: equal mean of full-record state agreement in PRESENT_CURRENT and ABSENT_EXPLICIT strata; invalid records and abstentions score zero.",
            "strict_micro_agreement": "Operational endpoint: fraction of all requests with a valid nonabstained full record and correct state.",
            "state_only_macro_agreement": "Diagnostic only: equal-stratum agreement of complete-JSON, matching-ID, valid-state predictions, ignoring other fields and abstention; not usable-record correctness.",
            "state_only_micro_agreement": "Diagnostic only: all-request agreement of the same state-only predictions.",
        },
        "bootstrap": {
            "unit": "vignette_case_id",
            "replicates_requested": bootstrap_replicates,
            "seed": bootstrap_seed,
            "generator": "numpy.random.default_rng / PCG64 multinomial cluster multiplicities",
            "numpy_version": np.__version__,
            "shared_draws_across_all_cells_and_models": True,
            "confidence": 0.95,
            "interval": "percentile; numpy.quantile method=linear",
            "macro_replicates_missing_reference_stratum": missing_stratum_replicates,
            "micro_replicates_excluded": 0,
            "multiplicity_adjustment": "none; intervals and sign reversals are descriptive",
        },
        "data": {
            "model_count": len(models),
            "conditions": list(CONDITIONS),
            "requests_per_cell": len(request_ids),
            "vignette_count": len(case_ids),
            "reference_state_counts": {state: int(gold_total[index]) for index, state in enumerate(REFERENCE_STATES)},
        },
        "interaction_definition": "(compact_constrained - compact_unconstrained) - (full_constrained - full_unconstrained)",
        "models": {},
        "contrasts": [],
    }
    for model in models:
        model_result: dict[str, Any] = {"conditions": {}, "compact_minus_full": {}, "constrained_minus_unconstrained": {}}
        for condition in CONDITIONS:
            model_result["conditions"][condition] = {
                **_diagnostics(aligned[model, condition]),
                **linear_summary([((model, condition), 1)]),
            }
        for decoder in ("unconstrained", "constrained"):
            model_result["compact_minus_full"][decoder] = linear_summary([
                ((model, f"compact_{decoder}"), 1), ((model, f"full_{decoder}"), -1)
            ])
        for contract in ("full", "compact"):
            model_result["constrained_minus_unconstrained"][contract] = linear_summary([
                ((model, f"{contract}_constrained"), 1), ((model, f"{contract}_unconstrained"), -1)
            ])
        model_result["contract_by_decoder_interaction"] = linear_summary([
            ((model, "compact_constrained"), 1), ((model, "compact_unconstrained"), -1),
            ((model, "full_constrained"), -1), ((model, "full_unconstrained"), 1),
        ])
        result["models"][model] = model_result

    for declaration in contrasts:
        a, b = declaration["models"]

        def pair_terms(condition: str, weight: int = 1) -> list[tuple[tuple[str, str], int]]:
            return [((b, condition), weight), ((a, condition), -weight)]

        contrast: dict[str, Any] = {
            **declaration,
            "direction": f"{b} minus {a}",
            "conditions": {condition: linear_summary(pair_terms(condition)) for condition in CONDITIONS},
            "decoder_shifts": {},
            "compact_minus_full": {},
            "descriptive_sign_reversals": {},
        }
        for contract in ("full", "compact"):
            contrast["decoder_shifts"][contract] = linear_summary(
                pair_terms(f"{contract}_constrained") + pair_terms(f"{contract}_unconstrained", -1)
            )
        for decoder in ("unconstrained", "constrained"):
            contrast["compact_minus_full"][decoder] = linear_summary(
                pair_terms(f"compact_{decoder}") + pair_terms(f"full_{decoder}", -1)
            )
        contrast["change_in_decoder_shift_compact_minus_full"] = linear_summary(
            pair_terms("compact_constrained") + pair_terms("compact_unconstrained", -1)
            + pair_terms("full_constrained", -1) + pair_terms("full_unconstrained")
        )
        for endpoint in ENDPOINTS:
            points = {condition: contrast["conditions"][condition][endpoint]["estimate"] for condition in CONDITIONS}
            contrast["descriptive_sign_reversals"][endpoint] = {
                "across_decoder_with_full_contract": points["full_unconstrained"] * points["full_constrained"] < 0,
                "across_decoder_with_compact_contract": points["compact_unconstrained"] * points["compact_constrained"] < 0,
                "across_contract_with_unconstrained_decoder": points["full_unconstrained"] * points["compact_unconstrained"] < 0,
                "across_contract_with_constrained_decoder": points["full_constrained"] * points["compact_constrained"] < 0,
            }
        contrast["sign_reversal_note"] = "Opposite nonzero point-estimate signs only; ties are not reversals; no significance or causal claim."
        result["contrasts"].append(contrast)
    return result
