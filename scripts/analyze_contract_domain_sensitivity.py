#!/usr/bin/env python3
"""Descriptive post-hoc domain sensitivity from saved responses; no inference.

Run: .venv/bin/python scripts/analyze_contract_domain_sensitivity.py
Only count/rate summaries are emitted; raw text and request/case IDs stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg2.slm_compact_contract import evaluate_contract_output

BASE = ROOT / "outputs/local_restricted/slm_benchmark_v2"
STATES = ("PRESENT_CURRENT", "ABSENT_EXPLICIT")
FIELDS = ("case_id", "domain", "criterion_id", "expected_state")
ENDPOINTS = ("state_macro_agreement", "strict_micro_agreement")
ORIGINAL_CONDITIONS = ("unconstrained", "json_schema_constrained")
FOLLOWUP_CONDITIONS = ("full_unconstrained", "full_constrained", "compact_unconstrained", "compact_constrained")
CORE = {
    "qwen_instruction_small": ("q15_pt_v1", "q15_it_v1"),
    "qwen_instruction_medium": ("q3_pt_v1", "q3_it_v1"),
    "qwen_size_pretrained": ("q15_pt_v1", "q3_pt_v1"),
    "qwen_size_instruction": ("q15_it_v1", "q3_it_v1"),
    "gemma_medical_adaptation_it": ("g4_it_v1", "mg4_it_v1"),
}
HEADLINE = ("qwen_instruction_small", "qwen_size_pretrained", "qwen_size_instruction")


def counts(rows: list[dict]) -> dict:
    return {
        "requests": len(rows),
        "vignettes": len({r["case_id"] for r in rows}),
        "reference_states": {
            state: {
                "requests": sum(r["expected_state"] == state for r in rows),
                "vignettes": len({r["case_id"] for r in rows if r["expected_state"] == state}),
            } for state in STATES
        },
        "missing_reference_states": [s for s in STATES if not any(r["expected_state"] == s for r in rows)],
    }


def metrics(rows: list[dict]) -> dict:
    """Keep every request; a macro requiring an absent stratum is undefined."""
    result = counts(rows)
    state_rates = []
    for state in STATES:
        selected = [r for r in rows if r["expected_state"] == state]
        correct = sum(r["state_correct"] for r in selected)
        rate = correct / len(selected) if selected else None
        result["reference_states"][state].update(correct=correct, agreement=rate)
        state_rates.append(rate)
    result.update(
        correct=sum(r["state_correct"] for r in rows),
        state_macro_agreement=None if None in state_rates else fmean(state_rates),
        strict_micro_agreement=sum(r["state_correct"] for r in rows) / len(rows) if rows else None,
        runtime_failure_count=sum(r["runtime_failed"] for r in rows),
        abstention_count=sum(r["parsed"]["abstained"] for r in rows),
        contract_invalid_count=sum(not r["parsed"]["logical_valid"] for r in rows),
    )
    return result


def aligned_cell(rows: list[dict], expected: dict[str, dict]) -> list[dict]:
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate response request IDs")
    if set(ids) != set(expected):
        raise ValueError("response request IDs differ from frozen dataset")
    indexed = {row["request_id"]: row for row in rows}
    for request_id, reference in expected.items():
        row = indexed[request_id]
        if any(row.get(key) != reference[key] for key in FIELDS):
            raise ValueError("paired response metadata differ from frozen dataset")
        if row["expected_state"] not in STATES:
            raise ValueError("unexpected reference state")
        if not isinstance(row.get("state_correct"), bool) or not isinstance(row.get("runtime_failed"), bool):
            raise ValueError("correctness and runtime flags must be Boolean")
    return [indexed[key] for key in sorted(expected)]


def rescore(rows: list[dict], contract: str, *, followup: bool) -> None:
    """Re-use the frozen full validator and declared compact adapter unchanged."""
    for row in rows:
        evaluation = evaluate_contract_output(
            row.get("raw_output"), contract=contract,
            expected_criterion_id=row["criterion_id"], runtime_failed=row["runtime_failed"],
        )
        parsed = evaluation["parsed"]
        if parsed != row["parsed"]:
            raise ValueError("recomputed parsing differs from stored response")
        correct = bool(parsed["logical_valid"] and not parsed["abstained"] and parsed["state"] == row["expected_state"])
        if row["state_correct"] is not correct:
            raise ValueError("recomputed full-record scoring differs from stored response")
        if followup and (any(row.get(key) != value for key, value in evaluation.items())
                         or row.get("state_only_correct") is not (evaluation["state_only_state"] == row["expected_state"])):
            raise ValueError("recomputed follow-up adapter/scoring differs from stored response")


def difference(b: float | None, a: float | None) -> float | None:
    return None if a is None or b is None else b - a


def reversal(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a * b < 0


def summarize(panel: dict, expected: dict, conditions: tuple, contrasts: dict = CORE) -> dict:
    reference = list(expected.values())
    domains = sorted({row["domain"] for row in reference})
    data = {"overall": counts(reference), "domains": {d: counts([r for r in reference if r["domain"] == d]) for d in domains}}
    data["weighting"] = {
        d: {
            "equal_domain_weight": 1 / len(domains),
            "pooled_micro_domain_weight": data["domains"][d]["requests"] / len(reference),
            "pooled_macro_domain_weight_within_state": {
                s: data["domains"][d]["reference_states"][s]["requests"] / data["overall"]["reference_states"][s]["requests"]
                if data["overall"]["reference_states"][s]["requests"] else None for s in STATES
            },
        } for d in domains
    }
    result = {"conditions": list(conditions), "data": data, "models": {}, "contrasts": {}}
    for model, cells in panel.items():
        if set(cells) != set(conditions):
            raise ValueError("incomplete condition panel")
        result["models"][model] = {}
        for condition in conditions:
            rows = aligned_cell(cells[condition], expected)
            by_domain = {domain: metrics([r for r in rows if r["domain"] == domain]) for domain in domains}
            equal = {
                endpoint: None if any(cell[endpoint] is None for cell in by_domain.values())
                else fmean(cell[endpoint] for cell in by_domain.values()) for endpoint in ENDPOINTS
            }
            equal["missing_reference_states_by_domain"] = {
                d: cell["missing_reference_states"] for d, cell in by_domain.items() if cell["missing_reference_states"]
            }
            result["models"][model][condition] = {"pooled": metrics(rows), "equal_domain": equal, "by_domain": by_domain}
    for contrast, (a, b) in contrasts.items():
        cells = {}
        for condition in conditions:
            model_a = result["models"][a][condition]
            model_b = result["models"][b][condition]
            cells[condition] = {
                weighting: {endpoint: difference(model_b[weighting][endpoint], model_a[weighting][endpoint]) for endpoint in ENDPOINTS}
                for weighting in ("pooled", "equal_domain")
            }
            cells[condition]["by_domain"] = {
                d: {endpoint: difference(model_b["by_domain"][d][endpoint], model_a["by_domain"][d][endpoint]) for endpoint in ENDPOINTS}
                for d in domains
            }
        result["contrasts"][contrast] = {"model_a": a, "model_b": b, "orientation": "model_b_minus_model_a", "conditions": cells}
    return result


def decoder_reversals(section: dict, u: str, c: str) -> dict:
    result = {}
    for name, contrast in section["contrasts"].items():
        cells = contrast["conditions"]
        entry = {w: reversal(cells[u][w]["state_macro_agreement"], cells[c][w]["state_macro_agreement"])
                 for w in ("pooled", "equal_domain")}
        entry["by_domain"] = {d: reversal(cells[u]["by_domain"][d]["state_macro_agreement"], cells[c]["by_domain"][d]["state_macro_agreement"])
                              for d in section["data"]["domains"]}
        entry["domain_exceptions_to_pooled_reversal"] = [d for d, value in entry["by_domain"].items() if entry["pooled"] is True and value is not True]
        entry["ties_by_domain_and_condition"] = {d: [condition for condition in (u, c) if cells[condition]["by_domain"][d]["state_macro_agreement"] == 0]
                                                for d in section["data"]["domains"]}
        result[name] = entry
    return result


def assert_close(observed, expected, label: str) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            raise ValueError(f"canonical mismatch: {label}")
    elif not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"canonical mismatch: {label}")


def build() -> dict:
    hashes = {}

    def read(path: Path, *, jsonl=False):
        content = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        return [json.loads(line) for line in content.splitlines() if line.strip()] if jsonl else json.loads(content)

    dataset = BASE / "e1/dataset"
    manifest = read(dataset / "manifest.json")
    blueprint = read(dataset / "blueprint.json")
    expected = {}
    case_ids = []
    for case in blueprint["cases"]:
        case_ids.append(case["case_id"])
        for criterion, request_id in case["request_ids"].items():
            if request_id in expected:
                raise ValueError("duplicate blueprint request IDs")
            expected[request_id] = {"case_id": case["case_id"], "domain": case["domain"], "criterion_id": criterion, "expected_state": case["expected_states"][criterion]}
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate blueprint vignette IDs")
    requests = read(dataset / "requests.jsonl", jsonl=True)
    if len(requests) != len(expected) or {r["request_id"] for r in requests} != set(expected):
        raise ValueError("frozen request/blueprint IDs differ")
    for row in requests:
        if any(row[key] != expected[row["request_id"]][key] for key in FIELDS if key != "expected_state"):
            raise ValueError("frozen request/blueprint metadata differ")
    for filename, key in (("requests.jsonl", "output_requests_sha256"), ("blueprint.json", "output_blueprint_sha256")):
        if hashes[str((dataset / filename).relative_to(ROOT))] != manifest[key]:
            raise ValueError("frozen dataset checksum differs")
    original, canonical = {}, {}
    for condition, folder in zip(ORIGINAL_CONDITIONS, ("e1", "e1_s2")):
        aggregate = read(BASE / folder / "aggregate/results.json")
        if aggregate["dataset_manifest_sha256"] != hashes[str((dataset / "manifest.json").relative_to(ROOT))]:
            raise ValueError("original dataset manifest differs")
        canonical[condition] = {m["manifest_id"]: m for m in aggregate["models"]}
        declaration_key = "paired_contrasts" if folder == "e1" else "s2_paired_model_contrasts"
        declared = {c["contrast_id"]: (c["model_a"], c["model_b"]) for c in aggregate[declaration_key]}
        if any(declared[name] != pair for name, pair in CORE.items()):
            raise ValueError("core contrast declaration differs")
        for model in canonical[condition]:
            rows = aligned_cell(read(BASE / folder / "results" / model / "responses.jsonl", jsonl=True), expected)
            rescore(rows, "full", followup=False)
            original.setdefault(model, {})[condition] = rows
    original_result = summarize(original, expected, ORIGINAL_CONDITIONS)
    validation_checks = 0
    for model, cells in original_result["models"].items():
        for condition, cell in cells.items():
            stored = canonical[condition][model]["metrics"]
            for computed, target in [(cell["pooled"], stored["overall"]), *[(v, stored["by_domain"][d]) for d, v in cell["by_domain"].items()]]:
                for metric, old in (("state_macro_agreement", "macro_state_agreement"), ("strict_micro_agreement", "state_agreement_unconditional"), ("requests", "n"), ("correct", "state_correct_unconditional")):
                    assert_close(computed[metric], target[old], "original pooled/domain " + metric)
                    validation_checks += 1
                for state in STATES:
                    assert_close(computed["reference_states"][state]["requests"], target["per_reference_state"].get(state, {}).get("n", 0), "original domain state count")
                    validation_checks += 1
    derived = read(ROOT / "paper_artifacts/derived/ml4h-findings-contract-analysis-v0.7.json")
    for model in derived["models"]:
        for condition in ORIGINAL_CONDITIONS:
            for metric in ENDPOINTS:
                assert_close(original_result["models"][model["manifest_id"]][condition]["pooled"][metric], model[condition][metric], "v0.7 derived model " + metric)
                validation_checks += 1
    original_result["decoder_reversals"] = decoder_reversals(original_result, *ORIGINAL_CONDITIONS)
    for contrast in derived["prespecified_core_contrasts"]:
        name = contrast["contrast_id"]
        for condition in ORIGINAL_CONDITIONS:
            assert_close(original_result["contrasts"][name]["conditions"][condition]["pooled"]["state_macro_agreement"], contrast[condition]["state_macro_difference"], "v0.7 derived contrast")
            validation_checks += 1
        if original_result["decoder_reversals"][name]["pooled"] != contrast["state_macro_selection_reversal"]:
            raise ValueError("v0.7 reversal differs")
    followup_aggregate = read(BASE / "contract_ablation_v1/aggregate/results.json")
    if set(followup_aggregate["models"]) != set(original):
        raise ValueError("original/follow-up model panels differ")
    declared = {c["contrast_id"]: tuple(c["models"]) for c in followup_aggregate["contrasts"]}
    if any(declared[name] != pair for name, pair in CORE.items()):
        raise ValueError("follow-up core contrast declaration differs")
    followup = {}
    for model in followup_aggregate["models"]:
        followup[model] = {}
        for condition in FOLLOWUP_CONDITIONS:
            rows = aligned_cell(read(BASE / "contract_ablation_v1/results" / model / condition / "responses.jsonl", jsonl=True), expected)
            if any(row.get("manifest_id") != model or row.get("condition") != condition for row in rows):
                raise ValueError("follow-up model/condition differs")
            rescore(rows, condition.split("_", 1)[0], followup=True)
            followup[model][condition] = rows
    followup_result = summarize(followup, expected, FOLLOWUP_CONDITIONS)
    for model, cells in followup_result["models"].items():
        for condition, cell in cells.items():
            for metric in ENDPOINTS:
                assert_close(cell["pooled"][metric], followup_aggregate["models"][model]["conditions"][condition][metric]["estimate"], "follow-up pooled " + metric)
                validation_checks += 1
    for contrast in followup_aggregate["contrasts"]:
        if contrast["contrast_id"] in CORE:
            for condition in FOLLOWUP_CONDITIONS:
                for metric in ENDPOINTS:
                    assert_close(followup_result["contrasts"][contrast["contrast_id"]]["conditions"][condition]["pooled"][metric], contrast["conditions"][condition][metric]["estimate"], "follow-up contrast " + metric)
                    validation_checks += 1
    followup_result["decoder_reversals"] = {contract: decoder_reversals(followup_result, contract + "_unconstrained", contract + "_constrained") for contract in ("full", "compact")}
    data = original_result["data"]
    assert_close(data["overall"]["requests"], manifest["requests"], "manifest request count")
    assert_close(data["overall"]["vignettes"], manifest["cases"], "manifest vignette count")
    assert_close(data["overall"]["requests"], followup_aggregate["data"]["requests_per_cell"], "follow-up request count")
    assert_close(data["overall"]["vignettes"], followup_aggregate["data"]["vignette_count"], "follow-up vignette count")
    for state in STATES:
        assert_close(data["overall"]["reference_states"][state]["requests"], manifest["state_counts"][state], "manifest reference-state count")
        assert_close(data["overall"]["reference_states"][state]["requests"], followup_aggregate["data"]["reference_state_counts"][state], "follow-up reference-state count")
    validation_checks += 8
    for d, values in data["domains"].items():
        assert_close(values["requests"], manifest["domain_request_counts"][d], "manifest domain requests")
        assert_close(values["vignettes"], manifest["domain_case_counts"][d], "manifest domain vignettes")
        validation_checks += 2
    return {
        "analysis": "descriptive_posthoc_contract_domain_sensitivity", "schema_version": "1.0",
        "status": "Descriptive post-hoc point estimates only; no confidence intervals or significance claims; no new inference.",
        "definitions": {
            "state_macro_agreement": "Mean of agreement in the two fixed reference states; all failures and abstentions are incorrect. Null if either reference state is absent.",
            "strict_micro_agreement": "Correct full records divided by every request, including failures and abstentions.",
            "pooled": "For each reference state pool correct/count across domains, then average the two state rates; domain weights therefore differ by reference state. Micro pools all requests.",
            "equal_domain": "Unweighted mean of within-domain state-macro (or within-domain strict micro); no domain is omitted. Macro is null if any domain lacks a reference state.",
            "reversal": "Strictly opposite nonzero signs across decoders; ties are not reversals, missing estimates yield null.",
            "vignettes": "Distinct case IDs per domain; state-specific vignette counts can overlap and must not be summed.",
        },
        "validation": {"all_request_and_gold_metadata_aligned": True, "all_responses_reparsed_and_rescored": True,
                       "responses_reparsed_and_rescored": sum(len(rows) for panel in (original, followup) for cells in panel.values() for rows in cells.values()),
                       "canonical_metric_count_checks": validation_checks, "canonical_agreement_tolerance": 1e-12,
                       "input_sha256": hashes},
        "original": original_result, "fresh_followup": followup_result,
        "original_three_headline_reversals": {name: original_result["decoder_reversals"][name] for name in HEADLINE},
    }


def table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def number(value, signed=False):
    return "NA (missing state)" if value is None else format(value, "+.4f" if signed else ".4f")


def render_report(result: dict) -> str:
    data = result["original"]["data"]
    lines = ["# Domain sensitivity of output-contract results", "", result["status"], "",
             "All values are proportions; differences are model B minus A. Every failure and abstention remains in the denominator. A missing reference state makes the required two-state macro undefined; no domain or state is silently dropped.", "",
             "Pooled macro gives each reference state weight 1/2, with domains weighted by their request count within that state. Equal-domain macro gives each domain weight 1/3 after computing its two-state macro. State-specific vignette counts overlap.", ""]
    lines += table(("Domain", "Requests", "Vignettes", "Present requests / vignettes", "Absent requests / vignettes", "Missing states"), [
        (domain, cell["requests"], cell["vignettes"], *[f"{cell['reference_states'][s]['requests']} / {cell['reference_states'][s]['vignettes']}" for s in STATES], ", ".join(cell["missing_reference_states"]) or "none")
        for domain, cell in [("Overall", data["overall"]), *data["domains"].items()]
    ])
    lines += ["", "## Original three headline reversals", "", "Signs describe point estimates only. Domain exceptions identify absence of the pooled reversal, including a tie.", ""]
    lines += table(("Contrast (B − A)", "Pooled U → C", "Equal-domain U → C", "Domain exceptions"), [
        (name, *[" → ".join(number(result["original"]["contrasts"][name]["conditions"][c][w]["state_macro_agreement"], True) for c in ORIGINAL_CONDITIONS) for w in ("pooled", "equal_domain")], ", ".join(result["original_three_headline_reversals"][name]["domain_exceptions_to_pooled_reversal"]) or "none") for name in HEADLINE
    ])
    for section_name in ("original", "fresh_followup"):
        section = result[section_name]
        conditions = section["conditions"]
        lines += ["", f"## {section_name}: model estimates", "", "Cells give state-macro / strict-micro agreement; equal-domain micro is additionally reported as a descriptive domain average.", ""]
        rows = []
        for model, cells in section["models"].items():
            for scope in ("pooled", "equal_domain", *data["domains"]):
                vals = []
                for condition in conditions:
                    cell = cells[condition][scope] if scope in ("pooled", "equal_domain") else cells[condition]["by_domain"][scope]
                    vals.append(" / ".join(number(cell[e]) for e in ENDPOINTS))
                rows.append((model, scope, *vals))
        lines += table(("Model", "Scope", *conditions), rows)
        lines += ["", f"## {section_name}: five core contrasts", "", "Cells give state-macro difference / strict-micro difference. Pair direction is explicitly model B minus model A.", ""]
        rows = []
        for name, contrast in section["contrasts"].items():
            for scope in ("pooled", "equal_domain", *data["domains"]):
                vals = []
                for condition in conditions:
                    cell = contrast["conditions"][condition]
                    metrics_ = cell[scope] if scope in ("pooled", "equal_domain") else cell["by_domain"][scope]
                    vals.append(" / ".join(number(metrics_[e], True) for e in ENDPOINTS))
                rows.append((f"{name}: {contrast['model_b']} − {contrast['model_a']}", scope, *vals))
        lines += table(("Contrast", "Scope", *conditions), rows)
    lines += ["", "## Validation", "", f"Reparsed and rescored {result['validation']['responses_reparsed_and_rescored']:,} saved responses with the existing full validator/compact adapter. Checked request IDs and case/domain/criterion/reference-state alignment against the frozen request file and blueprint; original pooled and domain summaries match their canonical aggregates, pooled original endpoints match v0.7 derived results, and fresh pooled endpoints and contrasts match the four-condition aggregate (absolute tolerance 1e-12). Input hashes are recorded in results.json.", "", "Run: `.venv/bin/python scripts/analyze_contract_domain_sensitivity.py`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "contract_domain_sensitivity_v1")
    args = parser.parse_args()
    result = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"Wrote count/rate summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
