#!/usr/bin/env python3
"""Print a compact Markdown summary of the matched ablation; write no files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/local_restricted/slm_benchmark_v2/contract_ablation_v1/aggregate/results.json"
MODELS = {
    "q15_pt_v1": "Qwen2.5 1.5B base",
    "q15_it_v1": "Qwen2.5 1.5B Instruct",
    "q3_pt_v1": "Qwen2.5 3B base",
    "q3_it_v1": "Qwen2.5 3B Instruct",
    "g4_it_v1": "Gemma 3 4B IT",
    "mg4_it_v1": "MedGemma 4B IT",
    "mg15_it_v1": "MedGemma 1.5 4B IT",
    "q35_ref_v1": "Qwen3.5 4B reference",
}
CONDITIONS = ("full_unconstrained", "full_constrained", "compact_unconstrained", "compact_constrained")
HEADERS = ("Full/U", "Full/C", "Compact/U", "Compact/C")
CORE_CONTRASTS = {
    "qwen_instruction_small": "1.5B: Instruct − base",
    "qwen_instruction_medium": "3B: Instruct − base",
    "qwen_size_pretrained": "Base: 3B − 1.5B",
    "qwen_size_instruction": "Instruct: 3B − 1.5B",
    "gemma_medical_adaptation_it": "4B: MedGemma − Gemma 3",
}


def _number(value, *, signed: bool = False, precision: int = 3) -> str:
    return "NA" if value is None else format(value, f"{'+' if signed else ''}.{precision}f")


def _interval(metric: dict) -> str:
    return f"{_number(metric['estimate'], signed=True)} [{_number(metric['ci_low'], signed=True)}, {_number(metric['ci_high'], signed=True)}]"


def _table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *[
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    ]]


def render_report(data: dict) -> str:
    """Render observed summaries only; inference and causal claims are excluded."""
    if data.get("analysis") != "paired_compact_contract_factorial_followup":
        raise ValueError("input is not a compact-contract factorial aggregate")
    if set(data["models"]) != set(MODELS):
        raise ValueError("report requires the complete declared eight-model panel")
    bootstrap = data["bootstrap"]
    lines = [
        "# Compact-output-contract ablation",
        "",
        f"{data['data']['requests_per_cell']} requests per cell; {data['data']['vignette_count']} vignette clusters; eight models, four conditions. U = unconstrained; C = JSON-schema constrained.",
        "Agreement and differences are proportions (not percentages or percentage points); token values are generated-token counts. Differences retain their displayed signs.",
        f"Intervals are 95% paired vignette-cluster bootstrap intervals ({bootstrap['replicates_requested']:,} shared draws; {bootstrap['macro_replicates_missing_reference_stratum']} draws excluded from macro endpoints for missing a reference stratum). No multiplicity adjustment.",
    ]
    for title, endpoint in (
        ("Primary: full-record state-macro agreement", "state_macro_agreement"),
        ("Operational: strict micro agreement", "strict_micro_agreement"),
    ):
        lines += ["", f"## {title}", ""]
        lines += _table(("Model", *HEADERS), [
            (label, *[_number(data["models"][model]["conditions"][cell][endpoint]["estimate"]) for cell in CONDITIONS])
            for model, label in MODELS.items()
        ])
    lines += ["", "## Compact minus full: primary endpoint and state-only diagnostic", "",
              "Macro columns give estimate [95% CI]. State-only changes ignore other fields and abstention and are not full-record usability scores.", ""]
    lines += _table(("Model", "Macro Δ, U", "Macro Δ, C", "State-only macro Δ, U / C"), [
        (label, *[_interval(data["models"][model]["compact_minus_full"][decoder]["state_macro_agreement"])
                   for decoder in ("unconstrained", "constrained")],
         " / ".join(_number(data["models"][model]["compact_minus_full"][decoder]["state_only_macro_agreement"]["estimate"], signed=True)
                    for decoder in ("unconstrained", "constrained")))
        for model, label in MODELS.items()
    ])
    lines += ["", "## Output diagnostics: full → compact", "",
              "Each count pair is schema-valid / logically-invalid-among-schema-valid. Token means include all requests with observed token counts.", ""]
    diagnostics = []
    for model, label in MODELS.items():
        for decoder, short in (("unconstrained", "U"), ("constrained", "C")):
            cells = [data["models"][model]["conditions"][f"{contract}_{decoder}"] for contract in ("full", "compact")]
            diagnostics.append((label, short,
                                " → ".join(f"{cell['schema_valid']['count']} / {cell['logical_invalid_among_schema_valid']['count']}" for cell in cells),
                                " → ".join(_number(cell["output_tokens"]["mean"], precision=1) for cell in cells)))
    lines += _table(("Model", "Decoder", "Schema-valid / logical-invalid", "Mean generated tokens"), diagnostics)
    lines += ["", "## Five core contrasts: primary state-macro agreement", "",
              "Every value is second declared model minus first (b − a); comparison labels spell out that direction. Signs are numerical descriptions, not significance or causal claims.", ""]
    contrasts = {contrast["contrast_id"]: contrast for contrast in data["contrasts"]}
    lines += _table(("Contrast", *HEADERS), [
        (label, *[_number(contrasts[contrast_id]["conditions"][condition]["state_macro_agreement"]["estimate"], signed=True)
                   for condition in CONDITIONS])
        for contrast_id, label in CORE_CONTRASTS.items()
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="complete aggregate/results.json")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print(render_report(data), end="")


if __name__ == "__main__":
    main()
