#!/usr/bin/env python3
"""Plot full-panel response-format contrasts for the v0.13 manuscript.

Panel A uses saved compact-minus-full changes in state-only and full-record
state-macro agreement under format enforcement. Panel B uses saved enforced-
minus-unenforced changes in note joint agreement under strict scoring or after
conservative fence normalization. All eight models appear in manuscript-table
order. Points and paired 95% intervals come directly from frozen aggregates.

Only aggregate JSON files are read, never raw responses or clinical text.
--check-only validates identities, denominators, all 32 displayed estimate/CI
triplets, and difference orientations without importing Matplotlib, rendering,
or writing files. No bootstrap resampling or new statistical analysis occurs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SYNTHETIC = (
    ROOT / "outputs/local_restricted/slm_benchmark_v2"
    / "contract_ablation_v1/aggregate/results.json"
)
DEFAULT_NOTES = (
    ROOT / "outputs/local_restricted/slm_benchmark_v2"
    / "e3_contract_comparison_v1/aggregate/results.json"
)
MODELS = (
    ("q15_pt_v1", "Qwen2.5 1.5B base"),
    ("q15_it_v1", "Qwen2.5 1.5B Instruct"),
    ("q3_pt_v1", "Qwen2.5 3B base"),
    ("q3_it_v1", "Qwen2.5 3B Instruct"),
    ("g4_it_v1", "Gemma 3 4B Instruct"),
    ("mg4_it_v1", "MedGemma 4B Instruct"),
    ("mg15_it_v1", "MedGemma 1.5 4B Instruct"),
    ("q35_ref_v1", "Qwen3.5 4B"),
)
MODEL_IDS = {model_id for model_id, _ in MODELS}
SYNTHETIC_CONDITIONS = {
    "full_unconstrained", "full_constrained",
    "compact_unconstrained", "compact_constrained",
}
REFERENCE_COUNTS = {"PRESENT_CURRENT": 255, "ABSENT_EXPLICIT": 132}
SYNTHETIC_METRICS = ("state_only_macro_agreement", "state_macro_agreement")
NOTE_VIEWS = ("strict", "fence_normalized")
Triplet = tuple[float, float, float]
Panel = dict[str, tuple[Triplet, ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, lower: float, upper: float, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and lower <= value <= upper,
        f"Invalid or out-of-bounds value: {label}: {value!r}",
    )
    return float(value)


def _same(value: float, expected: float, label: str) -> None:
    _require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12),
             f"Frozen-value or orientation mismatch: {label}: {value!r} != {expected!r}")


def _triplet(point: Any, low: Any, high: Any, expected_delta: float,
             label: str) -> Triplet:
    values = tuple(_finite(value, -1, 1, f"{label}/{name}")
                   for name, value in (("estimate", point), ("ci_low", low), ("ci_high", high)))
    estimate, ci_low, ci_high = values
    _require(ci_low <= estimate <= ci_high,
             f"Unordered interval or point outside stored interval: {label}")
    _same(estimate, expected_delta, label)
    return values


def load_synthetic(path: Path) -> Panel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload["schema_version"] == "1.0", "Synthetic schema version changed")
    _require(payload["analysis"] == "paired_compact_contract_factorial_followup",
             "Unexpected synthetic aggregate identity")
    data = payload["data"]
    _require((data["model_count"], data["requests_per_cell"], data["vignette_count"])
             == (8, 387, 249), "Unexpected synthetic benchmark dimensions")
    _require(data["reference_state_counts"] == REFERENCE_COUNTS,
             "Synthetic reference-state denominators changed")
    _require(set(data["conditions"]) == SYNTHETIC_CONDITIONS,
             "Synthetic conditions changed")
    _require(set(payload["models"]) == MODEL_IDS, "Synthetic model identities changed")
    bootstrap = payload["bootstrap"]
    _require((bootstrap["unit"], bootstrap["replicates_requested"], bootstrap["confidence"])
             == ("vignette_case_id", 10000, 0.95)
             and bootstrap["shared_draws_across_all_cells_and_models"] is True,
             "Unexpected synthetic paired-interval specification")

    panel: dict[str, list[Triplet]] = {metric: [] for metric in SYNTHETIC_METRICS}
    for model_id, _ in MODELS:
        model = payload["models"][model_id]
        conditions = model["conditions"]
        _require(set(conditions) == SYNTHETIC_CONDITIONS,
                 f"Incomplete synthetic panel: {model_id}")
        for condition, cell in conditions.items():
            _require(cell["n_requests"] == 387,
                     f"Synthetic request count changed: {model_id}/{condition}")
            _require(cell["reference_state_counts"] == REFERENCE_COUNTS,
                     f"Synthetic strata changed: {model_id}/{condition}")
        for condition in ("full_constrained", "compact_constrained"):
            cell = conditions[condition]
            _require(cell["schema_valid"] == {"count": 387, "denominator": 387, "rate": 1.0},
                     f"Constrained synthetic schema validity changed: {model_id}/{condition}")
            _require(cell["runtime_failure_count"] == 0
                     and cell["length_truncation"]["count"] == 0,
                     f"Unexpected constrained runtime failure or truncation: {model_id}/{condition}")
        for metric in SYNTHETIC_METRICS:
            full = _finite(conditions["full_constrained"][metric]["estimate"], 0, 1,
                           f"{model_id}/full/{metric}")
            compact = _finite(conditions["compact_constrained"][metric]["estimate"], 0, 1,
                              f"{model_id}/compact/{metric}")
            row = model["compact_minus_full"]["constrained"][metric]
            label = f"synthetic/{model_id}/{metric}"
            _require((row["confidence"], row["replicates_used"], row["replicates_excluded"])
                     == (0.95, 10000, 0), f"Unexpected stored interval metadata: {label}")
            panel[metric].append(_triplet(row["estimate"], row["ci_low"], row["ci_high"],
                                          compact - full, label))
    return {metric: tuple(values) for metric, values in panel.items()}


def load_notes(path: Path) -> Panel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload["schema_version"] == "1.0", "Note schema version changed")
    _require(payload["primary_endpoint"] == "strict_hierarchical_joint_exact_match",
             "Unexpected clinical-note endpoint")
    _require((payload["n_requests"], payload["n_models"]) == (2288, 8),
             "Unexpected clinical-note benchmark dimensions")
    clusters = {"notes": 150, "admissions": 130, "subjects": 130}
    _require(payload["clusters"] == clusters, "Clinical-note cluster counts changed")
    _require(set(payload["conditions"]) == {"unconstrained", "constrained"},
             "Clinical-note conditions changed")
    statistics = payload["statistics"]
    _require((statistics["cluster_unit"], statistics["bootstrap_replicates"],
              statistics["confidence_level"])
             == ("subject", 10000, 0.95)
             and statistics["paired_draws_shared_across_all_cells_and_scoring_views"] is True
             and statistics["decoding_difference_direction"] == "constrained minus unconstrained",
             "Unexpected note paired-interval specification or orientation")

    panel: dict[str, list[Triplet]] = {view: [] for view in NOTE_VIEWS}
    for view in NOTE_VIEWS:
        data = payload[view]
        _require(set(data["cells"]) == MODEL_IDS,
                 f"Clinical-note model identities changed: {view}")
        _require(set(data["within_model_decoding_effects"]) == MODEL_IDS,
                 f"Clinical-note contrast identities changed: {view}")
        for model_id, _ in MODELS:
            conditions = data["cells"][model_id]
            _require(set(conditions) == {"unconstrained", "constrained"},
                     f"Incomplete note conditions: {view}/{model_id}")
            rates = {}
            for condition, cell in conditions.items():
                label = f"{view}/{model_id}/{condition}"
                _require(cell["n"] == 2288 and cell["runtime_completed"] == 2288,
                         f"Note request count changed: {label}")
                _require(cell["clusters"] == clusters, f"Note clusters changed: {label}")
                joint = cell["hierarchical_joint"]
                correct = joint["correct"]
                _require(isinstance(correct, int) and not isinstance(correct, bool)
                         and 0 <= correct <= 2288, f"Invalid correct count: {label}")
                rates[condition] = _finite(joint["exact_match"], 0, 1, label)
                _same(rates[condition], correct / 2288, f"{label}/denominator")
            row = data["within_model_decoding_effects"][model_id]
            for condition in ("unconstrained", "constrained"):
                saved_rate = _finite(row[f"{condition}_agreement"], 0, 1,
                                     f"{view}/{model_id}/{condition}_agreement")
                _same(saved_rate, rates[condition], f"{view}/{model_id}/{condition}")
            limits = row["subject_cluster_ci95"]
            _require(isinstance(limits, list) and len(limits) == 2,
                     f"Invalid note interval shape: {view}/{model_id}")
            panel[view].append(_triplet(row["difference"], limits[0], limits[1],
                                        rates["constrained"] - rates["unconstrained"],
                                        f"notes/{view}/{model_id}"))
    return {view: tuple(values) for view, values in panel.items()}


def render(synthetic: Panel, notes: Panel, output: Path, png: Path | None) -> None:
    # Plotting imports and font-cache setup are deferred so --check-only writes nothing.
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cpg-ieee-matplotlib-cache")
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ticker import FormatStrFormatter

    findfont(FontProperties(family="Liberation Serif"), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.85), sharey=True)
    fig.subplots_adjust(left=0.265, right=0.985, bottom=0.18, top=0.72, wspace=0.22)
    specs = (
        (synthetic,
         (("state_only_macro_agreement", "State-only agreement", "o", "#1768AC"),
          ("state_macro_agreement", "Full-record agreement", "s", "#A85D12")),
         "(A) Synthetic vignettes (387 requests)\nFull vs. compact, with format enforcement",
         "Compact - full\nState-macro agreement difference",
         (-0.35, 0.50), [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]),
        (notes,
         (("strict", "Strict scoring", "o", "#444444"),
          ("fence_normalized", "Markdown markers removed", "s", "#7955A1")),
         "(B) MIMIC-III-Ext-Notes (2,288 candidates)\nEnforced vs. unenforced generation",
         "Enforced - unenforced\nJoint agreement difference",
         (-0.06, 0.70), [0.0, 0.2, 0.4, 0.6]),
    )
    for ax, (panel, series, title, xlabel, xlim, ticks) in zip(axes, specs, strict=True):
        for row in range(len(MODELS)):
            ax.axhline(row, color="#EEEEEE", linewidth=0.55, zorder=0)
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
        for offset, (key, label, marker, color) in zip((-0.14, 0.14), series, strict=True):
            triplets = panel[key]
            _require(len(triplets) == len(MODELS), f"Incomplete plotted series: {key}")
            _require(all(xlim[0] <= low <= point <= high <= xlim[1]
                         for point, low, high in triplets),
                     f"Stored interval exceeds display limits: {key}")
            points = [point for point, _, _ in triplets]
            errors = [[point - low for point, low, _ in triplets],
                      [high - point for point, _, high in triplets]]
            ax.errorbar(points, [row + offset for row in range(len(MODELS))],
                        xerr=errors, fmt=marker, color=color, ecolor=color,
                        markersize=3.7, linewidth=1.0, capsize=2.0,
                        label=label, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(len(MODELS) - 0.55, -0.55)
        ax.set_xlabel(xlabel, labelpad=6, linespacing=1.15)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", width=0.6, length=3)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  frameon=False, handlelength=1.6, borderaxespad=0,
                  labelspacing=0.35, handletextpad=0.6)
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, 0.965, title,
                 ha="center", va="top", fontsize=9, linespacing=1.2)
    axes[0].set_yticks(list(range(len(MODELS))), [label for _, label in MODELS])
    axes[1].tick_params(axis="y", labelleft=False)

    # Stop before export if any label falls outside the physical canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bounds = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if bounds.x0 < 0 or bounds.x1 > 1 or bounds.y0 < 0 or bounds.y1 > 1:
            plt.close(fig)
            raise ValueError(f"Text outside figure canvas: {artist.get_text()!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Creator": Path(__file__).name,
                                  "CreationDate": None, "ModDate": None})
    if png:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-analysis", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--notes-analysis", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper_artifacts/figures/agreement-diagnostics-v0.13-ieee.pdf")
    parser.add_argument("--png", type=Path, help="Optional 200-dpi preview")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate saved estimates and paired intervals; do not render or write files")
    args = parser.parse_args()
    synthetic = load_synthetic(args.synthetic_analysis)
    notes = load_notes(args.notes_analysis)
    print("Validated all 8 models and 32 saved estimate/95% CI triplets: "
          "387 synthetic requests and 2,288 note candidates per condition; "
          "contrast orientations match the saved condition estimates.")
    if args.check_only:
        return
    render(synthetic, notes, args.output, args.png)
    print(f"Rendered all-model agreement contrasts: {args.output}")


if __name__ == "__main__":
    main()
