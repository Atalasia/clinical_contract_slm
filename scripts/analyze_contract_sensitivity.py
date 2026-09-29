#!/usr/bin/env python3
"""Analyze existing frozen responses without model loading or inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cpg2.slm_contract_sensitivity import analyze_existing_outputs, render_report


ROOT = Path(__file__).resolve().parents[1]


def _write_new_or_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing diagnostic differs; choose a fresh output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/slm_benchmark/e1_contract_ablation.v1.json",
                        help="The exact config used for the saved run, including any local manifest binding")
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/local_restricted/slm_benchmark_v2/contract_ablation_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/local_restricted/slm_benchmark_v2/contract_sensitivity_v1")
    args = parser.parse_args()
    if args.output_root.resolve() == args.input_root.resolve() or args.input_root.resolve() in args.output_root.resolve().parents:
        parser.error("diagnostics must be written outside the frozen input directory")
    result = analyze_existing_outputs(args.input_root, repository_root=ROOT, config_path=args.config)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    report = render_report(result)
    _write_new_or_identical(args.output_root / "results.json", payload)
    _write_new_or_identical(args.output_root / "report.md", report)
    print(f"Verified {result['data']['model_count'] * 4} cells; diagnostics saved to {args.output_root}")


if __name__ == "__main__":
    main()
