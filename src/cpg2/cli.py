from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .slm_benchmark import audit_ext_notes_dataset, validate_benchmark_config
from .slm_e1 import (
    aggregate_e1_results,
    aggregate_e1_s2_results,
    audit_e1_tokens,
    prepare_e1_dataset,
    run_e1_model,
)
from .slm_e2 import (
    aggregate_e2_results,
    audit_e2_tokens,
    prepare_e2_dataset,
    run_e2_model,
)
from .slm_e3 import aggregate_e3_results, prepare_e3_dataset, run_e3_model


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpg2-paper",
        description="Reproduce the E1, E1-S2, E2, and E3 analyses reported in the paper.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("slm-benchmark-validate")
    validate.add_argument("--benchmark", required=True)
    validate.add_argument("--require-inference-ready", action="store_true")

    audit = commands.add_parser("slm-benchmark-audit-ext-notes")
    audit.add_argument("--dataset-root", required=True)
    audit.add_argument("--output", required=True)

    e1_prepare = commands.add_parser("slm-benchmark-e1-prepare")
    e1_prepare.add_argument("--config", required=True)
    e1_prepare.add_argument("--output-requests", required=True)
    e1_prepare.add_argument("--output-blueprint", required=True)
    e1_prepare.add_argument("--output-manifest", required=True)

    e1_tokens = commands.add_parser("slm-benchmark-e1-token-audit")
    e1_tokens.add_argument("--config", required=True)
    e1_tokens.add_argument("--dataset-manifest", required=True)
    e1_tokens.add_argument("--output", required=True)

    e1_run = commands.add_parser("slm-benchmark-e1-run")
    e1_run.add_argument("--config", required=True)
    e1_run.add_argument("--manifest-id", required=True)
    e1_run.add_argument("--dataset-manifest", required=True)
    e1_run.add_argument("--token-audit", required=True)
    e1_run.add_argument("--output-responses", required=True)
    e1_run.add_argument("--output-audit", required=True)

    e1_aggregate = commands.add_parser("slm-benchmark-e1-aggregate")
    e1_aggregate.add_argument("--config", required=True)
    e1_aggregate.add_argument("--dataset-manifest", required=True)
    e1_aggregate.add_argument("--results-root", required=True)
    e1_aggregate.add_argument("--output", required=True)

    s2_tokens = commands.add_parser("slm-benchmark-e1-s2-token-audit")
    s2_tokens.add_argument("--config", required=True)
    s2_tokens.add_argument("--sensitivity-config", required=True)
    s2_tokens.add_argument("--dataset-manifest", required=True)
    s2_tokens.add_argument("--output", required=True)

    s2_run = commands.add_parser("slm-benchmark-e1-s2-run")
    s2_run.add_argument("--config", required=True)
    s2_run.add_argument("--sensitivity-config", required=True)
    s2_run.add_argument("--manifest-id", required=True)
    s2_run.add_argument("--dataset-manifest", required=True)
    s2_run.add_argument("--token-audit", required=True)
    s2_run.add_argument("--output-responses", required=True)
    s2_run.add_argument("--output-audit", required=True)

    s2_aggregate = commands.add_parser("slm-benchmark-e1-s2-aggregate")
    s2_aggregate.add_argument("--sensitivity-config", required=True)
    s2_aggregate.add_argument("--results-root", required=True)
    s2_aggregate.add_argument("--output", required=True)

    e2_prepare = commands.add_parser("slm-benchmark-e2-prepare")
    e2_prepare.add_argument("--config", required=True)
    e2_prepare.add_argument("--output-cases", required=True)
    e2_prepare.add_argument("--output-blueprint", required=True)
    e2_prepare.add_argument("--output-manifest", required=True)

    e2_tokens = commands.add_parser("slm-benchmark-e2-token-audit")
    e2_tokens.add_argument("--config", required=True)
    e2_tokens.add_argument("--dataset-manifest", required=True)
    e2_tokens.add_argument("--output", required=True)

    e2_run = commands.add_parser("slm-benchmark-e2-run")
    e2_run.add_argument("--config", required=True)
    e2_run.add_argument("--manifest-id", required=True)
    e2_run.add_argument("--dataset-manifest", required=True)
    e2_run.add_argument("--token-audit", required=True)
    e2_run.add_argument("--output-decisions", required=True)
    e2_run.add_argument("--output-cases", required=True)
    e2_run.add_argument("--output-audit", required=True)

    e2_aggregate = commands.add_parser("slm-benchmark-e2-aggregate")
    e2_aggregate.add_argument("--config", required=True)
    e2_aggregate.add_argument("--dataset-manifest", required=True)
    e2_aggregate.add_argument("--results-root", required=True)
    e2_aggregate.add_argument("--output", required=True)

    e3_prepare = commands.add_parser("slm-benchmark-e3-prepare")
    e3_prepare.add_argument("--config", required=True)
    e3_prepare.add_argument("--output-requests", required=True)
    e3_prepare.add_argument("--output-manifest", required=True)

    e3_run = commands.add_parser("slm-benchmark-e3-run")
    e3_run.add_argument("--config", required=True)
    e3_run.add_argument("--manifest-id", required=True)
    e3_run.add_argument("--dataset-manifest", required=True)
    e3_run.add_argument("--output-responses", required=True)
    e3_run.add_argument("--output-audit", required=True)

    e3_aggregate = commands.add_parser("slm-benchmark-e3-aggregate")
    e3_aggregate.add_argument("--config", required=True)
    e3_aggregate.add_argument("--dataset-manifest", required=True)
    e3_aggregate.add_argument("--results-root", required=True)
    e3_aggregate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "slm-benchmark-validate":
            report = validate_benchmark_config(args.benchmark)
            _emit(report)
            if not report["structurally_valid"]:
                return 1
            if args.require_inference_ready and not report["inference_ready"]:
                return 1
            return 0
        if args.command == "slm-benchmark-audit-ext-notes":
            report = audit_ext_notes_dataset(args.dataset_root)
            from .io import write_json

            write_json(args.output, report)
            _emit(report)
            return 0
        if args.command == "slm-benchmark-e1-prepare":
            report = prepare_e1_dataset(
                args.config,
                output_requests=args.output_requests,
                output_blueprint=args.output_blueprint,
                output_manifest=args.output_manifest,
            )
        elif args.command == "slm-benchmark-e1-token-audit":
            report = audit_e1_tokens(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                output=args.output,
            )
        elif args.command == "slm-benchmark-e1-run":
            report = run_e1_model(
                args.config,
                manifest_id=args.manifest_id,
                dataset_manifest_path=args.dataset_manifest,
                token_audit_path=args.token_audit,
                output_responses=args.output_responses,
                output_audit=args.output_audit,
            )
        elif args.command == "slm-benchmark-e1-aggregate":
            report = aggregate_e1_results(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                results_root=args.results_root,
                output=args.output,
            )
        elif args.command == "slm-benchmark-e1-s2-token-audit":
            report = audit_e1_tokens(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                output=args.output,
                sensitivity_config_path=args.sensitivity_config,
            )
        elif args.command == "slm-benchmark-e1-s2-run":
            report = run_e1_model(
                args.config,
                manifest_id=args.manifest_id,
                dataset_manifest_path=args.dataset_manifest,
                token_audit_path=args.token_audit,
                output_responses=args.output_responses,
                output_audit=args.output_audit,
                sensitivity_config_path=args.sensitivity_config,
            )
        elif args.command == "slm-benchmark-e1-s2-aggregate":
            report = aggregate_e1_s2_results(
                args.sensitivity_config,
                results_root=args.results_root,
                output=args.output,
            )
        elif args.command == "slm-benchmark-e2-prepare":
            report = prepare_e2_dataset(
                args.config,
                output_cases=args.output_cases,
                output_blueprint=args.output_blueprint,
                output_manifest=args.output_manifest,
            )
        elif args.command == "slm-benchmark-e2-token-audit":
            report = audit_e2_tokens(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                output=args.output,
            )
        elif args.command == "slm-benchmark-e2-run":
            report = run_e2_model(
                args.config,
                manifest_id=args.manifest_id,
                dataset_manifest_path=args.dataset_manifest,
                token_audit_path=args.token_audit,
                output_decisions=args.output_decisions,
                output_cases=args.output_cases,
                output_audit=args.output_audit,
            )
        elif args.command == "slm-benchmark-e2-aggregate":
            report = aggregate_e2_results(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                results_root=args.results_root,
                output=args.output,
            )
        elif args.command == "slm-benchmark-e3-prepare":
            report = prepare_e3_dataset(
                args.config,
                output_requests=args.output_requests,
                output_manifest=args.output_manifest,
            )
        elif args.command == "slm-benchmark-e3-run":
            report = run_e3_model(
                args.config,
                manifest_id=args.manifest_id,
                dataset_manifest_path=args.dataset_manifest,
                output_responses=args.output_responses,
                output_audit=args.output_audit,
            )
        elif args.command == "slm-benchmark-e3-aggregate":
            report = aggregate_e3_results(
                args.config,
                dataset_manifest_path=args.dataset_manifest,
                results_root=args.results_root,
                output=args.output,
            )
        else:
            parser.error(f"unknown command {args.command}")
        _emit(report)
        return 0 if report.get("technical_completion", True) else 1
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
