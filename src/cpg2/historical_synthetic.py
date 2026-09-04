from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_json, write_json


SCHEMA_VERSION = "1.0"
END_LABEL = "End of Decision Tree"
NORMALIZED_END_LABELS = frozenset({"End", "No specific action required"})

ODS_NAMESPACES = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}


def normalize_terminal_label(label: str) -> str:
    value = label.strip()
    return END_LABEL if value in NORMALIZED_END_LABELS else value


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    total = sum(weights)
    return _safe_divide(sum(value * weight for value, weight in zip(values, weights)), total)


def _weighted_std(values: Sequence[float], weights: Sequence[int]) -> float:
    total = sum(weights)
    if not values:
        return 0.0
    if not total:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    mean = _weighted_mean(values, weights)
    return math.sqrt(
        sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights))
        / total
    )


def compute_classification_metrics(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    rows = [
        (normalize_terminal_label(expected), normalize_terminal_label(predicted))
        for expected, predicted in pairs
    ]
    if not rows:
        raise ValueError("at least one action/prediction pair is required")

    labels = sorted({value for row in rows for value in row})
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []
    for label in labels:
        true_positive = sum(expected == label and predicted == label for expected, predicted in rows)
        false_positive = sum(expected != label and predicted == label for expected, predicted in rows)
        false_negative = sum(expected == label and predicted != label for expected, predicted in rows)
        support = sum(expected == label for expected, _predicted in rows)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(_f1(precision, recall))
        supports.append(support)

    correct = sum(expected == predicted for expected, predicted in rows)
    weighted_precision = _weighted_mean(precisions, supports)
    weighted_recall = _weighted_mean(recalls, supports)
    weighted_f1 = _weighted_mean(f1_scores, supports)

    binary_rows = [
        (expected != END_LABEL, predicted != END_LABEL)
        for expected, predicted in rows
    ]
    binary_true_positive = sum(expected and predicted for expected, predicted in binary_rows)
    binary_true_negative = sum(not expected and not predicted for expected, predicted in binary_rows)
    binary_false_positive = sum(not expected and predicted for expected, predicted in binary_rows)
    binary_false_negative = sum(expected and not predicted for expected, predicted in binary_rows)
    binary_precision = _safe_divide(
        binary_true_positive, binary_true_positive + binary_false_positive
    )
    binary_recall = _safe_divide(
        binary_true_positive, binary_true_positive + binary_false_negative
    )
    binary_correct = sum(expected == predicted for expected, predicted in binary_rows)

    return {
        "n": len(rows),
        "normalization": {
            "end_labels_mapped_to_end_of_decision_tree": sorted(NORMALIZED_END_LABELS),
            "binary_positive_definition": "any terminal label other than End of Decision Tree",
        },
        "multiclass": {
            "accuracy": correct / len(rows),
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "weighted_class_precision_std": _weighted_std(precisions, supports),
            "weighted_class_recall_std": _weighted_std(recalls, supports),
            "weighted_class_f1_std": _weighted_std(f1_scores, supports),
            "true_label_count": len({expected for expected, _predicted in rows}),
            "predicted_label_count": len({predicted for _expected, predicted in rows}),
            "union_label_count": len(labels),
        },
        "binary": {
            "accuracy": binary_correct / len(rows),
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": _f1(binary_precision, binary_recall),
            "true_positive": binary_true_positive,
            "true_negative": binary_true_negative,
            "false_positive": binary_false_positive,
            "false_negative": binary_false_negative,
            "positive_reference_n": sum(expected for expected, _predicted in binary_rows),
            "negative_reference_n": sum(not expected for expected, _predicted in binary_rows),
        },
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_signature(path: str | Path, columns: Sequence[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    row_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: alignment columns missing: {missing}")
        for row in reader:
            encoded = json.dumps(
                [row[column] for column in columns],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(encoded)
            digest.update(b"\n")
            row_count += 1
    return {
        "row_count": row_count,
        "selected_column_sha256": digest.hexdigest(),
        "selected_columns": list(columns),
    }


def _prediction_pairs(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ("action", "diagnosis") if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: prediction columns missing: {missing}")
        for index, row in enumerate(reader, start=2):
            expected = row.get("action")
            predicted = row.get("diagnosis")
            if expected is None or predicted is None or not expected.strip() or not predicted.strip():
                raise ValueError(f"{path}:{index}: action and diagnosis must be nonempty")
            rows.append((expected, predicted))
    return rows


def _cell_text(cell: ET.Element) -> str:
    paragraphs = cell.findall("text:p", ODS_NAMESPACES)
    text_value = " ".join("".join(item.itertext()).strip() for item in paragraphs).strip()
    if text_value:
        return text_value
    office_value = f"{{{ODS_NAMESPACES['office']}}}value"
    return cell.get(office_value, "")


def read_ods_tables(path: str | Path) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    table_name_key = f"{{{ODS_NAMESPACES['table']}}}name"
    row_repeat_key = f"{{{ODS_NAMESPACES['table']}}}number-rows-repeated"
    column_repeat_key = f"{{{ODS_NAMESPACES['table']}}}number-columns-repeated"
    cell_tags = {
        f"{{{ODS_NAMESPACES['table']}}}table-cell",
        f"{{{ODS_NAMESPACES['table']}}}covered-table-cell",
    }
    output: dict[str, list[list[str]]] = {}
    for table in root.findall(".//table:table", ODS_NAMESPACES):
        name = table.get(table_name_key)
        if not name:
            continue
        rows: list[list[str]] = []
        for row in table.findall("table:table-row", ODS_NAMESPACES):
            values: list[str] = []
            for cell in list(row):
                if cell.tag not in cell_tags:
                    continue
                repeat = int(cell.get(column_repeat_key, "1"))
                value = _cell_text(cell)
                if value or repeat <= 100:
                    values.extend([value] * min(repeat, 100))
            while values and not values[-1]:
                values.pop()
            if not values:
                continue
            row_repeat = min(int(row.get(row_repeat_key, "1")), 100)
            rows.extend([list(values) for _ in range(row_repeat)])
        output[name] = rows
    return output


def _ods_metric_records(tables: Mapping[str, list[list[str]]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for sheet_name, rows in tables.items():
        if sheet_name not in {"multiclass_classification", "binary_classification"}:
            continue
        for row in rows:
            if len(row) < 5 or row[0].startswith("#") or row[0] in {"", "precision"}:
                continue
            records.append(
                {
                    "sheet": sheet_name,
                    "domain": row[0],
                    "precision": row[1],
                    "recall": row[2],
                    "f1": row[3],
                    "experiment": row[4],
                }
            )
    return records


def _reported_number(value: str) -> float:
    return float(value.split("±", maxsplit=1)[0].strip())


def _compare_to_ods(
    *,
    domain: str,
    experiment: str,
    metrics: Mapping[str, Any],
    ods_records: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    specifications = (
        (
            "multiclass_classification",
            {
                "precision": metrics["multiclass"]["weighted_precision"],
                "recall": metrics["multiclass"]["weighted_recall"],
                "f1": metrics["multiclass"]["weighted_f1"],
            },
        ),
        (
            "binary_classification",
            {
                "precision": metrics["binary"]["precision"],
                "recall": metrics["binary"]["recall"],
                "f1": metrics["binary"]["f1"],
            },
        ),
    )
    for sheet, expected_metrics in specifications:
        matches = [
            row
            for row in ods_records
            if row["sheet"] == sheet
            and row["domain"] == domain
            and row["experiment"] == experiment
        ]
        if len(matches) != 1:
            raise ValueError(
                f"ODS requires one {sheet}/{domain}/{experiment} row, found {len(matches)}"
            )
        row = matches[0]
        for metric_name, recomputed in expected_metrics.items():
            reported = _reported_number(row[metric_name])
            difference = recomputed - reported
            comparison = {
                "sheet": sheet,
                "metric": metric_name,
                "reported": reported,
                "recomputed": recomputed,
                "difference": difference,
                "matches_within_half_of_last_reported_decimal": abs(difference) <= 0.0005,
            }
            comparisons.append(comparison)
            if not comparison["matches_within_half_of_last_reported_decimal"]:
                discrepancies.append(
                    {
                        "domain": domain,
                        "experiment": experiment,
                        **comparison,
                    }
                )
    return comparisons, discrepancies


def _node_map(tree: Mapping[str, Any], *, sectioned: bool) -> dict[str, dict[str, Any]]:
    if not sectioned:
        nodes = tree.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("merged tree requires a nodes list")
        return {node["name"]: node for node in nodes}
    output: dict[str, dict[str, Any]] = {}
    for section in tree.values():
        if not isinstance(section, Mapping) or not isinstance(section.get("nodes"), list):
            continue
        for node in section["nodes"]:
            output[node["name"]] = node
    return output


def _tree_audit(domain: Mapping[str, Any]) -> dict[str, Any]:
    historical_path = Path(domain["historical_tree"])
    source_path = Path(domain["cpg2_source_tree"])
    normalized_path = Path(domain["cpg2_normalized_tree"])
    historical = read_json(historical_path)
    source = read_json(source_path)
    normalized = read_json(normalized_path)
    normalized_nodes = normalized.get("nodes", [])
    output: dict[str, Any] = {
        "comparison_mode": domain["tree_comparison"],
        "historical_tree": {
            "path": str(historical_path),
            "sha256": _sha256(historical_path),
        },
        "cpg2_source_tree": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "cpg2_normalized_tree": {
            "path": str(normalized_path),
            "sha256": _sha256(normalized_path),
            "tree_id": normalized.get("tree_id"),
            "node_count": len(normalized_nodes),
            "criterion_node_count": sum(
                node.get("node_type") == "criterion" for node in normalized_nodes
            ),
            "terminal_node_count": sum(
                node.get("node_type") == "terminal" for node in normalized_nodes
            ),
        },
    }
    if domain["tree_comparison"] == "exact_bytes":
        output["source_compatibility"] = {
            "exact_file_hash_match": output["historical_tree"]["sha256"]
            == output["cpg2_source_tree"]["sha256"],
            "json_object_match": historical == source,
        }
        return output
    if domain["tree_comparison"] != "merged_vs_sectioned":
        raise ValueError(f"unsupported tree comparison: {domain['tree_comparison']}")

    historical_nodes = _node_map(historical, sectioned=False)
    source_nodes = _node_map(source, sectioned=True)
    common = sorted(set(historical_nodes) & set(source_nodes))
    changed_nodes: list[dict[str, Any]] = []
    for name in common:
        changed_fields = sorted(
            field
            for field in set(historical_nodes[name]) | set(source_nodes[name])
            if historical_nodes[name].get(field) != source_nodes[name].get(field)
        )
        if changed_fields:
            changed_nodes.append({"node_name": name, "changed_fields": changed_fields})
    output["source_compatibility"] = {
        "historical_node_count": len(historical_nodes),
        "cpg2_source_node_count": len(source_nodes),
        "node_name_set_match": set(historical_nodes) == set(source_nodes),
        "identical_node_object_count": sum(
            historical_nodes[name] == source_nodes[name] for name in common
        ),
        "changed_nodes": changed_nodes,
        "interpretation": (
            "The same 23 named nodes are present, but two no-branch routing edges differ; "
            "the current normalized real-EHR tree is the headache red-flag subtask rather "
            "than the full historical synthetic traversal."
        ),
    }
    return output


def run_historical_synthetic_audit(config_path: str | Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported historical synthetic config schema")
    workbook_path = Path(config["result_workbook"])
    ods_tables = read_ods_tables(workbook_path)
    ods_records = _ods_metric_records(ods_tables)
    discrepancies: list[dict[str, Any]] = []
    domain_outputs: list[dict[str, Any]] = []

    for domain in config["domains"]:
        alignment_columns = domain["alignment_columns"]
        vignette_signature = _csv_signature(domain["vignettes"], alignment_columns)
        predictions: list[dict[str, Any]] = []
        for prediction in domain["predictions"]:
            path = Path(prediction["path"])
            signature = _csv_signature(path, alignment_columns)
            metrics = compute_classification_metrics(_prediction_pairs(path))
            comparisons, prediction_discrepancies = _compare_to_ods(
                domain=domain["ods_domain"],
                experiment=prediction["ods_experiment"],
                metrics=metrics,
                ods_records=ods_records,
            )
            discrepancies.extend(prediction_discrepancies)
            predictions.append(
                {
                    "model": prediction["model"],
                    "path": str(path),
                    "sha256": _sha256(path),
                    "row_alignment": {
                        "matches_vignette_selected_columns": signature
                        == vignette_signature,
                        "prediction_signature": signature,
                    },
                    "metrics": metrics,
                    "ods_comparison": comparisons,
                }
            )
        domain_outputs.append(
            {
                "domain": domain["domain"],
                "vignettes": {
                    "path": domain["vignettes"],
                    "sha256": _sha256(domain["vignettes"]),
                    "alignment_signature": vignette_signature,
                },
                "tree_audit": _tree_audit(domain),
                "predictions": predictions,
            }
        )

    paper_rows = [row for row in ods_records if row["experiment"] == "paper"]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "historical_cpgprompt_synthetic_recomputation",
        "benchmark_id": config["benchmark_id"],
        "claim_boundary": (
            "Historical synthetic action classification is reconstructed from supplied "
            "files. It is not directly comparable to real-EHR clinical correctness or "
            "uncertainty-preserving action-set resolution."
        ),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "workbook": {"path": str(workbook_path), "sha256": _sha256(workbook_path)},
        "paper_reference_provenance": config["paper_reference_provenance"],
        "paper_reference": config["paper_reference"],
        "paper_reference_rows": paper_rows,
        "model_provenance": config["models"],
        "methodology": {
            "multiclass": (
                "support-weighted one-vs-rest precision, recall, and F1 for local "
                "recomputation"
            ),
            "local_recomputation_multiclass": (
                "support-weighted one-vs-rest precision, recall, and F1"
            ),
            "published_reference_multiclass": (
                "macro-averaged precision, recall, and F1 as stated in the paper"
            ),
            "binary": "all non-end terminal actions are positive",
            "missing_or_malformed_historical_model_response": (
                "Traversal scripts defaulted to no; this audit recomputes only saved "
                "terminal predictions and does not rerun inference."
            ),
        },
        "domains": domain_outputs,
        "ods_discrepancies": discrepancies,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute aggregate historical synthetic CPGPrompt metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_historical_synthetic_audit(args.config)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": args.output,
                "domains": len(result["domains"]),
                "ods_discrepancies": len(result["ods_discrepancies"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
