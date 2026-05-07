#!/usr/bin/env python3
"""Generate CDSG strong-silver Module 3 labels from module3-ready bundles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.schema_validator import validate_instance
from src.rules.cdsg_executor import CDSGExecutor, load_cdsg_graph


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dimension", "label", "count", "fraction"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def _distribution_rows(counter: Counter[str], total: int, dimension: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, count in sorted(counter.items()):
        rows.append(
            {
                "dimension": dimension,
                "label": label,
                "count": count,
                "fraction": _rate(count, total),
            }
        )
    return rows


def _strong_silver_record(
    *,
    case_bundle: dict[str, Any],
    recommendation: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_bundle.get("case_id"),
        "subject_id": case_bundle.get("subject_id"),
        "patient_id": case_bundle.get("patient_id") or case_bundle.get("subject_id"),
        "case_bundle": case_bundle,
        "recommendation": recommendation.get("recommendation"),
        "recommendation_level": recommendation.get("recommendation_level"),
        "recommendation_action": recommendation.get("recommendation_action"),
        "followup_interval": recommendation.get("followup_interval"),
        "followup_modality": recommendation.get("followup_modality"),
        "lung_rads_category": recommendation.get("lung_rads_category"),
        "risk_category": recommendation.get("risk_category"),
        "reasoning_path": recommendation.get("reasoning_path") or [],
        "guideline_anchor": recommendation.get("guideline_anchor") or [],
        "missing_info": recommendation.get("missing_info") or [],
        "missing_information": recommendation.get("missing_information") or [],
        "evidence_quality": recommendation.get("evidence_quality"),
        "decision_path": recommendation.get("decision_path") or [],
        "visited_nodes": recommendation.get("visited_nodes") or [],
        "matched_edges": recommendation.get("matched_edges") or [],
        "failed_conditions": recommendation.get("failed_conditions") or [],
        "abstention_reason": recommendation.get("abstention_reason"),
        "label_source": "cdsg_strong_silver",
        "cdsg_graph_version": graph.get("version"),
        "cdsg_graph_id": graph.get("graph_id"),
        "cdsg_rules_version": graph.get("rules_version"),
        "recommendation_object": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Module 3 CDSG strong-silver labels.")
    parser.add_argument(
        "--input",
        default="outputs/phaseA3/datasets/module3_ready_case_bundles.jsonl",
    )
    parser.add_argument(
        "--graph",
        default="outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/phaseA3/datasets/module3_strong_silver.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="outputs/phaseA3/tables/module3_strong_silver_summary.csv",
    )
    parser.add_argument(
        "--label-distribution",
        default="outputs/phaseA3/tables/module3_strong_silver_label_distribution.csv",
    )
    args = parser.parse_args()

    graph = load_cdsg_graph(args.graph)
    executor = CDSGExecutor(graph)
    bundles = _load_jsonl(Path(args.input))

    records: list[dict[str, Any]] = []
    schema_valid = 0
    recommendation_level_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    abstention_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()

    for bundle in bundles:
        recommendation = executor.execute(bundle)
        schema_errors = validate_instance(recommendation, "module3_recommendation_schema.json")
        if not schema_errors:
            schema_valid += 1
        record = _strong_silver_record(
            case_bundle=bundle,
            recommendation=recommendation,
            graph=graph,
        )
        record["recommendation_schema_errors"] = schema_errors
        records.append(record)

        recommendation_level_counter[str(recommendation.get("recommendation_level"))] += 1
        category_counter[str(recommendation.get("lung_rads_category"))] += 1
        risk_counter[str(recommendation.get("risk_category"))] += 1
        abstention_counter[str(recommendation.get("abstention_reason"))] += 1
        missing_counter.update(recommendation.get("missing_information") or [])

    total = len(records)
    abstention_count = sum(1 for row in records if row.get("abstention_reason"))
    actionable_count = total - abstention_count
    anchor_count = sum(1 for row in records if row.get("guideline_anchor"))
    reasoning_count = sum(1 for row in records if row.get("reasoning_path"))

    summary_rows: list[dict[str, Any]] = [
        {"metric": "total_samples", "value": total},
        {"metric": "schema_valid_count", "value": schema_valid},
        {"metric": "schema_valid_rate", "value": _rate(schema_valid, total)},
        {"metric": "actionable_count", "value": actionable_count},
        {"metric": "actionable_rate", "value": _rate(actionable_count, total)},
        {"metric": "abstention_count", "value": abstention_count},
        {"metric": "abstention_rate", "value": _rate(abstention_count, total)},
        {"metric": "guideline_anchor_nonempty_count", "value": anchor_count},
        {"metric": "guideline_anchor_nonempty_rate", "value": _rate(anchor_count, total)},
        {"metric": "reasoning_path_nonempty_count", "value": reasoning_count},
        {"metric": "reasoning_path_nonempty_rate", "value": _rate(reasoning_count, total)},
        {"metric": "label_source", "value": "cdsg_strong_silver"},
        {"metric": "cdsg_graph_version", "value": graph.get("version")},
        {"metric": "cdsg_graph_id", "value": graph.get("graph_id")},
    ]
    for key, value in sorted(missing_counter.items()):
        summary_rows.append({"metric": f"missing_information.{key}", "value": value})

    distribution_rows: list[dict[str, Any]] = []
    distribution_rows.extend(_distribution_rows(recommendation_level_counter, total, "recommendation_level"))
    distribution_rows.extend(_distribution_rows(category_counter, total, "lung_rads_category"))
    distribution_rows.extend(_distribution_rows(risk_counter, total, "risk_category"))
    distribution_rows.extend(_distribution_rows(abstention_counter, total, "abstention_reason"))

    _write_jsonl(Path(args.output), records)
    _write_summary(Path(args.summary), summary_rows)
    _write_distribution(Path(args.label_distribution), distribution_rows)

    print(
        json.dumps(
            {
                "total_samples": total,
                "actionable_count": actionable_count,
                "abstention_count": abstention_count,
                "schema_valid_rate": _rate(schema_valid, total),
                "output": args.output,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
