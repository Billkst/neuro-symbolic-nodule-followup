#!/usr/bin/env python3
"""Build Module 3 Phase4 case subsets from M3-1 CDSG outputs."""

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

from src.rules.lung_rads_engine import generate_recommendation as generate_flat_recommendation


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv_by_case(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        return {row["case_id"]: row for row in csv.DictReader(fp) if row.get("case_id")}


def _load_metric_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        return {row["metric"]: row["value"] for row in csv.DictReader(fp) if row.get("metric")}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subset_name", "count", "fraction_of_total", "selection_rule"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _subset_record(
    *,
    subset_name: str,
    case_bundle: dict[str, Any],
    cdsg_recommendation: dict[str, Any],
    flat_recommendation: dict[str, Any],
    comparison_row: dict[str, str] | None,
    comparison_metrics: dict[str, str],
) -> dict[str, Any]:
    return {
        "case_id": case_bundle.get("case_id"),
        "subset_name": subset_name,
        "case_bundle": case_bundle,
        "cdsg_recommendation": cdsg_recommendation,
        "abstention_reason": cdsg_recommendation.get("abstention_reason"),
        "flat_lung_rads_output": flat_recommendation,
        "comparison_metadata": {
            "mismatch_row": comparison_row,
            "m3_1_total_cases": comparison_metrics.get("total_cases"),
            "m3_1_strict_hard_comparable_count": comparison_metrics.get(
                "strict_hard_comparable_count"
            ),
        },
    }


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "0.000000"
    return f"{count / total:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M3-2 case subsets from Phase4 bundles.")
    parser.add_argument("--case-bundles", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument(
        "--cdsg-recommendations",
        default="outputs/phaseA3/recommendations/cdsg_phase4_recommendations.jsonl",
    )
    parser.add_argument(
        "--comparison",
        default="outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_comparison.csv",
    )
    parser.add_argument(
        "--mismatches",
        default="outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_mismatches.csv",
    )
    parser.add_argument("--output-dir", default="outputs/phaseA3/datasets")
    parser.add_argument(
        "--summary",
        default="outputs/phaseA3/tables/module3_case_subset_summary.csv",
    )
    args = parser.parse_args()

    bundles = _load_jsonl(Path(args.case_bundles))
    recommendations = _load_jsonl(Path(args.cdsg_recommendations))
    rec_by_case = {str(row.get("case_id")): row for row in recommendations}
    mismatch_by_case = _load_csv_by_case(Path(args.mismatches))
    comparison_metrics = _load_metric_csv(Path(args.comparison))

    subset_rows: dict[str, list[dict[str, Any]]] = {
        "strict_hard_comparable": [],
        "missing_density_abstention": [],
        "missing_size_abstention": [],
        "no_structured_nodule_abstention": [],
    }

    reason_counter: Counter[str] = Counter()
    total = len(bundles)
    for bundle in bundles:
        case_id = str(bundle.get("case_id"))
        cdsg = rec_by_case.get(case_id)
        if cdsg is None:
            raise KeyError(f"missing CDSG recommendation for case_id={case_id}")
        flat = generate_flat_recommendation(bundle)
        flat_missing = set(flat.get("missing_information") or [])
        comparison_row = mismatch_by_case.get(case_id)
        abstention_reason = cdsg.get("abstention_reason")
        reason_counter[str(abstention_reason)] += 1

        if not abstention_reason and "density_category" not in flat_missing:
            subset_rows["strict_hard_comparable"].append(
                _subset_record(
                    subset_name="strict_hard_comparable",
                    case_bundle=bundle,
                    cdsg_recommendation=cdsg,
                    flat_recommendation=flat,
                    comparison_row=comparison_row,
                    comparison_metrics=comparison_metrics,
                )
            )
        elif abstention_reason == "missing_nodule_density":
            subset_rows["missing_density_abstention"].append(
                _subset_record(
                    subset_name="missing_density_abstention",
                    case_bundle=bundle,
                    cdsg_recommendation=cdsg,
                    flat_recommendation=flat,
                    comparison_row=comparison_row,
                    comparison_metrics=comparison_metrics,
                )
            )
        elif abstention_reason == "missing_nodule_size":
            subset_rows["missing_size_abstention"].append(
                _subset_record(
                    subset_name="missing_size_abstention",
                    case_bundle=bundle,
                    cdsg_recommendation=cdsg,
                    flat_recommendation=flat,
                    comparison_row=comparison_row,
                    comparison_metrics=comparison_metrics,
                )
            )
        elif abstention_reason == "no_structured_nodule":
            subset_rows["no_structured_nodule_abstention"].append(
                _subset_record(
                    subset_name="no_structured_nodule_abstention",
                    case_bundle=bundle,
                    cdsg_recommendation=cdsg,
                    flat_recommendation=flat,
                    comparison_row=comparison_row,
                    comparison_metrics=comparison_metrics,
                )
            )

    output_dir = Path(args.output_dir)
    for subset_name, rows in subset_rows.items():
        _write_jsonl(output_dir / f"{subset_name}.jsonl", rows)

    summary_rows = [
        {
            "subset_name": "strict_hard_comparable",
            "count": len(subset_rows["strict_hard_comparable"]),
            "fraction_of_total": _rate(len(subset_rows["strict_hard_comparable"]), total),
            "selection_rule": "cdsg_non_abstention_and_flat_missing_information_excludes_density_category",
        },
        {
            "subset_name": "missing_density_abstention",
            "count": len(subset_rows["missing_density_abstention"]),
            "fraction_of_total": _rate(len(subset_rows["missing_density_abstention"]), total),
            "selection_rule": "cdsg_abstention_reason == missing_nodule_density",
        },
        {
            "subset_name": "missing_size_abstention",
            "count": len(subset_rows["missing_size_abstention"]),
            "fraction_of_total": _rate(len(subset_rows["missing_size_abstention"]), total),
            "selection_rule": "cdsg_abstention_reason == missing_nodule_size",
        },
        {
            "subset_name": "no_structured_nodule_abstention",
            "count": len(subset_rows["no_structured_nodule_abstention"]),
            "fraction_of_total": _rate(
                len(subset_rows["no_structured_nodule_abstention"]), total
            ),
            "selection_rule": "cdsg_abstention_reason == no_structured_nodule",
        },
    ]
    _write_summary(Path(args.summary), summary_rows)

    print(
        json.dumps(
            {
                "total_cases": total,
                "subsets": {name: len(rows) for name, rows in subset_rows.items()},
                "abstention_reason_counts": dict(reason_counter),
                "summary": args.summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
