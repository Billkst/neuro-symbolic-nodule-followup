#!/usr/bin/env python3
"""Evaluate filled M3 manual audit labels.

The script is safe to run before annotation is complete. Empty or missing label
columns are reported as dry-run diagnostics instead of raising an exception.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    "gold_actionable",
    "gold_lung_rads_category",
    "gold_recommendation_level",
    "cdsg_recommendation_correct",
    "abstention_appropriate",
    "density_fact_correct",
    "size_fact_correct",
    "dominant_nodule_correct",
    "conflict_requires_manual_resolution",
    "under_followup_risk",
    "over_followup_risk",
    "annotator_notes",
]

ALLOWED_VALUES = {
    "gold_actionable": {"yes", "no", "uncertain"},
    "gold_lung_rads_category": {"1", "2", "3", "4A", "4B", "4X", "not_applicable", "uncertain"},
    "gold_recommendation_level": {
        "routine_screening",
        "short_interval_followup",
        "tissue_sampling",
        "insufficient_data",
        "uncertain",
    },
    "cdsg_recommendation_correct": {"yes", "no", "partially", "uncertain"},
    "abstention_appropriate": {"yes", "no", "not_applicable", "uncertain"},
    "density_fact_correct": {"yes", "no", "missing", "uncertain"},
    "size_fact_correct": {"yes", "no", "missing", "uncertain"},
    "dominant_nodule_correct": {"yes", "no", "uncertain"},
    "conflict_requires_manual_resolution": {"yes", "no", "uncertain"},
    "under_followup_risk": {"none", "low", "medium", "high", "uncertain"},
    "over_followup_risk": {"none", "low", "medium", "high", "uncertain"},
}

RATE_METRICS = [
    "cdsg_recommendation_correct",
    "abstention_appropriate",
    "density_fact_correct",
    "size_fact_correct",
    "dominant_nodule_correct",
    "conflict_requires_manual_resolution",
]

RISK_COLUMNS = ["under_followup_risk", "over_followup_risk"]


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def _nonempty(value: Any) -> bool:
    return value not in {None, "", "None"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _groups(row: dict[str, str]) -> list[str]:
    groups = [item for item in row.get("audit_group", "").split("|") if item]
    return groups or ["ungrouped"]


def _is_annotated(row: dict[str, str]) -> bool:
    return any(_nonempty(row.get(field)) for field in LABEL_COLUMNS if field != "annotator_notes")


def _effective_denominator_values(metric: str) -> set[str]:
    if metric == "conflict_requires_manual_resolution":
        return {"yes", "no"}
    if metric == "abstention_appropriate":
        return {"yes", "no"}
    if metric in {"density_fact_correct", "size_fact_correct"}:
        return {"yes", "no", "missing"}
    if metric == "cdsg_recommendation_correct":
        return {"yes", "no", "partially"}
    return {"yes", "no"}


def _positive_values(metric: str) -> set[str]:
    if metric == "conflict_requires_manual_resolution":
        return {"yes"}
    return {"yes"}


def _metric_rate(rows: list[dict[str, str]], metric: str) -> tuple[int, int, str]:
    denominator_values = _effective_denominator_values(metric)
    positive_values = _positive_values(metric)
    denominator = 0
    numerator = 0
    for row in rows:
        value = _clean(row.get(metric))
        if value in denominator_values:
            denominator += 1
            if value in positive_values:
                numerator += 1
    return numerator, denominator, _rate(numerator, denominator)


def _completion_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    total = len(rows)
    missing_columns = [field for field in LABEL_COLUMNS if field not in fieldnames]
    annotated_rows = [row for row in rows if _is_annotated(row)]
    output = [
        {
            "metric": "input_rows",
            "value": total,
            "numerator": total,
            "denominator": total,
            "rate": _rate(total, total),
            "note": "rows loaded from annotation file",
        },
        {
            "metric": "annotated_rows",
            "value": len(annotated_rows),
            "numerator": len(annotated_rows),
            "denominator": total,
            "rate": _rate(len(annotated_rows), total),
            "note": "rows with at least one non-empty manual label",
        },
        {
            "metric": "missing_annotation_columns",
            "value": "|".join(missing_columns),
            "numerator": len(missing_columns),
            "denominator": len(LABEL_COLUMNS),
            "rate": _rate(len(missing_columns), len(LABEL_COLUMNS)),
            "note": "dry-run warning if non-empty",
        },
    ]
    for field in LABEL_COLUMNS:
        if field not in fieldnames:
            continue
        filled = sum(1 for row in rows if _nonempty(row.get(field)))
        output.append(
            {
                "metric": f"label_completion.{field}",
                "value": filled,
                "numerator": filled,
                "denominator": total,
                "rate": _rate(filled, total),
                "note": "field-level annotation completion",
            }
        )
    return output


def _score_summary_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    summary = _completion_rows(rows, fieldnames)
    for metric in RATE_METRICS:
        if metric not in fieldnames:
            summary.append(
                {
                    "metric": f"{metric}_rate",
                    "value": "",
                    "numerator": 0,
                    "denominator": 0,
                    "rate": "0.000000",
                    "note": "missing annotation column",
                }
            )
            continue
        numerator, denominator, rate = _metric_rate(rows, metric)
        note = "positive rate among annotated non-uncertain values"
        if metric == "conflict_requires_manual_resolution":
            note = "manual-resolution-needed rate among yes/no labels"
        summary.append(
            {
                "metric": f"{metric}_rate",
                "value": numerator,
                "numerator": numerator,
                "denominator": denominator,
                "rate": rate,
                "note": note,
            }
        )
    for risk_col in RISK_COLUMNS:
        if risk_col not in fieldnames:
            continue
        counter = Counter(_clean(row.get(risk_col)) for row in rows if _nonempty(row.get(risk_col)))
        for label in ["none", "low", "medium", "high", "uncertain"]:
            count = counter.get(label, 0)
            summary.append(
                {
                    "metric": f"distribution.{risk_col}.{label}",
                    "value": count,
                    "numerator": count,
                    "denominator": sum(counter.values()),
                    "rate": _rate(count, sum(counter.values())),
                    "note": "risk distribution among non-empty labels",
                }
            )
    return summary


def _invalid_value_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field, allowed in ALLOWED_VALUES.items():
        if field not in fieldnames:
            continue
        invalid = [row for row in rows if _nonempty(row.get(field)) and _clean(row.get(field)) not in allowed]
        if invalid:
            output.append(
                {
                    "error_type": "invalid_label_value",
                    "field": field,
                    "label_value": "|".join(sorted({_clean(row.get(field)) for row in invalid})),
                    "count": len(invalid),
                    "fraction_of_rows": _rate(len(invalid), len(rows)),
                    "case_ids": "|".join(row.get("case_id", "") for row in invalid[:12]),
                    "note": f"allowed_values={sorted(allowed)}",
                }
            )
    return output


def _error_breakdown_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    output = _invalid_value_rows(rows, fieldnames)
    total = max(len(rows), 1)
    error_specs = [
        ("cdsg_recommendation_incorrect", "cdsg_recommendation_correct", {"no", "partially"}, "CDSG recommendation disagrees with audit label"),
        ("abstention_inappropriate", "abstention_appropriate", {"no"}, "abstention judged inappropriate"),
        ("density_fact_problem", "density_fact_correct", {"no", "missing"}, "density fact wrong or missing"),
        ("size_fact_problem", "size_fact_correct", {"no", "missing"}, "size fact wrong or missing"),
        ("dominant_nodule_problem", "dominant_nodule_correct", {"no"}, "dominant nodule selection judged wrong"),
        (
            "conflict_needs_manual_resolution",
            "conflict_requires_manual_resolution",
            {"yes"},
            "candidate conflict needs human resolution",
        ),
        ("under_followup_medium_high", "under_followup_risk", {"medium", "high"}, "medium/high under-follow-up risk"),
        ("over_followup_medium_high", "over_followup_risk", {"medium", "high"}, "medium/high over-follow-up risk"),
    ]
    for error_type, field, values, note in error_specs:
        if field not in fieldnames:
            continue
        matched = [row for row in rows if _clean(row.get(field)) in values]
        counter = Counter(_clean(row.get(field)) for row in matched)
        for label_value, count in sorted(counter.items()):
            case_ids = [row.get("case_id", "") for row in matched if _clean(row.get(field)) == label_value]
            output.append(
                {
                    "error_type": error_type,
                    "field": field,
                    "label_value": label_value,
                    "count": count,
                    "fraction_of_rows": _rate(count, total),
                    "case_ids": "|".join(case_ids[:12]),
                    "note": note,
                }
            )
    if not output:
        output.append(
            {
                "error_type": "no_scored_errors",
                "field": "",
                "label_value": "",
                "count": 0,
                "fraction_of_rows": "0.000000",
                "case_ids": "",
                "note": "no manual labels or no error labels found",
            }
        )
    return output


def _group_score_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for group in _groups(row):
            grouped[group].append(row)

    output: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        annotated_rows = [row for row in group_rows if _is_annotated(row)]
        base: dict[str, Any] = {
            "audit_group": group,
            "total_rows": len(group_rows),
            "annotated_rows": len(annotated_rows),
            "annotation_completion_rate": _rate(len(annotated_rows), len(group_rows)),
        }
        for metric in RATE_METRICS:
            if metric in fieldnames:
                numerator, denominator, rate = _metric_rate(group_rows, metric)
            else:
                numerator, denominator, rate = 0, 0, "0.000000"
            base[f"{metric}_numerator"] = numerator
            base[f"{metric}_denominator"] = denominator
            base[f"{metric}_rate"] = rate
        for risk_col in RISK_COLUMNS:
            if risk_col not in fieldnames:
                base[f"{risk_col}_medium_high_count"] = 0
                base[f"{risk_col}_medium_high_rate"] = "0.000000"
                continue
            denominator = sum(1 for row in group_rows if _nonempty(row.get(risk_col)))
            count = sum(1 for row in group_rows if _clean(row.get(risk_col)) in {"medium", "high"})
            base[f"{risk_col}_medium_high_count"] = count
            base[f"{risk_col}_medium_high_rate"] = _rate(count, denominator)
        output.append(base)
    return output


def _write_report(
    path: Path,
    *,
    input_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    summary_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_columns = [field for field in LABEL_COLUMNS if field not in fieldnames]
    annotated_count = sum(1 for row in rows if _is_annotated(row))
    dry_run = annotated_count == 0 or bool(missing_columns)

    def metric_rate(metric: str) -> str:
        for row in summary_rows:
            if row.get("metric") == f"{metric}_rate":
                return str(row.get("rate"))
        return "0.000000"

    lines = [
        "# M3-3C Manual Audit Label Evaluation",
        "",
        "## 定位",
        "",
        "本报告评估人工填写后的 M3 manual audit labels。该评分仅用于诊断 conservative CDSG agent 的人工审查结果，不是 learned-model 主实验，也不报告 Accuracy/F1。",
        "",
        "## 输入状态",
        "",
        f"- 输入文件：`{input_path}`",
        f"- 读取行数：{len(rows)}",
        f"- 已标注行数：{annotated_count}",
        f"- 缺失标注列：{', '.join(missing_columns) if missing_columns else '无'}",
    ]
    if dry_run:
        lines.extend(
            [
                "",
                "## Dry-run 结论",
                "",
                "当前输入尚未包含可评分的完整人工标签，脚本已正常生成空评分框架。完成人工标注后，用同一脚本重新运行即可得到正式 score summary、error breakdown 和 group-wise scores。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 主要评分",
                "",
                f"- CDSG recommendation correct rate：{metric_rate('cdsg_recommendation_correct')}",
                f"- Abstention appropriate rate：{metric_rate('abstention_appropriate')}",
                f"- Density fact correct rate：{metric_rate('density_fact_correct')}",
                f"- Size fact correct rate：{metric_rate('size_fact_correct')}",
                f"- Dominant nodule correct rate：{metric_rate('dominant_nodule_correct')}",
                f"- Conflict requires manual resolution rate：{metric_rate('conflict_requires_manual_resolution')}",
                "",
                "## 解释",
                "",
                "上述 rate 的分母为已标注且非 uncertain / not_applicable 的样本；`conflict_requires_manual_resolution` 的 rate 表示需要人工解决冲突的比例。风险分布和 group-wise scores 已写入对应 CSV。",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `outputs/phaseA3/tables/module3_manual_audit_score_summary.csv`",
            "- `outputs/phaseA3/tables/module3_manual_audit_error_breakdown.csv`",
            "- `outputs/phaseA3/tables/module3_manual_audit_group_scores.csv`",
            "",
            "## 后续使用",
            "",
            "若 pilot 20 显示 abstention 大多合理且 conflict 风险可控，可进入 soft matching / safer aggregation 设计；若 density 或 size fact 错误率较高，应先修正 fact recovery 或扩大人工审查集。不建议在缺少人工审查结论前进入 learned-model 主实验。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_manual_audit_label_template_filled.csv"),
        help="Filled manual audit CSV. The unfilled template can be used for dry-run.",
    )
    parser.add_argument(
        "--score-summary-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_manual_audit_score_summary.csv"),
    )
    parser.add_argument(
        "--error-breakdown-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_manual_audit_error_breakdown.csv"),
    )
    parser.add_argument(
        "--group-scores-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_manual_audit_group_scores.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/module3_manual_audit_evaluation_report.md"),
    )
    args = parser.parse_args()

    rows, fieldnames = _load_csv(args.input)
    if not rows and not args.input.exists():
        fieldnames = LABEL_COLUMNS

    summary_rows = _score_summary_rows(rows, fieldnames)
    error_rows = _error_breakdown_rows(rows, fieldnames)
    group_rows = _group_score_rows(rows, fieldnames)

    _write_csv(
        args.score_summary_output,
        summary_rows,
        ["metric", "value", "numerator", "denominator", "rate", "note"],
    )
    _write_csv(
        args.error_breakdown_output,
        error_rows,
        ["error_type", "field", "label_value", "count", "fraction_of_rows", "case_ids", "note"],
    )
    group_fieldnames = ["audit_group", "total_rows", "annotated_rows", "annotation_completion_rate"]
    for metric in RATE_METRICS:
        group_fieldnames.extend([f"{metric}_numerator", f"{metric}_denominator", f"{metric}_rate"])
    for risk_col in RISK_COLUMNS:
        group_fieldnames.extend([f"{risk_col}_medium_high_count", f"{risk_col}_medium_high_rate"])
    _write_csv(args.group_scores_output, group_rows, group_fieldnames)
    _write_report(
        args.report_output,
        input_path=args.input,
        rows=rows,
        fieldnames=fieldnames,
        summary_rows=summary_rows,
        error_rows=error_rows,
        group_rows=group_rows,
    )

    annotated_count = sum(1 for row in rows if _is_annotated(row))
    print(
        json.dumps(
            {
                "input": str(args.input),
                "rows": len(rows),
                "annotated_rows": annotated_count,
                "missing_annotation_columns": [field for field in LABEL_COLUMNS if field not in fieldnames],
                "dry_run": annotated_count == 0,
                "score_summary_output": str(args.score_summary_output),
                "error_breakdown_output": str(args.error_breakdown_output),
                "group_scores_output": str(args.group_scores_output),
                "report_output": str(args.report_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
