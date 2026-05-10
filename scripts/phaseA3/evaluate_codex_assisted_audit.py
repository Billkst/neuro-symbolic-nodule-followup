#!/usr/bin/env python3
"""Evaluate Codex-assisted evidence audit results for the M3 pilot set.

The output is a failure-mode audit summary. It is not a clinical gold benchmark
and does not compute learned-model performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CODEX_COLUMNS = [
    "codex_gold_actionable_suggestion",
    "codex_lung_rads_category_suggestion",
    "codex_recommendation_level_suggestion",
    "codex_cdsg_recommendation_correct_suggestion",
    "codex_abstention_appropriate_suggestion",
    "codex_under_followup_risk_suggestion",
    "codex_over_followup_risk_suggestion",
    "codex_confidence",
    "codex_rationale",
    "codex_cited_evidence",
]

REVIEW_COLUMNS = [
    "evidence_has_nodule",
    "evidence_has_size",
    "evidence_has_density",
    "evidence_has_location",
    "size_text_present",
    "density_text_present",
    "codex_cited_evidence_exists",
    "codex_rationale_supported_by_evidence",
    "obvious_codex_error",
    "needs_clinical_expert_review",
    "non_clinical_reviewer_notes",
]

UNCERTAINTY_COLUMNS = [
    "codex_gold_actionable_suggestion",
    "codex_lung_rads_category_suggestion",
    "codex_recommendation_level_suggestion",
    "codex_cdsg_recommendation_correct_suggestion",
    "codex_abstention_appropriate_suggestion",
    "codex_under_followup_risk_suggestion",
    "codex_over_followup_risk_suggestion",
]


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


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _nonempty(value: Any) -> bool:
    return value not in {None, "", "None"}


def _yes(value: Any) -> bool:
    return _clean(value).lower() in {"yes", "y", "true", "1"}


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def _groups(row: dict[str, str]) -> list[str]:
    return [item for item in row.get("audit_group", "").split("|") if item] or ["ungrouped"]


def _is_codex_annotated(row: dict[str, str]) -> bool:
    return any(_nonempty(row.get(field)) for field in CODEX_COLUMNS)


def _is_reviewed(row: dict[str, str]) -> bool:
    return any(_nonempty(row.get(field)) for field in REVIEW_COLUMNS)


def _uncertain_count(row: dict[str, str]) -> int:
    return sum(1 for field in UNCERTAINTY_COLUMNS if _clean(row.get(field)).lower() == "uncertain")


def _missing_columns(fieldnames: list[str]) -> list[str]:
    required = ["case_id", "audit_group"] + CODEX_COLUMNS + REVIEW_COLUMNS
    return [field for field in required if field not in fieldnames]


def _summary_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    total = len(rows)
    codex_annotated = [row for row in rows if _is_codex_annotated(row)]
    reviewed = [row for row in rows if _is_reviewed(row)]
    missing_cols = _missing_columns(fieldnames)

    output = [
        {
            "metric": "input_rows",
            "label": "all",
            "count": total,
            "denominator": total,
            "rate": _rate(total, total),
            "note": "pilot rows loaded",
        },
        {
            "metric": "codex_annotated_rows",
            "label": "any_codex_column_nonempty",
            "count": len(codex_annotated),
            "denominator": total,
            "rate": _rate(len(codex_annotated), total),
            "note": "preannotation completion",
        },
        {
            "metric": "non_clinical_reviewed_rows",
            "label": "any_review_column_nonempty",
            "count": len(reviewed),
            "denominator": total,
            "rate": _rate(len(reviewed), total),
            "note": "evidence verification completion",
        },
        {
            "metric": "missing_required_columns",
            "label": "|".join(missing_cols),
            "count": len(missing_cols),
            "denominator": len(["case_id", "audit_group"] + CODEX_COLUMNS + REVIEW_COLUMNS),
            "rate": _rate(len(missing_cols), len(["case_id", "audit_group"] + CODEX_COLUMNS + REVIEW_COLUMNS)),
            "note": "should be zero for a valid filled template",
        },
    ]

    confidence_counter = Counter(_clean(row.get("codex_confidence")) or "empty" for row in rows)
    for label, count in sorted(confidence_counter.items()):
        output.append(
            {
                "metric": "codex_confidence_distribution",
                "label": label,
                "count": count,
                "denominator": total,
                "rate": _rate(count, total),
                "note": "Codex confidence distribution",
            }
        )

    for field in [
        "needs_clinical_expert_review",
        "obvious_codex_error",
        "codex_cited_evidence_exists",
        "codex_rationale_supported_by_evidence",
    ]:
        yes_count = sum(1 for row in rows if _yes(row.get(field)))
        denominator = sum(1 for row in rows if _nonempty(row.get(field)))
        output.append(
            {
                "metric": field,
                "label": "yes",
                "count": yes_count,
                "denominator": denominator,
                "rate": _rate(yes_count, denominator),
                "note": "non-clinical evidence review yes-rate among filled values",
            }
        )

    cited_yes = sum(1 for row in rows if _yes(row.get("codex_cited_evidence_exists")))
    rationale_yes = sum(1 for row in rows if _yes(row.get("codex_rationale_supported_by_evidence")))
    grounding_denominator = sum(
        1
        for row in rows
        if _nonempty(row.get("codex_cited_evidence_exists"))
        and _nonempty(row.get("codex_rationale_supported_by_evidence"))
    )
    both_yes = sum(
        1
        for row in rows
        if _yes(row.get("codex_cited_evidence_exists"))
        and _yes(row.get("codex_rationale_supported_by_evidence"))
    )
    output.append(
        {
            "metric": "evidence_grounding_support_rate",
            "label": "cited_evidence_and_rationale_supported",
            "count": both_yes,
            "denominator": grounding_denominator,
            "rate": _rate(both_yes, grounding_denominator),
            "note": f"cited_yes={cited_yes}; rationale_yes={rationale_yes}",
        }
    )

    uncertain_rows = sum(1 for row in rows if _uncertain_count(row) > 0)
    output.append(
        {
            "metric": "rows_with_any_codex_uncertainty",
            "label": "any_uncertain",
            "count": uncertain_rows,
            "denominator": total,
            "rate": _rate(uncertain_rows, total),
            "note": "any uncertainty among Codex suggestion fields",
        }
    )

    grounding_rate_value = both_yes / grounding_denominator if grounding_denominator else 0.0
    obvious_error_count = sum(1 for row in rows if _yes(row.get("obvious_codex_error")))
    expert_review_count = sum(1 for row in rows if _yes(row.get("needs_clinical_expert_review")))
    low_confidence_count = sum(1 for row in rows if _clean(row.get("codex_confidence")).lower() == "low")
    recommend_expand = (
        total > 0
        and len(codex_annotated) == total
        and len(reviewed) == total
        and grounding_rate_value >= 0.8
        and obvious_error_count <= 2
        and expert_review_count <= max(2, total // 4)
        and low_confidence_count <= max(2, total // 4)
    )
    output.append(
        {
            "metric": "recommend_expand_to_133_cases",
            "label": "yes" if recommend_expand else "no",
            "count": int(recommend_expand),
            "denominator": 1,
            "rate": "1.000000" if recommend_expand else "0.000000",
            "note": (
                "requires complete pilot, high grounding support, low obvious Codex error count, "
                "and manageable low-confidence/expert-review burden"
            ),
        }
    )
    return output


def _risk_flag_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if _yes(row.get("needs_clinical_expert_review")):
            reasons.append("needs_clinical_expert_review")
        if _yes(row.get("obvious_codex_error")):
            reasons.append("obvious_codex_error")
        if _clean(row.get("codex_confidence")).lower() == "low":
            reasons.append("low_codex_confidence")
        if _clean(row.get("codex_under_followup_risk_suggestion")).lower() in {"medium", "high"}:
            reasons.append("medium_high_under_followup_risk_suggestion")
        if _clean(row.get("codex_over_followup_risk_suggestion")).lower() in {"medium", "high"}:
            reasons.append("medium_high_over_followup_risk_suggestion")
        if _nonempty(row.get("codex_cited_evidence_exists")) and not _yes(row.get("codex_cited_evidence_exists")):
            reasons.append("codex_cited_evidence_not_found")
        if _nonempty(row.get("codex_rationale_supported_by_evidence")) and not _yes(row.get("codex_rationale_supported_by_evidence")):
            reasons.append("codex_rationale_not_supported")
        if _uncertain_count(row) >= 3:
            reasons.append("high_codex_uncertainty")
        if not reasons:
            continue
        flags.append(
            {
                "case_id": row.get("case_id", ""),
                "audit_group": row.get("audit_group", ""),
                "pilot_stratum": row.get("pilot_stratum", ""),
                "flag_reasons": "|".join(reasons),
                "codex_confidence": row.get("codex_confidence", ""),
                "codex_under_followup_risk_suggestion": row.get("codex_under_followup_risk_suggestion", ""),
                "codex_over_followup_risk_suggestion": row.get("codex_over_followup_risk_suggestion", ""),
                "needs_clinical_expert_review": row.get("needs_clinical_expert_review", ""),
                "obvious_codex_error": row.get("obvious_codex_error", ""),
                "codex_rationale": row.get("codex_rationale", ""),
                "codex_cited_evidence": row.get("codex_cited_evidence", ""),
                "non_clinical_reviewer_notes": row.get("non_clinical_reviewer_notes", ""),
            }
        )
    if not flags:
        flags.append(
            {
                "case_id": "",
                "audit_group": "",
                "pilot_stratum": "",
                "flag_reasons": "no_risk_flags_or_no_completed_review",
                "codex_confidence": "",
                "codex_under_followup_risk_suggestion": "",
                "codex_over_followup_risk_suggestion": "",
                "needs_clinical_expert_review": "",
                "obvious_codex_error": "",
                "codex_rationale": "",
                "codex_cited_evidence": "",
                "non_clinical_reviewer_notes": "",
            }
        )
    return flags


def _group_uncertainty_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for group in _groups(row):
            grouped[group].append(row)

    output: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        total = len(group_rows)
        uncertain_rows = sum(1 for row in group_rows if _uncertain_count(row) > 0)
        expert_review = sum(1 for row in group_rows if _yes(row.get("needs_clinical_expert_review")))
        obvious_error = sum(1 for row in group_rows if _yes(row.get("obvious_codex_error")))
        grounding_ready = sum(
            1
            for row in group_rows
            if _nonempty(row.get("codex_cited_evidence_exists"))
            and _nonempty(row.get("codex_rationale_supported_by_evidence"))
        )
        grounding_supported = sum(
            1
            for row in group_rows
            if _yes(row.get("codex_cited_evidence_exists"))
            and _yes(row.get("codex_rationale_supported_by_evidence"))
        )
        output.append(
            {
                "audit_group": group,
                "rows": total,
                "codex_annotated_rows": sum(1 for row in group_rows if _is_codex_annotated(row)),
                "reviewed_rows": sum(1 for row in group_rows if _is_reviewed(row)),
                "rows_with_any_codex_uncertainty": uncertain_rows,
                "uncertainty_rate": _rate(uncertain_rows, total),
                "needs_clinical_expert_review_count": expert_review,
                "needs_clinical_expert_review_rate": _rate(expert_review, total),
                "obvious_codex_error_count": obvious_error,
                "obvious_codex_error_rate": _rate(obvious_error, total),
                "evidence_grounding_support_count": grounding_supported,
                "evidence_grounding_support_denominator": grounding_ready,
                "evidence_grounding_support_rate": _rate(grounding_supported, grounding_ready),
            }
        )
    return output


def _write_report(
    path: Path,
    *,
    input_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    summary_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_cols = _missing_columns(fieldnames)
    summary_index = {(row["metric"], row["label"]): row for row in summary_rows}

    def metric(metric_name: str, label: str = "yes") -> str:
        row = summary_index.get((metric_name, label))
        if not row:
            return "0"
        return f"{row.get('count')} / {row.get('denominator')} ({row.get('rate')})"

    expand_row = next((row for row in summary_rows if row["metric"] == "recommend_expand_to_133_cases"), None)
    expand_label = expand_row.get("label") if expand_row else "no"
    codex_annotated = sum(1 for row in rows if _is_codex_annotated(row))
    reviewed = sum(1 for row in rows if _is_reviewed(row))
    dry_run = codex_annotated == 0 or reviewed == 0
    confidence_counter = Counter(_clean(row.get("codex_confidence")) or "empty" for row in rows)
    expert_review_cases = [row.get("case_id", "") for row in rows if _yes(row.get("needs_clinical_expert_review"))]
    obvious_error_count = sum(1 for row in rows if _yes(row.get("obvious_codex_error")))
    low_confidence_count = confidence_counter.get("low", 0)
    expansion_blocker = "无"
    if expand_label != "yes":
        blockers: list[str] = []
        if len(rows) and codex_annotated != len(rows):
            blockers.append("Codex pre-annotation 未完成")
        if len(rows) and reviewed != len(rows):
            blockers.append("非医学 evidence self-check 未完成")
        grounding_row = summary_index.get(("evidence_grounding_support_rate", "cited_evidence_and_rationale_supported"))
        if grounding_row and float(grounding_row.get("rate", 0.0)) < 0.8:
            blockers.append("evidence grounding support rate 低于 0.8")
        if obvious_error_count > 2:
            blockers.append("obvious Codex error 数量超过阈值")
        if expert_review_cases:
            blockers.append("仍有多例需要 clinical expert review")
        if low_confidence_count > max(2, len(rows) // 4 if rows else 0):
            blockers.append("low confidence case 占比偏高")
        expansion_blocker = "；".join(blockers) if blockers else "pilot 仍存在未量化风险"

    lines = [
        "# M3-3C Codex-Assisted Pilot 20 Audit Report",
        "",
        "## 定位",
        "",
        "本报告是 Codex-assisted pre-annotation / model-assisted evidence audit 的 pilot 20 统计结果。它不是 clinical gold benchmark、不是 expert label、不是 manual gold，也不用于 learned-model performance table。",
        "",
        "没有医学专家参与；非医学复核列仅表示 Codex evidence-grounding self-check。结果只能用于 failure-mode audit。",
        "",
        "## 输入状态",
        "",
        f"- 输入文件：`{input_path}`",
        f"- 读取行数：{len(rows)}",
        f"- Codex 预标注完成行数：{codex_annotated}",
        f"- 非医学证据复核完成行数：{reviewed}",
        f"- 缺失必要列：{', '.join(missing_cols) if missing_cols else '无'}",
    ]
    if dry_run:
        lines.extend(
            [
                "",
                "## Dry-run 结论",
                "",
                "当前输入尚未包含完整 Codex 预标注或非医学证据复核。脚本已正常生成空统计框架；填完 `module3_codex_assisted_pilot_20_filled.csv` 后重新运行即可得到正式 pilot 评估。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 主要统计",
                "",
                f"- needs_clinical_expert_review：{metric('needs_clinical_expert_review')}",
                f"- obvious_codex_error：{metric('obvious_codex_error')}",
                f"- evidence grounding support：{metric('evidence_grounding_support_rate', 'cited_evidence_and_rationale_supported')}",
                f"- rows with Codex uncertainty：{metric('rows_with_any_codex_uncertainty', 'any_uncertain')}",
                f"- 是否建议扩展到 133 cases：{expand_label}",
                f"- 不建议扩展原因：{expansion_blocker if expand_label != 'yes' else '不适用'}",
                f"- 当前是否可进入 learned-model performance experiment：no",
            ]
        )
        lines.extend(["", "## Codex Confidence 分布", ""])
        for label in ["high", "medium", "low", "empty"]:
            if label in confidence_counter:
                lines.append(f"- {label}: {confidence_counter[label]}")
        lines.extend(["", "## Clinical Expert Review Queue", ""])
        if expert_review_cases:
            lines.extend(f"- {case_id}" for case_id in expert_review_cases)
        else:
            lines.append("- 无")
        lines.extend(["", "## Obvious Codex Error", "", f"- obvious Codex error 数量：{obvious_error_count}"])
    lines.extend(
        [
            "",
            "## Group-wise uncertainty",
            "",
        ]
    )
    for row in group_rows:
        lines.append(
            f"- {row['audit_group']}: uncertainty {row['rows_with_any_codex_uncertainty']}/{row['rows']} "
            f"({row['uncertainty_rate']}), expert review {row['needs_clinical_expert_review_count']}/{row['rows']}"
        )
    lines.extend(
        [
            "",
            "## 风险标记",
            "",
            f"- 风险标记行数：{0 if risk_rows and risk_rows[0].get('flag_reasons') == 'no_risk_flags_or_no_completed_review' else len(risk_rows)}",
            "",
            "## 使用限制",
            "",
            "Codex suggestion 和非医学复核结果只用于 failure-mode audit。若要形成 clinical gold benchmark，需要医学专家复核并重新定义 gold 标注协议。当前仍不能进入 learned-model performance experiment。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_codex_assisted_pilot20_summary.csv"),
    )
    parser.add_argument(
        "--risk-flags-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_codex_assisted_pilot20_risk_flags.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/module3_codex_assisted_pilot20_report.md"),
    )
    args = parser.parse_args()

    rows, fieldnames = _load_csv(args.input)
    if not rows and not args.input.exists():
        rows, fieldnames = [], []

    summary_rows = _summary_rows(rows, fieldnames)
    risk_rows = _risk_flag_rows(rows)
    group_rows = _group_uncertainty_rows(rows)

    _write_csv(
        args.summary_output,
        summary_rows + [
            {
                "metric": f"group_uncertainty.{row['audit_group']}",
                "label": "any_uncertain",
                "count": row["rows_with_any_codex_uncertainty"],
                "denominator": row["rows"],
                "rate": row["uncertainty_rate"],
                "note": "audit group-wise uncertainty",
            }
            for row in group_rows
        ],
        ["metric", "label", "count", "denominator", "rate", "note"],
    )
    _write_csv(
        args.risk_flags_output,
        risk_rows,
        [
            "case_id",
            "audit_group",
            "pilot_stratum",
            "flag_reasons",
            "codex_confidence",
            "codex_under_followup_risk_suggestion",
            "codex_over_followup_risk_suggestion",
            "needs_clinical_expert_review",
            "obvious_codex_error",
            "codex_rationale",
            "codex_cited_evidence",
            "non_clinical_reviewer_notes",
        ],
    )
    _write_report(
        args.report_output,
        input_path=args.input,
        rows=rows,
        fieldnames=fieldnames,
        summary_rows=summary_rows,
        risk_rows=risk_rows,
        group_rows=group_rows,
    )

    print(
        json.dumps(
            {
                "input": str(args.input),
                "rows": len(rows),
                "codex_annotated_rows": sum(1 for row in rows if _is_codex_annotated(row)),
                "reviewed_rows": sum(1 for row in rows if _is_reviewed(row)),
                "missing_required_columns": _missing_columns(fieldnames),
                "summary_output": str(args.summary_output),
                "risk_flags_output": str(args.risk_flags_output),
                "report_output": str(args.report_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
