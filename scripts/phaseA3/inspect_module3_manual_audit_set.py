#!/usr/bin/env python3
"""Inspect M3-3B manual audit set usability and build a pilot annotation file.

This script only reads existing audit-set CSV files and writes readability
diagnostics. It does not regenerate the audit set, train models, or call any
external inference service.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


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

PILOT_STRATA = [
    "actionable_recommendation",
    "missing_density",
    "missing_size",
    "no_structured_nodule",
    "high_risk_density_conflict",
]

FIELD_GROUPS = {
    "identity": ["case_id", "subject_id", "audit_group"],
    "decision": ["recommendation_level", "recommendation_action", "lung_rads_category", "abstention_reason"],
    "traceability": ["reasoning_path", "guideline_anchor"],
    "evidence": ["candidate_nodule_evidence", "candidate_count", "conflict_evidence", "conflict_fields"],
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader.fieldnames or [])


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
    return value not in {None, "", "None", "[]", "{}"}


def _groups(row: dict[str, str]) -> set[str]:
    return {item for item in row.get("audit_group", "").split("|") if item}


def _json_loads(value: str) -> Any:
    if not _nonempty(value):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _as_int(value: Any) -> int:
    if value in {None, ""}:
        return 0
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


def _case_examples(rows: list[dict[str, str]], limit: int = 8) -> str:
    return "|".join(str(row.get("case_id", "")) for row in rows[:limit] if row.get("case_id"))


def _check_row(
    *,
    check_id: str,
    check_name: str,
    status: str,
    affected_rows: int,
    total_rows: int,
    details: str,
    examples: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "status": status,
        "affected_rows": affected_rows,
        "total_rows": total_rows,
        "rate": _rate(affected_rows, total_rows),
        "details": details,
        "example_case_ids": _case_examples(examples or []),
    }


def _field_presence_checks(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    total = len(rows)
    for group_name, fields in FIELD_GROUPS.items():
        missing_columns = [field for field in fields if field not in fieldnames]
        checks.append(
            _check_row(
                check_id=f"field_group.{group_name}",
                check_name=f"{group_name} required columns",
                status="fail" if missing_columns else "pass",
                affected_rows=len(missing_columns),
                total_rows=len(fields),
                details=f"missing_columns={missing_columns}" if missing_columns else "all required columns present",
            )
        )
    for field in ["case_id", "subject_id", "audit_group", "candidate_nodule_evidence", "reasoning_path", "guideline_anchor"]:
        if field not in fieldnames:
            checks.append(
                _check_row(
                    check_id=f"field_missing.{field}",
                    check_name=f"{field} column missing",
                    status="fail",
                    affected_rows=total,
                    total_rows=total,
                    details="required column not found",
                )
            )
            continue
        empty_rows = [row for row in rows if not _nonempty(row.get(field))]
        required_all = field in {"case_id", "subject_id", "audit_group"}
        status = "fail" if required_all and empty_rows else "warn" if empty_rows else "pass"
        checks.append(
            _check_row(
                check_id=f"field_empty.{field}",
                check_name=f"{field} non-empty coverage",
                status=status,
                affected_rows=len(empty_rows),
                total_rows=total,
                details="empty values" if empty_rows else "non-empty for all rows",
                examples=empty_rows,
            )
        )
    return checks


def _decision_checks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    actionable = [row for row in rows if not _nonempty(row.get("abstention_reason"))]
    abstained = [row for row in rows if _nonempty(row.get("abstention_reason"))]
    actionable_missing_recommendation = [
        row for row in actionable if not _nonempty(row.get("recommendation_action")) or not _nonempty(row.get("lung_rads_category"))
    ]
    abstention_missing_reason = [row for row in abstained if not _nonempty(row.get("abstention_reason"))]
    return [
        _check_row(
            check_id="decision.actionable_recommendation_complete",
            check_name="actionable rows have recommendation and Lung-RADS category",
            status="fail" if actionable_missing_recommendation else "pass",
            affected_rows=len(actionable_missing_recommendation),
            total_rows=max(len(actionable), 1),
            details=f"actionable_rows={len(actionable)}",
            examples=actionable_missing_recommendation,
        ),
        _check_row(
            check_id="decision.abstention_reason_complete",
            check_name="abstained rows have abstention reason",
            status="fail" if abstention_missing_reason else "pass",
            affected_rows=len(abstention_missing_reason),
            total_rows=max(len(abstained), 1),
            details=f"abstained_rows={len(abstained)}; total_rows={total}",
            examples=abstention_missing_reason,
        ),
    ]


def _json_evidence_checks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    checks: list[dict[str, Any]] = []
    for field in ["candidate_nodule_evidence", "conflict_evidence", "reasoning_path", "guideline_anchor"]:
        malformed = [row for row in rows if _nonempty(row.get(field)) and _json_loads(row.get(field, "")) is None]
        status = "fail" if malformed and field in {"candidate_nodule_evidence", "reasoning_path", "guideline_anchor"} else "warn" if malformed else "pass"
        checks.append(
            _check_row(
                check_id=f"json_parse.{field}",
                check_name=f"{field} parseable compact JSON",
                status=status,
                affected_rows=len(malformed),
                total_rows=total,
                details="malformed JSON values" if malformed else "parseable when non-empty",
                examples=malformed,
            )
        )

    rows_with_mention_text: list[dict[str, str]] = []
    rows_with_readable_evidence: list[dict[str, str]] = []
    for row in rows:
        candidate_evidence = _json_loads(row.get("candidate_nodule_evidence", ""))
        conflict_evidence = _json_loads(row.get("conflict_evidence", ""))
        has_mention = False
        if isinstance(candidate_evidence, list):
            has_mention = any(_nonempty(item.get("mention_text")) for item in candidate_evidence if isinstance(item, dict))
        if has_mention:
            rows_with_mention_text.append(row)
        has_conflict_span = False
        if isinstance(conflict_evidence, list):
            has_conflict_span = any(_nonempty(item.get("example_evidence_spans")) for item in conflict_evidence if isinstance(item, dict))
        if has_mention or has_conflict_span or _nonempty(row.get("source_report_note_ids")):
            rows_with_readable_evidence.append(row)

    checks.append(
        _check_row(
            check_id="evidence.mention_text_coverage",
            check_name="rows with mention_text in candidate evidence",
            status="warn" if len(rows_with_mention_text) < total else "pass",
            affected_rows=len(rows_with_mention_text),
            total_rows=total,
            details="coverage count, not a failure for no-structured-nodule rows",
        )
    )
    checks.append(
        _check_row(
            check_id="evidence.readable_context_coverage",
            check_name="rows with mention text, conflict span, or source note id",
            status="fail" if not rows_with_readable_evidence else "pass" if len(rows_with_readable_evidence) == total else "warn",
            affected_rows=len(rows_with_readable_evidence),
            total_rows=total,
            details="coverage count for any human-readable context field",
        )
    )
    return checks


def _length_checks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    total = len(rows)
    for field, warn_threshold, fail_threshold in [
        ("candidate_nodule_evidence", 2400, 6000),
        ("conflict_evidence", 1600, 4000),
        ("reasoning_path", 1200, 3000),
        ("guideline_anchor", 1200, 3000),
    ]:
        lengths = [(row, len(row.get(field, ""))) for row in rows]
        warn_rows = [row for row, length in lengths if length > warn_threshold]
        fail_rows = [row for row, length in lengths if length > fail_threshold]
        max_len = max((length for _, length in lengths), default=0)
        status = "fail" if fail_rows else "warn" if warn_rows else "pass"
        checks.append(
            _check_row(
                check_id=f"length.{field}",
                check_name=f"{field} length readability",
                status=status,
                affected_rows=len(warn_rows),
                total_rows=total,
                details=f"warn_threshold={warn_threshold}; fail_threshold={fail_threshold}; max_len={max_len}",
                examples=fail_rows or warn_rows,
            )
        )
    return checks


def _conflict_checks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    high_risk_rows = [
        row
        for row in rows
        if "high_risk_density_conflict" in _groups(row)
        or ("density_category" in row.get("conflict_fields", "") and "high" in row.get("conflict_risk_levels", ""))
    ]
    missing_summary = [
        row
        for row in high_risk_rows
        if _as_int(row.get("conflict_rows")) <= 0 or not _nonempty(row.get("conflict_evidence"))
    ]
    return [
        _check_row(
            check_id="conflict.high_risk_summary_complete",
            check_name="high-risk density conflict rows have conflict evidence",
            status="fail" if missing_summary else "pass",
            affected_rows=len(missing_summary),
            total_rows=max(len(high_risk_rows), 1),
            details=f"high_risk_density_conflict_rows={len(high_risk_rows)}",
            examples=missing_summary,
        )
    ]


def _label_template_checks(label_fieldnames: list[str], label_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    missing = [field for field in LABEL_COLUMNS if field not in label_fieldnames]
    empty_labels = 0
    if label_rows:
        empty_labels = sum(1 for row in label_rows if all(not _nonempty(row.get(field)) for field in LABEL_COLUMNS))
    return [
        _check_row(
            check_id="label_template.columns",
            check_name="manual label template columns",
            status="fail" if missing else "pass",
            affected_rows=len(missing),
            total_rows=len(LABEL_COLUMNS),
            details=f"missing_label_columns={missing}" if missing else "all label columns present",
        ),
        _check_row(
            check_id="label_template.empty_ready_for_annotation",
            check_name="label template rows currently empty",
            status="pass",
            affected_rows=empty_labels,
            total_rows=len(label_rows),
            details="empty labels are expected before manual annotation",
        ),
    ]


def build_readability_checks(
    audit_rows: list[dict[str, str]],
    audit_fieldnames: list[str],
    label_rows: list[dict[str, str]],
    label_fieldnames: list[str],
) -> list[dict[str, Any]]:
    case_ids = [row.get("case_id", "") for row in audit_rows if row.get("case_id")]
    duplicate_case_rows = [row for row in audit_rows if case_ids.count(row.get("case_id", "")) > 1]
    checks = [
        _check_row(
            check_id="file.audit_set_nonempty",
            check_name="audit set CSV has rows",
            status="pass" if audit_rows else "fail",
            affected_rows=len(audit_rows),
            total_rows=len(audit_rows),
            details="input audit set loaded" if audit_rows else "input audit set has no rows or is missing",
        ),
        _check_row(
            check_id="identity.case_id_unique",
            check_name="case_id uniqueness",
            status="fail" if duplicate_case_rows else "pass",
            affected_rows=len(duplicate_case_rows),
            total_rows=len(audit_rows),
            details=f"unique_case_ids={len(set(case_ids))}",
            examples=duplicate_case_rows,
        ),
    ]
    checks.extend(_field_presence_checks(audit_rows, audit_fieldnames))
    checks.extend(_decision_checks(audit_rows))
    checks.extend(_json_evidence_checks(audit_rows))
    checks.extend(_length_checks(audit_rows))
    checks.extend(_conflict_checks(audit_rows))
    checks.extend(_label_template_checks(label_fieldnames, label_rows))
    return checks


def _stratum_predicate(stratum: str) -> Callable[[dict[str, str]], bool]:
    def has_group(row: dict[str, str], group: str) -> bool:
        return group in _groups(row)

    if stratum == "actionable_recommendation":
        return lambda row: has_group(row, "actionable_recommendation") or not _nonempty(row.get("abstention_reason"))
    if stratum == "missing_density":
        return lambda row: has_group(row, "missing_density_priority_sample") or "density" in row.get("missing_info", "")
    if stratum == "missing_size":
        return lambda row: has_group(row, "missing_size_priority_sample") or "size" in row.get("missing_info", "")
    if stratum == "no_structured_nodule":
        return lambda row: has_group(row, "no_structured_nodule") or row.get("abstention_reason") == "no_structured_nodule"
    if stratum == "high_risk_density_conflict":
        return lambda row: has_group(row, "high_risk_density_conflict") or (
            "density_category" in row.get("conflict_fields", "") and "high" in row.get("conflict_risk_levels", "")
        )
    raise ValueError(f"unknown stratum: {stratum}")


def _pilot_sort_key(stratum: str, row: dict[str, str]) -> tuple[Any, ...]:
    if stratum == "missing_density":
        return (
            -_as_int(row.get("stage2_subtype_but_not_applicable_mentions")),
            -_as_int(row.get("stage2_subtype_mentions")),
            -_as_int(row.get("candidate_with_density")),
            row.get("case_id", ""),
        )
    if stratum == "missing_size":
        return (
            -_as_int(row.get("size_mm_prediction_mentions")),
            -_as_int(row.get("pulmonary_cue_prediction_mentions")),
            -_as_int(row.get("candidate_with_size")),
            row.get("case_id", ""),
        )
    if stratum == "high_risk_density_conflict":
        return (-_as_int(row.get("conflict_rows")), row.get("case_id", ""))
    return (_as_int(row.get("audit_priority")), row.get("case_id", ""))


def build_pilot_rows(rows: list[dict[str, str]], *, pilot_size: int = 20, quota_per_stratum: int = 4) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    ordered_case_ids: list[str] = []

    pools: dict[str, list[dict[str, str]]] = {}
    for stratum in PILOT_STRATA:
        predicate = _stratum_predicate(stratum)
        pools[stratum] = sorted([row for row in rows if predicate(row)], key=lambda row, s=stratum: _pilot_sort_key(s, row))

    for stratum in PILOT_STRATA:
        chosen_for_stratum = 0
        for row in pools[stratum]:
            case_id = row.get("case_id", "")
            if not case_id:
                continue
            if case_id in selected:
                strata = set(str(selected[case_id]["pilot_stratum"]).split("|"))
                strata.add(stratum)
                selected[case_id]["pilot_stratum"] = "|".join(sorted(strata))
                continue
            item = dict(row)
            item["pilot_stratum"] = stratum
            selected[case_id] = item
            ordered_case_ids.append(case_id)
            chosen_for_stratum += 1
            if chosen_for_stratum >= quota_per_stratum:
                break

    if len(selected) < pilot_size:
        for stratum in PILOT_STRATA:
            for row in pools[stratum]:
                if len(selected) >= pilot_size:
                    break
                case_id = row.get("case_id", "")
                if not case_id or case_id in selected:
                    continue
                item = dict(row)
                item["pilot_stratum"] = stratum
                selected[case_id] = item
                ordered_case_ids.append(case_id)
            if len(selected) >= pilot_size:
                break

    return [selected[case_id] for case_id in ordered_case_ids[:pilot_size]]


def write_annotation_guide(path: Path, *, audit_rows: int, pilot_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# M3-3C 模块3人工审查标注指南

## 定位

本指南服务于 `module3_manual_audit_set` 的人工审查。该 audit set 不是训练集，也不是最终 gold benchmark；它用于判断 conservative CDSG agent 的 recommendation、abstention、candidate conflict 与后续 soft matching / safer aggregation 是否合理。

建议先标注 pilot 20，再根据一致性和字段可读性决定是否标完整 {audit_rows} 条。本轮生成的 pilot 文件包含 {pilot_rows} 条。

## 标注前阅读顺序

1. 先看 `case_id`、`subject_id` 和 `audit_group`，确认该样本的审查目的。
2. 再看 `recommendation_action`、`lung_rads_category`、`abstention_reason` 和 `missing_info`。
3. 查看 `candidate_nodule_evidence`，重点关注 mention_text、size_mm、density_category、location 和各自 confidence。
4. 如存在 `conflict_fields` / `conflict_evidence`，判断冲突是否会改变 Lung-RADS 路径。
5. 最后参考 `reasoning_path` 与 `guideline_anchor`，判断 CDSG 轨迹是否与证据一致。

## 字段含义与允许取值

### gold_actionable

含义：人工判断该 case 是否有足够事实给出 Lung-RADS/CDSG 随访建议。
允许取值：`yes` / `no` / `uncertain`。

### gold_lung_rads_category

含义：人工根据可见证据判断的 Lung-RADS 类别。
允许取值：`1` / `2` / `3` / `4A` / `4B` / `4X` / `not_applicable` / `uncertain`。

### gold_recommendation_level

含义：人工判断的推荐强度/随访层级。
允许取值：`routine_screening` / `short_interval_followup` / `tissue_sampling` / `insufficient_data` / `uncertain`。

### cdsg_recommendation_correct

含义：CDSG 输出的 recommendation 与人工判断是否一致。
允许取值：`yes` / `no` / `partially` / `uncertain`。
判断方法：如果 CDSG category、recommendation level 与关键事实均一致，标 `yes`；如果 category 或随访强度明显错误，标 `no`；如果方向正确但存在事实遗漏或强度边界争议，标 `partially`。

### abstention_appropriate

含义：CDSG abstain 是否合理。
允许取值：`yes` / `no` / `not_applicable` / `uncertain`。
判断方法：actionable case 通常填 `not_applicable`。若缺失 size/density/no nodule 确实无法安全决策，标 `yes`；若证据中已有可用事实但被系统漏用，标 `no`。

### density_fact_correct

含义：系统使用或候选中的 density fact 是否正确。
允许取值：`yes` / `no` / `missing` / `uncertain`。
判断方法：比较 mention_text 中是否明确描述 solid、part-solid、ground-glass、calcified 等；若文本可支持系统值，标 `yes`；若系统值与文本不符，标 `no`；若文本没有可判定 density，标 `missing`。

### size_fact_correct

含义：系统使用或候选中的 size_mm 是否正确。
允许取值：`yes` / `no` / `missing` / `uncertain`。
判断方法：优先看 nodule mention 附近的毫米数值和是否对应同一个结节；若只有 has-size 概率但无可抽取 size_mm，不应视为正确 size fact。

### dominant_nodule_correct

含义：被 CDSG 或 candidate selection 视为主导结节的候选是否合理。
允许取值：`yes` / `no` / `uncertain`。
判断方法：通常最大且最可疑的结节应作为 dominant nodule；若 size、density 或 note_id 明显来自不同结节且被错误合并，标 `no`。

### conflict_requires_manual_resolution

含义：候选事实冲突是否需要人工解决。
允许取值：`yes` / `no` / `uncertain`。
判断方法：如果 density/size/location 冲突可能改变 Lung-RADS category 或随访强度，标 `yes`；如果只是同义位置或不影响路径的小差异，标 `no`。

### under_followup_risk

含义：系统输出相对人工判断是否有随访不足风险。
允许取值：`none` / `low` / `medium` / `high` / `uncertain`。
判断方法：将高风险结节降为低风险、错误 abstain 导致应随访未随访、或漏掉更大/更可疑结节时风险升高。

### over_followup_risk

含义：系统输出相对人工判断是否有过度随访风险。
允许取值：`none` / `low` / `medium` / `high` / `uncertain`。
判断方法：将低风险结节升为短期复查或侵入性处置、把非结节误作结节时风险升高。

### annotator_notes

含义：自由文本备注。建议记录关键证据片段、争议点、是否需要第二人复核。

## 如何判断 CDSG recommendation 是否正确

CDSG recommendation 必须同时满足三点：使用的 dominant nodule 与证据一致；size/density/location 等硬事实可由文本支持；reasoning_path 和 guideline_anchor 指向的规则与最终 category/recommendation 一致。不要把 CDSG 自己生成的 strong-silver label 当作 gold。

## 如何判断 abstention 是否合理

abstention 是 conservative CDSG 的安全机制。若缺失关键 size_mm、明确 density，或没有可靠 structured nodule，abstention 通常合理。若 candidate_nodule_evidence 或 conflict_evidence 已经包含清晰事实但 adapter/CDSG 未使用，则 abstention 可能不合理。

## 风险标注建议

优先关注 under-followup risk。任何可能把 4A/4B/4X 降到较低类别、或把应短期复查/取样的 case 变成 routine/abstain 的情况，都应标为 medium 或 high。over-followup risk 则关注不必要短期复查或侵入性建议。

## 推荐流程

先完成 `module3_manual_audit_pilot_20.csv` 的 20 条 pilot annotation。若标注者对字段含义和证据可读性无明显问题，再标完整 133 条；若 pilot 中大量样本因 evidence 过长或缺少原文无法判断，应先改进审查表而不是进入 learned-model 实验。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-set",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_manual_audit_set.csv"),
    )
    parser.add_argument(
        "--label-template",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_manual_audit_label_template.csv"),
    )
    parser.add_argument(
        "--readability-output",
        type=Path,
        default=Path("outputs/phaseA3/tables/module3_manual_audit_readability_check.csv"),
    )
    parser.add_argument(
        "--pilot-output",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_manual_audit_pilot_20.csv"),
    )
    parser.add_argument(
        "--guide-output",
        type=Path,
        default=Path("reports/module3_manual_audit_annotation_guide.md"),
    )
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--quota-per-stratum", type=int, default=4)
    args = parser.parse_args()

    audit_rows = _load_csv(args.audit_set)
    audit_fieldnames = _fieldnames(args.audit_set)
    label_rows = _load_csv(args.label_template)
    label_fieldnames = _fieldnames(args.label_template)

    checks = build_readability_checks(audit_rows, audit_fieldnames, label_rows, label_fieldnames)
    _write_csv(
        args.readability_output,
        checks,
        ["check_id", "check_name", "status", "affected_rows", "total_rows", "rate", "details", "example_case_ids"],
    )

    pilot_source = label_rows if label_rows else audit_rows
    pilot_rows = build_pilot_rows(
        pilot_source,
        pilot_size=args.pilot_size,
        quota_per_stratum=args.quota_per_stratum,
    )
    pilot_fieldnames = ["pilot_stratum"] + [field for field in (label_fieldnames or audit_fieldnames) if field != "pilot_stratum"]
    _write_csv(args.pilot_output, pilot_rows, pilot_fieldnames)
    write_annotation_guide(args.guide_output, audit_rows=len(audit_rows), pilot_rows=len(pilot_rows))

    group_counter: Counter[str] = Counter()
    for row in pilot_rows:
        for stratum in str(row.get("pilot_stratum", "")).split("|"):
            if stratum:
                group_counter[stratum] += 1

    print(
        json.dumps(
            {
                "audit_rows": len(audit_rows),
                "readability_checks": len(checks),
                "readability_failures": sum(1 for row in checks if row["status"] == "fail"),
                "readability_warnings": sum(1 for row in checks if row["status"] == "warn"),
                "pilot_rows": len(pilot_rows),
                "pilot_group_counts": dict(sorted(group_counter.items())),
                "readability_output": str(args.readability_output),
                "pilot_output": str(args.pilot_output),
                "guide_output": str(args.guide_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
