#!/usr/bin/env python3
"""Build the M3-3D closure report and summary tables.

This closure step only consolidates the Codex-assisted pilot 20 audit. It does
not train models, run GPU jobs, expand to 133 cases, or start learned-model
experiments.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INPUT_CSV = Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.csv")

CONFIDENCE_SUMMARY_CSV = Path("outputs/phaseA3/tables/module3_codex_pilot20_confidence_summary.csv")
EXPERT_REVIEW_CASES_CSV = Path("outputs/phaseA3/tables/module3_codex_pilot20_expert_review_cases.csv")
GROUPWISE_SUMMARY_CSV = Path("outputs/phaseA3/tables/module3_codex_pilot20_groupwise_summary.csv")
CURRENT_STATUS_SUMMARY_CSV = Path("outputs/phaseA3/tables/module3_module3_current_status_summary.csv")
CLOSURE_REPORT = Path("reports/module3_codex_assisted_audit_closure.md")

UNCERTAINTY_COLUMNS = [
    "codex_gold_actionable_suggestion",
    "codex_lung_rads_category_suggestion",
    "codex_recommendation_level_suggestion",
    "codex_cdsg_recommendation_correct_suggestion",
    "codex_under_followup_risk_suggestion",
    "codex_over_followup_risk_suggestion",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _yes(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _rate(count: int, denominator: int) -> str:
    return f"{count / denominator:.6f}" if denominator else "0.000000"


def _groups(row: dict[str, str]) -> list[str]:
    return [item for item in row.get("audit_group", "").split("|") if item] or ["ungrouped"]


def _has_uncertainty(row: dict[str, str]) -> bool:
    return any(str(row.get(col, "")).strip().lower() == "uncertain" for col in UNCERTAINTY_COLUMNS)


def _grounded(row: dict[str, str]) -> bool:
    return _yes(row.get("codex_cited_evidence_exists")) and _yes(row.get("codex_rationale_supported_by_evidence"))


def _uncertainty_driver(row: dict[str, str]) -> str:
    group = row.get("audit_group", "")
    missing_info = row.get("missing_info", "")
    conflict_fields = row.get("conflict_fields", "")
    if "high_risk_density_conflict" in group:
        return "high-risk density conflict affecting possible Lung-RADS path"
    if "density_category" in missing_info:
        return "missing density prevents safe category assignment"
    if "nodule_size" in missing_info:
        return "missing size prevents safe category assignment"
    if "nodule" in missing_info:
        return "no structured pulmonary nodule evidence"
    if conflict_fields:
        return f"candidate conflict fields: {conflict_fields}"
    return "clinical uncertainty noted by Codex-assisted self-check"


def _confidence_interpretation(label: str) -> str:
    if label == "high":
        return "rare stable prelabel; still not clinical gold"
    if label == "medium":
        return "evidence-grounded but not sufficient for expert replacement"
    if label == "low":
        return "requires caution; often tied to conflicts or missing facts"
    return "unrecognized confidence label"


def _methodological_conclusion(group: str, expert_count: int, total: int, uncertain_count: int) -> str:
    if group == "high_risk_density_conflict":
        return "most sensitive group; all cases require clinical expert review before any gold use"
    if group == "missing_density_priority_sample":
        return "density absence makes clinical adjudication unsafe without expert review"
    if group == "missing_size_priority_sample":
        return "size absence supports conservative abstention but blocks category-level claims"
    if group == "no_structured_nodule":
        return "abstention appears methodologically stable in this pilot"
    if expert_count / total >= 0.5 or uncertain_count / total >= 0.5:
        return "actionable-looking cases still contain substantial clinical uncertainty"
    return "lower uncertainty in this pilot but still not a gold benchmark"


def build_confidence_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(row.get("codex_confidence", "empty") or "empty" for row in rows)
    output: list[dict[str, Any]] = []
    for label in ["high", "medium", "low"]:
        count = counts.get(label, 0)
        output.append(
            {
                "confidence": label,
                "count": count,
                "denominator": total,
                "rate": _rate(count, total),
                "interpretation": _confidence_interpretation(label),
            }
        )
    return output


def build_expert_review_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not _yes(row.get("needs_clinical_expert_review")):
            continue
        output.append(
            {
                "case_id": row.get("case_id", ""),
                "pilot_stratum": row.get("pilot_stratum", ""),
                "audit_group": row.get("audit_group", ""),
                "codex_confidence": row.get("codex_confidence", ""),
                "codex_under_followup_risk_suggestion": row.get("codex_under_followup_risk_suggestion", ""),
                "codex_over_followup_risk_suggestion": row.get("codex_over_followup_risk_suggestion", ""),
                "clinical_uncertainty_driver": _uncertainty_driver(row),
                "codex_rationale": row.get("codex_rationale", ""),
                "codex_cited_evidence": row.get("codex_cited_evidence", ""),
            }
        )
    return output


def build_groupwise_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for group in _groups(row):
            grouped[group].append(row)

    output: list[dict[str, Any]] = []
    for group in sorted(grouped):
        group_rows = grouped[group]
        total = len(group_rows)
        low_conf = sum(1 for row in group_rows if row.get("codex_confidence") == "low")
        expert = sum(1 for row in group_rows if _yes(row.get("needs_clinical_expert_review")))
        uncertain = sum(1 for row in group_rows if _has_uncertainty(row))
        grounded = sum(1 for row in group_rows if _grounded(row))
        output.append(
            {
                "audit_group": group,
                "rows": total,
                "low_confidence_count": low_conf,
                "low_confidence_rate": _rate(low_conf, total),
                "needs_clinical_expert_review_count": expert,
                "needs_clinical_expert_review_rate": _rate(expert, total),
                "rows_with_any_codex_uncertainty": uncertain,
                "uncertainty_rate": _rate(uncertain, total),
                "evidence_grounding_support_count": grounded,
                "evidence_grounding_support_denominator": total,
                "evidence_grounding_support_rate": _rate(grounded, total),
                "methodological_conclusion": _methodological_conclusion(group, expert, total, uncertain),
            }
        )
    return output


def build_current_status_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    expert = sum(1 for row in rows if _yes(row.get("needs_clinical_expert_review")))
    grounded = sum(1 for row in rows if _grounded(row))
    low = sum(1 for row in rows if row.get("codex_confidence") == "low")
    return [
        {
            "module3_component": "CDSG hard-match deterministic agent",
            "current_status": "implemented and auditable",
            "decision_for_now": "retain as deterministic evidence-path engine",
            "rationale": "rules and reasoning paths can be inspected; output remains conservative rather than clinical gold",
            "next_condition": "expert-validated labels are required before clinical performance claims",
        },
        {
            "module3_component": "module2-to-CDSG fact adapter",
            "current_status": "implemented with observable missing-fact and conflict modes",
            "decision_for_now": "retain for failure-mode analysis",
            "rationale": "adapter surfaces missing density, missing size, and candidate conflicts that explain abstentions",
            "next_condition": "improve fact selection only after expert-reviewed error taxonomy is available",
        },
        {
            "module3_component": "conservative abstention mechanism",
            "current_status": "methodologically supported by pilot",
            "decision_for_now": "retain as safety mechanism",
            "rationale": "missing density/size and no structured nodule groups are safer as insufficient-data outputs",
            "next_condition": "calibrate only against clinical expert review, not Codex prelabels",
        },
        {
            "module3_component": "evidence-grounded failure-mode audit",
            "current_status": f"completed for pilot 20 with grounding {grounded}/{total}",
            "decision_for_now": "use as module3 reporting artifact",
            "rationale": "Codex cited evidence and rationale support were internally consistent in the pilot",
            "next_condition": "continue to label as model-assisted failure-mode audit",
        },
        {
            "module3_component": "Codex-assisted pilot audit",
            "current_status": f"completed; low confidence {low}/{total}; expert review needed {expert}/{total}",
            "decision_for_now": "do not expand to gold benchmark",
            "rationale": "evidence grounding is usable, but medical adjudication confidence is insufficient",
            "next_condition": "clinical expert review protocol and adjudicated reference labels",
        },
        {
            "module3_component": "complete 133-case Codex expansion",
            "current_status": "not started",
            "decision_for_now": "do not expand",
            "rationale": "pilot has too many low-confidence and expert-review cases",
            "next_condition": "only reconsider after expert-reviewed pilot and revised conflict handling",
        },
        {
            "module3_component": "learned-model performance experiment",
            "current_status": "blocked",
            "decision_for_now": "do not run",
            "rationale": "Codex prelabels are not clinical gold and cannot serve as evaluation labels",
            "next_condition": "expert-adjudicated benchmark with fixed protocol and train/test separation",
        },
        {
            "module3_component": "soft matching formal integration",
            "current_status": "not ready for formal process",
            "decision_for_now": "allow design discussion only; no formal integration",
            "rationale": "soft matching needs expert-approved targets and acceptance criteria to avoid masking fact errors",
            "next_condition": "define expert-reviewed matching targets, failure thresholds, and audit gates",
        },
        {
            "module3_component": "paper positioning",
            "current_status": "suitable as method and audit contribution",
            "decision_for_now": "frame as model-assisted failure-mode audit",
            "rationale": "current artifacts demonstrate deterministic CDSG, adapter, abstention, and need for expert review",
            "next_condition": "avoid claims of expert annotation, clinical gold, or learned-model performance",
        },
    ]


def _report_lines(
    rows: list[dict[str, str]],
    confidence_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> list[str]:
    total = len(rows)
    grounded = sum(1 for row in rows if _grounded(row))
    obvious = sum(1 for row in rows if _yes(row.get("obvious_codex_error")))
    confidence = {row["confidence"]: row["count"] for row in confidence_rows}
    hardest_groups = sorted(
        group_rows,
        key=lambda item: (
            float(item["needs_clinical_expert_review_rate"]),
            float(item["uncertainty_rate"]),
            float(item["low_confidence_rate"]),
        ),
        reverse=True,
    )

    return [
        "# Module 3 M3-3D Codex-Assisted Audit Closure",
        "",
        "## 定位",
        "",
        "本报告是 M3-3D 的阶段性收口，固化 Codex-assisted pilot audit（Codex 辅助试点审计）的工程与方法学结论。这里的 prelabels（预标注）只能称为 Codex-assisted pre-annotation、model-assisted evidence audit（模型辅助证据审计）或 failure-mode audit（失败模式审计）。",
        "",
        "这些结果不是 clinical gold（临床金标准）、不是 expert annotation（专家标注）、不是 manual gold benchmark（人工金标准基准）。没有医学专家参与，`non_clinical_reviewer_notes` 仅表示 Codex self-check; not human reviewed。",
        "",
        "## Pilot 20 完成状态",
        "",
        f"- Codex-assisted pilot 20 已完成：{total}/{total} completed。",
        f"- Codex confidence 分布：high={confidence.get('high', 0)}，medium={confidence.get('medium', 0)}，low={confidence.get('low', 0)}。",
        f"- needs_clinical_expert_review：{len(expert_rows)}/{total}。",
        f"- obvious_codex_error：{obvious}。",
        f"- evidence grounding support rate：{grounded}/{total}。",
        "",
        "结论：Codex evidence grounding 表现可用，但医学裁决信心不足。",
        "",
        "## 必答问题",
        "",
        "1. Codex-assisted pilot 20 是否完成？",
        "",
        f"已完成。20 条 case 均已填写 Codex suggestion 和 evidence-grounding self-check 字段，评估脚本也已生成 summary、risk flags 和 report。",
        "",
        "2. Codex 是否可以替代医学专家？",
        "",
        "不可以。Codex 能辅助定位证据和暴露失败模式，但不能替代医学专家进行 clinical adjudication（临床裁决）。10/20 case 需要 clinical expert review，因此不能把 Codex prelabels 扩展为 gold benchmark。",
        "",
        "3. 为什么不扩展到完整 133？",
        "",
        "不扩展。虽然 evidence grounding 为 20/20，但 low confidence 为 7/20，clinical expert review 为 10/20，且 high-risk density conflict 是高不确定性来源。继续扩展只会扩大非专家 prelabels 的规模，不能解决 gold validity（标签有效性）问题。",
        "",
        "4. 哪些 audit group 最容易出现 clinical uncertainty？",
        "",
    ] + [
        f"- {row['audit_group']}: expert review {row['needs_clinical_expert_review_count']}/{row['rows']}，uncertainty {row['rows_with_any_codex_uncertainty']}/{row['rows']}，low confidence {row['low_confidence_count']}/{row['rows']}。"
        for row in hardest_groups
    ] + [
        "",
        "最突出的是 `high_risk_density_conflict` 和 `missing_density_priority_sample`。前者 5/5 需要 expert review；后者 4/4 需要 expert review 且 4/4 有 Codex uncertainty。",
        "",
        "5. 为什么这些 prelabels 不能作为 clinical gold？",
        "",
        "原因是本轮没有医学专家参与，Codex 只做 evidence-grounded pre-annotation。missing density、dominant nodule selection、solid/part-solid/ground-glass 冲突、非肺部 nodule 混入等问题会改变 Lung-RADS 路径，必须由专家按固定协议裁决。",
        "",
        "6. 当前模块3是否可以进入 learned-model 实验？",
        "",
        "不可以。当前模块3不进入 learned-model 主实验，也不生成 learned-model performance table。Codex prelabels 不能作为训练或评估 gold。",
        "",
        "7. 当前模块3是否可以进入 soft matching 正式接入？",
        "",
        "不可以进入正式接入。soft matching（软匹配）可以作为后续工程设计讨论，但正式接入需要专家审核目标、可接受误差阈值和审计门禁；否则 soft matching 可能掩盖 fact adapter 的事实错误。",
        "",
        "8. 当前模块3最稳妥的论文定位是什么？",
        "",
        "最稳妥定位是 model-assisted failure-mode audit，而不是 expert annotation。论文中可报告阶段性工程成果：",
        "",
        "1. CDSG hard-match deterministic agent；",
        "2. module2-to-CDSG fact adapter；",
        "3. conservative abstention mechanism；",
        "4. evidence-grounded failure-mode audit；",
        "5. Codex-assisted pilot audit demonstrating need for expert review。",
        "",
        "9. 后续如果要做临床性能评估，需要什么条件？",
        "",
        "- 医学专家参与并签署明确标注协议。",
        "- 对 high-risk density conflict、missing density、missing size、dominant nodule selection 进行专家裁决。",
        "- 固定 clinical gold benchmark，并明确 train/validation/test 分离。",
        "- 建立双人或多轮 adjudication 机制，记录不一致与裁决。",
        "- 在 gold benchmark 完成后再定义 learned-model performance experiment 和 soft matching acceptance criteria。",
        "",
        "## 阶段性结论",
        "",
        "当前 M3-3C/M3-3D 的结论是：Codex evidence grounding 表现可用，但医学裁决信心不足。10/20 需要 clinical expert review，因此不能把 Codex prelabels 扩展为 gold benchmark。当前模块3不进入 learned-model 主实验，也不进入 soft matching 正式流程。",
        "",
        "模块3当前阶段性成果应表述为 deterministic CDSG pipeline plus evidence-grounded failure-mode audit，展示了保守 abstention 与专家复核需求，而不是展示 clinical performance。",
    ]


def main() -> int:
    rows = _read_rows(INPUT_CSV)
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 pilot rows, found {len(rows)}")

    confidence_rows = build_confidence_summary(rows)
    expert_rows = build_expert_review_cases(rows)
    group_rows = build_groupwise_summary(rows)
    status_rows = build_current_status_summary(rows)

    _write_csv(
        CONFIDENCE_SUMMARY_CSV,
        confidence_rows,
        ["confidence", "count", "denominator", "rate", "interpretation"],
    )
    _write_csv(
        EXPERT_REVIEW_CASES_CSV,
        expert_rows,
        [
            "case_id",
            "pilot_stratum",
            "audit_group",
            "codex_confidence",
            "codex_under_followup_risk_suggestion",
            "codex_over_followup_risk_suggestion",
            "clinical_uncertainty_driver",
            "codex_rationale",
            "codex_cited_evidence",
        ],
    )
    _write_csv(
        GROUPWISE_SUMMARY_CSV,
        group_rows,
        [
            "audit_group",
            "rows",
            "low_confidence_count",
            "low_confidence_rate",
            "needs_clinical_expert_review_count",
            "needs_clinical_expert_review_rate",
            "rows_with_any_codex_uncertainty",
            "uncertainty_rate",
            "evidence_grounding_support_count",
            "evidence_grounding_support_denominator",
            "evidence_grounding_support_rate",
            "methodological_conclusion",
        ],
    )
    _write_csv(
        CURRENT_STATUS_SUMMARY_CSV,
        status_rows,
        ["module3_component", "current_status", "decision_for_now", "rationale", "next_condition"],
    )
    CLOSURE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLOSURE_REPORT.write_text(
        "\n".join(_report_lines(rows, confidence_rows, expert_rows, group_rows)) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "closure_report": str(CLOSURE_REPORT),
            "confidence_summary": str(CONFIDENCE_SUMMARY_CSV),
            "expert_review_cases": str(EXPERT_REVIEW_CASES_CSV),
            "groupwise_summary": str(GROUPWISE_SUMMARY_CSV),
            "current_status_summary": str(CURRENT_STATUS_SUMMARY_CSV),
            "rows": len(rows),
            "expert_review_cases_count": len(expert_rows),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
