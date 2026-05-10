#!/usr/bin/env python3
"""Build the M3-3B manual audit set for conservative CDSG evaluation.

This script is data construction only. It does not train models, call LLMs, or
modify the official conservative v2 case bundles.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AUDIT_LABEL_COLUMNS = [
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

AUDIT_CSV_FIELDS = [
    "case_id",
    "subject_id",
    "audit_group",
    "audit_priority",
    "recommendation_level",
    "recommendation_action",
    "lung_rads_category",
    "risk_category",
    "abstention_reason",
    "missing_info",
    "selected_size_mm",
    "selected_density",
    "selected_note_id",
    "candidate_count",
    "candidate_with_density",
    "candidate_with_size",
    "candidate_with_location",
    "stage1_explicit_density_mentions",
    "stage2_subtype_mentions",
    "stage2_applicable_true_mentions",
    "stage2_subtype_but_not_applicable_mentions",
    "size_mm_prediction_mentions",
    "pulmonary_cue_prediction_mentions",
    "density_root_cause",
    "size_root_cause",
    "no_structured_nodule_root_cause",
    "conflict_fields",
    "conflict_risk_levels",
    "conflict_rows",
    "reasoning_path",
    "guideline_anchor",
    "candidate_nodule_evidence",
    "candidate_evidence_truncated",
    "conflict_evidence",
    "source_report_note_ids",
]


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
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fp.flush()


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _truncate(value: Any, limit: int = 180) -> str | None:
    if value in {None, ""}:
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _is_abstained(row: dict[str, Any]) -> bool:
    return row.get("abstention_reason") not in {None, "", "None"}


def _as_int(value: Any) -> int:
    if value in {None, ""}:
        return 0
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _split_pipe(value: Any) -> list[str]:
    if value in {None, ""}:
        return []
    return [item for item in str(value).split("|") if item]


def _index_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            indexed[case_id] = row
    return indexed


def _group_by_case(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id:
            grouped[case_id].append(row)
    return grouped


def _fact_source_summary(fact_sources: dict[str, Any] | None, field: str) -> dict[str, Any] | None:
    if not isinstance(fact_sources, dict):
        return None
    source = fact_sources.get(field)
    if not isinstance(source, dict):
        return None
    return {
        "source": source.get("source"),
        "confidence": source.get("confidence"),
        "confidence_value": source.get("confidence_value"),
        "probability": source.get("probability"),
        "model_tag": source.get("model_tag"),
        "mention_id": source.get("mention_id"),
        "original_text": source.get("original_text"),
    }


def _iter_nodules(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    nodules: list[dict[str, Any]] = []
    for report in bundle.get("radiology_facts") or []:
        note_id = report.get("note_id")
        for nodule in report.get("nodules") or []:
            item = dict(nodule)
            item["_note_id"] = note_id
            nodules.append(item)
    return nodules


def _nodule_priority(nodule: dict[str, Any]) -> tuple[int, float, str]:
    size = nodule.get("size_mm")
    try:
        size_value = float(size) if size is not None else -1.0
    except (TypeError, ValueError):
        size_value = -1.0
    has_density = int(nodule.get("density_category") not in {None, "", "unclear"})
    has_size = int(size_value >= 0)
    dominant_rank = nodule.get("dominant_candidate_rank")
    rank_bonus = 1 if dominant_rank in {1, "1"} else 0
    return (rank_bonus + has_density + has_size, size_value, str(nodule.get("mention_id") or ""))


def _candidate_summary(nodule: dict[str, Any]) -> dict[str, Any]:
    fact_sources = nodule.get("fact_sources")
    mention_text = nodule.get("mention_text") or nodule.get("evidence_span")
    for field in ("density_category", "size_mm", "location_lobe"):
        source = _fact_source_summary(fact_sources, field)
        if source and source.get("original_text"):
            mention_text = mention_text or source.get("original_text")
    return {
        "note_id": nodule.get("_note_id"),
        "nodule_id": nodule.get("nodule_id_in_report"),
        "candidate_id": nodule.get("module2_candidate_id"),
        "mention_id": nodule.get("mention_id"),
        "mention_text": mention_text,
        "size_mm": nodule.get("size_mm"),
        "size_text": nodule.get("size_text"),
        "density_category": nodule.get("density_category"),
        "density_text": nodule.get("density_text"),
        "location": nodule.get("location_lobe") or nodule.get("location_text"),
        "confidence": nodule.get("confidence"),
        "source_confidence": nodule.get("source_confidence"),
        "missing_flags": nodule.get("missing_flags") or [],
        "size_source": _fact_source_summary(fact_sources, "size_mm"),
        "density_source": _fact_source_summary(fact_sources, "density_category"),
        "location_source": _fact_source_summary(fact_sources, "location_lobe"),
    }


def _candidate_csv_summary(summary: dict[str, Any]) -> dict[str, Any]:
    size_source = summary.get("size_source") or {}
    density_source = summary.get("density_source") or {}
    location_source = summary.get("location_source") or {}
    return {
        "note_id": summary.get("note_id"),
        "mention_id": summary.get("mention_id"),
        "mention_text": _truncate(summary.get("mention_text")),
        "size_mm": summary.get("size_mm"),
        "size_text": summary.get("size_text"),
        "density_category": summary.get("density_category"),
        "location": summary.get("location"),
        "confidence": summary.get("confidence"),
        "source_confidence": summary.get("source_confidence"),
        "size_confidence": size_source.get("confidence_value"),
        "density_confidence": density_source.get("confidence_value"),
        "location_confidence": location_source.get("confidence_value"),
        "size_model": size_source.get("model_tag"),
        "density_model": density_source.get("model_tag"),
        "location_model": location_source.get("model_tag"),
    }


def _candidate_evidence(bundle: dict[str, Any], *, max_csv_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    summaries = [_candidate_summary(nodule) for nodule in sorted(_iter_nodules(bundle), key=_nodule_priority, reverse=True)]
    csv_summaries = [_candidate_csv_summary(summary) for summary in summaries[:max_csv_candidates]]
    truncated = max(0, len(summaries) - len(csv_summaries))
    return summaries, csv_summaries, truncated


def _note_ids(bundle: dict[str, Any]) -> list[str]:
    note_ids: list[str] = []
    for report in bundle.get("radiology_facts") or []:
        note_id = report.get("note_id")
        if note_id:
            note_ids.append(str(note_id))
    return sorted(set(note_ids))


def _density_sample_score(row: dict[str, str]) -> tuple[int, int, int, int, int, str]:
    return (
        _as_int(row.get("stage2_subtype_but_not_applicable_mentions")),
        _as_int(row.get("candidate_with_density")),
        int(_as_bool(row.get("candidate_selection_missed_density"))),
        int(_as_bool(row.get("density_stage2_without_adapter_density"))),
        _as_int(row.get("stage2_subtype_mentions")),
        row.get("case_id", ""),
    )


def _size_sample_score(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    return (
        _as_int(row.get("size_prediction_not_in_candidate_mentions")),
        int(_as_bool(row.get("candidate_selection_missed_size"))),
        _as_int(row.get("size_mm_prediction_mentions")),
        _as_int(row.get("pulmonary_cue_prediction_mentions")),
        row.get("case_id", ""),
    )


def _select_missing_density(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, str]]:
    return sorted(rows, key=_density_sample_score, reverse=True)[:sample_size]


def _select_missing_size(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, str]]:
    return sorted(rows, key=_size_sample_score, reverse=True)[:sample_size]


def _add_group(
    selected: dict[str, set[str]],
    case_id: str,
    group: str,
    source_order: list[tuple[str, str]],
) -> None:
    selected[case_id].add(group)
    source_order.append((case_id, group))


def _priority(groups: set[str]) -> int:
    weights = {
        "actionable_recommendation": 10,
        "high_risk_density_conflict": 20,
        "no_structured_nodule": 30,
        "missing_density_priority_sample": 40,
        "missing_size_priority_sample": 50,
    }
    return min(weights.get(group, 99) for group in groups)


def _reasoning_csv_summary(reasoning_path: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step in reasoning_path or []:
        if not isinstance(step, dict):
            continue
        output.append(
            {
                "step": step.get("step"),
                "node_id": step.get("node_id"),
                "edge_id": step.get("edge_id"),
                "condition": step.get("condition"),
                "match_type": step.get("match_type"),
            }
        )
    return output


def _anchor_csv_summary(anchors: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for anchor in anchors or []:
        if not isinstance(anchor, dict):
            continue
        output.append(
            {
                "anchor_id": anchor.get("anchor_id"),
                "graph_element_id": anchor.get("graph_element_id"),
                "graph_element_type": anchor.get("graph_element_type"),
                "condition": anchor.get("condition"),
            }
        )
    return output


def _audit_row(
    *,
    case_id: str,
    groups: set[str],
    bundle: dict[str, Any],
    strong: dict[str, Any],
    abstention_audit: dict[str, str] | None,
    conflict_rows: list[dict[str, str]],
    max_csv_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_full, candidate_csv, truncated = _candidate_evidence(bundle, max_csv_candidates=max_csv_candidates)
    conflict_fields = sorted({row.get("field", "") for row in conflict_rows if row.get("field")})
    conflict_risks = sorted({row.get("risk_level", "") for row in conflict_rows if row.get("risk_level")})
    conflict_csv = [
        {
            "note_id": row.get("note_id"),
            "field": row.get("field"),
            "selected_value": row.get("selected_value"),
            "alternative_values": row.get("alternative_values"),
            "risk_level": row.get("risk_level"),
            "source_mention_ids": row.get("source_mention_ids"),
            "example_evidence_spans": _truncate(row.get("example_evidence_spans"), 260),
        }
        for row in conflict_rows[:5]
    ]
    audit = abstention_audit or {}
    csv_row = {
        "case_id": case_id,
        "subject_id": strong.get("subject_id") or bundle.get("subject_id"),
        "audit_group": "|".join(sorted(groups)),
        "audit_priority": _priority(groups),
        "recommendation_level": strong.get("recommendation_level"),
        "recommendation_action": strong.get("recommendation_action"),
        "lung_rads_category": strong.get("lung_rads_category"),
        "risk_category": strong.get("risk_category"),
        "abstention_reason": strong.get("abstention_reason"),
        "missing_info": "|".join(str(item) for item in (strong.get("missing_info") or strong.get("missing_information") or [])),
        "selected_size_mm": audit.get("selected_size_mm", ""),
        "selected_density": audit.get("selected_density", ""),
        "selected_note_id": audit.get("selected_note_id", ""),
        "candidate_count": audit.get("candidate_count", len(candidate_full)),
        "candidate_with_density": audit.get("candidate_with_density", ""),
        "candidate_with_size": audit.get("candidate_with_size", ""),
        "candidate_with_location": audit.get("candidate_with_location", ""),
        "stage1_explicit_density_mentions": audit.get("stage1_explicit_density_mentions", ""),
        "stage2_subtype_mentions": audit.get("stage2_subtype_mentions", ""),
        "stage2_applicable_true_mentions": audit.get("stage2_applicable_true_mentions", ""),
        "stage2_subtype_but_not_applicable_mentions": audit.get("stage2_subtype_but_not_applicable_mentions", ""),
        "size_mm_prediction_mentions": audit.get("size_mm_prediction_mentions", ""),
        "pulmonary_cue_prediction_mentions": audit.get("pulmonary_cue_prediction_mentions", ""),
        "density_root_cause": audit.get("density_root_cause", ""),
        "size_root_cause": audit.get("size_root_cause", ""),
        "no_structured_nodule_root_cause": audit.get("no_structured_nodule_root_cause", ""),
        "conflict_fields": "|".join(conflict_fields),
        "conflict_risk_levels": "|".join(conflict_risks),
        "conflict_rows": len(conflict_rows),
        "reasoning_path": _json_compact(_reasoning_csv_summary(strong.get("reasoning_path") or [])),
        "guideline_anchor": _json_compact(_anchor_csv_summary(strong.get("guideline_anchor") or [])),
        "candidate_nodule_evidence": _json_compact(candidate_csv),
        "candidate_evidence_truncated": truncated,
        "conflict_evidence": _json_compact(conflict_csv),
        "source_report_note_ids": "|".join(_note_ids(bundle)),
    }
    json_record = {
        "case_id": case_id,
        "subject_id": strong.get("subject_id") or bundle.get("subject_id"),
        "audit_group": sorted(groups),
        "audit_priority": _priority(groups),
        "recommendation": strong.get("recommendation"),
        "recommendation_level": strong.get("recommendation_level"),
        "recommendation_action": strong.get("recommendation_action"),
        "lung_rads_category": strong.get("lung_rads_category"),
        "risk_category": strong.get("risk_category"),
        "abstention_reason": strong.get("abstention_reason"),
        "missing_info": strong.get("missing_info") or strong.get("missing_information") or [],
        "reasoning_path": strong.get("reasoning_path") or [],
        "guideline_anchor": strong.get("guideline_anchor") or [],
        "candidate_nodules": candidate_full,
        "candidate_evidence_truncated_in_csv": truncated,
        "conflict_rows": conflict_rows,
        "abstention_audit": audit,
        "source_report_note_ids": _note_ids(bundle),
        "case_bundle": bundle,
    }
    return csv_row, json_record


def _template_rows(csv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in csv_rows:
        item = dict(row)
        for column in AUDIT_LABEL_COLUMNS:
            item[column] = ""
        output.append(item)
    return output


def _summary_rows(
    *,
    total_cases: int,
    selected: dict[str, set[str]],
    source_group_counts: Counter[str],
    selected_group_counts: Counter[str],
    raw_selection_events: int,
    conflict_examples: list[dict[str, str]],
    missing_density_selected: int,
    missing_size_selected: int,
) -> list[dict[str, Any]]:
    rows = [
        {"metric": "total_cases", "value": total_cases, "note": "conservative v2 full set"},
        {"metric": "audit_set_unique_cases", "value": len(selected), "note": "deduplicated case-level audit set"},
        {"metric": "raw_selection_events", "value": raw_selection_events, "note": "group additions before case-level deduplication"},
        {"metric": "deduplicated_overlap_events", "value": raw_selection_events - len(selected), "note": "multi-label audit cases retained once"},
        {"metric": "missing_density_sample_target", "value": missing_density_selected, "note": "priority deterministic sample"},
        {"metric": "missing_size_sample_target", "value": missing_size_selected, "note": "priority deterministic sample"},
        {
            "metric": "high_risk_density_conflict_source_rows",
            "value": sum(
                1
                for row in conflict_examples
                if row.get("field") == "density_category" and row.get("risk_level") == "high"
            ),
            "note": "conflict rows, not unique cases",
        },
    ]
    for group, count in sorted(source_group_counts.items()):
        rows.append({"metric": f"source_pool.{group}", "value": count, "note": "eligible source rows or cases"})
    for group, count in sorted(selected_group_counts.items()):
        rows.append({"metric": f"selected_group.{group}", "value": count, "note": "unique audit cases carrying group label"})
    overlap_counter = Counter(len(groups) for groups in selected.values())
    for label_count, count in sorted(overlap_counter.items()):
        rows.append({"metric": f"cases_with_{label_count}_audit_groups", "value": count, "note": "deduplication profile"})
    return rows


def _write_report(
    path: Path,
    *,
    total_cases: int,
    audit_cases: int,
    selected_group_counts: Counter[str],
    source_group_counts: Counter[str],
    raw_selection_events: int,
    output_csv: Path,
    output_jsonl: Path,
    template_csv: Path,
    summary_csv: Path,
) -> None:
    lines = [
        "# M3-3B Manual Audit Set Design",
        "",
        "## 定位",
        "",
        "M3-3B 构建的是人工审查集，不是训练集，也不是最终 gold benchmark。本轮不训练、不调用 LLM/NLI/RAG，也不改变 conservative v2 case bundle。审查集的目标是验证 CDSG hard-match recommendation、abstention、candidate conflict 与后续 soft matching / safer aggregation 设计是否合理。",
        "",
        "## 为什么需要人工 audit set",
        "",
        "M3-3A 显示 CDSG agent 的 schema、guideline anchor、reasoning path 和 decision path 都完整，但 actionable 仅 76/253，abstention 主要来自 missing density 和 missing size。仅靠 CDSG strong-silver 不能判断 abstention 是否临床合理，也不能判断 density conflict 是否会改变 Lung-RADS 路径，因此需要人工抽查。",
        "",
        "## Audit group 设计",
        "",
        f"- actionable_recommendation: {selected_group_counts.get('actionable_recommendation', 0)} cases。全部纳入，用于检查 CDSG 终态 recommendation 是否合规。",
        f"- high_risk_density_conflict: {selected_group_counts.get('high_risk_density_conflict', 0)} cases。全部纳入，用于判断 density subtype 冲突是否改变 Lung-RADS 路径。",
        f"- no_structured_nodule: {selected_group_counts.get('no_structured_nodule', 0)} cases。全部纳入，用于判断是否确实没有可靠 pulmonary nodule candidate。",
        f"- missing_density_priority_sample: {selected_group_counts.get('missing_density_priority_sample', 0)} cases。优先选择 stage2 有 subtype 但 stage1 非 explicit、或存在 density candidate 但未被 CDSG 使用的样本。",
        f"- missing_size_priority_sample: {selected_group_counts.get('missing_size_priority_sample', 0)} cases。优先选择存在 size evidence / pulmonary cue 但无法形成 size_mm hard fact 的样本。",
        "",
        "## 去重结果",
        "",
        f"- Full set: {total_cases} cases。",
        f"- Raw selection events: {raw_selection_events}。",
        f"- Deduplicated audit set: {audit_cases} cases。",
        f"- Overlap removed: {raw_selection_events - audit_cases} events。",
        "",
        "## 如何使用 audit 结果",
        "",
        "- 如果 missing_density 中人工确认 density subtype 明确，但 stage1 标为 non-explicit，应优先修 density stage1 gating。",
        "- 如果 high-risk density conflict 多数来自不同结节混合，不能采用 same-case union 或 dominant size-first aggregation。",
        "- 如果 confidence-gated same-note aggregation 在人工审查中事实错配率低，可作为 v2.1 conservative extension 候选。",
        "- 如果 no_structured_nodule 中人工发现明显 nodule cue，需要回查 mention alignment 和 pulmonary cue filtering。",
        "- 如果人工审查显示 current 76 actionable 的 recommendation 合规，可进入 M3-3C / M3-3 soft matching 设计；否则先修 CDSG fact routing。",
        "",
        "## 当前判断",
        "",
        "当前仍不建议进入 learned-model 主实验。原因是可行动样本少、类别不平衡，并且 strong-silver label 来自 CDSG 自身，不能作为公平性能标签。下一步更适合进入人工审查执行，或基于人工结果设计 soft matching / safer aggregation。",
        "",
        "## 输出文件",
        "",
        f"- Audit CSV: `{output_csv}`",
        f"- Audit JSONL: `{output_jsonl}`",
        f"- Label template: `{template_csv}`",
        f"- Summary: `{summary_csv}`",
        "",
        "## Source pool",
        "",
    ]
    for group, count in sorted(source_group_counts.items()):
        lines.append(f"- {group}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M3-3B manual audit set.")
    parser.add_argument("--case-bundles", default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl")
    parser.add_argument("--strong-silver", default="outputs/phaseA3/datasets/module3_strong_silver_v2.jsonl")
    parser.add_argument("--abstention-audit", default="outputs/phaseA3/tables/module3_abstention_case_audit.csv")
    parser.add_argument("--missing-density-audit", default="outputs/phaseA3/tables/module3_missing_density_audit.csv")
    parser.add_argument("--missing-size-audit", default="outputs/phaseA3/tables/module3_missing_size_audit.csv")
    parser.add_argument("--no-structured-nodule-audit", default="outputs/phaseA3/tables/module3_no_structured_nodule_audit.csv")
    parser.add_argument("--conflict-summary", default="outputs/phaseA3/tables/module3_candidate_conflict_summary.csv")
    parser.add_argument("--conflict-examples", default="outputs/phaseA3/tables/module3_candidate_conflict_examples.csv")
    parser.add_argument("--missing-density-sample-size", type=int, default=30)
    parser.add_argument("--missing-size-sample-size", type=int, default=20)
    parser.add_argument("--max-csv-candidates", type=int, default=12)
    parser.add_argument("--output-csv", default="outputs/phaseA3/audit_sets/module3_manual_audit_set.csv")
    parser.add_argument("--output-jsonl", default="outputs/phaseA3/audit_sets/module3_manual_audit_set.jsonl")
    parser.add_argument("--label-template", default="outputs/phaseA3/audit_sets/module3_manual_audit_label_template.csv")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module3_manual_audit_set_summary.csv")
    parser.add_argument("--report", default="reports/module3_manual_audit_set_design.md")
    args = parser.parse_args()

    bundles = _index_by_case(_load_jsonl(Path(args.case_bundles)))
    strong_rows = _load_jsonl(Path(args.strong_silver))
    strong = _index_by_case(strong_rows)
    abstention_audit_rows = _load_csv(Path(args.abstention_audit))
    abstention_audit = _index_by_case(abstention_audit_rows)
    missing_density_rows = _load_csv(Path(args.missing_density_audit))
    missing_size_rows = _load_csv(Path(args.missing_size_audit))
    no_structured_rows = _load_csv(Path(args.no_structured_nodule_audit))
    _ = _load_csv(Path(args.conflict_summary))
    conflict_examples = _load_csv(Path(args.conflict_examples))
    conflict_by_case = _group_by_case(conflict_examples)

    selected: dict[str, set[str]] = defaultdict(set)
    selection_events: list[tuple[str, str]] = []
    source_group_counts: Counter[str] = Counter()

    actionable_ids = [row["case_id"] for row in strong_rows if row.get("case_id") and not _is_abstained(row)]
    source_group_counts["actionable_recommendation"] = len(actionable_ids)
    for case_id in actionable_ids:
        _add_group(selected, case_id, "actionable_recommendation", selection_events)

    high_risk_density_ids = sorted(
        {
            row.get("case_id", "")
            for row in conflict_examples
            if row.get("field") == "density_category" and row.get("risk_level") == "high" and row.get("case_id")
        }
    )
    source_group_counts["high_risk_density_conflict"] = len(high_risk_density_ids)
    for case_id in high_risk_density_ids:
        _add_group(selected, case_id, "high_risk_density_conflict", selection_events)

    no_structured_ids = [row["case_id"] for row in no_structured_rows if row.get("case_id")]
    source_group_counts["no_structured_nodule"] = len(no_structured_ids)
    for case_id in no_structured_ids:
        _add_group(selected, case_id, "no_structured_nodule", selection_events)

    density_sample = _select_missing_density(missing_density_rows, args.missing_density_sample_size)
    source_group_counts["missing_density_pool"] = len(missing_density_rows)
    source_group_counts["missing_density_priority_sample"] = len(density_sample)
    for row in density_sample:
        _add_group(selected, row["case_id"], "missing_density_priority_sample", selection_events)

    size_sample = _select_missing_size(missing_size_rows, args.missing_size_sample_size)
    source_group_counts["missing_size_pool"] = len(missing_size_rows)
    source_group_counts["missing_size_priority_sample"] = len(size_sample)
    for row in size_sample:
        _add_group(selected, row["case_id"], "missing_size_priority_sample", selection_events)

    csv_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    missing_case_ids: list[str] = []
    for case_id, groups in sorted(selected.items(), key=lambda item: (_priority(item[1]), item[0])):
        bundle = bundles.get(case_id)
        strong_row = strong.get(case_id)
        if not bundle or not strong_row:
            missing_case_ids.append(case_id)
            continue
        csv_row, json_record = _audit_row(
            case_id=case_id,
            groups=groups,
            bundle=bundle,
            strong=strong_row,
            abstention_audit=abstention_audit.get(case_id),
            conflict_rows=conflict_by_case.get(case_id, []),
            max_csv_candidates=args.max_csv_candidates,
        )
        csv_rows.append(csv_row)
        json_rows.append(json_record)

    selected_group_counts: Counter[str] = Counter()
    for row in csv_rows:
        for group in _split_pipe(row["audit_group"]):
            selected_group_counts[group] += 1

    summary_rows = _summary_rows(
        total_cases=len(strong_rows),
        selected={row["case_id"]: set(_split_pipe(row["audit_group"])) for row in csv_rows},
        source_group_counts=source_group_counts,
        selected_group_counts=selected_group_counts,
        raw_selection_events=len(selection_events),
        conflict_examples=conflict_examples,
        missing_density_selected=len(density_sample),
        missing_size_selected=len(size_sample),
    )
    if missing_case_ids:
        summary_rows.append(
            {
                "metric": "selected_cases_missing_inputs",
                "value": len(missing_case_ids),
                "note": "|".join(missing_case_ids),
            }
        )

    _write_csv(Path(args.output_csv), csv_rows, AUDIT_CSV_FIELDS)
    _write_jsonl(Path(args.output_jsonl), json_rows)
    _write_csv(Path(args.label_template), _template_rows(csv_rows), AUDIT_CSV_FIELDS + AUDIT_LABEL_COLUMNS)
    _write_csv(Path(args.summary), summary_rows, ["metric", "value", "note"])
    _write_report(
        Path(args.report),
        total_cases=len(strong_rows),
        audit_cases=len(csv_rows),
        selected_group_counts=selected_group_counts,
        source_group_counts=source_group_counts,
        raw_selection_events=len(selection_events),
        output_csv=Path(args.output_csv),
        output_jsonl=Path(args.output_jsonl),
        template_csv=Path(args.label_template),
        summary_csv=Path(args.summary),
    )

    print(
        json.dumps(
            {
                "audit_set_unique_cases": len(csv_rows),
                "raw_selection_events": len(selection_events),
                "selected_group_counts": dict(sorted(selected_group_counts.items())),
                "output_csv": args.output_csv,
                "output_jsonl": args.output_jsonl,
                "label_template": args.label_template,
                "summary": args.summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
