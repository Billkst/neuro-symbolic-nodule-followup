#!/usr/bin/env python3
"""Build a Codex-assisted evidence audit template for M3-3C.

This script does not produce clinical gold labels. It preserves the pilot case
context and evidence fields, removes the old non-expert gold-style label
columns, and adds Codex suggestion plus non-clinical evidence verification
columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEGACY_MANUAL_LABEL_COLUMNS = [
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

CODEX_SUGGESTION_COLUMNS = [
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

NON_CLINICAL_REVIEW_COLUMNS = [
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

NON_GOLD_NOTICE_COLUMN = "audit_workflow_note"


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


def _groups(row: dict[str, str]) -> list[str]:
    return [item for item in row.get("audit_group", "").split("|") if item]


def _build_template_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    retained_fields = [field for field in fieldnames if field not in LEGACY_MANUAL_LABEL_COLUMNS]
    output_fields = [NON_GOLD_NOTICE_COLUMN] + retained_fields + CODEX_SUGGESTION_COLUMNS + NON_CLINICAL_REVIEW_COLUMNS
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            NON_GOLD_NOTICE_COLUMN: "model_assisted_failure_mode_audit_not_clinical_gold",
        }
        for field in retained_fields:
            item[field] = row.get(field, "")
        for field in CODEX_SUGGESTION_COLUMNS + NON_CLINICAL_REVIEW_COLUMNS:
            item[field] = ""
        output_rows.append(item)
    return output_rows, output_fields


def _write_protocol(
    path: Path,
    *,
    pilot_input: Path,
    template_output: Path,
    guide_input: Path,
    prompt_output: Path,
    rows: list[dict[str, str]],
) -> None:
    group_counter: Counter[str] = Counter()
    for row in rows:
        for group in _groups(row):
            group_counter[group] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M3-3C Codex-Assisted Evidence Audit Protocol",
        "",
        "## 定位",
        "",
        "该流程是 model-assisted evidence audit，不是 gold label construction。Codex 只生成医学预标注建议，非医学专家只复核 evidence grounding 与内部一致性。任何 Codex suggestion 或非医学专家复核结果都不能命名为 clinical gold，也不能作为 learned-model performance table 的 gold benchmark。",
        "",
        "## 输入与输出",
        "",
        f"- Pilot 输入：`{pilot_input}`",
        f"- 旧标注指南参考：`{guide_input}`",
        f"- Codex-assisted 模板：`{template_output}`",
        f"- Codex 预标注 prompt：`{prompt_output}`",
        f"- Pilot case 数：{len(rows)}",
        "",
        "## Audit group 覆盖",
        "",
    ]
    for group, count in sorted(group_counter.items()):
        lines.append(f"- {group}: {count}")
    lines.extend(
        [
            "",
            "## Codex 预标注列的含义",
            "",
            "Codex suggestion 列是模型辅助判断，不是 gold。`codex_gold_actionable_suggestion` 保留该列名只是为了兼容本轮指定模板字段；该列不得解释为 gold label。Codex 必须基于 evidence 判断，不允许把 CDSG recommendation 当作 gold，不允许凭空补 size_mm，不允许把 missing density 默认当 solid。",
            "",
            "Codex 需要填写：",
        ]
    )
    for column in CODEX_SUGGESTION_COLUMNS:
        lines.append(f"- `{column}`")
    lines.extend(
        [
            "",
            "## 非医学专家只填写的证据复核列",
            "",
            "非医学专家不做最终临床判断，只检查证据是否存在、Codex 引用是否能在 evidence 字段中找到、Codex rationale 是否被证据支持，以及是否有明显错误或需要医学专家复核。",
        ]
    )
    for column in NON_CLINICAL_REVIEW_COLUMNS:
        lines.append(f"- `{column}`")
    lines.extend(
        [
            "",
            "## 推荐流程",
            "",
            "1. 将 `module3_codex_pilot20_preannotation_prompt.md` 和 `module3_codex_assisted_pilot_20_template.csv` 提供给 Codex 进行预标注。",
            "2. 将 Codex 输出填回模板中的 Codex suggestion 列。",
            "3. 非医学专家只填写 evidence review 列，重点检查 cited evidence 是否存在、rationale 是否支持、是否有 obvious error。",
            "4. 运行 `scripts/phaseA3/evaluate_codex_assisted_audit.py` 汇总 evidence grounding 和风险标记。",
            "5. 只有 pilot 20 的 evidence grounding 质量足够稳定时，才扩展到完整 133 cases。",
            "",
            "## 结果用途",
            "",
            "该结果仅用于 failure mode analysis，例如识别 missing density 是否来自 stage1 gating、size_mm 是否缺失、high-risk conflict 是否需要专家复核。若要形成 clinical gold benchmark，必须由具备资质的医学专家复核并明确标注协议。",
            "",
            "## 当前限制",
            "",
            "本流程不训练模型，不跑 GPU，不接 LLM/NLI/RAG 推理服务，也不输出 learned-model performance table。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prompt(path: Path, *, template_output: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns_csv = ",".join(CODEX_SUGGESTION_COLUMNS)
    text = f"""# Codex Pilot 20 Preannotation Prompt

你将收到 `module3_codex_assisted_pilot_20_template.csv` 的 20 条 case。请只填写 Codex 预标注建议列，并输出可直接粘贴回 CSV 的结果。

## 关键边界

1. 这不是最终 gold label，也不是 clinical gold benchmark。
2. 不允许把 CDSG recommendation 或 strong-silver label 当作 gold。
3. 必须基于表格中的 evidence 字段判断，尤其是 `candidate_nodule_evidence`、`conflict_evidence`、`reasoning_path`、`guideline_anchor`、`missing_info`。
4. 不允许凭空补 `size_mm`；如果 evidence 中没有明确 size_mm 或 size text，应标 uncertain 或指出 missing size。
5. 不允许把 missing density 默认当 solid；如果 evidence 中没有明确 density，应标 uncertain 或指出 missing density。
6. 不确定就标 `uncertain`，不要强行给临床结论。
7. 必须给出 `codex_cited_evidence`，引用应能在输入 evidence 中找到。
8. 必须输出 `codex_confidence`，允许取值建议为 `low` / `medium` / `high`。
9. 对 high-risk density conflict 要保守；如果 density conflict 可能改变 Lung-RADS 路径，应倾向标记需要专家关注，并降低 confidence。
10. 输出必须是 CSV 可填入模板；不要输出 narrative report。

## 只填写以下列

`{columns_csv}`

不要填写人工证据复核列：`evidence_has_nodule`、`evidence_has_size`、`evidence_has_density`、`evidence_has_location`、`size_text_present`、`density_text_present`、`codex_cited_evidence_exists`、`codex_rationale_supported_by_evidence`、`obvious_codex_error`、`needs_clinical_expert_review`、`non_clinical_reviewer_notes`。

## 允许取值建议

- `codex_gold_actionable_suggestion`: `yes` / `no` / `uncertain`。注意：该列名中的 gold 只是模板兼容名，不代表 gold label。
- `codex_lung_rads_category_suggestion`: `1` / `2` / `3` / `4A` / `4B` / `4X` / `not_applicable` / `uncertain`。
- `codex_recommendation_level_suggestion`: `routine_screening` / `short_interval_followup` / `tissue_sampling` / `insufficient_data` / `uncertain`。
- `codex_cdsg_recommendation_correct_suggestion`: `yes` / `no` / `partially` / `uncertain`。
- `codex_abstention_appropriate_suggestion`: `yes` / `no` / `not_applicable` / `uncertain`。
- `codex_under_followup_risk_suggestion`: `none` / `low` / `medium` / `high` / `uncertain`。
- `codex_over_followup_risk_suggestion`: `none` / `low` / `medium` / `high` / `uncertain`。
- `codex_confidence`: `low` / `medium` / `high`。
- `codex_rationale`: 一句话说明，必须基于 evidence。
- `codex_cited_evidence`: 引用具体 mention_text、size_text、density_text 或 conflict evidence；如果没有可引用证据，写 `no_direct_evidence_found`。

## 输出格式

请输出 CSV，第一列必须是 `case_id`，后面是 Codex 预标注列：

```csv
case_id,{columns_csv}
...
```

如果某个 case 证据不足，请用 `uncertain` 和 `low` confidence，不要臆造医学事实。

## 输入文件

请使用：`{template_output}`
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-input",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_manual_audit_pilot_20.csv"),
    )
    parser.add_argument(
        "--annotation-guide",
        type=Path,
        default=Path("reports/module3_manual_audit_annotation_guide.md"),
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv"),
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=Path("reports/module3_codex_assisted_audit_protocol.md"),
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("reports/module3_codex_pilot20_preannotation_prompt.md"),
    )
    args = parser.parse_args()

    rows, fieldnames = _load_csv(args.pilot_input)
    if not rows:
        raise SystemExit(f"No pilot rows loaded from {args.pilot_input}")
    if not args.annotation_guide.exists():
        raise SystemExit(f"Annotation guide not found: {args.annotation_guide}")

    template_rows, template_fields = _build_template_rows(rows, fieldnames)
    _write_csv(args.template_output, template_rows, template_fields)
    _write_protocol(
        args.protocol_output,
        pilot_input=args.pilot_input,
        template_output=args.template_output,
        guide_input=args.annotation_guide,
        prompt_output=args.prompt_output,
        rows=rows,
    )
    _write_prompt(args.prompt_output, template_output=args.template_output)

    print(
        json.dumps(
            {
                "pilot_rows": len(rows),
                "template_rows": len(template_rows),
                "dropped_legacy_gold_style_columns": [field for field in LEGACY_MANUAL_LABEL_COLUMNS if field in fieldnames],
                "codex_suggestion_columns": CODEX_SUGGESTION_COLUMNS,
                "non_clinical_review_columns": NON_CLINICAL_REVIEW_COLUMNS,
                "template_output": str(args.template_output),
                "protocol_output": str(args.protocol_output),
                "prompt_output": str(args.prompt_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
