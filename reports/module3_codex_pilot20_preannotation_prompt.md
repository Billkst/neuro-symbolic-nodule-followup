# Codex Pilot 20 Preannotation Prompt

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

`codex_gold_actionable_suggestion,codex_lung_rads_category_suggestion,codex_recommendation_level_suggestion,codex_cdsg_recommendation_correct_suggestion,codex_abstention_appropriate_suggestion,codex_under_followup_risk_suggestion,codex_over_followup_risk_suggestion,codex_confidence,codex_rationale,codex_cited_evidence`

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
case_id,codex_gold_actionable_suggestion,codex_lung_rads_category_suggestion,codex_recommendation_level_suggestion,codex_cdsg_recommendation_correct_suggestion,codex_abstention_appropriate_suggestion,codex_under_followup_risk_suggestion,codex_over_followup_risk_suggestion,codex_confidence,codex_rationale,codex_cited_evidence
...
```

如果某个 case 证据不足，请用 `uncertain` 和 `low` confidence，不要臆造医学事实。

## 输入文件

请使用：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv`
