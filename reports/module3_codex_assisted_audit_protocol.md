# M3-3C Codex-Assisted Evidence Audit Protocol

## 定位

该流程是 model-assisted evidence audit，不是 gold label construction。Codex 只生成医学预标注建议，非医学专家只复核 evidence grounding 与内部一致性。任何 Codex suggestion 或非医学专家复核结果都不能命名为 clinical gold，也不能作为 learned-model performance table 的 gold benchmark。

## 输入与输出

- Pilot 输入：`outputs/phaseA3/audit_sets/module3_manual_audit_pilot_20.csv`
- 旧标注指南参考：`reports/module3_manual_audit_annotation_guide.md`
- Codex-assisted 模板：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv`
- Codex 预标注 prompt：`reports/module3_codex_pilot20_preannotation_prompt.md`
- Pilot case 数：20

## Audit group 覆盖

- actionable_recommendation: 8
- high_risk_density_conflict: 5
- missing_density_priority_sample: 4
- missing_size_priority_sample: 4
- no_structured_nodule: 4

## Codex 预标注列的含义

Codex suggestion 列是模型辅助判断，不是 gold。`codex_gold_actionable_suggestion` 保留该列名只是为了兼容本轮指定模板字段；该列不得解释为 gold label。Codex 必须基于 evidence 判断，不允许把 CDSG recommendation 当作 gold，不允许凭空补 size_mm，不允许把 missing density 默认当 solid。

Codex 需要填写：
- `codex_gold_actionable_suggestion`
- `codex_lung_rads_category_suggestion`
- `codex_recommendation_level_suggestion`
- `codex_cdsg_recommendation_correct_suggestion`
- `codex_abstention_appropriate_suggestion`
- `codex_under_followup_risk_suggestion`
- `codex_over_followup_risk_suggestion`
- `codex_confidence`
- `codex_rationale`
- `codex_cited_evidence`

## 非医学专家只填写的证据复核列

非医学专家不做最终临床判断，只检查证据是否存在、Codex 引用是否能在 evidence 字段中找到、Codex rationale 是否被证据支持，以及是否有明显错误或需要医学专家复核。
- `evidence_has_nodule`
- `evidence_has_size`
- `evidence_has_density`
- `evidence_has_location`
- `size_text_present`
- `density_text_present`
- `codex_cited_evidence_exists`
- `codex_rationale_supported_by_evidence`
- `obvious_codex_error`
- `needs_clinical_expert_review`
- `non_clinical_reviewer_notes`

## 推荐流程

1. 将 `module3_codex_pilot20_preannotation_prompt.md` 和 `module3_codex_assisted_pilot_20_template.csv` 提供给 Codex 进行预标注。
2. 将 Codex 输出填回模板中的 Codex suggestion 列。
3. 非医学专家只填写 evidence review 列，重点检查 cited evidence 是否存在、rationale 是否支持、是否有 obvious error。
4. 运行 `scripts/phaseA3/evaluate_codex_assisted_audit.py` 汇总 evidence grounding 和风险标记。
5. 只有 pilot 20 的 evidence grounding 质量足够稳定时，才扩展到完整 133 cases。

## 结果用途

该结果仅用于 failure mode analysis，例如识别 missing density 是否来自 stage1 gating、size_mm 是否缺失、high-risk conflict 是否需要专家复核。若要形成 clinical gold benchmark，必须由具备资质的医学专家复核并明确标注协议。

## 当前限制

本流程不训练模型，不跑 GPU，不接 LLM/NLI/RAG 推理服务，也不输出 learned-model performance table。
