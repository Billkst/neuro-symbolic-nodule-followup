# M3-3C 模块3人工审查标注指南

## 定位

本指南服务于 `module3_manual_audit_set` 的人工审查。该 audit set 不是训练集，也不是最终 gold benchmark；它用于判断 conservative CDSG agent 的 recommendation、abstention、candidate conflict 与后续 soft matching / safer aggregation 是否合理。

建议先标注 pilot 20，再根据一致性和字段可读性决定是否标完整 133 条。本轮生成的 pilot 文件包含 20 条。

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
