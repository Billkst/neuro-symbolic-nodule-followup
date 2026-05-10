# M3-3B Manual Audit Set Design

## 定位

M3-3B 构建的是人工审查集，不是训练集，也不是最终 gold benchmark。本轮不训练、不调用 LLM/NLI/RAG，也不改变 conservative v2 case bundle。审查集的目标是验证 CDSG hard-match recommendation、abstention、candidate conflict 与后续 soft matching / safer aggregation 设计是否合理。

## 为什么需要人工 audit set

M3-3A 显示 CDSG agent 的 schema、guideline anchor、reasoning path 和 decision path 都完整，但 actionable 仅 76/253，abstention 主要来自 missing density 和 missing size。仅靠 CDSG strong-silver 不能判断 abstention 是否临床合理，也不能判断 density conflict 是否会改变 Lung-RADS 路径，因此需要人工抽查。

## Audit group 设计

- actionable_recommendation: 76 cases。全部纳入，用于检查 CDSG 终态 recommendation 是否合规。
- high_risk_density_conflict: 19 cases。全部纳入，用于判断 density subtype 冲突是否改变 Lung-RADS 路径。
- no_structured_nodule: 7 cases。全部纳入，用于判断是否确实没有可靠 pulmonary nodule candidate。
- missing_density_priority_sample: 30 cases。优先选择 stage2 有 subtype 但 stage1 非 explicit、或存在 density candidate 但未被 CDSG 使用的样本。
- missing_size_priority_sample: 20 cases。优先选择存在 size evidence / pulmonary cue 但无法形成 size_mm hard fact 的样本。

## 去重结果

- Full set: 253 cases。
- Raw selection events: 152。
- Deduplicated audit set: 133 cases。
- Overlap removed: 19 events。

## 如何使用 audit 结果

- 如果 missing_density 中人工确认 density subtype 明确，但 stage1 标为 non-explicit，应优先修 density stage1 gating。
- 如果 high-risk density conflict 多数来自不同结节混合，不能采用 same-case union 或 dominant size-first aggregation。
- 如果 confidence-gated same-note aggregation 在人工审查中事实错配率低，可作为 v2.1 conservative extension 候选。
- 如果 no_structured_nodule 中人工发现明显 nodule cue，需要回查 mention alignment 和 pulmonary cue filtering。
- 如果人工审查显示 current 76 actionable 的 recommendation 合规，可进入 M3-3C / M3-3 soft matching 设计；否则先修 CDSG fact routing。

## 当前判断

当前仍不建议进入 learned-model 主实验。原因是可行动样本少、类别不平衡，并且 strong-silver label 来自 CDSG 自身，不能作为公平性能标签。下一步更适合进入人工审查执行，或基于人工结果设计 soft matching / safer aggregation。

## 输出文件

- Audit CSV: `outputs/phaseA3/audit_sets/module3_manual_audit_set.csv`
- Audit JSONL: `outputs/phaseA3/audit_sets/module3_manual_audit_set.jsonl`
- Label template: `outputs/phaseA3/audit_sets/module3_manual_audit_label_template.csv`
- Summary: `outputs/phaseA3/tables/module3_manual_audit_set_summary.csv`

## Source pool

- actionable_recommendation: 76
- high_risk_density_conflict: 19
- missing_density_pool: 104
- missing_density_priority_sample: 30
- missing_size_pool: 66
- missing_size_priority_sample: 20
- no_structured_nodule: 7
