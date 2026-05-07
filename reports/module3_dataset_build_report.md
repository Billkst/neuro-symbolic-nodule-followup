# Module 3 M3-2 数据集构建报告

日期：2026-05-07

范围：本轮只做数据构建、事实缺失诊断、module2-to-case_bundle 事实适配、CDSG strong-silver 标签生成和 case/patient-level split。未训练、未跑 GPU、未接 LLM/NLI/RAG。

## 1. M3-1 Abstention 子集

输入为 M3-1 的 `cdsg_phase4_recommendations.jsonl`、CDSG vs flat comparison/mismatch 表和原始 Phase4 `case_bundles_eval.jsonl`。共 253 条。

| subset | count | fraction |
| --- | ---: | ---: |
| strict_hard_comparable | 20 | 0.079051 |
| missing_density_abstention | 122 | 0.482213 |
| missing_size_abstention | 76 | 0.300395 |
| no_structured_nodule_abstention | 7 | 0.027668 |

strict_hard_comparable 的定义沿用 M3-1：CDSG 非 abstention，且 flat baseline 的 missing_information 不包含 density_category。

## 2. Module2-to-Case Bundle 适配结果

适配层输出 `outputs/phaseA3/datasets/module3_ready_case_bundles.jsonl`，共 253 条，全部通过 `module3_case_bundle_schema`。

关键统计：

| metric | value |
| --- | ---: |
| Phase4 cases | 253 |
| Phase4 radiology notes | 500 |
| Phase4-aligned module2 mention rows | 2018 |
| cases with module2 candidates | 199 |
| appended candidate nodules | 978 |
| candidates with density | 213 |
| candidates with size | 648 |
| candidates with location | 724 |
| unmatched rows | 1040 |

重要限制：模块2最终 density/location 结果 JSON 只有聚合指标，没有逐样本 prediction/logit 文件。因此本轮适配没有伪造逐样本模型输出，而是把 Phase5 mention 数据集中的逐 mention fact 字段作为可追溯来源追加到 case_bundle；Has-size Wave5 可用逐样本概率，使用 `size_wave5_lexical_alone` 概率作为 size 事实置信度来源。所有追加字段均记录 source、confidence、model_tag、mention_id、original_text 和 source path。

适配层不覆盖原始 case_bundle 的结节事实，也不从文本规则推断缺失字段。无法可靠使用的 mention 已写入 `module2_to_case_bundle_unmatched.csv`，主要原因是：

| reason | count |
| --- | ---: |
| no_direct_module2_fact | 604 |
| no_pulmonary_nodule_cue | 258 |
| non_pulmonary_mention_filtered | 178 |

## 3. Strong-Silver 标签

在 module3-ready case_bundle 上重新运行 Lung-RADS v2022 CDSG MVP，生成 `module3_strong_silver.jsonl`，共 253 条，recommendation schema valid rate 为 1.0。

M3-1 原始 CDSG actionable 为 48，abstention 为 205。M3-2 适配后 actionable 为 51，abstention 为 202。降低幅度很小，说明当前瓶颈仍然是逐 mention density/size 的可用事实覆盖，而不是 CDSG executor。

剩余缺失：

| missing / abstention | count |
| --- | ---: |
| missing_nodule_density | 129 |
| missing_nodule_size | 66 |
| no_structured_nodule | 7 |

注意：missing_density 从 M3-1 的 122 增至 129，是因为适配层追加了部分有 size 但 density 仍不明确的候选结节；CDSG 在全 abstention 情况下会优先返回更接近决策边界的缺失原因。因此 size 缺失下降但 density 缺失上升，净 actionable 只增加 3 条。

strong-silver label 分布：

| label | count |
| --- | ---: |
| insufficient_data | 202 |
| routine_screening / Lung-RADS 2 | 37 |
| short_interval_followup | 12 |
| tissue_sampling | 2 |
| Lung-RADS 3 | 7 |
| Lung-RADS 4A | 5 |
| Lung-RADS 4B | 2 |

## 4. Split 结果

输出：

| split | count |
| --- | ---: |
| train | 176 |
| val | 38 |
| test | 39 |

split 单位为 patient_id/subject_id，若缺失则退化为 case_id。本轮共有 253 个 patient/case groups，split leakage count 为 0。

标签极不平衡：actionable 只有 51 条，最小 actionable category 只有 2 条。因此当前 split 可以用于 M3-3 hard-match agent 的数据管线和确定性参考评估，但不适合作为 learned model 正式训练集。

## 5. 是否进入 M3-3

可以进入 M3-3 hard-match agent 主实验的基础设施阶段：CDSG graph、executor、module3-ready bundle、strong-silver label 和 split 都已具备。

不建议进入训练或 learned-model 主表实验。主要风险包括：

1. density/location 缺少逐样本最终模型预测文件，本轮只能使用 Phase5 mention fact 字段并严格标注来源。
2. strong-silver actionable 样本少，且 Lung-RADS 4B 只有 2 条。
3. 适配后 abstention 仍高达 202/253。
4. 多 mention 到 case 的候选结节尚未做人工核查，dominant selection 可能受低质量 candidate 影响。

仍需要人工 gold label 或小规模人工审查集。建议优先审查 51 条 actionable strong-silver、全部 7 条 no_structured_nodule、以及 missing_density/missing_size 中各抽样 30 条，用于确认 module2-to-case_bundle 对齐质量和 CDSG strong-silver 标签可靠性。
