# M3-2.5 模块2逐样本预测导出与 case_bundle 事实恢复报告

本轮只执行审计、导出脚本、adapter v2、CDSG strong-silver v2 dry-run。未训练、未跑 GPU、未接 LLM/NLI/RAG，也未在本地加载 Transformer 模型推理。

## 1. 模块2逐样本预测是否可导出

当前本地只能完整导出 size/Has-size Wave5 的逐样本概率；density stage1、density stage2、location 的 PhaseA2 final result 均存在，但本地缺少对应 final model 目录，结果 JSON 只含聚合指标，没有 final model 逐样本 prediction/logit 文件。

审计产物见 `outputs/phaseA3/tables/module2_prediction_availability.csv`。关键结论：

| task | final result | local final model | existing exported rows | final-model rows | 结论 |
|---|---:|---:|---:|---:|---|
| density_stage1 | yes | no | 2018 | 0 | 只能导出 constructed stage1 fact；需 HCF 重导出 final model predictions |
| density_stage2 | yes | no | 2018 | 0 | 只能导出 constructed subtype fact；需 HCF 重导出 final model predictions |
| size | yes | yes | 2018 | 523 | Has-size 概率部分可用；size_mm 只能来自已有 mention fact |
| location | yes | no | 2018 | 0 | 只能导出 constructed location fact；需 HCF 重导出 final model predictions |

## 2. 本地 prediction export 结果

导出脚本为 `scripts/phaseA3/export_module2_predictions_for_module3.py`。本地 dry-run 默认只导出 Phase4 case bundle 可对齐的 mention，避免生成全量 Phase5 大文件。

输出：

- `outputs/phaseA3/module2_predictions/module2_density_stage1_predictions.jsonl`: 2018 行，全部为 `constructed_fact_not_final_model`
- `outputs/phaseA3/module2_predictions/module2_density_stage2_predictions.jsonl`: 2018 行，全部为 `constructed_fact_not_final_model`
- `outputs/phaseA3/module2_predictions/module2_size_predictions.jsonl`: 2018 行，其中 523 行有 Wave5 final probability，1495 行无概率、仅 constructed fact
- `outputs/phaseA3/module2_predictions/module2_location_predictions.jsonl`: 2018 行，全部为 `constructed_fact_not_final_model`

如需真正恢复 density/location final model predictions，应在 HCF 上同步模型目录后运行：

```bash
conda run -n follow-up python scripts/phaseA3/export_module2_predictions_for_module3.py --allow-model-inference --tasks density_stage1,density_stage2,location --output-dir outputs/phaseA3/module2_predictions --batch-size 128 --max-length 128
conda run -n follow-up python scripts/phaseA3/adapt_module2_predictions_to_case_bundle_v2.py
conda run -n follow-up python scripts/phaseA3/build_module3_strong_silver_labels.py --input outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl --output outputs/phaseA3/datasets/module3_strong_silver_v2.jsonl --summary outputs/phaseA3/tables/module3_strong_silver_v2_summary.csv --label-distribution outputs/phaseA3/tables/module3_strong_silver_v2_label_distribution.csv
```

## 3. mention-to-case 对齐

adapter v2 使用 `note_id` 对齐 Phase4 case bundle。导出的 Phase4-aligned prediction 行全部可对齐：

- exported rows per task: 2018
- case-aligned rows per task: 2018
- alignment rate: 1.000000
- Phase4 radiology notes: 500
- aligned unique mentions: 2018

adapter v2 产物：

- `outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl`
- `outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_summary.csv`
- `outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_unmatched.csv`
- `outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_conflicts.csv`

adapter v2 结果：

- output cases: 253
- cases with module2 candidates: 200
- candidate nodules appended: 1122
- note-level aggregate candidates appended: 119
- candidate_with_density: 323
- candidate_with_size: 756
- candidate_with_location: 840
- conflict rows: 289
- unmatched rows: 1015

unmatched 主要原因：

- no_direct_module2_fact: 604
- non_pulmonary_mention_filtered: 247
- no_pulmonary_nodule_cue: 164

conflict 没有静默覆盖；adapter v2 记录字段冲突，并将 selected value 作为候选结节的 dominant selection 字段保留。

## 4. v2 是否降低 abstention

CDSG strong-silver v2 生成结果：

- total: 253
- schema valid rate: 1.000000
- actionable: 75
- abstention: 178
- guideline anchor non-empty rate: 1.000000
- reasoning path non-empty rate: 1.000000

v1 vs v2：

| version | actionable | abstention | missing_density | missing_size | no_structured_nodule |
|---|---:|---:|---:|---:|---:|
| v1 | 51 | 202 | 129 | 66 | 7 |
| v2 | 75 | 178 | 105 | 66 | 7 |
| change | +24 | -24 | -24 | 0 | 0 |

v2 显著降低了 density-driven abstention，但提升来自跨 mention candidate aggregation 和已有 constructed density facts 的更好组合，不来自 density/location final model predictions。

## 5. v2 label 分布

| dimension | label | count |
|---|---|---:|
| recommendation_level | insufficient_data | 178 |
| recommendation_level | routine_screening | 54 |
| recommendation_level | short_interval_followup | 17 |
| recommendation_level | tissue_sampling | 4 |
| lung_rads_category | 2 | 54 |
| lung_rads_category | 3 | 9 |
| lung_rads_category | 4A | 8 |
| lung_rads_category | 4B | 4 |
| lung_rads_category | None | 178 |

剩余无法决策的主要缺失：

- density_category: 105
- nodule_size: 66
- nodule: 7
- solid_component_mm: 1
- smoking_eligibility: 242

其中 smoking_eligibility 当前不是 CDSG MVP 的主要 abstention reason，但会影响后续准入/风险分层扩展。

## 6. 是否可以进入 M3-3

可以进入 M3-3 hard-match agent 的确定性管线阶段：CDSG executor、schema、adapter v2、strong-silver v2 均已跑通 253 cases，且 schema valid rate 为 1.0。

但不建议进入 learned-model 主实验，也不建议把当前 v2 当作 predicted-module2 final-model facts 的最终版本。原因是 density/location 仍然没有真正的 final model 逐样本预测，本地 v2 的事实恢复仍以 constructed facts 为主。

进入 M3-3 前的建议边界：

- 可做：hard-match CDSG agent、路径有效性、schema validity、abstention analysis、deterministic reference。
- 暂缓：learned recommendation model、LLM/RAG/NLI fallback、公平主表 learned baseline。
- 如 M3-3 需要 predicted module2 final facts，应先在 HCF 跑 density/location prediction export，再重跑 adapter v2 和 strong-silver v2。

## 7. 是否需要人工审查集

仍然需要。建议最小人工审查集：

- 75 个 v2 actionable cases 全量审查，确认 note-level fact aggregation 是否错误合并不同结节。
- 105 个 missing_density 中抽样 30 个，判断是否是 final model 可恢复还是报告确实缺失。
- 66 个 missing_size 中抽样 30 个，判断 size fact 是否被 has-size threshold 或 mention 对齐丢失。
- 289 个 conflict rows 中抽样 50 个，重点检查跨 mention 聚合的临床合理性。
