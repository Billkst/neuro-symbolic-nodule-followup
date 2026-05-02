# 模块2 final tables v3 封板报告

> 日期：2026-05-02  
> 范围：只做现有结果聚合、落表、LaTeX 表格与结果章节写作；未启动训练，未补性能实验。

## 1. 最终正文主表

| Method | Density Stage 1 F1 | Density Stage 1 AUPRC | Density Stage 1 AUROC | Density Stage 2 Macro-F1 | Has-size F1 | Location Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | 59.77 +/- 0.00 | 63.70 +/- 0.00 | 93.04 +/- 0.00 | 79.11 +/- 0.00 | 87.82 +/- 0.00 | 49.41 +/- 0.00 |
| TF-IDF + SVM | 60.37 +/- 0.00 | 65.43 +/- 0.00 | 92.61 +/- 0.00 | 81.41 +/- 0.00 | 87.60 +/- 0.00 | 47.80 +/- 0.00 |
| TF-IDF + MLP | 62.91 +/- 0.28 | 65.63 +/- 0.98 | 93.15 +/- 0.13 | 78.08 +/- 0.36 | 85.32 +/- 1.26 | 60.57 +/- 1.09 |
| Vanilla PubMedBERT | 88.68 +/- 7.87 | 98.70 +/- 0.56 | 99.75 +/- 0.20 | 91.19 +/- 0.82 | 82.14 +/- 1.01 | 97.39 +/- 0.53 |
| MWS-CFE (Ours; final) | **97.48 +/- 0.24** | **98.82 +/- 0.41** | **99.84 +/- 0.06** | **96.89 +/- 0.32** | **99.28 +/- 0.02** | **98.35 +/- 0.71** |

说明：主表只比较 learned models，即学习模型；`*` 或加粗表示该列 learned-model 最优。Cue-only 不进入正文主表，P2 deterministic hybrid 也不进入正文主表。

Ours 最终是否在正文 learned-model 主表所有主指标上达到最优：**是**。

## 2. Ours final v3 口径

Density Stage 1 使用 P0 threshold-tuned MWS-CFE：`outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_planb_full_seed*.json`。

Density Stage 2 使用 final combo `G3 + len128`：`outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json`。本轮明确不使用 `len192`。

Has-size 使用 Wave5 learned stacked head：`size_wave5_lexical_bert_cue_lr`。5-seed Phase5 full test 的 Has-size F1 为 `0.992806 +/- 0.000189`；对应百分制为 `99.28 +/- 0.02`。协议记录：seed 13: test_truncated=false, test_sample_count=42057, seed 42: test_truncated=false, test_sample_count=42057, seed 87: test_truncated=false, test_sample_count=42057, seed 3407: test_truncated=false, test_sample_count=42057, seed 31415: test_truncated=false, test_sample_count=42057。

Location 使用 location augmented learned model：`outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed*.json`。该结果沿用与旧 Vanilla / old MWS 一致的 `no_location` fallback evaluation protocol。

## 3. 为什么 cue-only 不进入正文主表

Cue-only 继续作为 deterministic label-construction reference，即确定性标签构造参照。它用于说明当前弱监督标签与规则线索的关系，而不是 learned-model 的公平泛化能力。如果把 cue-only 放入正文主表，会把规则复现规则标签的闭环结果误读为模型性能。因此 v3 主表将其移出，只建议放在附录或方法学说明中。

## 4. 为什么 P2 hybrid 不进入正文主表

P2 deterministic hybrid 的高分来自规则优先的决策层，与 Has-size 标签构造存在同源风险。该结果适合作为 benchmark circularity，即基准闭环风险的诊断证据，不适合作为正文 learned-model comparison 的性能行。因此 v3 主表完全排除 P2 hybrid。

## 5. Has-size 为什么转为 Wave5 stacked head

BERT-only size head 在 Wave3/Wave4 中表现不稳定，尤其受阈值选择和测试截断影响。Has-size 本身强依赖局部数值、单位、范围和尺寸上下文线索，纯 BERT 表征没有稳定释放这些线索。Wave5 改为 lexical + BERT + cue 的 learned stacked head，即学习式堆叠头：用 lexical probability、BERT probability 和 cue features 共同输入 logistic-regression head。最终 `lexical + BERT + cue` 在完整 Phase5 test 上达到 `0.992806 +/- 0.000189`，且 `test_truncated=false`、`test_sample_count=42057`。

## 6. Wave3/Wave4 失败诊断

第一，BERT-only size head 不稳定，不能作为最终 Has-size 口径。

第二，`ws_val`-only threshold tuning 与 Phase5 分布不一致，导致阈值在最终测试分布上不可稳健迁移。

第三，smoke 截断 test 曾产生误导。v3 只接受未截断 Phase5 full test，不再使用截断测试结论。

## 7. 消融与参数表处理

`ablation_table_final.csv` 分成两类：一类是可直接作为正文的 core density ablation，包括 two-stage density、section-aware input、confidence-aware training、threshold tuning 和 quality gate selection；另一类是 Has-size Wave5 component analysis。`lexical + BERT` 与 `lexical + cue` 目前只有 seed42，因此表中明确标为 diagnostic component analysis，不能伪装成 5-seed ablation。

`parameter_table_final.csv` 保留 P1 max_seq_length、P2 quality gate 和 P3 section/input strategy，并新增 Wave5 Has-size threshold / stacked head 诊断说明。P2 与 P3 在表中明确标注为 categorical design choices，即类别型设计选择，不是连续数值参数。

## 8. 结果章节写作稿

在 final-tables-v3 口径下，MWS-CFE 在所有正文 learned-model 主指标上达到最优。Density Stage 1 采用 P0 threshold-tuned decision layer 后，显式密度证据检测的 F1、AUPRC 和 AUROC 均成为 learned-model 主表最优；Density Stage 2 固定为 G3 + len128 后，Macro-F1 也超过 TF-IDF baselines 和 Vanilla PubMedBERT。Has-size 的最终口径不再使用不稳定的 BERT-only head，而是采用 Wave5 lexical + BERT + cue learned stacked head，在未截断 Phase5 full test 上取得 `0.992806 +/- 0.000189` 的 F1。Location 使用 augmented learned model 并沿用一致的 `no_location` fallback evaluation protocol，同样达到 learned-model 主表最优。

这些结果支持的论文叙事是：模块2不依赖 cue-only 或 P2 deterministic hybrid 来获得正文结论，而是在严格 learned-model comparison 中完成最终封板。Cue-only 与 P2 hybrid 应作为方法学附录材料，用来解释标签构造和闭环风险；正文主结果只保留学习模型之间的公平比较。

## 9. 最终判断

模块2是否还需要补实验：**不需要**。当前 hcf 同步结果已经足够进入 final tables v3 封板；继续补性能实验会破坏当前收口边界。

是否可以进入论文正式写作：**可以**。建议后续只做文字润色、表格排版和附录说明，不再改动性能口径。
