# 模块2 final tables v6 论文呈现报告

> 日期：2026-05-07
> 范围：只重构表格和报告呈现；未启动训练，未修改任何实验结果。

## 1. v6 主表

v6 主表沿用 v5 learned-model 主表口径：正文只保留 learned models，即学习模型；cue-only 和 P2 deterministic hybrid 继续排除。Ours Density Stage 1 使用 `mws_cfe_density_stage1_results_density_final_g3_len128_seed*.json`，Density Stage 2 继续使用 `mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json`，不使用 len192。

| Method | Density Stage 1 F1 | Density Stage 1 AUPRC | Density Stage 1 AUROC | Density Stage 2 Macro-F1 | Has-size F1 | Location Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | 59.77 +/- 0.00 | 63.70 +/- 0.00 | 93.04 +/- 0.00 | 79.11 +/- 0.00 | 87.82 +/- 0.00 | 49.41 +/- 0.00 |
| TF-IDF + SVM | 60.37 +/- 0.00 | 65.43 +/- 0.00 | 92.61 +/- 0.00 | 81.41 +/- 0.00 | 87.60 +/- 0.00 | 47.80 +/- 0.00 |
| TF-IDF + MLP | 62.91 +/- 0.28 | 65.63 +/- 0.98 | 93.15 +/- 0.13 | 78.08 +/- 0.36 | 85.32 +/- 1.26 | 60.57 +/- 1.09 |
| Vanilla PubMedBERT | 88.68 +/- 7.87 | 98.70 +/- 0.56 | 99.75 +/- 0.20 | 91.19 +/- 0.82 | 82.14 +/- 1.01 | 97.39 +/- 0.53 |
| SciBERT | 93.50 +/- 3.06 | 99.23 +/- 0.18 | 99.74 +/- 0.05 | 90.54 +/- 1.04 | 83.28 +/- 1.03 | 97.52 +/- 0.64 |
| BioClinicalBERT / ClinicalBERT | 90.58 +/- 1.61 | 98.47 +/- 0.84 | 99.73 +/- 0.15 | 90.56 +/- 0.91 | 80.94 +/- 0.73 | 97.33 +/- 0.23 |
| MWS-CFE (Ours; final) | **98.43 +/- 0.45** | **99.36 +/- 0.39** | **99.88 +/- 0.09** | **96.89 +/- 0.32** | **99.28 +/- 0.02** | **98.35 +/- 0.71** |

Ours 是否在正文 learned-model 主表所有主指标上最优：**是**。

## 2. 主表 primary metrics 口径

主表不同任务使用不同 primary metrics，即主要评价指标，是因为任务定义和可解释性不同。

1. Density Stage 1 是二分类证据检测；本文做了 threshold calibration，即阈值校准，因此正文报告 F1、AUPRC 和 AUROC。
2. Density Stage 2 是多类别密度亚型分类，因此正文报告 Macro-F1，避免大类掩盖小类表现。
3. Has-size 是二分类字段抽取，因此正文报告 F1，直接反映 has_size 正类抽取质量。
4. Location 是多类别位置抽取，因此正文报告 Macro-F1，避免频繁肺叶类别主导平均结果。
5. Accuracy、Precision、Recall、AUPRC、AUROC 等完整指标进入附录，避免正文主表过宽。

## 3. 任务级完整指标附录

v6 将 appendix full metrics 拆成 4 张任务级表，而不是继续使用一张大总表。原因是四个任务可计算和应报告的指标不同；拆表后每张表只包含该任务适用指标，不再出现大量空白列。若旧结果 JSON 未提供某个适用指标，表中写作 N/A，表示未计算而非任务不适用。

CSV 路径：
- `outputs/phaseA2_planB/final_tables_v6/appendix_density_stage1_full_metrics.csv`
- `outputs/phaseA2_planB/final_tables_v6/appendix_density_stage2_full_metrics.csv`
- `outputs/phaseA2_planB/final_tables_v6/appendix_size_full_metrics.csv`
- `outputs/phaseA2_planB/final_tables_v6/appendix_location_full_metrics.csv`

LaTeX 路径：
- `outputs/phaseA2_planB/final_tables_latex_v6/appendix_density_stage1_full_metrics.tex`
- `outputs/phaseA2_planB/final_tables_latex_v6/appendix_density_stage2_full_metrics.tex`
- `outputs/phaseA2_planB/final_tables_latex_v6/appendix_size_full_metrics.tex`
- `outputs/phaseA2_planB/final_tables_latex_v6/appendix_location_full_metrics.tex`

## 4. 消融表拆分

Density ablation 和 Has-size Wave5 component analysis 在 v6 中分开。Density ablation 是 Full vs w/o 格式，用来解释 Stage 1/Stage 2 density pipeline 的关键组件；Has-size Wave5 表是组件诊断，不伪装成严格 5-seed ablation，其中 `lexical + BERT` 和 `lexical + cue` 明确标为 seed42 诊断结果。

Density ablation：`outputs/phaseA2_planB/final_tables_v6/density_ablation_table_final.csv`，共 5 行。

Has-size Wave5 component analysis：`outputs/phaseA2_planB/final_tables_v6/size_wave5_component_table_final.csv`，共 4 行。

## 5. 参数表拆分

Density 参数表只讨论 P1/P2/P3，并只保留 Density Stage 1 AUPRC 与 Density Stage 2 Macro-F1。这里不放 Has-size F1，也不放 Density Stage 1 F1，因为很多参数设置没有统一做 P0 threshold tuning，F1 不适合作为公平参数扫描指标。

P2 quality gate 和 P3 section/input strategy 是 categorical design choices，即类别型设计选择，不是连续数值参数；因此 v6 报告把它们作为设计选项比较，而不是数值超参数曲线。

Density parameter table：`outputs/phaseA2_planB/final_tables_v6/density_parameter_table_final.csv`，共 16 行。

Has-size Wave5 diagnostic parameter table：`outputs/phaseA2_planB/final_tables_v6/size_wave5_diagnostic_parameter_table.csv`，共 6 行。

## 6. 参数图进入正文和附录

正文推荐 3 张参数图：
- `p1_max_seq_length_stage_2_macro_f1.svg`
- `p2_quality_gate_stage_2_macro_f1.svg`
- `p3_section_input_strategy_stage_2_macro_f1.svg`

附录放 3 张参数图：
- `p1_max_seq_length_stage_1_auprc.svg`
- `p2_quality_gate_stage_1_auprc.svg`
- `p3_section_input_strategy_stage_1_auprc.svg`

如果版面紧张，正文只保留 `p2_quality_gate_stage_2_macro_f1.svg` 和 `p3_section_input_strategy_stage_2_macro_f1.svg`；P1 图移入附录。完整索引见 `outputs/phaseA2_planB/final_figures_v6/figure_manifest.csv`。

## 7. 最终判断

模块2是否还需要补实验：**不需要**。当前已有结果足够支持 Module 2 v6 表格封板；不需要重新训练或补 Ours 实验。
