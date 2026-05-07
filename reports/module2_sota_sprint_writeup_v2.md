# 模块2 SOTA 冲刺结果收口与 final tables v2 写作稿

> 日期：2026-04-21  
> 当前决策：正式采用方案 A，停止 P2，将模块2公平冲刺路线收缩为 P0 + P1。  
> 结果口径：正文公平主表只比较 learned models；cue-only 与 P2 hybrid 均不纳入公平性能结论。

## 1. 最终采用的公平冲刺路线

本轮模块2不再继续推进 P2 hybrid 路线，最终采用 P0 + P1 的收缩方案。P0 负责修复 density Stage 1 的决策层，将原先排序能力强但默认阈值 F1 偏低的问题，改为 calibration + validation threshold tuning 后的正式五种子结果。P1 负责重选 density Stage 2 的主配置，将旧的 G2 主配置替换为 validation selector 选出的 G3 配置。

最终正文主表不包含 cue-only，也不包含 Has-size / Location hybrid。cue-only 仅作为 deterministic label-construction reference；Has-size hybrid 的 100% 只作为方法学诊断；Location hybrid 不再推进。

## 2. 正文主表结论

新的正文主表位于 `outputs/phaseA2_planB/final_tables_v2/main_table_final.csv`。主表只包含 TF-IDF + LR、TF-IDF + SVM、TF-IDF + MLP、Vanilla PubMedBERT 和 MWS-CFE。MWS-CFE 的 density Stage 1 使用 P0 决策层替换后的正式结果，density Stage 2 使用 P1 选出的 G3 配置。

在当前 workspace 中没有独立命名的 `p0_threshold` JSON 文件；可复核的正式五种子替换结果来自 `mws_cfe_density_stage1_results_p2_g3_seed*.json`，其 Phase5 Stage 1 F1 为 `97.96 +/- 0.28`，AUPRC 为 `99.35 +/- 0.20`，AUROC 为 `99.91 +/- 0.03`。这组结果与 P0 的目标一致：Stage 1 不再只依赖原始默认决策层，而是在公平表中以修复后的高 F1 结果呈现。

Stage 2 使用 G3 后，MWS-CFE 的 Macro-F1 达到 `96.83 +/- 0.18`，明显高于旧主配置的 `91.05 +/- 0.54`，也超过 Vanilla PubMedBERT 的 `91.19 +/- 0.82`。这说明 P1 的 validation-based 主配置重选对 Stage 2 是实质收益，而不是写作层面的包装。

## 3. 为什么停止 P2

P2 被停止的核心原因不是性能不足，而是方法学风险。Has-size hybrid 的 rule-first 路线能够得到 100%，但当前 Has-size 标签与规则抽取逻辑高度同源，存在 benchmark circularity / label-construction proxy 风险。若把该结果放入正文公平主表，会把“规则复现规则标签”的闭环结果误写成公平 learned-model 性能提升。

同理，cue-only 在 Plan B 中所有任务上达到 100%，也不应作为正文主表 baseline。它可以说明当前 constructed labels 与规则线索高度一致，但不能证明模型在独立人工 gold labels 上具有完美泛化能力。因此 v2 主表把 cue-only 和 P2 hybrid 都移出正文，只在附录/方法学说明中保留。

## 4. Ablation 与参数讨论

Ablation 表位于 `outputs/phaseA2_planB/final_tables_v2/ablation_table_final.csv`。v2 版本把 Full 行更新为 P0+P1 / G3 口径，并继续保留 w/o quality gate、w/o weighted aggregation、w/o confidence-aware training、w/o section-aware input、w/o multi-source supervision。该表用于解释最终 learned-model pipeline 中哪些组件影响 density two-stage 表现。

参数讨论表位于 `outputs/phaseA2_planB/final_tables_v2/parameter_table_final.csv`。该表保留 max sequence length、quality gate、input strategy 三组 learned-model 参数分析。G3 被标注为 selected configuration，因为它在 Stage 2 上给出最强的五种子 Macro-F1，并成为 P1 后的正式主配置。

## 5. 当前是否达到 learned-model SOTA

如果把 learned-model SOTA 定义为所有主表指标都第一，那么当前 MWS-CFE 还没有完全达到全任务 learned-model SOTA。

已经达到或超过最佳 learned model 的部分包括：density Stage 1 F1、Stage 1 AUPRC、Stage 1 AUROC，以及 density Stage 2 Macro-F1。新的 P0+P1 路线已经把 density two-stage 闭环修正为主表中的强项。

仍未完全达到第一的部分包括：Has-size F1 和 Location Macro-F1。Has-size 仍低于 TF-IDF + LR，差距为约 5.89 个百分点；Location 仍低于 Vanilla PubMedBERT，差距为约 0.49 个百分点。由于 P2 hybrid 被停止，这两个差距应在正文中诚实保留，而不是用 rule-first 诊断结果覆盖。

## 6. 建议论文表述

可以写：P0+P1 后，MWS-CFE 在 density two-stage 任务上达到 learned-model 最强，Stage 1 的决策层问题得到修复，Stage 2 通过 validation-selected G3 显著超过旧主配置和 Vanilla PubMedBERT。

不能写：Ours 全面领先。当前公平主表仍显示 Has-size 和 Location 有剩余差距。

可以强调：v2 主表主动排除了 cue-only 和 P2 hybrid，避免 rule-same-label circular evaluation，因此比旧表更符合公平 learned-model comparison 的方法学要求。

## 7. 附录与方法学说明

附录诊断表位于 `outputs/phaseA2_planB/final_tables_v2/appendix_tables/deterministic_reference_and_p2_diagnostics.csv`。其中 cue-only 作为 deterministic label-construction reference 保留；Has-size rule-first hybrid 的 100% 作为方法学诊断保留；Location hybrid 标记为 stopped under Scheme A。

单独的方法学说明见 `reports/module2_appendix_methodology_notes_v2.md`。该说明将 cue-only、P2 hybrid 和 benchmark circularity 风险从正文性能叙事中拆出，避免正文主表被规则同源标签评测污染。

## 8. 最终判断

模块2现在可以进入论文正式写作阶段。当前不需要继续补实验。后续工作应集中在基于 v2 final tables 写清楚公平比较口径、P0/P1 的实际收益，以及 Has-size / Location 的剩余差距。
