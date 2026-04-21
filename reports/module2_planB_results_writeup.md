# 模块2 Plan B 结果验收、论文落表与结果章节草稿

> 日期：2026-04-21  
> 阶段：结果审计、论文版 final tables、结果章节写作  
> 数据来源：`outputs/phaseA2_planB/tables` 与 `outputs/phaseA2_planB/figures`  
> 本报告只做审表、落表和写作，不补跑实验，不切换模块3。

## 1. 新增/更新文件

正文 CSV：

1. `outputs/phaseA2_planB/final_tables/main_table_final.csv`
2. `outputs/phaseA2_planB/final_tables/ablation_table_final.csv`
3. `outputs/phaseA2_planB/final_tables/parameter_table_final.csv`

附录 CSV：

1. `outputs/phaseA2_planB/final_tables/appendix_tables/main_table_full_metrics_with_seed_counts.csv`
2. `outputs/phaseA2_planB/final_tables/appendix_tables/best_method_audit.csv`
3. `outputs/phaseA2_planB/final_tables/appendix_tables/ablation_table_with_tags.csv`
4. `outputs/phaseA2_planB/final_tables/appendix_tables/parameter_table_with_tags.csv`
5. `outputs/phaseA2_planB/final_tables/appendix_tables/manifest_completeness_summary.csv`
6. `outputs/phaseA2_planB/final_tables/appendix_tables/figure_manifest.csv`

LaTeX：

1. `outputs/phaseA2_planB/final_tables_latex/main_table.tex`
2. `outputs/phaseA2_planB/final_tables_latex/ablation_table.tex`
3. `outputs/phaseA2_planB/final_tables_latex/parameter_table.tex`
4. `outputs/phaseA2_planB/final_tables_latex/appendix_tables.tex`

报告：

1. `reports/module2_planB_results_writeup.md`

## 2. 结果审计

### 2.1 完整性

`planb_manifest_report.json` 显示 `records=310`，`unmatched_files=0`，manifest 条目 `68` 个，其中 incomplete 条目 `0` 个。预期 seeds 为 `13, 42, 87, 3407, 31415`。因此，Plan B 当前没有缺 seed、缺文件或聚合未闭环的硬缺口。

`outputs/phaseA2_planB/figures/` 下共有 `6` 个 SVG，覆盖 P1、P2、P3 三组参数，每组都有 Stage 1 AUPRC 和 Stage 2 Macro-F1 图。所有 SVG 均非空，字节数已写入 `figure_manifest.csv`。

### 2.2 主表是否支持 “Ours 全面领先”

不支持。必须正面承认：当前主表不能写成 “Ours 全面领先”。

全部方法比较下，`Cue-only rules` 在所有主表指标上均为 `100.00 +/- 0.00`，因此全部方法中的最强者是 cue-only/rule baseline。这个结果说明当前标签构造与显式规则线索高度一致，也说明 cue-only 必须被解释为确定性规则参照，而不是可学习模型。

排除 cue-only/rule baseline 后，学习模型中的最强者按指标分裂：

1. Stage 1 AUPRC：`MWS-CFE (Ours)` 最强，`98.82 +/- 0.41`。
2. Stage 1 AUROC：`MWS-CFE (Ours)` 最强，`99.84 +/- 0.06`。
3. Stage 1 F1：`Vanilla PubMedBERT` 最强，`88.68 +/- 7.87`；Ours 为 `60.29 +/- 6.93`。
4. Stage 2 Macro-F1：`Vanilla PubMedBERT` 最强，`91.19 +/- 0.82`；Ours 为 `91.05 +/- 0.54`，非常接近但不是最高。
5. Has-size F1：`TF-IDF + LR` 最强，`87.82 +/- 0.00`；Ours 为 `81.93 +/- 1.31`。
6. Location Macro-F1：`Vanilla PubMedBERT` 最强，`97.39 +/- 0.53`；Ours 为 `96.90 +/- 0.50`。

因此，论文中可以写：Ours 在学习模型里的 Stage 1 排序指标有优势，Stage 2 与 Vanilla PubMedBERT 接近；但不能写 Ours 全局最强或全面领先。

### 2.3 消融表格式与结论

`planb_ablation_table.csv` 已经符合标准 `Full vs w/o ...` 论文格式。本轮生成的 `ablation_table_final.csv` 删除了 `Tag`、`N` 等工程列，只保留 `Variant`、`Stage 1 AUPRC`、`Stage 2 Macro-F1`。

消融结论不能包装成所有组件均有效。`w/o quality gate` 的两阶段结果高于 Full，`w/o multi-source supervision` 的 Stage 2 Macro-F1 也高于 Full。真正稳定的正向证据是 section-aware input：去掉 section-aware input 后，Stage 1 AUPRC 从 `98.82 +/- 0.41` 降到 `79.33 +/- 4.64`，Stage 2 Macro-F1 从 `91.05 +/- 0.54` 降到 `86.76 +/- 0.32`。

### 2.4 参数讨论覆盖情况

参数讨论已经完整覆盖 P1/P2/P3：

1. P1 `max_seq_length`：64、96、128、160、192。
2. P2 `quality_gate`：G1、G2、G3、G4、G5。
3. P3 `section/input strategy`：mention text、section-aware text、findings text、impression text、findings + impression text、full text。

每个设置均有 Stage 1 AUPRC 与 Stage 2 Macro-F1，且 `N=5`。P2 中 G3 是参数扫描最强设置；P3 中 section-aware text 是当前最稳的主配置；P1 中 192 在 Stage 2 Macro-F1 上最高，但增益有限。

### 2.5 正文与附录建议

适合正文：

1. `main_table_final.csv`：使用紧凑主表，只保留 Stage 1 AUPRC/AUROC、Stage 2 Macro-F1、Has-size F1、Location Macro-F1。
2. `ablation_table_final.csv`：标准 Full vs w/o 消融表。
3. `parameter_table_final.csv`：P1/P2/P3 参数讨论表；若版面紧张，可正文写结论，完整表放附录。

更适合附录：

1. `main_table_full_metrics_with_seed_counts.csv`：完整指标和 seed count。
2. `best_method_audit.csv`：明确区分全部方法最强与排除 cue-only 后学习模型最强。
3. 带 `Source Tag`、`N` 的消融和参数表。
4. manifest 完整性表和 figure manifest。

### 2.6 是否需要补实验

不需要。当前没有真实硬缺口；需要处理的是论文叙事边界，而不是继续训练。

## 3. 结果章节草稿

### 3.1 主结果：two-stage density 闭环

Plan B 将 density 任务从旧的单阶段分类改为 two-stage density protocol。Stage 1 是 explicit density evidence detection，即显式密度证据检测，用于判断报告文本中是否存在明确的密度描述。Stage 2 是 density subtype classification，即密度亚型分类，在显式密度证据成立后区分具体密度类别。该设计将“是否有明确密度证据”和“密度属于哪一类”拆开，避免旧单阶段设置把证据缺失与亚型分类混在同一个标签空间中。

主表显示，cue-only rules 在所有指标上达到 `100.00 +/- 0.00`。这一行不能被当作普通学习模型解读，而应作为确定性规则参照基线。它说明当前 Plan B 标签与显式 cue 的一致性很高，也提醒我们：如果把所有方法放在一起比较，最强方法不是 Ours，而是 cue-only/rule baseline。

排除 cue-only 后，MWS-CFE 在学习模型中的 Stage 1 排序指标上表现最好。Ours 的 Stage 1 AUPRC 为 `98.82 +/- 0.41`，AUROC 为 `99.84 +/- 0.06`，均略高于 Vanilla PubMedBERT 的 `98.70 +/- 0.56` 和 `99.75 +/- 0.20`。这表明，多源弱监督与置信度感知训练对显式密度证据检测的排序能力有帮助。

但这种优势没有转化为全面领先。Stage 1 F1 上，Vanilla PubMedBERT 为 `88.68 +/- 7.87`，明显高于 Ours 的 `60.29 +/- 6.93`。Stage 2 Macro-F1 上，Vanilla PubMedBERT 为 `91.19 +/- 0.82`，Ours 为 `91.05 +/- 0.54`。两者非常接近，但严格最高不是 Ours。因此，论文不能写 “Ours 全面领先”，只能写 “Ours 在学习模型的 Stage 1 排序指标上最强，Stage 2 与 Vanilla PubMedBERT 接近”。

Has-size 和 location 进一步说明不同结构化字段偏好的模型不同。Has-size F1 最高的是 TF-IDF + LR，达到 `87.82 +/- 0.00`，说明尺寸存在性检测主要依赖数字、单位和固定词面模式。Location Macro-F1 最高的是 Vanilla PubMedBERT，达到 `97.39 +/- 0.53`，Ours 为 `96.90 +/- 0.50`，处于接近区间但不是最高。

综上，Plan B 主表支持的结论是：two-stage density 闭环成立，学习模型里 Ours 在 Stage 1 排序指标上有优势，Stage 2 与 Vanilla PubMedBERT 接近；但 Ours 不是全局最强，也不是所有任务和指标上的全面领先方法。

### 3.2 标准消融：Full vs w/o component

标准消融表采用 `Full vs w/o ...` 格式。Full MWS-CFE 的 Stage 1 AUPRC 为 `98.82 +/- 0.41`，Stage 2 Macro-F1 为 `91.05 +/- 0.54`。

最明确的正向组件是 section-aware input。去掉 section-aware input 后，Stage 1 AUPRC 降到 `79.33 +/- 4.64`，Stage 2 Macro-F1 降到 `86.76 +/- 0.32`。这说明报告段落结构和输入组织对 two-stage density 非常关键，尤其影响 Stage 1 的显式密度证据检测。

同时，消融也暴露出 Full 配置并非每个组件都带来稳定增益。`w/o quality gate` 的 Stage 1 AUPRC 为 `99.09 +/- 0.18`，Stage 2 Macro-F1 为 `94.58 +/- 0.37`，均高于 Full。`w/o multi-source supervision` 的 Stage 2 Macro-F1 为 `97.26 +/- 0.19`，也高于 Full。这说明当前质量门控和多源监督配置仍存在任务特异校准问题，不能把 Full 写成所有组件均优的最优组合。

`w/o weighted aggregation` 与 Full 接近，Stage 1 AUPRC 为 `98.79 +/- 0.37`，Stage 2 Macro-F1 为 `91.32 +/- 0.81`。这说明 weighted aggregation 不是当前性能变化的主要来源。`w/o confidence-aware training` 的 Stage 2 Macro-F1 为 `90.70 +/- 0.77`，略低于 Full，说明 confidence-aware training 有一定稳定作用，但效应有限。

因此，消融部分应围绕两点展开：第一，section-aware input 是最关键组件之一；第二，quality gate 和 multi-source supervision 的当前实现不是无条件收益，需要在论文中作为局限和后续优化方向诚实说明。

### 3.3 参数讨论：P1 / P2 / P3

P1、P2、P3 参数讨论已经完整覆盖，并均在 two-stage density 协议下报告 Stage 1 AUPRC 与 Stage 2 Macro-F1。

P1 `max_seq_length` 显示，Stage 1 AUPRC 在 128、160、192 三个设置下接近，分别为 `98.82 +/- 0.41`、`98.79 +/- 0.41`、`98.80 +/- 0.51`。Stage 2 Macro-F1 随长度增加有小幅提升，192 达到 `91.70 +/- 0.49`，高于主配置 128 的 `91.05 +/- 0.54`。因此，128 可解释为性能和计算成本之间的主配置折中，而不是全局最优长度。

P2 `quality_gate` 是最敏感的参数组。G3 的 Stage 1 AUPRC 为 `99.35 +/- 0.20`，Stage 2 Macro-F1 为 `96.83 +/- 0.18`，均高于主配置 G2。这说明质量门控对 Plan B two-stage density 有强影响，但也意味着不能把 G2 主配置写成参数扫描最优点。合理写法是：G2 是预先锁定的主配置，P2 显示 gate 选择具有显著敏感性。

P3 `section/input strategy` 显示，section-aware text 是当前最稳输入策略。mention text 的 Stage 1 AUPRC 只有 `41.84 +/- 8.80`，Stage 2 Macro-F1 为 `74.09 +/- 2.31`；findings text、findings + impression text 和 full text 也明显低于 section-aware text。这说明 two-stage density 不只是需要更长文本，而是需要保留并组织报告段落上下文。

参数讨论的总体结论是：P1 说明长度对 Stage 2 有有限收益；P2 说明质量门控非常敏感；P3 证明 section-aware input 是关键设计。完整参数表适合正文或附录，取决于版面空间。

### 3.4 可直接用于论文或导师汇报的结论段

Plan B 已经将 density 抽取重构为 two-stage density 任务：Stage 1 识别显式密度证据，Stage 2 进行密度亚型分类。完整 5-seed 聚合显示，cue-only/rule baseline 在当前标签口径下达到 100%，因此全部方法比较中 Ours 不是全局最强。排除 cue-only 后，MWS-CFE 在学习模型的 Stage 1 AUPRC/AUROC 上取得最高结果，并在 Stage 2 Macro-F1 上与 Vanilla PubMedBERT 非常接近，但没有超过 Vanilla PubMedBERT。标准消融进一步显示，section-aware input 是最关键组件之一；去掉该组件会显著降低 Stage 1 和 Stage 2 表现。参数讨论完整覆盖 P1/P2/P3，并表明 quality gate 和输入策略对结果高度敏感。因此，模块2可以进入论文写作阶段，但论文叙事必须从“全面领先”调整为“two-stage 闭环成立、Stage 1 排序优势、Stage 2 接近强语义模型、消融揭示关键组件和局限”。

## 4. 最终判断

模块2 Plan B 已经可以进入论文写作阶段。当前不需要继续补实验。

理由是：实验完整性已经闭环，主表、标准消融、参数讨论和图件均已具备；硬缺口不存在。唯一需要控制的是论文表述边界：不能写 Ours 全面领先，必须如实写 cue-only 全部方法最强、Ours 只在学习模型 Stage 1 排序指标上最强、Stage 2 与 Vanilla PubMedBERT 接近。
