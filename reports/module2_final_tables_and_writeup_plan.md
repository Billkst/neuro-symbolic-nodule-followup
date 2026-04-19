# 模块2最终论文表格与结果写作支撑方案

> 日期：2026-04-19  
> 阶段：模块2最终论文表格生成与结果写作支撑  
> 正式数据来源：`outputs/phaseA2/tables`  
> 审核依据：`reports/module2_final_table_audit.md`  
> 约束：不继续训练，不切换模块3编码，不使用旧 `scripts/phaseA2/build_main_table.py`。

---

## 0. 本轮生成文件

### 0.1 论文版 CSV 表格

正文表：

1. `outputs/phaseA2/final_tables/main_table_final.csv`
2. `outputs/phaseA2/final_tables/efficiency_table_final.csv`
3. `outputs/phaseA2/final_tables/ablation_table_final.csv`

附录表：

1. `outputs/phaseA2/final_tables/appendix_tables/a2_quality_gate_final.csv`
2. `outputs/phaseA2/final_tables/appendix_tables/a3_aggregation_final.csv`
3. `outputs/phaseA2/final_tables/appendix_tables/p1_max_seq_length_final.csv`
4. `outputs/phaseA2/final_tables/appendix_tables/p3_section_input_final.csv`

### 0.2 LaTeX 表格

正文 LaTeX：

1. `outputs/phaseA2/final_tables_latex/main_table.tex`
2. `outputs/phaseA2/final_tables_latex/efficiency_table.tex`
3. `outputs/phaseA2/final_tables_latex/ablation_table.tex`

附录 LaTeX：

1. `outputs/phaseA2/final_tables_latex/appendix_tables.tex`

LaTeX 文件均使用普通 `tabular` 和 `\hline`，未依赖 `booktabs`。数值中的均值方差统一写作 `$\pm$`，避免直接写入 Unicode `±`。

---

## 1. 最终表格处理口径

### 1.1 正式主表

来源：`outputs/phaseA2/tables/a2_5_main_table.csv`  
输出：`outputs/phaseA2/final_tables/main_table_final.csv` 与 `outputs/phaseA2/final_tables_latex/main_table.tex`

处理规则：

1. 删除 `Density N`、`Has_size N`、`Location N`。
2. `Has_size` 统一改为 `Has-size`。
3. 保留 4 个正式方法：`TF-IDF + LR`、`TF-IDF + SVM`、`Vanilla PubMedBERT`、`MWS-CFE (Ours)`。
4. 保留三任务的 Acc. / F1 指标。
5. 表注说明所有结果为 5 seeds mean ± std，且数值乘以 100。

最终表头：

| Method | Density Acc. | Density Macro-F1 | Has-size Acc. | Has-size F1 | Location Acc. | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|

### 1.2 正文效率表

来源：`outputs/phaseA2/tables/a2_5_efficiency_table.csv`  
输出：`outputs/phaseA2/final_tables/efficiency_table_final.csv` 与 `outputs/phaseA2/final_tables_latex/efficiency_table.tex`

处理规则：

1. 删除 `Tag`、`N`。
2. `Has_size` 统一改为 `Has-size`。
3. 保留 `Train Samples`、`Train Time / seed (s)`、`Eval Time / seed (s)`、`Best Epoch`、`Peak GPU Memory (GB)`。
4. 不把 `Eval Time / seed` 改写成 `Inference Time (ms/sample)`。
5. MWS-CFE 的 `Peak GPU Memory` 保留为 `not recorded` 或 `—`，并在脚注说明是历史 JSON 元数据未记录。

最终表头：

| Method | Task | Train Samples | Train Time / seed (s) | Eval Time / seed (s) | Best Epoch | Peak GPU Memory (GB) |
|---|---|---:|---:|---:|---:|---:|

### 1.3 正文核心消融表

来源：

1. `outputs/phaseA2/tables/a2_quality_gate_summary.csv`
2. `outputs/phaseA2/tables/a3_aggregation_summary.csv`

输出：`outputs/phaseA2/final_tables/ablation_table_final.csv` 与 `outputs/phaseA2/final_tables_latex/ablation_table.tex`

处理规则：

1. 不完整搬运所有消融行，而是生成正文核心摘要。
2. A2 quality-gate 只保留主配置、最强配置和明显失败配置。
3. A3 aggregation 保留 weighted/main 和 uniform 两个配置，用于说明差异较小。
4. 删除 `Tag`、`N`、`Main Metric` 等工程列。

正文核心消融表包含：

1. Quality gate: density 的 G2/main 与 G3。
2. Quality gate: location 的 G2/main、G1 与 G5。
3. Aggregation: density 与 location 上的 weighted vote/main 与 uniform vote。

### 1.4 附录表

附录表保留完整结果，但仍删除工程字段：

1. A2 quality-gate：完整 G1/G2/G3/G4/G5，density 与 location。
2. A3 aggregation：weighted vote/main 与 uniform vote，density 与 location。
3. P1 max_seq_length：density-only，64/96/128/160/192。
4. P3 section/input strategy：density-only，mention_text/findings/impression/findings_impression/full_text。

---

## 2. 结果章节写作提纲

### 2.1 主表应该怎么写

主表要证明什么：

1. 模块2已在统一 multi-source weak supervision 口径下完成正式公平比较。
2. 4 个 trainable 方法使用同一 split、同一 G2 主配置、同一 5-seed 汇总规则。
3. 三个正式任务分别覆盖 density、has-size、location。

推荐写法：

1. “表 X 汇总了模块2在三个结构化抽取任务上的 5-seed 测试结果。”
2. “所有方法均在统一 multi-source weak supervision 数据和 subject-level split 下评估，因此主表数值具有可比性。”
3. “Vanilla PubMedBERT 在 density Macro-F1 和 location Macro-F1 上取得最高结果，说明当前语义编码器对密度与位置分类仍具有较强优势。”
4. “TF-IDF + LR/SVM 在 has-size 任务上表现较强，表明尺寸是否存在具有明显词面模式。”
5. “MWS-CFE 完成了统一弱监督训练流程下的三任务闭环，但在当前主表口径下并未全面超过所有 baseline。”

不能写的句子：

1. “MWS-CFE 在所有任务上全面领先。”
2. “MWS-CFE 显著优于 PubMedBERT。”
3. “MWS-CFE 是当前最优模型。”
4. “模块2主表证明了质量感知训练必然提升性能。”

可以写的保守结论：

1. “MWS-CFE 提供了一个可审计的多源弱监督训练框架。”
2. “主表结果表明，仅引入 MWS-CFE 训练配置并不自动带来三任务性能优势，这也提示弱标签噪声与任务差异仍是后续优化重点。”
3. “后续消融结果进一步说明，质量门控和输入策略对不同任务存在明显影响。”

### 2.2 效率表应该怎么写

效率表要证明什么：

1. 展示各方法训练成本和评估成本。
2. 说明传统 ML baseline 与神经模型在计算成本上的差异。
3. 说明 MWS-CFE 与 Vanilla PubMedBERT 训练时间处于同一量级。

推荐写法：

1. “表 X 报告了每个任务上的训练时间和整套 evaluation 时间。”
2. “TF-IDF 系列方法运行在 CPU 路径上，训练时间明显低于神经模型。”
3. “MWS-CFE 在三个任务上的训练时间与 Vanilla PubMedBERT 处于同一量级，未观察到额外数量级的训练开销。”
4. “需要注意，Eval Time / seed 表示整套测试集评估耗时，而不是严格的单样本推理延迟。”

不能写的句子：

1. “MWS-CFE 推理速度最快。”
2. “Eval Time / seed 等价于 Inference Time (ms/sample)。”
3. “MWS-CFE 显存占用低于 PubMedBERT。”

必须脚注说明：

1. MWS-CFE 的 GPU memory 在历史 JSON 元数据中未记录。
2. `—` 或 `not recorded` 不表示 0。
3. 传统 ML 的 GPU memory 记为 0，因为 CPU 路径运行。

### 2.3 消融表应该怎么写

A2 quality-gate 主要结论：

1. quality gate 是当前模块2最有讨论价值的消融。
2. Density 上 G3 明显优于 G2/main：Macro-F1 从 `23.17 ± 4.47` 提高到 `39.84 ± 3.60`。
3. Location 上 G1 最强：Macro-F1 为 `99.31 ± 0.22`。
4. G5 在 location 上明显退化：Macro-F1 为 `67.49 ± 2.11`，说明过强质量门控可能损害覆盖或改变训练分布。
5. 不同任务的最优 gate 不同，说明弱监督质量控制具有任务依赖性。

A3 aggregation 主要结论：

1. Weighted vote/main 与 uniform vote 在 density 和 location 上差异都很小。
2. Density Macro-F1：weighted `23.17 ± 4.47`，uniform `23.24 ± 8.34`。
3. Location Macro-F1：weighted `96.63 ± 0.33`，uniform `96.53 ± 0.39`。
4. 因此当前瓶颈不主要来自简单投票权重，而更可能来自标签噪声、类别不平衡、gate 选择或输入表征。

P1 参数讨论：

1. P1 是 density-only，不应泛化到 has-size/location。
2. 64 长度较弱，192 的 Macro-F1 最高，为 `24.77 ± 6.10`。
3. 128/main 接近 160/192，是计算成本与性能之间的折中。
4. 不应写“长度越长一定越好”，因为方差较大且 Accuracy 没有同步提升。

P3 参数讨论：

1. P3 是 density-only。
2. `impression` 的 Macro-F1 最高，为 `31.27 ± 7.21`，说明 impression 对 density 更集中。
3. `full_text` 的 Accuracy 最高，为 `22.14 ± 4.70`，但 Macro-F1 接近主配置，不能只看 Accuracy 得出 full_text 最优。
4. 结论应写为“输入段落选择会影响 density，impression 是值得进一步探索的输入策略”。

### 2.4 局限性怎么写

Ours 未全面领先如何解释：

1. MWS-CFE 当前主要验证了多源弱监督训练闭环与质量控制机制，而不是已经取得全任务性能领先。
2. 主表结果表明，弱标签噪声、类别分布不均衡和任务差异仍会限制质量感知训练的收益。
3. Vanilla PubMedBERT 在 density 和 location 上更强，说明强语义编码器本身仍是关键因素。
4. TF-IDF 在 has-size 上更强，说明该任务更依赖显式词面线索，复杂模型不一定占优。

为什么研究仍然成立：

1. 模块2完成了统一 multi-source weak supervision 的正式 5-seed 公平比较。
2. 结果不是单次运行，而是 strict manifest 闭环后的稳定汇总。
3. 消融表揭示了质量门控、聚合方式、输入长度和输入段落对性能的真实影响。
4. 这些结果能支撑论文中“结构化事实抽取是模块3图谱智能体输入基础”的系统工程贡献。
5. 诚实报告 Ours 未全面领先，反而能增强实验可信度，避免过度包装。

当前模块2真正贡献点：

1. 建立了从多源弱监督标签到三任务抽取模型的完整实验闭环。
2. 给出了 density、has-size、location 三类字段的统一评估口径。
3. 完成了 5-seed strict 聚合，避免单 seed 结果偶然性。
4. 明确发现不同任务对 quality gate 和输入策略的敏感性不同。
5. 为模块3提供可追踪、可审计的结构化输入字段。

---

## 3. 正文表与附录表最终布局

### 3.1 正文

**表1：模块2正式主表**

文件：

1. `outputs/phaseA2/final_tables/main_table_final.csv`
2. `outputs/phaseA2/final_tables_latex/main_table.tex`

用途：展示 4 个正式方法在三个任务上的公平比较结果。必须放正文。

**表2：模块2训练与评估成本表**

文件：

1. `outputs/phaseA2/final_tables/efficiency_table_final.csv`
2. `outputs/phaseA2/final_tables_latex/efficiency_table.tex`

用途：展示训练样本、训练时间、评估时间、best epoch 和显存记录状态。可以放正文，但必须保留脚注；若正文篇幅紧张，可降为附录。

**表3：模块2核心消融摘要表**

文件：

1. `outputs/phaseA2/final_tables/ablation_table_final.csv`
2. `outputs/phaseA2/final_tables_latex/ablation_table.tex`

用途：正文展示最关键消融结论。建议放正文，因为 A2 quality-gate 是当前最能支撑方法分析的结果。

### 3.2 附录

**表A1：完整 A2 quality-gate 消融**

文件：`outputs/phaseA2/final_tables/appendix_tables/a2_quality_gate_final.csv`

用途：展示 G1/G2/G3/G4/G5 的完整结果。

**表A2：完整 A3 aggregation 消融**

文件：`outputs/phaseA2/final_tables/appendix_tables/a3_aggregation_final.csv`

用途：展示 weighted vote/main 与 uniform vote 的完整结果。

**表A3：P1 max_seq_length 参数讨论**

文件：`outputs/phaseA2/final_tables/appendix_tables/p1_max_seq_length_final.csv`

用途：density-only 参数讨论。

**表A4：P3 section/input strategy 参数讨论**

文件：`outputs/phaseA2/final_tables/appendix_tables/p3_section_input_final.csv`

用途：density-only 输入策略讨论。

附录 LaTeX 合并文件：

```text
outputs/phaseA2/final_tables_latex/appendix_tables.tex
```

---

## 4. 最关键结论

1. 模块2可以直接进入论文结果章节写作，不需要继续优先补跑训练。
2. 正文主表已经清洗完成，删除了 `N` 工程列，并统一使用 `Has-size`。
3. MWS-CFE 没有全面超过 baseline，论文必须避免“全面领先”叙述。
4. Vanilla PubMedBERT 是 density 和 location 主指标上的最强方法。
5. TF-IDF + LR/SVM 在 has-size 上更强，说明该任务具有强词面模式。
6. 效率表可用于训练/评估成本说明，但不能当作严格推理延迟表。
7. A2 quality-gate 是正文最值得展示的消融，能说明任务依赖性和 gate 选择影响。
8. A3 aggregation 差异较小，可作为“weighted vote 不是当前瓶颈”的证据。
9. P1/P3 都是 density-only，适合附录或正文摘趋势，不应泛化到全部任务。
10. 模块2真正贡献应写为多源弱监督抽取闭环、严格聚合、消融洞察和模块3结构化输入支撑。

---

## 5. 下一步建议

下一步应以最终落表和结果章节写作为主：

1. 将 `main_table.tex`、`efficiency_table.tex`、`ablation_table.tex` 放入论文结果章节。
2. 将 `appendix_tables.tex` 放入附录。
3. 根据本报告第 2 节撰写结果分析段落。
4. 在局限性中明确说明 MWS-CFE 未全面领先，并解释当前实验贡献边界。
5. 不继续补跑模块2实验，除非导师明确要求补充严格 per-sample inference benchmark 或修复 MWS-CFE 显存元数据。
