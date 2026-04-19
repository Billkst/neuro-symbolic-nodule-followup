# 模块2论文结果章节草稿

> 适用位置：论文第 4 章实验结果中“模块2：放射学报告结构化信息抽取”相关小节  
> 依据文件：`reports/module2_final_table_audit.md` 与 `outputs/phaseA2/final_tables/*`  
> 写作约束：本草稿不声称 MWS-CFE 全面领先；所有结论均与 final tables 保持一致。

---

## 4.x Main Results

表 X 给出了模块2在三个结构化信息抽取任务上的正式结果。所有方法均在统一的 multi-source weak supervision 设置下进行比较，并使用相同的 subject-level split、相同的 G2 主配置和 5 个随机种子进行汇总。三个任务分别为 `density`、`Has-size` 和 `location`，其中 `Has-size` 表示报告片段中是否能够抽取到结节尺寸信息。表中结果均以 5-seed mean ± std 的形式报告。

**表 X：模块2正式主表**

| Method | Density Acc. | Density Macro-F1 | Has-size Acc. | Has-size F1 | Location Acc. | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 13.13 ± 0.00 | 17.64 ± 0.00 | 90.06 ± 0.00 | 87.83 ± 0.00 | 98.49 ± 0.00 | 95.95 ± 0.00 |
| TF-IDF + SVM | 13.07 ± 0.00 | 18.52 ± 0.00 | 89.76 ± 0.00 | 87.53 ± 0.00 | 98.70 ± 0.00 | 96.73 ± 0.00 |
| Vanilla PubMedBERT | 12.76 ± 0.09 | 43.76 ± 6.92 | 84.29 ± 1.36 | 82.11 ± 1.28 | 98.74 ± 0.37 | 97.13 ± 0.83 |
| MWS-CFE (Ours) | 12.77 ± 0.06 | 23.17 ± 4.47 | 83.91 ± 1.33 | 81.75 ± 1.26 | 98.47 ± 0.19 | 96.63 ± 0.33 |

表注建议：所有结果均为 5 seeds mean ± std，数值单位为百分比。`Has-size` 对应代码任务 `has_size`。

总体来看，不同任务呈现出明显不同的方法偏好。在 density 任务上，Vanilla PubMedBERT 取得最高的 Macro-F1，为 `43.76 ± 6.92`，显著高于 TF-IDF + LR 的 `17.64 ± 0.00`、TF-IDF + SVM 的 `18.52 ± 0.00` 以及 MWS-CFE 的 `23.17 ± 4.47`。这一结果说明，密度类别识别更依赖上下文语义建模能力，而不仅是浅层词面匹配。MWS-CFE 在该任务上优于两个 TF-IDF baseline，但没有超过 Vanilla PubMedBERT，因此不能将其表述为 density 任务上的最佳方法。

在 Has-size 任务上，传统线性方法反而表现更强。TF-IDF + LR 获得最高 F1，为 `87.83 ± 0.00`，TF-IDF + SVM 的 F1 为 `87.53 ± 0.00`。相比之下，Vanilla PubMedBERT 和 MWS-CFE 分别为 `82.11 ± 1.28` 和 `81.75 ± 1.26`。这一现象表明，尺寸是否存在具有较强的显式词面模式，例如数字、单位和固定表达，因此轻量级词袋特征已经能够捕捉大量判别信息。对于该任务，复杂神经模型并未带来性能优势。

在 location 任务上，所有方法的准确率都较高。Vanilla PubMedBERT 取得最高 Macro-F1，为 `97.13 ± 0.83`；TF-IDF + SVM 为 `96.73 ± 0.00`；MWS-CFE 为 `96.63 ± 0.33`；TF-IDF + LR 为 `95.95 ± 0.00`。MWS-CFE 在 location 任务上与其他方法处于相近区间，但仍不是最高结果。该结果说明 location 任务本身具有较高可抽取性，同时也提示当前 MWS-CFE 配置尚未在该任务上形成稳定优势。

综合三个任务，MWS-CFE 在统一弱监督框架下完成了 density、Has-size 和 location 的 5-seed 正式闭环，但其主要价值不应被表述为“在所有任务上全面领先”。更准确的表述是：MWS-CFE 提供了一个可复现、可审计的多源弱监督训练框架，并通过后续消融实验揭示了质量门控、聚合方式和输入策略对结构化抽取性能的影响。主表也暴露出一个重要事实：不同抽取任务的最优方法并不一致，模块2的性能瓶颈与任务类型、标签噪声和输入表示均有关。

可以直接写入论文的表述：

“MWS-CFE does not dominate all baselines in the main table. Instead, the main comparison shows that different clinical fact extraction tasks favor different modeling assumptions: semantic encoding is more effective for density, lexical baselines are strong for has-size detection, and all methods perform competitively on location extraction.”

中文论文可改写为：

“主表结果表明，MWS-CFE 并未在所有任务上全面优于 baseline。不同字段抽取任务对模型能力的需求存在差异：密度识别更依赖上下文语义编码，尺寸存在性判断具有较强词面模式，而位置抽取在多种方法下均表现较稳定。”

---

## 4.x Efficiency Analysis

表 X 汇总了模块2各方法在三个任务上的训练与评估成本。需要注意的是，表中的 `Eval Time / seed` 表示完整测试集评估耗时，而不是严格意义上的单样本推理延迟。因此，在论文表述中应使用“evaluation time”或“评估耗时”，不应将其包装为 `Inference Time (ms/sample)`。

**表 X：模块2训练与评估成本表**

| Method | Task | Train Samples | Train Time / seed (s) | Eval Time / seed (s) | Best Epoch | Peak GPU Memory (GB) |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | Density | 36038.0 ± 0.0 | 47.3 ± 1.2 | 2.9 ± 0.2 | — | 0.00 ± 0.00 |
| TF-IDF + LR | Has-size | 168433.0 ± 0.0 | 12.6 ± 2.2 | 4.2 ± 0.5 | — | 0.00 ± 0.00 |
| TF-IDF + LR | Location | 131233.0 ± 0.0 | 112.8 ± 3.9 | 4.1 ± 0.4 | — | 0.00 ± 0.00 |
| TF-IDF + SVM | Density | 36038.0 ± 0.0 | 56.3 ± 2.3 | 2.8 ± 0.2 | — | 0.00 ± 0.00 |
| TF-IDF + SVM | Has-size | 168433.0 ± 0.0 | 19.3 ± 1.1 | 4.1 ± 0.5 | — | 0.00 ± 0.00 |
| TF-IDF + SVM | Location | 131233.0 ± 0.0 | 290.1 ± 11.7 | 4.2 ± 0.3 | — | 0.00 ± 0.00 |
| Vanilla PubMedBERT | Density | 36038.0 ± 0.0 | 343.1 ± 79.2 | 26.8 ± 0.3 | 3.0 ± 1.4 | 9.25 ± 0.01 |
| Vanilla PubMedBERT | Has-size | 168433.0 ± 0.0 | 1865.0 ± 366.1 | 48.2 ± 2.2 | 5.0 ± 1.6 | 6.14 ± 0.01 |
| Vanilla PubMedBERT | Location | 131233.0 ± 0.0 | 843.7 ± 227.6 | 78.9 ± 1.1 | 1.8 ± 1.3 | 6.14 ± 0.02 |
| MWS-CFE (Ours) | Density | 36038.0 ± 0.0 | 271.7 ± 105.4 | 26.0 ± 10.1 | 3.0 ± 3.4 | — |
| MWS-CFE (Ours) | Has-size | 168433.0 ± 0.0 | 1745.8 ± 518.7 | 45.1 ± 5.7 | 5.6 ± 2.6 | — |
| MWS-CFE (Ours) | Location | 131233.0 ± 0.0 | 763.4 ± 104.8 | 87.8 ± 18.8 | 1.4 ± 0.5 | — |

表注建议：`Eval Time / seed` 表示完整评估耗时，不是严格单样本推理延迟。MWS-CFE 的 GPU memory 字段未在历史 JSON 元数据中记录，`—` 不表示 0。

从训练时间看，TF-IDF + LR 和 TF-IDF + SVM 的计算成本明显低于神经模型。例如在 density 任务上，TF-IDF + LR 和 TF-IDF + SVM 的训练时间分别为 `47.3 ± 1.2` 秒和 `56.3 ± 2.3` 秒；Vanilla PubMedBERT 和 MWS-CFE 分别为 `343.1 ± 79.2` 秒和 `271.7 ± 105.4` 秒。在 Has-size 与 location 任务上也可以观察到类似趋势，传统线性模型运行更快，而神经模型需要更高训练成本。

MWS-CFE 与 Vanilla PubMedBERT 的训练成本处于同一量级，并未表现出额外数量级的训练开销。具体而言，MWS-CFE 在 density、Has-size 和 location 上的训练时间分别为 `271.7 ± 105.4` 秒、`1745.8 ± 518.7` 秒和 `763.4 ± 104.8` 秒；Vanilla PubMedBERT 分别为 `343.1 ± 79.2` 秒、`1865.0 ± 366.1` 秒和 `843.7 ± 227.6` 秒。可以保守地表述为：在当前实现中，MWS-CFE 的训练耗时与 Vanilla PubMedBERT 相近，部分任务略低。

从评估耗时看，神经模型的评估时间明显高于 TF-IDF 系列方法。例如 location 任务上，TF-IDF + LR 和 TF-IDF + SVM 的评估时间均约为 4 秒，而 Vanilla PubMedBERT 与 MWS-CFE 分别为 `78.9 ± 1.1` 秒和 `87.8 ± 18.8` 秒。这说明神经模型在评估阶段也具有更高计算成本。

需要保守处理的是 GPU memory 元数据。当前 final efficiency table 中，Vanilla PubMedBERT 记录了 peak GPU memory，而 MWS-CFE 的对应字段为未记录状态。因此，正文中不应比较 MWS-CFE 与 Vanilla PubMedBERT 的显存高低，也不应推断 MWS-CFE 的显存优势。可在表注中说明：MWS-CFE 的 GPU memory 字段未在历史 JSON 元数据中记录，缺失值不表示显存为 0。

推荐写法：

“The efficiency table reports training time and full evaluation time per seed. These values quantify the cost of the experimental pipeline, but should not be interpreted as strict per-sample inference latency.”

中文论文可改写为：

“效率表反映的是每个 seed 下的训练耗时与完整评估耗时，而非严格的单样本推理延迟。结果显示，TF-IDF 系列方法计算成本最低，神经模型成本更高；MWS-CFE 与 Vanilla PubMedBERT 的训练耗时处于同一量级，未引入额外数量级开销。”

---

## 4.x Ablation Study

### 4.x.1 Effect of Quality Gate

质量门控是模块2消融实验中最值得讨论的部分。表 X 中的 A2 quality-gate 结果表明，不同任务对 gate 配置的敏感性不同，且更严格的质量门控并不总是带来更优性能。

**表 X：模块2核心消融摘要表**

| Experiment | Variant | Task | Acc. | Macro-F1 |
|---|---|---|---:|---:|
| Quality gate | G2 / main | Density | 12.77 ± 0.06 | 23.17 ± 4.47 |
| Quality gate | G3 | Density | 19.28 ± 4.26 | 39.84 ± 3.60 |
| Quality gate | G2 / main | Location | 98.47 ± 0.19 | 96.63 ± 0.33 |
| Quality gate | G1 | Location | 99.75 ± 0.09 | 99.31 ± 0.22 |
| Quality gate | G5 | Location | 90.11 ± 0.47 | 67.49 ± 2.11 |
| Aggregation | Weighted vote / main | Density | 12.77 ± 0.06 | 23.17 ± 4.47 |
| Aggregation | Uniform vote | Density | 12.75 ± 0.07 | 23.24 ± 8.34 |
| Aggregation | Weighted vote / main | Location | 98.47 ± 0.19 | 96.63 ± 0.33 |
| Aggregation | Uniform vote | Location | 98.41 ± 0.18 | 96.53 ± 0.39 |

在 density 任务上，G3 的表现明显优于主配置 G2。G2/main 的 Accuracy 和 Macro-F1 分别为 `12.77 ± 0.06` 和 `23.17 ± 4.47`，而 G3 提升到 `19.28 ± 4.26` 和 `39.84 ± 3.60`。这说明 density 任务对弱监督标签质量较为敏感，适当调整 gate 可以显著改善类别均衡表现。由于主表采用预先锁定的 G2 作为统一公平配置，因此不能事后把 G3 替换为主结果；但 G3 的提升为后续优化 MWS-CFE 提供了明确方向。

在 location 任务上，G1 表现最好，Accuracy 为 `99.75 ± 0.09`，Macro-F1 为 `99.31 ± 0.22`，高于 G2/main 的 `98.47 ± 0.19` 和 `96.63 ± 0.33`。相反，G5 在 location 上明显退化，Macro-F1 降至 `67.49 ± 2.11`。这表明过强的质量门控可能降低覆盖率或改变训练样本分布，从而损害位置抽取效果。换言之，质量门控不是越严格越好，而需要根据任务特征进行调节。

该消融结果支持一个更稳妥的结论：MWS-CFE 的关键价值之一在于提供了可分析的 weak supervision quality control 机制。虽然主表中 MWS-CFE 未全面领先，但 quality-gate 消融揭示了性能变化的可解释来源，说明当前系统已经能够定位弱监督配置对下游抽取任务的影响。

### 4.x.2 Effect of Aggregation Strategy

A3 aggregation 消融比较了 weighted vote/main 和 uniform vote。结果显示，两种聚合方式在 density 和 location 上的差异较小。

在 density 任务上，weighted vote/main 的 Macro-F1 为 `23.17 ± 4.47`，uniform vote 为 `23.24 ± 8.34`。两者均值几乎一致，但 uniform vote 的方差更大。在 location 任务上，weighted vote/main 的 Macro-F1 为 `96.63 ± 0.33`，uniform vote 为 `96.53 ± 0.39`，差异同样很小。

这说明，在当前数据和 labeling function 设置下，简单改变投票权重并不是性能的主要决定因素。当前瓶颈更可能来自弱标签噪声、类别不平衡、质量门控选择、输入文本策略或模型训练目标，而不是 weighted vote 与 uniform vote 的差别本身。因此，A3 aggregation 更适合作为“当前 weighted vote 不是主要瓶颈”的证据，而不应被包装成显著提升。

推荐写法：

“The aggregation ablation shows that weighted and uniform voting produce very similar results, suggesting that the current performance bottleneck is unlikely to be solved by changing vote weights alone.”

---

## 4.x Parameter Discussion

P1 和 P3 均为 density-only 参数讨论，不能泛化为所有任务的结论。详细数值建议放入附录，正文只保留主要趋势。

**表 A：P1 max_seq_length 参数讨论表（density-only）**

| Max seq length | Task | Acc. | Macro-F1 |
|---|---|---:|---:|
| 64 | Density | 12.75 ± 0.05 | 19.21 ± 3.52 |
| 96 | Density | 12.79 ± 0.13 | 22.51 ± 5.74 |
| 128 / main | Density | 12.77 ± 0.06 | 23.17 ± 4.47 |
| 160 | Density | 12.75 ± 0.09 | 23.12 ± 4.63 |
| 192 | Density | 12.71 ± 0.04 | 24.77 ± 6.10 |

**表 A：P3 section/input strategy 参数讨论表（density-only）**

| Input strategy | Task | Acc. | Macro-F1 |
|---|---|---:|---:|
| mention_text / main | Density | 12.77 ± 0.06 | 23.17 ± 4.47 |
| findings | Density | 12.81 ± 0.04 | 19.03 ± 2.23 |
| impression | Density | 14.12 ± 1.86 | 31.27 ± 7.21 |
| findings_impression | Density | 12.74 ± 0.05 | 22.90 ± 3.68 |
| full_text | Density | 22.14 ± 4.70 | 23.22 ± 4.62 |

P1 max_seq_length 结果显示，更长输入在 density 任务上整体有一定帮助，但收益并不稳定。max_seq_length 为 64 时 Macro-F1 为 `19.21 ± 3.52`，低于主配置 128 的 `23.17 ± 4.47`。当长度增加到 192 时，Macro-F1 达到最高的 `24.77 ± 6.10`。不过，各长度的 Accuracy 差异很小，且 192 的方差较大，因此不能简单得出“输入越长越好”的结论。更合理的解释是：较长上下文可能为密度判断提供更多信息，但同时也引入更高不确定性；128 作为主配置是性能和计算成本之间的折中。

P3 section/input strategy 结果显示，输入段落选择对 density 任务有明显影响。`impression` 的 Macro-F1 最高，为 `31.27 ± 7.21`，高于 `mention_text / main` 的 `23.17 ± 4.47` 和 `findings` 的 `19.03 ± 2.23`。这说明 impression 段落可能包含更集中、更具诊断价值的密度描述。另一方面，`full_text` 的 Accuracy 最高，为 `22.14 ± 4.70`，但 Macro-F1 仅为 `23.22 ± 4.62`，与主配置接近。这提示 full_text 可能提升了多数类或整体匹配表现，但没有明显改善类别均衡表现。

因此，参数讨论部分可以写为：density 任务对输入长度和段落选择较敏感，尤其 impression 段落值得后续进一步探索；但当前 P1/P3 均为 density-only 实验，不能直接作为 has-size 或 location 的结论。

---

## 4.x Discussion / Limitations

模块2实验的一个重要结果是：MWS-CFE 并没有在所有任务上全面超过 baseline。这一点应在论文中正面承认，而不是通过选择性叙述回避。主表显示，Vanilla PubMedBERT 在 density 和 location 的 Macro-F1 上更强，TF-IDF + LR/SVM 在 Has-size 任务上更强，MWS-CFE 在三个正式任务上均未取得最高主指标。

这一结果并不否定模块2的研究价值。首先，模块2完成了一个统一的 multi-source weak supervision 闭环，将 density、Has-size 和 location 三个字段纳入同一实验框架，并完成了 5-seed strict 聚合。相比单次实验或旧版不公平主表，这一结果更稳定，也更适合支撑论文中的实验分析。

其次，消融实验提供了主表之外的重要洞察。A2 quality-gate 表明，不同任务对弱监督质量门控具有明显不同的敏感性：density 在 G3 下显著提升，而 location 在过强门控 G5 下明显退化。这说明 MWS-CFE 的价值不仅在于最终分数，也在于提供了可解释的弱监督质量控制分析框架。A3 aggregation 进一步说明，单纯改变 vote weighting 不能解释主要性能差异，后续优化应更关注 gate、输入策略、标签噪声和任务特异性。

第三，P1/P3 参数讨论揭示了 density 任务对输入上下文和报告段落选择的敏感性。尤其是 impression 输入在 Macro-F1 上优于主配置，提示放射学报告不同段落承载的信息密度不同。这一发现对后续模块2优化和模块3输入构建都有实际意义。

最后，模块2的系统价值在于为模块3提供结构化、可追踪的输入基础。即使 MWS-CFE 当前没有在所有任务上取得最高性能，模块2仍然完成了从非结构化放射学报告到结构化字段的实验闭环，为后续图谱智能体使用 `density`、`Has-size`、`location`、字段置信度和证据 span 生成随访建议奠定基础。

可写入局限性的表述：

“A limitation of the current Module 2 results is that MWS-CFE does not outperform all baselines across all tasks. This suggests that multi-source weak supervision alone is insufficient to guarantee superior predictive performance under noisy silver labels. Nevertheless, the framework provides a reproducible extraction pipeline and exposes task-specific effects of quality gating and input construction, which are essential for downstream graph-agent reasoning.”

中文论文可改写为：

“当前模块2的局限在于，MWS-CFE 并未在所有任务上全面超过 baseline。这说明在存在弱标签噪声和类别不平衡的情况下，多源弱监督框架本身并不能保证性能绝对领先。然而，该框架完成了稳定可复现的结构化抽取闭环，并通过消融实验揭示了质量门控和输入构造的任务依赖性，为后续图谱智能体提供了必要的结构化输入基础。”

---

## 导师汇报版简短提纲

### 主表怎么讲

模块2主表已经完成 4 个方法、3 个任务、5 个 seed 的公平比较。结果不是“MWS-CFE 全面赢”，而是不同任务最优方法不同：density 和 location 上 Vanilla PubMedBERT 更强，Has-size 上 TF-IDF + LR/SVM 更强，MWS-CFE 完成了统一多源弱监督闭环，但当前性能不是全任务最优。

### 消融怎么讲

最值得讲的是 A2 quality-gate。Density 上 G3 从主配置的 `23.17` Macro-F1 提升到 `39.84`，说明密度任务对弱监督质量门控很敏感；location 上 G1 最好，但 G5 明显掉到 `67.49` Macro-F1，说明门控不是越严格越好。A3 aggregation 差异很小，说明 weighted vote 不是当前主要瓶颈。

### 参数讨论怎么讲

P1/P3 都是 density-only。长度从 64 到 192 对 Macro-F1 有一定提升趋势，但不稳定；impression 输入的 Macro-F1 最高，说明报告摘要段落对密度判断更集中。这些结果适合放附录或作为优化方向，不作为主结论。

### 最关键 takeaway

1. 模块2已经完成 strict 5-seed 聚合，可以进入论文写作。
2. MWS-CFE 不能写全面领先，必须诚实承认主表结果。
3. 模块2的主要贡献是多源弱监督闭环、严格聚合和消融洞察。
4. A2 quality-gate 是最有说服力的正文消融。
5. 模块2为模块3图谱智能体提供结构化字段和证据基础。

---

## 不能写 / 建议写

### 不建议写

1. “我们的方法全面优于所有 baseline。”
2. “MWS-CFE 在所有任务上均达到最佳性能。”
3. “MWS-CFE 显著优于 Vanilla PubMedBERT。”
4. “MWS-CFE 推理速度最快。”
5. “Eval Time / seed 可以代表单样本推理延迟。”
6. “质量门控越严格，性能越好。”
7. “weighted vote 明显优于 uniform vote。”
8. “P1/P3 的 density-only 结论可以推广到所有任务。”

### 建议写

1. “在统一 multi-source weak supervision 框架下，我们完成了模块2三项任务的公平 5-seed 比较。”
2. “主表显示，不同任务的最优方法并不一致，MWS-CFE 未在所有任务上全面领先。”
3. “MWS-CFE 的主要价值体现在弱监督训练闭环、可审计聚合和消融实验洞察。”
4. “A2 quality-gate 结果表明，弱监督质量控制具有明显任务依赖性。”
5. “A3 aggregation 结果显示，weighted vote 与 uniform vote 差异较小，当前瓶颈不主要来自投票权重。”
6. “P1/P3 参数讨论提示 density 任务对上下文长度和报告段落选择敏感，但该结论不应泛化到所有任务。”
7. “模块2为后续模块3图谱智能体提供了结构化事实、置信度和证据 span 的输入基础。”
