# 项目三实验结果总报告

> 生成日期：2026-04-07
> 数据来源：MIMIC-IV-Note 2.2 放射学报告 + 出院小结
> 实验阶段：Phase 4 (Baseline) → Phase 5 (主模型) → Phase 5.1 (Gold 验证)

---

## Section 1. 实验总体概览

本项目旨在从非结构化放射学报告中提取肺结节的结构化临床事实（密度、尺寸、位置等），并基于提取结果生成符合 Lung-RADS 指南的随访建议。实验按以下阶段推进：

**Phase 4 — Baseline 固定与评测协议建立**
- 建立了基于复杂正则表达式的 Baseline 提取系统
- 设计了 Silver Standard 评测协议（因缺乏大规模人工标注）
- 完成了三组对比实验：section-aware vs full-text、smoking fallback 策略、recommendation 规则推理 vs cue 抓取
- 确定了后续主模型实验的优先字段排序

**Phase 5 — 主模型实验**
- 在 288,894 条 mentions（33,149 名受试者）上训练并评测了 4 种方法：Regex、TF-IDF+LR、TF-IDF+SVM、PubMedBERT
- 覆盖 3 个核心任务：density_category（5 分类）、has_size（二分类）、location_lobe（9 分类）
- 完成了 3 组消融实验和 3 组参数讨论实验
- 发现了 Silver Label 循环性问题（Regex 在 silver 上 1.0 是标签来源导致的，非真实完美）

**Phase 5.1 — 人工 Gold 小样本验证**
- 由研究者对 62 条复杂案例进行人工标注，构建 Gold Standard
- 正式证实了 Silver Ceiling 现象：density 宏 F1 从 silver 上的 >0.99 骤降至 gold 上的 0.70
- 明确了各字段的真实质量和后续投入价值

**Phase 5.1a — size_mm 口径修正**
- 明确 size_mm 评测来自所有方法共享的同一正则解析模块，不构成方法间独立比较
- 该评测反映的是共享解析模块的稳定性

---

## Section 2. 对比实验结果总表

### 2.1 Phase 4 Baseline 对比

#### 2.1.1 full_text_regex vs section_aware_regex（放射学提取）

| 指标 | full_text_regex | section_aware_regex | Delta |
|------|----------------|---------------------|-------|
| size_mm 提取率 | 0.3769 | 0.4094 | +3.3% |
| density 提取率 | 0.1251 | 0.1337 | +0.9% |
| location 提取率 | 0.3523 | 0.3863 | +3.4% |
| 检测结节总数 | 2319 | 1473 | -846 |
| 每报告平均结节数 | 4.638 | 2.946 | -1.692 |

**结论**：section_aware_regex 在 size、density、location 三个关键字段的提取率上均优于 full_text_regex。full_text_regex 检测到更多结节（2319 vs 1473），但其中大量是非肺部病变（如肝脏、肾脏）的误匹配。section_aware 通过限制搜索范围到 FINDINGS/IMPRESSION 段落，减少了 57.5% 的假阳性结节。这证明 section 解析是有价值的设计决策。

#### 2.1.2 social_history_only vs plus_fallback（吸烟史提取）

| 指标 | social_history_only | plus_fallback | Delta |
|------|--------------------|--------------:|-------|
| non_unknown_rate | 0.1060 | 0.6880 | +58.2pp |
| pack_year_parse_rate | 0.1509 | 0.1076 | -4.3pp |
| evidence_quality: high | — | 2.8% | — |
| evidence_quality: low | — | 70.6% | — |

**结论**：fallback 策略将吸烟状态覆盖率从 10.6% 大幅提升到 68.8%，但引入的样本 evidence quality 较低（70.6% 为 low），定量字段解析率反而下降。这是一个典型的覆盖率-质量 trade-off。由于 MIMIC 脱敏导致 97.9% 的 Social History 被掩码，smoking 模块整体受数据限制严重，不适合作为主实验核心。

#### 2.1.3 cue_only vs structured_rule（随访建议生成）

| 指标 | cue_only | structured_rule | Delta |
|------|----------|----------------|-------|
| actionable_rate | 0.5415 | 0.4071 | -13.4pp |
| monitoring_rate | 0.0000 | 0.2648 | +26.5pp |
| guideline_anchor_presence_rate | 0.0000 | 0.6719 | +67.2pp |
| triggered_rules_nonempty_rate | 0.5415 | 1.0000 | +45.9pp |

**结论**：structured_rule 引入了 monitoring 类别（26.5%），cue_only 无法区分 monitoring 和 actionable。structured_rule 的 guideline_anchor_presence_rate 为 67.2%（cue_only 为 0%），可解释性显著提升。结构化规则推理在稳定性和可解释性上均优于直接抓 cue。

### 2.2 Phase 5 主模型对比

数据规模：288,894 mentions, 33,149 subjects。切分：train 201,947 / val 44,890 / test 42,057（按患者 ID 切分，无重叠）。

#### 2.2.1 Density Classification（5 类）

| 方法 | Test Accuracy | Test Macro F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9982 | 0.9938 | 32.5s |
| TF-IDF + SVM | 0.9992 | 0.9975 | 48.3s |
| PubMedBERT | 0.9993 | 0.9985 | 2383s |

#### 2.2.2 Size Detection（二分类）

| 方法 | Test Accuracy | Test F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9920 | 0.9890 | 8.6s |
| TF-IDF + SVM | 0.9927 | 0.9900 | 14.4s |
| PubMedBERT | 0.9994 | 0.9992 | 2439s |

#### 2.2.3 Location Classification（9 类）

| 方法 | Test Accuracy | Test Macro F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9872 | 0.9849 | 24.2s |
| TF-IDF + SVM | 0.9983 | 0.9972 | 16.8s |
| PubMedBERT | 0.9999 | 0.9998 | 2416s |

#### 2.2.4 关键解读

**Silver Label 循环性问题（必须理解）**：Regex 在所有任务上均达到 1.0 的完美分数，这不是因为 Regex 是完美模型，而是因为 Silver Label 本身就是由 Regex 生成的。ML/DL 模型本质上是在学习复制 Regex 的行为模式。这种"天花板效应"是 Silver Label 评测框架的固有特性。

**模型一致性**：三种方法在测试集上的预测一致性极高（density 99.91%、size 99.27%、location 99.83%）。PubMedBERT 与 Regex 几乎完全重合，说明 BERT 更忠实地还原了 Regex 的判断逻辑。

**Regex 失败区域分析**：在 Regex 无法提取的样本上，BERT 表现极度保守——density 36,481 个 unclear 样本中仅 21 个被重新分类；size 26,915 个 no_size 样本中仅 24 个被重新分类；location 26,134 个 no_location 样本中 0 个被重新分类。BERT 未能显著突破 Regex 的覆盖边界。

---

## Section 3. 消融实验总结

### 3.1 A1: Mention-centered Window vs Full-text（核心消融）

| 任务 | Mention Window (Macro F1) | Full Text (Macro F1) | Delta |
|---|---|---|---|
| density | 0.9975 | 0.4785 | **+0.5190** |
| size | 0.9900 | 0.6080 | **+0.3821** |
| location | 0.9972 | 0.3645 | **+0.6327** |

**结论**：基于 mention 为中心的窗口特征在所有三个任务上均大幅优于全文特征（delta 从 +0.38 到 +0.63）。这是本项目最重要的消融结果之一，证明了 section-aware 窗口设计不是拍脑袋，而是系统性能的关键支撑。全文输入引入了大量无关噪声，导致模型无法聚焦到目标 mention 的局部上下文。

**对系统设计的意义**：mention-centered window 是整个提取流水线的基础设计决策，消融结果证明这一决策带来了 38–63 个百分点的性能增益，是不可替代的。

### 3.2 A2: Exam Name 特征

| 任务 | Plain (Macro F1) | +Exam Name (Macro F1) | Delta |
|---|---|---|---|
| density | 0.9975 | 0.9975 | -0.0000 |
| size | 0.9900 | 0.9899 | -0.0001 |
| location | 0.9972 | 0.9974 | +0.0002 |

**结论**：加入 exam_name 特征对性能几乎无影响（delta < 0.001）。这说明前期的 Chest CT 过滤已经保证了输入数据的标准化，exam_name 不再携带额外的判别信息。

**对系统设计的意义**：可以安全地省略 exam_name 特征，简化特征工程流程。前期数据过滤的质量控制是有效的。

### 3.3 A3: Explicit-only vs All Silver Labels

| 任务 | All Data (Macro F1) | Explicit Only (Macro F1) | Delta |
|---|---|---|---|
| density | 0.9975 | 0.4409 | **-0.5566** |
| size | 0.9900 | 0.7810 | **-0.2090** |
| location | 0.9972 | 0.5622 | **-0.4350** |

**结论**：仅使用显式描述样本训练会导致性能大幅下降（delta 从 -0.21 到 -0.56）。这证明了保留全部 silver labels（包含隐式推断样本）对维持模型覆盖率至关重要。仅依赖显式样本会严重限制训练数据的多样性和规模。

**对系统设计的意义**：silver label 的弱监督策略虽然引入了标签噪声，但其覆盖率优势远大于噪声代价。这为"先用 silver 大规模训练、再用 gold 小样本校准"的两阶段策略提供了实验依据。

---

## Section 4. 参数讨论实验总结

以下参数讨论均基于 TF-IDF + LR 模型，在 Silver 测试集上评测。

### 4.1 max_features（TF-IDF 特征维度）

| max_features | density Macro F1 | size F1 | location Macro F1 |
|---|---|---|---|
| 5000 | 0.9969 | 0.9898 | 0.9972 |
| **10000** | **0.9975** | **0.9900** | **0.9972** |
| 20000 | 0.9964 | 0.9905 | 0.9967 |

**结论**：10000 为最优配置。5000 在 density 上略差（-0.0006），20000 在 density 和 location 上反而退化，且训练时间增加 30%。增加特征维度并未带来收益，反而增加了稀疏性。

### 4.2 ngram_range（文本粒度）

| ngram_range | density Macro F1 | size F1 | location Macro F1 |
|---|---|---|---|
| (1,1) unigram | 0.9911 | 0.9893 | 0.9511 |
| **(1,2) bigram** | **0.9975** | **0.9900** | **0.9972** |
| (1,3) trigram | 0.9948 | 0.9896 | 0.9991 |

**结论**：(1,2) 效果最好。仅使用 unigram 会丢失关键的上下文组合（如 "ground glass"、"right upper"），尤其在 location 任务上退化严重（0.9511 vs 0.9972）。加入 trigram 在 location 上略有提升但在 density 上退化，整体不如 bigram 稳定。

### 4.3 C（SVM 正则化强度）

| C | density Macro F1 | size F1 | location Macro F1 |
|---|---|---|---|
| 0.1 | 0.9933 | 0.9899 | 0.9906 |
| **1.0** | **0.9975** | **0.9900** | **0.9972** |
| 10.0 | 0.9957 | 0.9868 | 0.9970 |

**结论**：C=1.0 为最优。C=0.1 导致欠拟合（density -0.0042, location -0.0066）；C=10.0 导致 size 任务退化（-0.0032）且训练时间增加 2.5 倍，无性能收益。

### 4.4 参数讨论总结论

**参数并不是当前系统的瓶颈。** 默认配置（max_features=10000, ngram_range=(1,2), C=1.0）已处于性能平坦区的最优位置。所有参数变化带来的增益均在 0.2% 以内，而退化风险更大。当前系统的主要瓶颈在于 silver label 的标签质量和部分字段（如 density）的可提取性，而非模型超参数。

---

## Section 5. Phase 5.1 人工 Gold 验证总结

### 5.1 Gold 数据构建

- **数据源**：Phase 5 测试集（42,057 条 mentions, 4,973 名受试者）
- **采样策略**：分层抽样（seed=42），覆盖明确阳性、信息缺失、边界模糊、罕见类别四个层级
- **人工标注**：初始 80 条候选，剔除 13 条非肺部目标（甲状腺结节、弥漫性病变等），最终保留 N=62 条
- **标注置信度**：高置信度 35 条，中等置信度 27 条
- **无泄漏保证**：62 条样本均来自测试集，与训练/验证集受试者完全不重叠

### 5.2 为什么要做 Gold 评测

Phase 5 的 Silver 评测显示各模型 Macro F1 均 >0.99，但这种"完美"表现源于 Silver 标签本身由 Regex 生成的循环论证。如果不引入独立的人工标注，无法判断模型是否真正理解了临床语义，还是仅仅复制了规则系统的输出逻辑。

### 5.3 核心发现：Silver Ceiling 正式证实

**这是本次评测最重要的发现。** 在 62 条 Gold 样本上，所有五种评估方法（silver / regex / ml_lr / ml_svm / pubmedbert）在密度预测结果上完全一致，差异为零。这意味着所有模型已完全习得了 Regex 的逻辑。

| 评测基准 | density Macro F1 |
|---|---|
| Silver 测试集 | >0.99 |
| Gold 测试集 | **0.7003** |
| **性能缺口** | **~29 个百分点** |

这种巨大的性能缺口完全源于 Silver 标签的噪声，而非模型能力差异。Silver 评测具有严重的误导性。

### 5.4 各字段 Gold 评测详细结果

#### density_category（N=62）

| 指标 | 值 |
|---|---|
| Accuracy | 0.7258 |
| Macro F1 | 0.7003 |
| Weighted F1 | 0.6989 |

| 类别 | Precision | Recall | F1 | Gold N |
|---|---|---|---|---|
| solid | 0.8333 | 0.8824 | 0.8571 | 17 |
| ground_glass | 0.6818 | 1.0000 | 0.8108 | 15 |
| calcified | 0.8333 | 0.8333 | 0.8333 | 6 |
| part_solid | 0.6000 | 0.6000 | 0.6000 | 10 |
| unclear | 0.6667 | 0.2857 | 0.4000 | 14 |

**关键发现**：unclear 类别召回率最低（0.2857），反映出规则系统对模糊表述存在过度细化倾向——当报告同时描述多种密度时，Regex 倾向于强行指定具体类别，而 Gold 标准标注为 unclear。

#### has_size（N=62）

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| silver/regex/ml_lr/pubmedbert | 0.9355 | 1.0000 | 0.9259 | 0.9615 |
| ml_svm | 0.9194 | 1.0000 | 0.9074 | 0.9515 |

**关键发现**：所有方法精确率均为 1.0（无误报）。4 例漏报中，尺寸信息出现在上下文而非 mention 文本本身，或使用了非标准表达。has_size 是当前最稳定的字段。

#### size_mm（N=54, 仅 gold_has_size=yes）— 共享正则解析模块评测

> **重要口径说明**：所有五种方法的 size_mm 预测均来自同一个共享正则尺寸解析模块（predict_regex_size_mm），因此指标完全一致。此处评测的是该共享模块的 Gold 标准表现，不是方法间的独立建模能力对比。

| 指标 | 值 |
|---|---|
| MAE | 0.8389 mm |
| Median AE | 0.0000 mm |
| Exact match | 85.19% |
| Within ±1mm | 90.74% |

8 例错误中：4 例因 mention 边界截断（正则返回 0.0），2 例因范围值解释差异（Gold 取最大值 vs 正则取中值），2 例因精度差异。

#### location_lobe（N=62）

| 方法 | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| silver/regex/pubmedbert | 0.8226 | 0.8730 | 0.8094 |
| ml_lr | 0.8387 | 0.8547 | 0.8262 |
| ml_svm | 0.8387 | 0.8886 | 0.8261 |

| 类别 | F1 | Gold N |
|---|---|---|
| RML / RLL / LUL / lingula | 1.0000 | 4/7/8/1 |
| LLL | 0.9231 | 7 |
| RUL | 0.7407 | 12 |
| no_location | 0.8000 | 10 |
| **bilateral** | **0.5333** | **11** |
| unclear | 1.0000 | 2 |

**关键发现**：bilateral（双肺）类别 F1 最低（0.5333）。Regex 通常只提取首个出现的叶位，而 Gold 标准在多叶位提及下标注为 bilateral。这是 mention 级提取的结构性限制。

### 5.5 Silver vs Gold 一致性总览

| 字段 | 一致率 | 分歧数/总数 |
|---|---|---|
| density_category | 72.58% | 17/62 |
| has_size | 93.55% | 4/62 |
| size_mm | 87.10% | 8/62 |
| location_lobe | 82.26% | 11/62 |

### 5.6 主要错误模式

1. **多密度混合表述**（17 例密度错误中的 10 例）：如 "solid and ground-glass"，Regex 抓取首个关键字，Gold 标注为 unclear
2. **隐式密度描述**（4 例）：如 "sub solid" 被误判为 solid（Gold 为 part_solid）
3. **多叶位提及导致 bilateral 漏判**（11 例位置错误中的 7 例）：Regex 仅提取首个叶位
4. **mention 边界截断**（4 例尺寸 + 4 例位置）：信息存在于上下文但未被包含在 mention 窗口内
5. **范围值解释差异**（4 例尺寸）：Gold 取最大值 vs 正则取中值

### 5.7 各字段投入价值评估

| 字段 | 建议 | 理由 |
|---|---|---|
| density_category | **继续投入** | Macro F1 0.70 仍有提升空间，需优化多密度表述处理逻辑 |
| location_lobe | **继续投入** | bilateral 问题可通过多标签分类或层次化方法解决 |
| has_size | 边际收益低 | F1 已达 0.96，进一步优化空间有限 |
| size_mm | 当前不构成方法间对比 | 共享解析模块，90.7% 已在容差范围内，剩余误差属数据定义层面 |

---

## Section 6. 最终综合结论

### 核心结论 1：系统整体可用，mention-centered 设计是关键
当前方案已经做出来了。基于 section-aware + mention-centered window 的提取流水线在 size 和单叶位 location 上表现可靠（Gold F1 > 0.80）。消融实验证明 mention-centered 设计带来了 38–63 个百分点的性能增益，是整个系统的基石。

### 核心结论 2：Silver Ceiling 是当前最重要的科学发现
Silver 评测中 >0.99 的 Macro F1 具有严重误导性。Gold 评测揭示 density 真实 Macro F1 仅为 0.70，存在约 29 个百分点的性能缺口。所有模型（包括 PubMedBERT）在 Gold 集上的密度预测完全一致，证实它们只是在复现规则系统，而非理解临床语义。这一发现本身具有方法论价值，可作为论文的核心贡献之一。

### 核心结论 3：density_category 是最成功也最值得继续投入的字段
density 在 Phase 4 时 86.6% 为 unclear（提取率极低），经过 Phase 5 主模型实验和 Phase 5.1 Gold 验证，已经明确了其真实性能边界（Gold Macro F1 = 0.70）和主要错误模式（多密度混合表述）。这是当前最有改进空间的字段。

### 核心结论 4：部分字段因数据限制无法深入
- smoking 模块受 MIMIC 脱敏限制（97.9% Social History 被掩码），不适合作为主实验核心
- size_mm 当前评测基于共享解析模块，不构成方法间独立比较
- bilateral 位置提取受 mention 级粒度的结构性限制

### 核心结论 5：论文可以写，优先级从"扩实验"转为"写论文"
当前实验链（Baseline → 主模型 → 消融 → 参数讨论 → Gold 验证）已经完整，覆盖了对比、消融、参数讨论、人工验证四个维度。Silver Ceiling 的发现和 mention-centered 设计的消融验证是两个可以写进论文的硬结论。不需要再大规模扩实验。

---

## Section 7. 当前局限性与下一步

### 7.1 真实局限性

1. **Smoking 模块受脱敏限制**：97.9% 的 Social History 被掩码，fallback 策略虽提升覆盖率但质量低（70.6% 为 low evidence），不适合再当主实验核心
2. **Silver Label Ceiling 限制解释空间**：在 silver 框架下，所有模型表现趋同，无法区分方法间的真实能力差异。只有 Gold 评测才能揭示真实性能
3. **Gold 样本量有限**：N=62 的结果应视为方向性参考，罕见类别（如 calcified N=6, lingula N=1）指标波动较大
4. **Mention 级粒度的结构性限制**：bilateral 和多密度混合表述问题本质上是 mention 级提取的固有局限，需要更高层次的聚合机制

### 7.2 低成本下一步（聚焦论文写作）

1. **优先级已从"继续大做实验"转为"写论文、整理图表、汇报导师"**
2. 如需小幅补充实验，density 的多密度表述处理和 location 的 bilateral 多标签分类是最值得投入的两个方向，且属于低成本逻辑调整，无需大规模重训
3. 不建议开新坑、不建议大规模新训练、不建议脱离当前项目主线

---

*本报告所有数据均来自仓库中已完成的实验结果，未包含任何未执行的实验。*
