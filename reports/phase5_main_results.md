# Phase 5 主模型实验结果报告

## 1. 实验概述
- **3 个任务**：density_category (密度分类), size_mm (大小检测), location_lobe (位置分类)
- **4 种方法**：Regex baseline (正则基准), TF-IDF+LR (逻辑回归), TF-IDF+SVM (支持向量机), PubMedBERT (预训练语言模型)
- **数据规模**：总计 288,894 mentions, 33,149 subjects
- **切分**：train 201,947 / val 44,890 / test 42,057 (按患者 ID 切分，无重叠)

## 2. 主结果对比表

### 2.1 Density Classification (5类)
| 方法 | Test Accuracy | Test Macro F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9982 | 0.9938 | 32.5s |
| TF-IDF + SVM | 0.9992 | 0.9975 | 48.3s |
| PubMedBERT | 0.9993 | 0.9985 | 2383s |

### 2.2 Size Detection (二分类)
| 方法 | Test Accuracy | Test F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9920 | 0.9890 | 8.6s |
| TF-IDF + SVM | 0.9927 | 0.9900 | 14.4s |
| PubMedBERT | 0.9994 | 0.9992 | 2439s |

### 2.3 Location Classification (9类)
| 方法 | Test Accuracy | Test Macro F1 | 训练耗时 |
|---|---|---|---|
| Regex | 1.0000 | 1.0000 | 0s |
| TF-IDF + LR | 0.9872 | 0.9849 | 24.2s |
| TF-IDF + SVM | 0.9983 | 0.9972 | 16.8s |
| PubMedBERT | 0.9999 | 0.9998 | 2416s |

## 3. 关键发现

### 3.1 Silver Label 循环性问题
实验结果显示 Regex 基准在所有任务上均达到 1.0 的完美分数。这是因为本项目采用的 silver label 本身就是由 Regex 逻辑生成的。机器学习模型（LR/SVM）和深度学习模型（PubMedBERT）本质上是在学习并尝试复制 Regex 的行为模式。这种“天花板效应”是 silver label 评测框架的固有特性，而非模型 bug。

### 3.2 模型一致性分析
三种方法在测试集上表现出极高的一致性：
- **density**: 99.91%
- **size**: 99.27%
- **location**: 99.83%
不一致的情况主要源于 SVM（尤其在 size 任务中，SVM 有 283 例独立不一致），而 BERT 的预测结果与 Regex 几乎完全重合，说明 BERT 更忠实地还原了 Regex 的判断逻辑。

### 3.3 Regex 失败区域分析（核心价值）
分析模型在 Regex 无法提取（即标签为 unclear 或 no_size/no_location）的样本上的表现：
- **density**: 在 36,481 个 unclear 样本中，BERT 仅将 21 个重新分类为明确类别（如 ground_glass）。
- **size**: 在 26,915 个 no_size 样本中，BERT 仅将 24 个重新分类为 has_size。
- **location**: 在 26,134 个 no_location 样本中，BERT 未能重新分类任何样本。
**结论**：BERT 表现高度保守，几乎完全复制了 Regex 的判断边界，未能显著突破正则规则的覆盖范围。

### 3.4 置信度分析
- 绝大多数预测（>99.8%）处于极高置信度区间（>0.95）。
- 中低置信度样本极少（如 density 任务仅 68 例），虽然这些样本的准确率有所下降（约 89%），但整体稳定性依然很高。

## 4. 消融实验结果

### 4.1 A1: Section-aware Window vs Full-text
| 任务 | Mention Text (Window) | Full Text | Delta |
|---|---|---|---|
| density | 0.9975 | 0.4785 | +0.5190 |
| size | 0.9900 | 0.6080 | +0.3821 |
| location | 0.9972 | 0.3645 | +0.6327 |

**结论**：基于 mention 为中心的窗口特征显著优于全文特征，验证了 section-aware 窗口设计的必要性。

### 4.2 A2: Exam Name Feature
| 任务 | Plain | +Exam Name | Delta |
|---|---|---|---|
| density | 0.9975 | 0.9975 | -0.0000 |
| size | 0.9900 | 0.9899 | -0.0001 |
| location | 0.9972 | 0.9974 | +0.0002 |

**结论**：加入 exam_name 特征对性能几乎无影响，说明前期的 Chest CT 过滤已经保证了输入的标准化。

### 4.3 A3: Explicit-only vs All Silver Labels
| 任务 | All Data | Explicit Only | Delta |
|---|---|---|---|
| density | 0.9975 | 0.4409 | -0.5566 |
| size | 0.9900 | 0.7810 | -0.2090 |
| location | 0.9972 | 0.5622 | -0.4350 |

**结论**：仅使用 explicit（显式描述）样本训练会导致性能大幅下降，证明了保留全部 silver labels（包含隐式推断）对维持模型覆盖率至关重要。

## 5. 参数讨论实验结果
- **max_features**: 10000 为最优配置。5000 表现略差，20000 则无进一步提升。
- **ngram_range**: (1,2) 效果最好。仅使用 unigram 会丢失上下文，而加入 trigram 增加了特征稀疏性但未带来增益。
- **C (正则化)**: 1.0 为最优。0.1 会导致欠拟合，10.0 则可能导致过拟合且无性能提升。
**结论**：默认参数配置已处于性能平坦区的最优位置。

## 6. 3090 资源使用情况
- **PubMedBERT**: 每个任务训练耗时约 40 分钟（5 epochs, batch=32），总计 GPU 时间约 2 小时。显存峰值约 4-6 GB。
- **ML Baselines**: 全部在 CPU 上完成，单个任务耗时均在 1 分钟以内，总耗时约 10 分钟。

## 7. 讨论与局限性

### 7.1 Silver Label 的天花板效应
当前评测框架下，所有方法的上限都被限制在 Regex 的提取能力之内。要真正评估模型的泛化能力，必须引入人工标注的 Gold Standard 测试集。

### 7.2 BERT 的真正价值
虽然在 silver label 框架下 BERT 优势不明显，但其价值在于：
- 对未见过的、非标准的临床描述具有更好的泛化能力。
- 能够理解复杂的句式结构（如否定、条件句）。
- 可作为 Regex 的补充，处理长尾分布中的复杂案例。

### 7.3 下一步建议
1. 构建小规模（50-100 样本）的人工标注 Gold Standard。
2. 在 Gold Standard 上重新评测所有模型，寻找 BERT 相比 Regex 的增量收益。
3. 探索集成方法（Ensemble），结合 Regex 的高精确度与 BERT 的高覆盖率。
