# 项目三实验结果汇报提纲

> 适用场景：导师面对面汇报 / 组会汇报
> 建议时长：10–15 分钟

---

## 1. 项目三目前做完了什么

我们从 MIMIC-IV 放射学报告中提取肺结节的结构化临床事实，目标是支撑 Lung-RADS 指南级别的随访建议生成。

已完成三个阶段：
- **Phase 4**：建立了 Regex Baseline 和 Silver Standard 评测协议，完成了 section-aware vs full-text、smoking fallback、recommendation 规则推理三组对比实验
- **Phase 5**：在 28.9 万条 mentions 上训练了 4 种方法（Regex / TF-IDF+LR / TF-IDF+SVM / PubMedBERT），覆盖密度分类、尺寸检测、位置分类三个任务，并完成了消融和参数讨论
- **Phase 5.1**：人工标注 62 条 Gold 样本，正式验证了 Silver Ceiling 现象

---

## 2. 最关键的对比实验结果

**Phase 4 核心结论**：
- Section-aware 设计有效：限制搜索范围到 FINDINGS/IMPRESSION 后，减少了 57.5% 的假阳性结节，关键字段提取率均提升
- 结构化规则推理优于直接抓 cue：guideline anchor 从 0% 提升到 67.2%

**Phase 5 核心结论**：
- 在 Silver 测试集上，所有方法表现趋同（Macro F1 均 >0.99），Regex 达到 1.0
- **这不是 Regex 完美，而是 Silver Label 由 Regex 生成导致的循环性问题**
- PubMedBERT 在 Regex 失败区域表现极度保守，未能突破规则覆盖边界

---

## 3. 最关键的消融实验结果

三组消融中最重要的一组：

**Mention-centered Window vs Full-text**：
| 任务 | Window | Full-text | Delta |
|---|---|---|---|
| density | 0.9975 | 0.4785 | **+0.52** |
| location | 0.9972 | 0.3645 | **+0.63** |

窗口设计带来了 38–63 个百分点的增益，是系统性能的关键支撑。

另外两组消融：
- exam_name 特征无效（delta < 0.001），说明前期数据过滤已足够
- 仅用 explicit 样本训练会导致性能暴跌（-0.21 到 -0.56），证明 silver label 弱监督策略的覆盖率优势

---

## 4. 最关键的 Gold 验证结果

**Silver Ceiling 被正式证实**：
- 62 条人工标注样本上，所有 5 种方法的 density 预测完全一致（差异为零）
- density Macro F1 从 silver 上的 >0.99 骤降至 gold 上的 **0.70**，缺口达 29 个百分点
- 模型学会的是"复制规则系统"，而非"理解临床语义"

**各字段真实质量**：
| 字段 | Silver-Gold 一致率 | Gold 表现 | 投入价值 |
|---|---|---|---|
| density_category | 72.58% | Macro F1 = 0.70 | 最值得继续优化 |
| has_size | 93.55% | F1 = 0.96 | 已较稳定 |
| location_lobe | 82.26% | Macro F1 = 0.87 | bilateral 是真实难点 |
| size_mm | 87.10% | 85% 完全匹配 | 共享模块评测，非方法对比 |

---

## 5. 当前最大局限

1. **Silver Ceiling**：在 silver 框架下无法区分方法间真实能力差异，限制了主模型实验的解释空间
2. **Smoking 模块受脱敏限制**：97.9% Social History 被掩码，不适合当主实验核心
3. **Gold 样本量有限**：N=62 为方向性参考，罕见类别指标波动较大
4. **Mention 级粒度限制**：bilateral 和多密度混合表述是结构性难点

---

## 6. 接下来准备做什么

**当前优先级已从"继续大做实验"转为"写论文、整理图表、汇报导师"。**

具体计划：
1. 基于本报告整理论文实验章节的图表和数据
2. 如需小幅补充，density 多密度处理和 location bilateral 多标签分类是最值得投入的两个低成本方向
3. 不开新坑，不做大规模新训练

---

## 附：5 条核心结论（一句话版）

1. Section-aware + mention-centered 窗口设计是系统性能的基石，消融证明带来 38–63pp 增益
2. Silver Label 评测存在严重循环性问题，Regex 的 1.0 不代表真实完美
3. Gold 评测证实 density 真实 Macro F1 仅 0.70，所有模型表现完全一致，Silver Ceiling 被正式确认
4. 当前系统瓶颈在标签质量和字段可提取性，而非模型架构或超参数
5. density_category 和 location bilateral 是最值得继续投入的两个方向

