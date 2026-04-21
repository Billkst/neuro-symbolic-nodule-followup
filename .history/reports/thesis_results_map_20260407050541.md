# 论文图表与结果映射清单 (Thesis Results Map)

本文档旨在列出毕业论文各章节中“最值得放进正文”与“适合放入附录”的图表清单，明确每一张图表在论文叙事中承担的具体“证明责任”，避免图表堆砌或重复证明。

## 一、 正文主图清单 (Top 5 核心图)

### 1. 神经符号双层系统总体架构图 (System Architecture Diagram)
- **存放位置**：第三章 方法
- **来源建议**：需新画
- **证明目的**：展示整个 Pipeline 的全景。体现“非结构化文本 → (神经抽取网络) → JSON/Case Bundle → (符号规则引擎) → 标准化指南随访建议”的设计思想。

### 2. 共享尺寸解析模块数据流图 (Shared Size Parser Data Flow)
- **存放位置**：第三章 方法 (3.5节)
- **来源建议**：需新画（可基于 Phase 5.1a 修正口径说明）
- **证明目的**：说明为什么预测模型只提取“提及文本(Mention Text)”，而数值换算交由 Python 规则完成。证明工程设计的严谨性。

### 3. Phase 4 vs Phase 5 主实验对比柱状图 (Main Results Benchmark)
- **存放位置**：第五章 实验结果
- **来源建议**：可根据 `phase5_main_results.md` 中的 F1 / AUPRC 表格整理生成柱状图。
- **证明目的**：直观展示各种预测模型（Phase 5）在银标准上如何逼近并高度拟合 Regex 基线的表现极限。

### 4. 银标准与金标准性能落差对比图 (Silver vs Gold Performance Drop)
- **存放位置**：第五章 实验结果 (5.4 节)
- **来源建议**：将 Phase 5 (Silver) 和 Phase 5.1 (Gold) 中 `density_category` 的 Macro-F1 放进同一张对比图中。
- **证明目的**：这是全文最重要的发见之一。视觉化展示“Silver Ceiling”现象，证明模型在真实医学长尾语义理解上仍有待突破。

### 5. Density Category 混淆矩阵 (Density Confusion Matrix)
- **存放位置**：第五章 误差分析
- **来源建议**：基于 Phase 5 阶段模型对 `density_category` 的预测结果绘制。
- **证明目的**：用数据支撑“多密度混合描述是核心难点”的结论。观察模型是更倾向于将复杂描述分到 `solid` 还是 `unclear`。

---

## 二、 正文主表清单 (Top 5 核心表)

### 表1：模块间核心 Schema 字段定义摘要表
- **存放位置**：第三章 方法
- **内容来源**：`reports/schema_design.md` 简化版。
- **证明目的**：精简展示 `case_bundle` 中与随访推断最强相关的字段（density, size_mm, location, age, smoking）。

### 表2：Phase 5 主模型银标准评测核心指标汇总表
- **存放位置**：第五章 实验结果
- **内容来源**：基于 `phase5_main_results.md` 产出的 AUPRC 和 Macro-F1 结果整理生成。
- **证明目的**：用数字证明各预测模型在银标准测试中高度拟合了规则标注结果（天花板效应）。

### 表3：Phase 5.1 人工金标准 (Gold Eval) 评测对照表
- **存放位置**：第五章 实验结果 (5.4 节)
- **内容来源**：`reports/phase5_1_gold_eval.md` 的汇总数据。
- **证明目的**：小样本下（N=62），展示各字段的真实临床抽取准确率。

### 表4：典型误差案例分类表 (Error Case Taxonomy)
- **存放位置**：第五章 误差分析
- **内容来源**：基于 `reports/phase5_error_analysis.md` 整理。
- **要求格式**：包含【误差类型】、【原文本节选】、【模型预测】、【人工Ground Truth】、【原因剖析】。
- **证明目的**：增加医学可解释性，如“Bilateral 被错误解析为单一叶位”、“多结节混合表述干扰”。

### 表5：不同模块对于系统整体鲁棒性的影响表（消融/模块对比）
- **存放位置**：第五章 实验结果
- **内容来源**：基于共享解析器 (Shared Parser) 开启前后的对比口径（Phase 5.1a）。
- **证明目的**：证明在工程约束中，“让大模型做自己擅长的事（分类/匹配）”与“让代码做自己擅长的事（算术运算）”结合的必要性。

---

## 三、 附录清单 (Appendices)

这些内容对于保持系统开源严谨性至关重要，但放正文会干扰阅读节奏。

1. **附录 A：完整的 JSON Schema 定义与示例**
   - 包含完整的 `radiology_fact_schema.json`、`smoking_eligibility_schema.json` 与 `case_bundle_schema.json` 完整示例。
2. **附录 B：Phase 3 Baseline 核心正则表达式与规则清单**
   - 列举传统医学 NLP 抽取 Baseline 的具体规则，方便他人复现。
3. **附录 C：Fleischner 肺结节随访指南 (2017版) 规则逻辑映射树**
   - 以树状图或伪代码形式，详细说明本项目规则引擎是如何完整覆盖指南中的推荐逻辑的。
4. **附录 D：全量62条金标准(Gold Eval)的评测对比明细**
   - 如有必要，可放入小样本评测的完整 case 明细供同行审查。
