# 第三章 基于神经符号架构的肺结节随访建议生成方法 (Methodology Draft)

## 3.1 本章引言
针对真实世界放射学报告（如MIMIC-IV-Note）存在表述口语化、句法结构复杂、信息分布碎片化的问题，完全依赖规则系统往往难以获得高召回率；而直接使用生成式大语言模型以端到端（End-to-End）的方式生成随访建议，则面临临床规则一致性差、黑盒推理不可靠（幻觉问题）以及难以溯源审计的致命挑战。
为此，本项目提出了一种**“神经符号双层架构”（Neuro-Symbolic Architecture）**，将整个系统解耦为“神经感知层”（Neural Perception Layer，负责非结构化文本的结构化特征抽取）与“符号决策层”（Symbolic Decision Layer，负责基于临床指南的确定性逻辑推理）。

## 3.2 总体架构设计与 Schema 契约
本系统的核心数据流由四个明确的 JSON Schema 定义，分别对应不同的模块与处理阶段，确保了不同模块间的接口契约（Data Contract）与状态隔离：
1. **`radiology_fact_schema`**：定义单份放射报告的结构化事实（结节大小、密度、位置等）。
2. **`smoking_eligibility_schema`**：定义从出院小结中推断的患者级吸烟史与高危属性。
3. **`recommendation_schema`**：定义最终由符号规则引擎生成的随访建议结论与触发路径。
4. **`case_bundle_schema`**：作为系统的核心运行容器，将患者及不同时间、不同维度的多次检查记录（Radiology Facts）与患者属性（Smoking Eligibility）聚合为一个完整的实验对象（Experiment Unit），并为最终的规则引擎提供统一输入。

数据主链路为：`Radiology Report / Discharge Note → Case Bundle → Recommendation`。

## 3.3 放射学特征抽取模块 (Neural Perception)
本模块是连接非结构化文本与结构化指南的核心感知组件。我们的目标是精准抽取影响随访定级（如 Fleischner Society Guidelines）的三个核心结节属性：**结节密度（Density）、结节大小（Size_mm）和结节位置（Location）**。

### 3.3.1 预处理与候选过滤
并非所有胸部影像报告均包含肺结节。通过构建基于临床医学术语库的初筛逻辑，系统过滤无结节报告，保留存在结节或相关模糊描述的候选记录，并对原始文本进行段落解析（如提取 FINDINGS 与 IMPRESSION 部分）。

### 3.3.2 基于预训练语言模型的结构化抽取
区别于纯粹依赖硬编码正则表达式的基线系统（Phase 3 Baseline），我们引入预训练语言模型（如 PubMedBERT）以及传统机器学习分类器（TF-IDF + LR/SVM）作为核心抽取器，使用正则基线生成的银标准数据进行训练。该抽取器旨在学习和泛化规则逻辑，尝试解决以下复杂临床文本难题：
- **同义替换与模式泛化**：如学习“glassy opacity”到“ground_glass”的映射规律。
- **高维特征映射**：探索模型在多结节合并与混合描述拆解中的表现。
- **密度类型分类**：将复杂文本映射至标准枚举值 `solid`、`part_solid`、`ground_glass` 等。

## 3.4 患者属性（吸烟史）推断模块的脱敏约束与回退策略
在指南推理中，吸烟史是判定高危人群的重要特征。然而，在真实使用脱敏医疗数据集（如 MIMIC-IV）时，遇到了极其严重的脱敏限制——出院小结中关于吸烟史的具体年份或包年数（Pack-year）大量被 `___` 掩码脱敏（脱敏率接近 98%）。
针对这一客观限制，本模块设计为“弱监督推断与回退（Fallback）机制”。当由于数据脱敏无法获得确切数值时，系统保守评估，并在 `case_bundle` 中标注 `is_desensitized: true` 与 `uncertainty_level: high`。这不仅避免了让预测模型 去“猜”脱敏数据带来的幻觉问题，也保证了下游规则引擎在缺乏明确高危证据时可以执行安全的保守建议（如按照常规风险处理）。

## 3.5 符号化指南推理引擎 (Symbolic Reasoning)
神经感知层将信息沉淀在 `case_bundle` 后，交由符号引擎进行确切推理。以 Fleischner Society 肺结节随访指南为核心逻辑构建了基于属性树的决策引擎（Rule Engine）。

### 3.5.1 共享尺寸解析器与逻辑分离
由于 `size_mm` 经常以各种格式（如 “6 x 5 mm”, “0.8 cm”, “1.2x1.0x0.9cm”）出现，纯文本生成易带来数值不稳定性。系统引入了一个“共享正则尺寸解析模块”（Shared Parser）。模型仅负责提取提及尺寸的“边界原文”（Mention Text），具体最大径（Max Diameter）的毫米级换算由共享的 Python 逻辑处理。此设计避免了让序列标注模型处理复杂的浮点运算，从而显著提升了数值相关规则判断的鲁棒性。

### 3.5.2 确定性随访生成
规则引擎读取 `case_bundle` 内的核心字段（Density, Max Size, Age, Smoking），严格按照预定义逻辑树（Decision Tree）落入对应分类节点，并输出具体的随访建议区间（如“3-6个月复查CT”），同时输出完整的 `trigger_path`，确保每一次医学判断 100% 可解释、可溯源审计。
