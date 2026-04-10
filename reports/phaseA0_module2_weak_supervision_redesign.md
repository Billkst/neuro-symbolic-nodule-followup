# Phase A0：模块2方法重构与弱监督升级定案

> 生成日期：2026-04-10
> 状态：**定案版**——后续 Phase A1/A2 执行阶段以本文件为唯一权威参考
> 约束：本阶段仅产出计划，不写代码、不跑实验、不改 schema、不改现有报告正文
> 前置文件：`reports/final_experiment_redesign_plan.md`（方案三总体计划定案版）

---

# 1. 执行摘要

模块2（结构化信息抽取）当前已有完整的 Phase 5/5.1 实验链，但存在一个核心方法论缺陷：**当前 silver labels 完全由单一 Regex teacher 生成，导致所有下游模型（LR/SVM/PubMedBERT）在 silver test 上的高分仅反映对 teacher 的拟合，而非真实临床语义理解能力。** Phase 5.1 Gold 评测已实证确认这一问题——density Macro F1 从 silver 的 >0.99 骤降至 gold 的 0.70，且五种方法预测完全一致。

本计划将模块2从"Regex 打标签 + PubMedBERT 微调"的工程 pipeline，升级为一个**正式的多源弱监督信息抽取框架**。核心升级包括：

1. **方法身份重定义**：从 "Task-Specific Pipeline Framework" 升级为 **"Multi-Source Weak Supervision Framework for Clinical Fact Extraction"（MWS-CFE）**
2. **弱监督升级**：将单一 Regex teacher 拆解为 5-8 个独立 labeling functions，引入 label aggregation 和 quality gate
3. **实验矩阵重构**：新增弱监督消融维度，补齐 Vanilla PubMedBERT baseline，重组参数讨论
4. **模块3接口保障**：确保输出字段、置信度、缺失性标记足以支撑后续图谱智能体

**资源约束**：单张 RTX 3090（24GB），所有设计均在此约束下可行。弱监督升级的核心开销在 CPU 端（labeling function 执行 + label aggregation），不额外消耗 GPU。

---

# 2. 当前模块2的现状与问题

## 2.1 现有资产盘点

| 资产 | 状态 | 位置 |
|------|------|------|
| 数据集（288,894 mentions, 33,149 subjects, train/val/test 按 subject 切分无重叠） | 已构建 | `outputs/phase5/datasets/` |
| Regex baseline（density/size/location 三任务） | 已完成 | `outputs/phase5/results/regex_*.json` |
| TF-IDF+LR baseline（三任务） | 已完成 | `outputs/phase5/results/ml_lr_*.json` |
| TF-IDF+SVM baseline（三任务） | 已完成 | `outputs/phase5/results/ml_svm_*.json` |
| Method-PubMedBERT（mention-centered + section-aware，三任务） | 已完成 | `outputs/phase5/models/` + `results/pubmedbert_*.json` |
| 消融实验 A1（window vs full-text）、A2（exam_name）、A3（explicit-only vs all） | 已完成 | `outputs/phase5/results/ablation_*.json` |
| 参数讨论（max_features, ngram_range, C） | 已完成 | `outputs/phase5/results/param_sweep_*.json` |
| Gold 评测（N=62，人工标注） | 已完成 | `outputs/phase5_1/` |
| 错误分析 | 已完成 | `reports/phase5_error_analysis.md` |
| Section parser | 已实现 | `src/parsers/section_parser.py` |
| Mention segmentation | 已实现 | `src/extractors/nodule_extractor.py::segment_nodule_mentions()` |
| Label quality 三级分层（explicit/silver/weak） | 已实现 | `scripts/phase5/build_datasets.py::_infer_label_quality()` |

## 2.2 Silver Ceiling 问题（核心缺陷）

### 问题本质

当前 silver labels 的生成链路为：

```
报告文本 → section_parser → segment_nodule_mentions → extract_density/size/location (Regex)
                                                        ↓
                                                   silver labels
                                                        ↓
                                              TF-IDF/PubMedBERT 训练
                                                        ↓
                                              silver test 评测 → 虚高分数
```

这条链路的根本问题是：**评测标签与训练标签来自同一个 Regex teacher**。模型学到的是"如何复制 Regex 的行为"，而非"如何理解临床语义"。

### 实证证据

| 证据 | 数据 | 结论 |
|------|------|------|
| Silver test 上 Regex 得分 1.0 | Phase 5 主结果 | 循环论证——标签就是 Regex 生成的 |
| Silver test 上所有方法 Macro F1 > 0.99 | Phase 5 主结果 | 模型完美拟合了 teacher |
| Gold test 上 density Macro F1 = 0.70 | Phase 5.1 | 真实性能比 silver 低 29pp |
| Gold test 上五种方法预测完全一致 | Phase 5.1 | 所有模型都只是 Regex 的复制品 |
| PubMedBERT 在 Regex 失败区域几乎无突破 | Phase 5 错误分析 | BERT 未学到超越 Regex 的能力 |

### 为什么"Regex 打标签再训练 PubMedBERT"不能直接包装为弱监督方法

1. **单一 teacher 不构成弱监督**：弱监督的核心前提是多个独立、有噪声的信号源通过聚合产生比任何单一源更好的标签。当前只有一个 Regex teacher，不满足这一前提。
2. **无 label denoising**：当前 `_infer_label_quality()` 仅基于字段是否存在做三级分层，不涉及标签噪声估计或修正。
3. **无 teacher-student 解耦**：PubMedBERT 直接吃 Regex 的硬标签训练，没有 soft label、confidence weighting 或 curriculum 策略。
4. **论文审稿风险**：如果在论文中声称"弱监督"但实际只有一个 Regex teacher，审稿人会直接指出这不是真正的弱监督。

## 2.3 当前方法线不足

当前模块2的"方法"实质上是一个**工程 pipeline**：

- Section parsing → mention segmentation → Regex extraction → PubMedBERT fine-tuning

其中：
- **mention-centered window** 是有价值的设计贡献（消融证明 +38-63pp）
- **section-aware preprocessing** 是有效的工程决策
- 但 **标签策略** 只是"用 Regex 打标签"，缺乏方法论深度

要让模块2在论文中站住脚，必须在标签策略维度引入真正的方法性内容——即多源弱监督。

---

# 3. 模块2"我们的方法"重定义

## 3.1 为什么不能再只叫"pipeline"

1. **论文定位要求**：模块2占方案三实验权重的 ~45%，是核心感知模块。如果方法只是"工程 pipeline"，论文第三/四章缺乏方法论贡献。
2. **审稿人预期**：计算机方向硕士论文需要展示方法设计能力，不能只是"调用 Regex + 微调 BERT"。
3. **Silver ceiling 暴露了方法缺陷**：Gold 评测证明当前方法的高分是虚假的，必须在方法层面解决，而非仅靠"补一个 Vanilla baseline"。

## 3.2 方法命名与定义

**正式名称**：**MWS-CFE**（Multi-source Weak Supervision for Clinical Fact Extraction）

**论文中的表述**：

> 我们提出 MWS-CFE，一个面向放射报告结构化信息抽取的多源弱监督框架。该框架将传统的单一正则提取器拆解为多个独立的标注函数（labeling functions），通过标签聚合模型（label aggregation model）融合多源噪声信号，并结合质量门控（quality gate）策略进行训练数据筛选，最终在 PubMedBERT 上进行微调。

**与旧版 "Method-PubMedBERT" 的关系**：

- 旧版 Method-PubMedBERT 是 MWS-CFE 框架的一个特例（单源 Regex teacher + 无 aggregation + 无 quality gate）
- 新版 MWS-CFE 是对旧版的正式升级，不是推翻

## 3.3 核心组成（5 个不可分割的组件）

| # | 组件 | 具体设计 | 对应消融/参数讨论 | 与旧版的差异 |
|---|------|---------|-----------------|------------|
| C1 | Section-aware Preprocessing | 限定 FINDINGS/IMPRESSION 段落 | A-section（旧 A2 升级） | 无变化，沿用 |
| C2 | Mention-centered Text Window | 以结节 mention 为中心截取局部上下文 | A-window（旧 A1 升级） | 无变化，沿用 |
| C3 | Multi-source Labeling Functions | 5-8 个独立 LF 替代单一 Regex teacher | A-lf（新增消融） | **核心新增** |
| C4 | Label Aggregation + Quality Gate | 多源信号融合 + 训练数据质量筛选 | A-agg（新增消融）+ P3 参数讨论 | **核心新增** |
| C5 | PubMedBERT Fine-tuning | 在聚合后的高质量标签上微调 | 主结果对比 | 训练数据质量提升 |

## 3.4 各组件中的角色定位

- **PubMedBERT**：可替换的主干模型（backbone），不是方法的核心贡献。其价值在于对非标准表述的泛化能力。
- **弱监督（C3+C4）**：方法的核心方法论贡献。将"如何获得高质量训练标签"从工程问题提升为方法问题。
- **Mention-centered window（C2）**：方法的核心工程贡献。消融已证明其 +38-63pp 的增益。
- **Section-aware（C1）**：必要的预处理步骤，非独立贡献点。

## 3.5 相较当前版本，新增的真正方法性内容

| 维度 | 当前版本 | MWS-CFE 升级版 |
|------|---------|---------------|
| 标签来源 | 单一 Regex teacher | 5-8 个独立 labeling functions |
| 标签融合 | 无（直接使用 Regex 输出） | Label aggregation model（加权投票或概率模型） |
| 标签质量控制 | 三级离散分层（explicit/silver/weak） | Quality gate + confidence-weighted filtering |
| 训练策略 | 全量 silver 直接训练 | 分层训练 / curriculum / quality-aware sampling |
| 方法论深度 | 工程 pipeline | 正式的弱监督学习框架 |


---

# 4. 弱监督升级方案设计

## 4.1 当前单一 Teacher Silver 的局限

| 局限 | 具体表现 | 影响 |
|------|---------|------|
| 单点故障 | Regex 无法识别的模式（如 "possibly part-solid"、"7mmsince"），所有下游模型同样无法识别 | Gold density Macro F1 仅 0.70 |
| 系统性偏差 | Regex 对多密度混合表述（"solid and ground-glass"）总是取第一个匹配，Gold 评测中 10/17 density 错误源于此 | unclear 召回率仅 0.29 |
| 无冲突信号 | 只有一个信号源，无法检测标签噪声 | 无法估计标签置信度 |
| 覆盖率受限 | Regex 对非标准缩写（"GGO"、"subsolid"）和拼写错误（"4mmm"）覆盖不全 | Phase 5 错误分析中 24 例 size 错误全部是 Regex 漏检 |
| 位置提取单一化 | Regex 只取第一个匹配的叶位，无法处理多叶位（bilateral） | Gold location bilateral F1 仅 0.53 |

## 4.2 Labeling Functions 设计

### 设计原则

1. **独立性**：每个 LF 必须基于不同的信号源或不同的提取策略，不能是同一 Regex 的简单变体
2. **可组合性**：LF 之间可以冲突（conflict），冲突本身是有价值的信号
3. **覆盖率互补**：不同 LF 覆盖不同的文本模式，联合覆盖率应显著高于单一 Regex
4. **ABSTAIN 机制**：每个 LF 在无法判断时必须返回 ABSTAIN，而非强行给出标签

### 4.2.1 Density 任务的 Labeling Functions

| LF ID | 名称 | 信号源 | 策略 | 预期覆盖 | 预期精度 |
|-------|------|--------|------|---------|---------|
| LF-D1 | Keyword-Exact | 精确关键词匹配 | 当前 `extract_density()` 的核心逻辑：part-solid > ground-glass > calcified > solid | ~60% mentions | 高（>0.90） |
| LF-D2 | Keyword-Fuzzy | 模糊/非标准关键词 | 扩展匹配：sub-solid, hazy, frosted, attenuating 等同义词和非标准表述 | ~10% 额外 | 中（~0.75） |
| LF-D3 | Negation-Aware | 否定词检测 | 检测 "not definitely calcified", "no evidence of solid" 等否定模式，输出 ABSTAIN 或修正标签 | ~5% mentions | 高（>0.85） |
| LF-D4 | Multi-Density-Detector | 多密度共现检测 | 当 mention 中同时出现 2+ 密度关键词时，输出 unclear 而非取第一个 | ~8% mentions | 高（>0.90） |
| LF-D5 | Impression-Section-Cue | IMPRESSION 段落线索 | 从 IMPRESSION 段落中提取放射科医生的总结性密度判断（通常比 FINDINGS 更准确） | ~30% mentions | 高（>0.85） |

**设计理由**：
- LF-D1 是当前 Regex 的核心，保留作为高精度基础源
- LF-D2 扩展覆盖率，捕获 Regex 遗漏的非标准表述
- LF-D3 直接解决 Gold 评测中最大的错误模式（否定词识别失效，22/29 density 错误）
- LF-D4 直接解决 Gold 评测中第二大错误模式（多密度混合表述，10/17 gold density 错误）
- LF-D5 利用报告结构信息（IMPRESSION 通常是放射科医生的最终判断）

### 4.2.2 Has_Size 任务的 Labeling Functions

| LF ID | 名称 | 信号源 | 策略 | 预期覆盖 | 预期精度 |
|-------|------|--------|------|---------|---------|
| LF-S1 | Regex-Standard | 标准尺寸模式 | 当前 `extract_size()` 的核心逻辑：N mm, NxN mm, N-N mm, N cm | ~85% has_size | 高（>0.95） |
| LF-S2 | Regex-Tolerant | 容错尺寸模式 | 处理拼写错误（"7mmsince"、"4mmm"）、缺空格、非标准单位 | ~3% 额外 | 高（>0.90） |
| LF-S3 | Numeric-Context | 数字+上下文 | 检测 mention 中出现数字且上下文含 "measuring", "diameter", "size" 等词 | ~90% has_size | 中（~0.80） |
| LF-S4 | Subcentimeter-Cue | 定性尺寸描述 | 检测 "subcentimeter", "tiny", "small", "large" 等定性尺寸词 | ~15% mentions | 中（~0.70） |
| LF-S5 | No-Size-Negative | 无尺寸负信号 | 当 mention 中无任何数字且无定性尺寸词时，输出 no_size | ~30% mentions | 高（>0.90） |

### 4.2.3 Location 任务的 Labeling Functions

| LF ID | 名称 | 信号源 | 策略 | 预期覆盖 | 预期精度 |
|-------|------|--------|------|---------|---------|
| LF-L1 | Lobe-Exact | 精确叶位匹配 | 当前 `extract_location()` 的核心逻辑 | ~70% mentions | 高（>0.90） |
| LF-L2 | Multi-Lobe-Detector | 多叶位检测 | 当 mention 中出现 2+ 不同叶位时，输出 bilateral | ~8% mentions | 高（>0.85） |
| LF-L3 | Bilateral-Keyword | 双肺关键词 | 检测 "bilateral", "both lungs", "scattered throughout", "diffuse" | ~5% mentions | 高（>0.90） |
| LF-L4 | Laterality-Inference | 侧别推断 | 从 "right lung" / "left lung"（无具体叶位）推断侧别，输出 unclear 而非 None | ~5% mentions | 中（~0.75） |
| LF-L5 | Context-Window-Location | 上下文窗口位置 | 当 mention 本身无位置信息但前一句/后一句有叶位描述时，继承位置 | ~10% mentions | 中（~0.70） |

## 4.3 Label Aggregation 方案

### 方案选择：加权多数投票（Weighted Majority Vote）

**为什么不用 Snorkel Label Model**：
1. Snorkel 的概率标签模型（data programming）需要大量 LF 之间的冲突/覆盖统计来估计 LF 精度，当 LF 数量仅 5 个时统计量不足
2. Snorkel 的安装和调试成本较高，对硕士论文工期不友好
3. 加权多数投票在 LF 数量 < 10 时与 Snorkel Label Model 性能差异不大（文献支撑：Ratner et al. 2017 的消融实验）

**为什么不用简单多数投票**：
1. 不同 LF 的精度差异很大（LF-D1 精度 >0.90，LF-D2 精度 ~0.75）
2. 简单多数投票会让低精度 LF 拉低整体质量

### 聚合算法

对每个 mention 的每个任务（density/size/location）：

```
输入：K 个 LF 的输出 {(label_k, weight_k) | k=1..K}，其中 label_k 可以是 ABSTAIN
步骤：
1. 过滤掉 ABSTAIN 的 LF
2. 对剩余 LF 按 label 分组，计算每个 label 的加权得分：score(l) = sum(weight_k for k where label_k == l)
3. 选择得分最高的 label 作为聚合标签
4. 计算聚合置信度：confidence = score(winner) / sum(all scores)
5. 如果所有 LF 都 ABSTAIN，输出 ABSTAIN（该样本不参与训练）
```

### LF 权重确定

**初始权重**：基于 Gold 评测集（N=62）上的 LF 精度估计。具体做法：
1. 在 Gold 集上运行每个 LF
2. 计算每个 LF 的 precision（仅在非 ABSTAIN 样本上）
3. 以 precision 作为权重

**权重校准**：如果 Gold 集太小（N=62）导致权重估计不稳定，可采用以下 fallback：
- LF-*1（精确匹配类）：权重 1.0
- LF-*2/3（扩展/否定类）：权重 0.8
- LF-*4/5（上下文/推断类）：权重 0.6

### 与现有 explicit/silver/weak 三层的关系

当前 `_infer_label_quality()` 的三级分层逻辑：
- `explicit`：density_text + size_text + location_text 全部存在
- `silver`：至少一个 *_text 存在
- `weak`：所有 *_text 均为 None

**升级后的关系**：

| 旧层级 | 新框架中的对应 | 处理方式 |
|--------|-------------|---------|
| explicit | 多个 LF 高置信度一致 → 聚合置信度高 | 直接使用，权重最高 |
| silver | 部分 LF 有输出，部分 ABSTAIN → 聚合置信度中等 | 通过 quality gate 筛选 |
| weak | 大部分 LF ABSTAIN → 聚合置信度低 | 可选择排除或降权 |

**关键设计决策**：旧的三级分层不废弃，而是作为 quality gate 的一个维度被吸收到新框架中。新框架增加了 LF 一致性维度和聚合置信度维度。

## 4.4 Quality Gate 设计

Quality Gate 是训练数据筛选策略，决定哪些样本进入 PubMedBERT 训练集。

### Gate 维度

| 维度 | 计算方式 | 阈值范围 |
|------|---------|---------|
| 聚合置信度 | `score(winner) / sum(all scores)` | 0.5 ~ 1.0 |
| LF 覆盖数 | 非 ABSTAIN 的 LF 数量 | 1 ~ K |
| LF 一致性 | 非 ABSTAIN LF 中投票给 winner 的比例 | 0.5 ~ 1.0 |
| 旧 label_quality | explicit / silver / weak | 离散 |

### 预设 Gate 策略（用于参数讨论实验）

| Gate 策略 | 筛选条件 | 预期训练集规模 | 预期标签质量 |
|----------|---------|-------------|------------|
| G1: No Gate（全量） | 所有非 ABSTAIN 样本 | ~250K | 基线 |
| G2: Confidence ≥ 0.7 | 聚合置信度 ≥ 0.7 | ~200K | 中等提升 |
| G3: Coverage ≥ 2 | 至少 2 个 LF 非 ABSTAIN | ~180K | 中等提升 |
| G4: Agreement ≥ 0.8 | LF 一致性 ≥ 0.8 | ~150K | 显著提升 |
| G5: Strict（Conf ≥ 0.8 AND Coverage ≥ 2） | 同时满足置信度和覆盖数 | ~120K | 最高 |

## 4.5 训练策略建议

### 推荐方案：Quality-Aware Sampling

不采用 curriculum learning（实现复杂度高，工期风险大），而是采用更简单的 quality-aware sampling：

1. **Phase 1**：在 G5（Strict gate）筛选后的高质量子集上训练 PubMedBERT（~120K 样本，~2h GPU）
2. **Phase 2**：用 Phase 1 模型对全量数据做推理，获取模型置信度
3. **Phase 3**：将模型置信度与 LF 聚合置信度结合，构建最终训练集（self-training 一轮）

**简化版（如果工期紧张）**：只做 Phase 1，跳过 self-training。这已经比当前的"全量 silver 直接训练"有方法论提升。

### 3090 可行性评估

| 步骤 | 计算资源 | 预估耗时 |
|------|---------|---------|
| 运行 5 个 LF（per task） | CPU only | ~10 min |
| Label aggregation | CPU only | ~5 min |
| Quality gate filtering | CPU only | ~1 min |
| PubMedBERT 训练（per task） | GPU（3090） | ~40 min |
| 推理获取置信度（可选 self-training） | GPU（3090） | ~15 min |
| **总计（3 tasks）** | | **~3.5h（无 self-training）/ ~5h（含 self-training）** |

**结论**：完全在 3090 约束内可行。


---

# 5. 模块2最终实验矩阵

## 5.1 最终方法集合

| # | 方法 | 论文中命名 | 类型 | 标签来源 | 状态 |
|---|------|----------|------|---------|------|
| 1 | Regex Baseline | Regex | 规则基线 | N/A（规则直接输出） | 已有结果 |
| 2 | TF-IDF + LR | TF-IDF + LR | ML 基线 | 单源 Regex silver | 已有结果 |
| 3 | TF-IDF + SVM | TF-IDF + SVM | ML 基线 | 单源 Regex silver | 已有结果 |
| 4 | Vanilla PubMedBERT | Vanilla PubMedBERT | 深度学习基线 | 单源 Regex silver, full-text 输入 | **必须补跑** |
| 5 | MWS-CFE (Ours) | MWS-CFE (Ours) | 主模型 | 多源 LF 聚合标签, mention-centered 输入 | **必须补跑** |

**与旧版 Method-PubMedBERT 的关系**：

- 旧版 Method-PubMedBERT 的已有结果（Phase 5）可作为 "Single-Source Baseline" 在消融实验中复用
- 新版 MWS-CFE 使用多源聚合标签训练，是正式的 "Ours"
- 论文主结果表中，MWS-CFE 替代旧版 Method-PubMedBERT 作为 "Ours"

## 5.2 最终主结果任务

| 任务 | 标签类型 | 类别数 | 主指标 | 论文地位 |
|------|---------|--------|--------|---------|
| density_category | 5 分类（solid/part_solid/ground_glass/calcified/unclear） | 5 | Macro F1 | **正式主任务** |
| has_size | 二分类（has_size / no_size） | 2 | F1 | **正式主任务** |
| location_lobe | 9 分类（RUL/RML/RLL/LUL/LLL/lingula/bilateral/unclear/no_location） | 9 | Macro F1 | **正式主任务** |
| size_mm | 回归（连续值） | N/A | MAE, ±1mm 覆盖率 | **共享解析器补充指标**（非方法间对比） |

**size_mm 口径说明**：所有方法的 size_mm 预测均来自同一共享正则尺寸解析器（`extract_size()`），不构成方法间独立对比。在论文中作为"共享解析模块的 Gold 标准表现"单独报告，不列入主结果对比表。

## 5.3 主结果表设计

### 表 M2-1：Silver Test 主结果

| 方法 | density Macro F1 | has_size F1 | location Macro F1 |
|------|-----------------|------------|------------------|
| Regex | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + LR | 0.9938 | 0.9890 | 0.9849 |
| TF-IDF + SVM | 0.9975 | 0.9900 | 0.9972 |
| Vanilla PubMedBERT | — | — | — |
| MWS-CFE (Ours) | — | — | — |

> 注：Regex 在 silver test 上得分 1.0 是因为 silver labels 由 Regex 生成（循环论证）。此表的价值在于展示 ML/DL 方法对 teacher 的拟合程度。

### 表 M2-2：Gold Test 主结果（N=62）

| 方法 | density Macro F1 | has_size F1 | location Macro F1 |
|------|-----------------|------------|------------------|
| Regex / Silver | 0.7003 | 0.9615 | 0.8730 |
| TF-IDF + LR | 0.7003 | 0.9615 | 0.8547 |
| TF-IDF + SVM | 0.7003 | 0.9515 | 0.8886 |
| Vanilla PubMedBERT | — | — | — |
| MWS-CFE (Ours) | — | — | — |

> 注：当前 Regex/Silver/LR/PubMedBERT 在 density 上预测完全一致（均为 0.7003），这正是 silver ceiling 的实证。MWS-CFE 的目标是在 Gold test 上打破这一 ceiling。

### 表 M2-3：Silver vs Gold 性能差距分析

| 任务 | Silver Macro F1（最佳方法） | Gold Macro F1（最佳方法） | Gap | 主要错误模式 |
|------|--------------------------|------------------------|-----|------------|
| density | 0.9985 | 0.7003 | -0.2982 | 多密度混合、否定词 |
| has_size | 0.9992 | 0.9615 | -0.0377 | mention 边界截断 |
| location | 0.9998 | 0.8730 | -0.1268 | 多叶位→bilateral |

## 5.4 消融实验设计

### A-window：Mention-centered Window vs Full-text

**目的**：验证 mention-centered window 设计的核心价值。

| 设置 | 输入方式 | 标签来源 | 模型 |
|------|---------|---------|------|
| MWS-CFE (mention window) | mention-centered text | 多源聚合 | PubMedBERT |
| MWS-CFE (full-text) | 完整报告文本 | 多源聚合 | PubMedBERT |

**可复用**：Phase 5 A1 消融（TF-IDF+LR 上的 window vs full-text）可作为补充证据引用。

### A-section：Section-aware vs No Section-aware

**目的**：验证限定 FINDINGS/IMPRESSION 段落的价值。

| 设置 | Section 策略 | 标签来源 | 模型 |
|------|-------------|---------|------|
| MWS-CFE (section-aware) | 仅 FINDINGS + IMPRESSION | 多源聚合 | PubMedBERT |
| MWS-CFE (all sections) | 全部段落 | 多源聚合 | PubMedBERT |

**可复用**：Phase 4 section-aware vs full-text 结果可作为补充证据引用。

### A-lf：Multi-source LF vs Single-source Regex（核心新增消融）

**目的**：验证多源 labeling functions 相比单一 Regex teacher 的增益。这是弱监督升级的核心消融。

| 设置 | 标签来源 | LF 数量 | 模型 |
|------|---------|--------|------|
| Single-source (Regex only) | 仅 LF-*1（当前 Regex） | 1 per task | PubMedBERT |
| Multi-source (All LFs) | 全部 LF 聚合 | 5 per task | PubMedBERT |

**关键对比**：Single-source 设置等价于旧版 Method-PubMedBERT（Phase 5 已有结果可直接复用），Multi-source 设置是新版 MWS-CFE。

### A-agg：Label Aggregation 策略消融（核心新增消融）

**目的**：验证加权投票相比简单策略的增益。

| 设置 | 聚合策略 | 模型 |
|------|---------|------|
| Majority Vote | 简单多数投票（等权） | PubMedBERT |
| Weighted Vote | 加权多数投票（精度加权） | PubMedBERT |
| Best-LF Only | 仅使用精度最高的单个 LF | PubMedBERT |

### A-qg：Quality Gate 消融

**目的**：验证训练数据质量筛选的价值。

| 设置 | Quality Gate | 预期训练集规模 | 模型 |
|------|-------------|-------------|------|
| No Gate | 全量非 ABSTAIN 样本 | ~250K | PubMedBERT |
| Confidence Gate | 聚合置信度 ≥ 0.7 | ~200K | PubMedBERT |
| Strict Gate | Conf ≥ 0.8 AND Coverage ≥ 2 | ~120K | PubMedBERT |

## 5.5 参数讨论设计

### P1: Mention Window Size / max_seq_length

控制 mention-centered 文本窗口的长度。

| 设置 | max_seq_length | 预期效果 |
|------|---------------|---------|
| 64 | 64 tokens | 可能截断长 mention |
| 96 | 96 tokens | 覆盖大部分 mention |
| 128（默认） | 128 tokens | 当前设置 |
| 160 | 160 tokens | 包含更多上下文 |
| 192 | 192 tokens | 可能引入噪声 |

**状态**：基于 TF-IDF+LR 的 window size 参数讨论可直接复用 Phase 5 已有结果。PubMedBERT 上的 max_seq_length 讨论需补跑。
**预估耗时**：5 settings × 3 tasks × ~40min = ~10h GPU（可通过减少 epochs 到 3 来压缩到 ~6h）。

### P2: Section Strategy

控制输入文本的段落范围。

| 设置 | 包含的段落 | 预期效果 |
|------|----------|---------|
| FINDINGS only | 仅 FINDINGS | 最聚焦，可能丢失 IMPRESSION 中的总结 |
| IMPRESSION only | 仅 IMPRESSION | 放射科医生总结，但可能缺少细节 |
| FINDINGS + IMPRESSION（默认） | FINDINGS + IMPRESSION | 当前设置 |
| All clinical sections | INDICATION + FINDINGS + IMPRESSION | 增加临床背景 |
| Full report | 完整报告文本 | 最大上下文，可能引入噪声 |

**状态**：基于 TF-IDF+LR 的 section strategy 参数讨论需补跑。
**预估耗时**：5 settings × 3 tasks × ~2min = ~30min CPU。

### P3: Quality Gate Strength（弱监督核心参数）

控制训练数据质量筛选的严格程度。

| 设置 | 筛选条件 | 预期训练集规模 |
|------|---------|-------------|
| G1: No Gate | 全量非 ABSTAIN | ~250K |
| G2: Confidence ≥ 0.6 | 低阈值 | ~220K |
| G3: Confidence ≥ 0.7 | 中阈值 | ~200K |
| G4: Confidence ≥ 0.8 | 高阈值 | ~160K |
| G5: Strict（Conf ≥ 0.8 AND Coverage ≥ 2） | 最严格 | ~120K |

**状态**：全部需补跑（依赖 LF 实现和 aggregation 完成）。
**预估耗时**：5 settings × 3 tasks × ~40min = ~10h GPU（可通过减少 epochs 压缩）。

**实现依赖说明**：P3 的全部 5 个设置都依赖 §4 中的 labeling functions 和 aggregation 实现完成。这是 Phase A1 的核心产出。


---

# 6. 可复用结果与必须补跑内容

## 6.1 可直接复用（无需重新训练或运行）

| 资产 | 来源 | 复用方式 | 理由 |
|------|------|---------|------|
| Regex baseline 三任务结果 | Phase 5 | 直接填入表 M2-1, M2-2 | 规则固定，结果不变 |
| TF-IDF+LR 三任务结果 | Phase 5 | 直接填入表 M2-1, M2-2 | 训练数据和参数不变 |
| TF-IDF+SVM 三任务结果 | Phase 5 | 直接填入表 M2-1, M2-2 | 训练数据和参数不变 |
| Phase 5 旧版 PubMedBERT 结果 | Phase 5 | 作为消融 A-lf 的 "Single-source" 行 | 等价于单源 Regex teacher + mention-centered |
| A1 消融（window vs full-text, LR） | Phase 5 | 作为 A-window 消融的补充证据 | 已验证 window 设计在 ML 模型上的价值 |
| A3 消融（explicit-only vs all, LR） | Phase 5 | 作为 A-qg 消融的参考基线 | 已验证标签质量分层的两端效果 |
| A2 消融（exam_name, LR） | Phase 5 | 移入附录 | 结论明确（无增益），不需重跑 |
| Gold 评测集（N=62） | Phase 5.1 | 用于 LF 权重校准 + 新方法 Gold 评测 | 标注不变 |
| Gold 评测中 Regex/LR/SVM 结果 | Phase 5.1 | 直接填入表 M2-2 | 结果不变 |
| 错误分析报告 | Phase 5 | 直接复用于论文 §5.5 | 错误模式分析仍然有效 |
| Silver ceiling 分析 | Phase 5.1 | 直接复用于论文 §5.4 | 核心发现不变 |
| Section parser 实现 | `src/parsers/section_parser.py` | 直接复用 | 无需修改 |
| Mention segmentation 实现 | `src/extractors/nodule_extractor.py` | 直接复用 | 无需修改 |
| 数据集构建框架 | `scripts/phase5/build_datasets.py` | 扩展复用（增加 LF 输出字段） | 核心逻辑不变，增加新字段 |
| PubMedBERT 训练框架 | `scripts/phase5/train_pubmedbert_common.py` | 直接复用 | 训练逻辑不变，仅输入数据变化 |

## 6.2 必须补跑（新实验）

| 实验 | 优先级 | 依赖项 | 预估耗时（3090） | 理由 |
|------|--------|--------|----------------|------|
| **Labeling Functions 实现**（5 LF × 3 tasks） | **P0 必须** | 无 | ~2 天开发 + ~10min 运行 | 弱监督升级的基础 |
| **Label Aggregation 实现** | **P0 必须** | LF 完成 | ~1 天开发 + ~5min 运行 | 弱监督升级的核心 |
| **Quality Gate 实现** | **P0 必须** | Aggregation 完成 | ~0.5 天开发 + ~1min 运行 | 训练数据筛选 |
| **MWS-CFE 训练**（3 tasks） | **P0 必须** | Quality Gate 完成 | ~2h GPU | 新版 "Ours" 的正式结果 |
| **MWS-CFE Gold 评测** | **P0 必须** | MWS-CFE 训练完成 | ~10min GPU | 验证是否打破 silver ceiling |
| **Vanilla PubMedBERT 训练**（3 tasks） | **P0 必须** | 无 | ~2h GPU | 缺少此行则无法证明方法框架价值 |
| **Vanilla PubMedBERT Gold 评测** | **P0 必须** | Vanilla 训练完成 | ~10min GPU | Gold 表需要 Vanilla 行 |
| **A-lf 消融**（single vs multi-source） | **P1 必须** | MWS-CFE 训练完成 | ~0（复用已有结果 vs 新结果） | 弱监督升级的核心消融 |
| **A-agg 消融**（3 aggregation 策略） | **P1 必须** | LF + Aggregation 完成 | ~6h GPU（3 策略 × 3 tasks） | 验证聚合策略选择 |
| **A-qg 消融**（3 gate 策略） | **P1 必须** | Quality Gate 完成 | ~6h GPU（3 策略 × 3 tasks） | 验证质量门控价值 |
| **P1 max_seq_length 参数讨论** | **P1 必须** | MWS-CFE 基础训练完成 | ~10h GPU（5 settings × 3 tasks） | 参数讨论表核心项 |
| **P2 Section Strategy 参数讨论** | **P1 必须** | 无 | ~30min CPU（基于 LR） | 参数讨论表核心项 |
| **P3 Quality Gate Strength 参数讨论** | **P1 必须** | Quality Gate 完成 | ~10h GPU（5 settings × 3 tasks） | 弱监督核心参数 |

## 6.3 仅需重组表格（无需重新运行）

| 内容 | 来源 | 重组方式 |
|------|------|---------|
| Phase 5 ML baseline 参数讨论（max_features/ngram_range/C） | Phase 5 | 移入附录，作为 ML baseline 的参数选择依据 |
| Phase 4 section-aware vs full-text 对比 | Phase 4 | 引用为 A-section 消融的历史证据 |
| Phase 5 模型一致性分析 | Phase 5 | 引用为 silver ceiling 的补充证据 |
| Phase 5 置信度分析 | Phase 5 | 引用为 PubMedBERT 预测特性的补充分析 |

## 6.4 暂时不做（明确排除）

| 内容 | 排除理由 |
|------|---------|
| 多 PLM 横向刷表（BioBERT/ClinicalBERT/GatorTron） | 资源受限（3090），且与论文核心论点（弱监督框架 vs 裸模型）关系不大 |
| BiLSTM-CRF / SpERT / DyGIE++ | 任务为分类而非序列标注/关系抽取，架构不匹配 |
| LLM few-shot 做模块2抽取对比 | 放在模块3更合适（模块3的外部范式比较已规划 LLM-only baseline） |
| Snorkel Label Model（概率标签模型） | LF 数量仅 5 个，统计量不足以训练概率模型；加权投票在此规模下效果相当 |
| Self-training 多轮迭代 | 工期风险大，一轮 quality-aware training 已足够展示方法论贡献 |
| 大规模 Gold 标注扩展 | 人力成本高，N=62 已足够做方向性验证 |

---

# 7. 模块2到模块3的接口约束

## 7.1 模块3最关键的输入字段

模块3（图谱智能体）后续需要从模块2获取以下字段来执行 CDSG 图遍历和推荐生成：

| 字段 | 模块3用途 | 优先级 | 当前质量 |
|------|---------|--------|---------|
| `density_category` | CDSG 图的第一个分支条件（solid vs sub-solid vs GGO） | **最高** | Gold Macro F1 = 0.70（需提升） |
| `size_mm` | CDSG 图的尺寸阈值判断（6mm/8mm/30mm 等） | **最高** | Gold MAE = 0.84mm, ±1mm 覆盖 90.7% |
| `has_size` | 决定是否可以进入尺寸相关的图分支 | **高** | Gold F1 = 0.96 |
| `location_lobe` | 影响风险评估和随访建议的具体化 | **中** | Gold Macro F1 = 0.87 |
| `change_status` | 影响 Lung-RADS 分类升降级 | **中** | 未做 Gold 评测 |
| `morphology`（spiculation/lobulation 等） | 影响恶性概率估计 | **低** | 布尔标记，未做 Gold 评测 |

## 7.2 弱监督升级后，哪些字段最值得优先保证质量

| 优先级 | 字段 | 理由 |
|--------|------|------|
| **P0** | `density_category` | 当前 Gold Macro F1 仅 0.70，是模块3图遍历的第一个分支条件。如果密度判断错误，后续所有推理路径都会偏离。弱监督升级的 LF-D3（否定词）和 LF-D4（多密度检测）直接针对此字段的主要错误模式。 |
| **P0** | `size_mm` / `has_size` | 尺寸是 Lung-RADS 分类的核心阈值。当前共享解析器质量已较高（±1mm 90.7%），但 LF-S2（容错模式）可进一步提升覆盖率。 |
| **P1** | `location_lobe` | 位置影响随访建议的具体化，但不是 CDSG 图的核心分支条件。LF-L2（多叶位检测）和 LF-L3（bilateral 关键词）可改善 bilateral 的低 F1。 |
| **P2** | `change_status` | 影响 Lung-RADS 升降级，但当前无 Gold 评测数据，无法量化改进空间。 |

## 7.3 当前不适合作为模块3强依赖输入的字段

| 字段 | 原因 | 建议处理 |
|------|------|---------|
| `smoking_status`（来自模块1） | MIMIC 脱敏率 97.9%，数据基础极弱 | 模块3应将 smoking_status 视为可选输入，缺失时使用保守默认值 |
| `morphology` 各布尔标记 | 未做 Gold 评测，精度未知 | 模块3可使用但不应作为硬约束条件 |
| `recommendation_cue` | 来自报告原文的推荐线索，不是模块2的预测输出 | 模块3可作为参考但不应替代图推理结果 |

## 7.4 模块2输出中需要提前考虑的附加信息

### 7.4.1 Uncertainty / Confidence

**当前状态**：`radiology_extractor.py` 中的 `_confidence()` 函数基于字段完整性（size + density + location 各 +1 分）计算 high/medium/low 三级置信度。这是字段级完整性评估，不是预测置信度。

**升级需求**：
- MWS-CFE 的 label aggregation 天然产出聚合置信度（`score(winner) / sum(all scores)`）
- PubMedBERT 的 softmax 输出天然产出模型置信度
- 模块3需要这两个置信度来决定：(a) 是否信任模块2的输出进入图推理，(b) 是否触发 abstention

**建议**：在模块2输出的每个字段旁附加 `confidence` 字段，取 LF 聚合置信度和模型 softmax 置信度的较低值。

### 7.4.2 Missingness

**当前状态**：`radiology_extractor.py` 已有 `missing_flags` 列表，记录哪些字段为 None。

**升级需求**：模块3的 abstention 机制需要知道哪些字段缺失。当前 `missing_flags` 已满足需求，无需额外修改。

### 7.4.3 Evidence Span

**当前状态**：`radiology_extractor.py` 已有 `evidence_span` 字段（= mention text）。

**升级需求**：模块3的 explanation grounding 需要知道每个字段的证据来源。当前 evidence_span 是 mention 级别的，不是字段级别的。

**建议**：在 LF 实现中，每个 LF 返回 `(label, evidence_text)` 对。聚合后保留 winner LF 的 evidence_text 作为字段级 evidence span。这为模块3的 guideline anchor grounding 提供直接支撑。

### 7.4.4 Section Source

**当前状态**：`build_datasets.py` 中的 `section_source` 字段记录 mention 来自 FINDINGS 还是 IMPRESSION。

**升级需求**：模块3可能需要区分来自不同段落的信息权重（IMPRESSION 通常更权威）。当前字段已满足需求。

## 7.5 如何保证模块2输出足够支撑模块3

### 支撑 Graph-aware Retrieval

模块3的 graph-aware retrieval 需要根据当前图状态检索相关的临床事实。模块2需要提供：
- 结构化字段（density, size, location）→ 用于图节点条件匹配
- 置信度 → 用于决定是否需要额外检索
- evidence span → 用于检索相关上下文

**MWS-CFE 的贡献**：多源聚合置信度比单源 Regex 的二值输出（有/无）提供更丰富的信号。

### 支撑 Path Ranking

模块3的 path ranking 需要对多条候选推理路径打分。模块2需要提供：
- 字段级置信度 → 影响路径中每个节点的匹配置信度
- 字段间一致性 → 如果 density 和 size 的置信度都高，路径更可信

**MWS-CFE 的贡献**：LF 一致性指标（多少个 LF 同意）直接反映字段的可靠性。

### 支撑 Explanation Grounding

模块3的 explanation grounding 需要将推理路径中的每个决策关联到原始报告文本。模块2需要提供：
- 字段级 evidence span → 每个字段的文本证据
- LF 来源标记 → 哪个 LF 产出了最终标签（可追溯性）

**MWS-CFE 的贡献**：每个 LF 返回 `(label, evidence_text)` 对，天然支持 explanation grounding。

---

# 8. Phase A1 / A2 前置条件

## 8.1 Phase A1 开始前必须满足的条件

Phase A1 = 弱监督基础设施实现（LF + Aggregation + Quality Gate + 新数据集构建）

| # | 前置条件 | 验证方式 | 当前状态 |
|---|---------|---------|---------|
| 1 | 方法命名锁定为 MWS-CFE | 本文件 §3 已定义 | ✅ 已满足 |
| 2 | 三任务的 LF 列表锁定（density 5个, size 5个, location 5个） | 本文件 §4.2 已定义 | ✅ 已满足 |
| 3 | Aggregation 方案锁定为加权多数投票 | 本文件 §4.3 已定义 | ✅ 已满足 |
| 4 | Quality Gate 维度和预设策略锁定 | 本文件 §4.4 已定义 | ✅ 已满足 |
| 5 | 训练目标锁定（PubMedBERT, 同 Phase 5 超参数） | 本文件 §5.1 + Phase 5 configs | ✅ 已满足 |
| 6 | Gold 评测集可用（N=62） | `outputs/phase5_1/` | ✅ 已满足 |
| 7 | 现有 Regex 提取器可用作 LF-*1 的基础 | `src/extractors/nodule_extractor.py` | ✅ 已满足 |
| 8 | 本文件经用户/导师确认 | 用户确认 | ⏳ 待确认 |

### Phase A1 的具体产出物

1. `src/weak_supervision/labeling_functions/` — 15 个 LF 实现（5 per task）
2. `src/weak_supervision/aggregation.py` — 加权多数投票聚合器
3. `src/weak_supervision/quality_gate.py` — Quality Gate 筛选器
4. `scripts/phaseA1/build_ws_datasets.py` — 新数据集构建脚本（在现有 `build_datasets.py` 基础上扩展）
5. `outputs/phaseA1/datasets/` — 新数据集（含 LF 输出、聚合标签、置信度、quality gate 标记）
6. `outputs/phaseA1/lf_analysis/` — LF 覆盖率、冲突率、精度估计报告

### Phase A1 预估工期

| 步骤 | 预估耗时 |
|------|---------|
| LF 实现（15 个） | 2-3 天 |
| Aggregation + Quality Gate 实现 | 1 天 |
| 新数据集构建 + LF 分析报告 | 0.5 天 |
| 单元测试 + 验证 | 0.5 天 |
| **总计** | **4-5 天** |

## 8.2 Phase A2 开始前必须满足的条件

Phase A2 = 模块2正式实验执行（训练 + 评测 + 消融 + 参数讨论 + 表格重组）

| # | 前置条件 | 验证方式 | 依赖 |
|---|---------|---------|------|
| 1 | Phase A1 全部产出物可用 | LF 分析报告无异常 | Phase A1 |
| 2 | 新数据集通过 sanity check | 标签分布合理、无全 ABSTAIN 任务 | Phase A1 |
| 3 | LF 在 Gold 集上的精度估计完成 | LF 权重已校准 | Phase A1 |
| 4 | Vanilla PubMedBERT 训练入口定义清楚 | 训练脚本已准备（full-text 输入, 单源 silver 标签） | Phase A1 |
| 5 | 表格模板与指标定义锁定 | 本文件 §5.3 已定义 | ✅ 已满足 |
| 6 | 消融实验矩阵锁定 | 本文件 §5.4 已定义 | ✅ 已满足 |
| 7 | 参数讨论矩阵锁定 | 本文件 §5.5 已定义 | ✅ 已满足 |

### Phase A2 的具体产出物

1. Vanilla PubMedBERT 三任务训练结果 + Gold 评测结果
2. MWS-CFE 三任务训练结果 + Gold 评测结果
3. 5 组消融实验结果（A-window, A-section, A-lf, A-agg, A-qg）
4. 3 组参数讨论结果（P1 max_seq_length, P2 section strategy, P3 quality gate strength）
5. 最终表格（M2-1, M2-2, M2-3 + 消融表 + 参数讨论表）
6. 更新后的错误分析（MWS-CFE vs 旧版的错误模式对比）

### Phase A2 预估工期

| 步骤 | 预估耗时 |
|------|---------|
| Vanilla PubMedBERT 训练 + 评测 | 0.5 天（~2.5h GPU） |
| MWS-CFE 训练 + 评测 | 0.5 天（~2.5h GPU） |
| 消融实验（5 组） | 2 天（~20h GPU，可部分并行） |
| 参数讨论（3 组） | 2 天（~20h GPU，可部分并行） |
| 表格重组 + 错误分析更新 | 1 天 |
| **总计** | **6-7 天** |

### Phase A1 + A2 总工期

**10-12 天**（含开发、训练、评测、分析）。在单张 3090 约束下可行。

---

# 9. 最终结论

## 9.1 模块2为什么必须重构？

因为当前模块2的 silver labels 完全由单一 Regex teacher 生成，导致：(1) silver test 上的高分（>0.99）是循环论证的虚假结果；(2) Gold 评测证实真实 density Macro F1 仅 0.70；(3) 五种方法在 Gold 集上预测完全一致，说明所有模型都只是 Regex 的复制品。如果不重构，模块2在论文中无法展示真正的方法论贡献，只是一个工程 pipeline。

## 9.2 模块2"我们的方法"最终如何定义？

**MWS-CFE**（Multi-source Weak Supervision for Clinical Fact Extraction）：一个面向放射报告结构化信息抽取的多源弱监督框架，包含 5 个不可分割的组件——Section-aware Preprocessing、Mention-centered Text Window、Multi-source Labeling Functions（5 LF per task）、Label Aggregation + Quality Gate（加权多数投票 + 置信度/覆盖数筛选）、PubMedBERT Fine-tuning。核心方法论贡献在于 C3（多源 LF）和 C4（聚合 + 质量门控），核心工程贡献在于 C2（mention-centered window, +38-63pp）。

## 9.3 弱监督升级准备怎么做？

将单一 Regex teacher 拆解为每任务 5 个独立 labeling functions（density: 精确匹配/模糊匹配/否定词检测/多密度检测/IMPRESSION线索；size: 标准模式/容错模式/数字上下文/定性描述/无尺寸负信号；location: 精确叶位/多叶位检测/bilateral关键词/侧别推断/上下文窗口位置）。通过加权多数投票聚合多源信号，以 Gold 集上的 LF 精度作为权重。Quality Gate 基于聚合置信度和 LF 覆盖数筛选高质量训练数据。

## 9.4 哪些结果还能直接用？

Regex/LR/SVM 三任务的 silver 和 gold 结果全部可直接复用。Phase 5 旧版 PubMedBERT 结果可作为消融 A-lf 的 "Single-source" 基线。A1/A2/A3 消融结果可作为新消融的补充证据。Gold 评测集（N=62）可直接用于 LF 权重校准和新方法评测。Section parser、mention segmentation、PubMedBERT 训练框架全部可直接复用。

## 9.5 哪些必须重跑？

必须补跑的核心实验：(1) 15 个 LF 实现 + aggregation + quality gate（Phase A1）；(2) Vanilla PubMedBERT 训练 + Gold 评测；(3) MWS-CFE 训练 + Gold 评测；(4) 5 组消融（A-window, A-section, A-lf, A-agg, A-qg）；(5) 3 组参数讨论（P1 max_seq_length, P2 section strategy, P3 quality gate strength）。总 GPU 时间约 45h，总工期约 10-12 天。

## 9.6 模块2下一步最先做什么？

**Phase A1**：实现弱监督基础设施（LF + Aggregation + Quality Gate + 新数据集构建）。这是所有后续实验的前置依赖。预估 4-5 天。同时可并行启动 Vanilla PubMedBERT 训练（无依赖）。

## 9.7 模块2现在绝对不要做什么？

1. **不要**开始写模块3的代码——先完成模块2全部实验，锁定模块2结果
2. **不要**引入 Snorkel Label Model——LF 数量仅 5 个，统计量不足，加权投票已够用
3. **不要**引入新的 PLM（BioBERT/ClinicalBERT/GatorTron）——资源受限且偏离核心论点
4. **不要**做 self-training 多轮迭代——工期风险大，一轮 quality-aware training 已足够
5. **不要**扩展 Gold 标注集——N=62 已足够做方向性验证，人力成本不值得
6. **不要**修改已有的 Phase 5/5.1 实验结果——结果已锁定，作为历史基线复用
7. **不要**在 LF 实现完成前就开始训练 MWS-CFE——LF 是训练数据的基础

---

> **本文件为模块2方法重构与弱监督升级的定案版。后续 Phase A1/A2 执行阶段以本文件为唯一权威参考。任何偏离本计划的决策需要明确记录理由。**
