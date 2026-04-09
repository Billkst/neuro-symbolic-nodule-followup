# 方案三最终实验重构计划

> 生成日期：2026-04-09
> 状态：**定案版**——后续执行阶段以本文件为唯一权威参考
> 约束：本阶段仅产出计划，不写代码、不跑实验、不改 schema

---

# 1. 执行摘要

本计划将方案三从"三个独立模块各自实验"收敛为**一个完整的神经符号随访建议生成系统**的统一实验框架。核心决策如下：

1. **模块1（吸烟史）降级为辅助属性模块**，仅保留 coarse-grained smoking status sanity check，不再承担筛查资格判定的主实验责任。原因：MIMIC 出院小结 97.9% Social History 被脱敏，无法稳定恢复 pack-year/quit years。
2. **模块2（结构化信息抽取）明确"我们的方法"定义**：Task-Specific Pipeline Framework（mention-centered window + section-aware preprocessing + silver label weak supervision）+ PubMedBERT backbone。Vanilla PubMedBERT（full-text 微调）作为必补 baseline。
3. **模块3（图谱智能体）必须从 flat rule engine 升级为最小可行图谱智能体原型**，包含显式决策图、graph executor、guideline anchor、soft match、abstention 机制。实验分两层：同体系内部比较 + 外部范式比较（LLM-only / RAG）。
4. **筛查 vs 偶发分流**解耦为 report-intent router 层，作为模块3的前置组件，不再由模块1独立承担。
5. **所有 baseline 现在锁死**，后续执行阶段不允许漂移。

**资源约束**：单张 RTX 3090（24GB），所有实验设计均在此约束下可行。

---

# 2. 方案三最终系统定位

## 2.1 三模块最终职责

| 模块 | 论文中的角色 | 最终职责 | 实验权重 |
|------|-------------|---------|---------|
| 模块1：吸烟史推断 | 辅助属性模块 | 从出院小结提取 current/former/never/unknown 状态及 evidence sentence，为模块3提供患者风险分级输入 | ~10% |
| 模块2：结构化信息抽取 | 核心感知模块 | 从放射报告提取 density/size/location 等结构化事实，是整个系统的信息基座 | ~45% |
| 模块3：图谱智能体推理 | 核心决策模块 | 基于临床决策状态图（CDSG）执行指南约束推理，生成可解释的随访建议 | ~45% |

## 2.2 路由层定位

**report-intent router**（报告意图路由器）作为模块3的前置组件：
- 输入：case_bundle（含 radiology_facts + smoking_eligibility）
- 功能：判断当前报告属于"筛查场景"还是"偶发发现场景"，选择对应的指南分支（Lung-RADS vs Fleischner）
- 实现方式：基于 exam_name + ordering context + smoking eligibility 的规则路由，不需要独立训练模型
- 论文口径：资格判定与报告场景路由解耦，路由器是系统工程组件而非独立模块

## 2.3 与开题报告的一致性与调整说明

| 维度 | 开题报告设计 | 最终实现 | 调整理由 | 答辩口径 |
|------|------------|---------|---------|---------|
| 模块1架构 | ClinicalBERT + 双通道正交投影 + 因果解耦 | Regex 吸烟提取 + coarse status check | MIMIC 脱敏率 97.9%，定量字段（pack-year/PPD/quit years）几乎不可用，因果解耦架构失去数据基础 | 数据现实约束导致的合理降级，论文中如实说明限制 |
| 模块2架构 | Clinical-Longformer + CRF + 跨度注意力池化 + 模式约束指针解码 | PubMedBERT 分类 + mention-centered window + 共享尺寸解析器 | 任务本质是分类而非序列标注（density/location 是枚举分类，size 是二分类+共享解析器），PubMedBERT 在 3090 上可行且已验证 | 根据任务特性选择更匹配的建模方式，mention-centered 设计经消融验证带来 38-63pp 增益 |
| 模块3架构 | LLM 构建 CDSG + 符号硬匹配 + 语义软匹配 + 轨迹链报告 | 需从 flat rule engine 升级为 graph-state executor + guideline anchor + soft match + abstention | 当前仅实现了 flat IF-THEN 规则引擎，缺少图结构和智能体能力，必须补齐 | 保持与开题报告方向一致，是实现层面的渐进式补齐 |
| 分流逻辑 | 由模块1的筛查资格判定承担 | 解耦为独立的 report-intent router | 模块1数据受限无法独立承担分流，路由逻辑本质是工程组件 | 更合理的系统设计，资格判定与场景路由是两个独立关注点 |

**核心答辩论点**：开题报告定义的是系统目标和技术方向，实现层面根据数据现实（MIMIC 脱敏）和任务特性（分类 vs 序列标注）做了合理调整。三模块的功能职责和系统闭环保持不变，调整的是具体技术路径。

---

# 3. 模块1最终保留方案

## 3.1 任务目标

从 MIMIC-IV 出院小结中提取患者吸烟状态（current / former / never / unknown），并提供 evidence sentence 支撑。**不再尝试**精确恢复 pack-year、quit years 或 USPSTF 定量资格判定。

**理由**：
- MIMIC 出院小结 Social History 段 97.9% 被 `___` 掩码脱敏
- pack_year_parse_rate 仅 ~0.5%，ppd_parse_rate 仅 ~15.7%
- 定量字段极度稀疏，无法支撑因果解耦建模（开题报告中的双通道正交投影架构失去数据基础）
- fallback 策略虽将覆盖率从 10.6% 提升到 68.8%，但 70.6% 为 low evidence quality

## 3.2 实验保留范围

### 保留实验

| 实验 | 内容 | 数据 | 状态 |
|------|------|------|------|
| E1.1 Coarse Status Extraction | 在 researcher-reviewed subset 上评估 smoking_status_norm 的 4 分类准确率 | researcher-reviewed subset（约 50-100 条人工核验样本） | **需新建** subset 并补跑 |
| E1.2 Fallback Strategy Ablation | social_history_only vs plus_fallback 的覆盖率-质量 tradeoff | 已有 Phase 4 结果 | **可直接复用**，仅需表格重组 |
| E1.3 Evidence Quality Distribution | 展示 high/medium/low/none 的分布比例 | 已有 Phase 4 结果 | **可直接复用** |

### 不再做的实验

| 实验 | 理由 |
|------|------|
| Pack-year 精确回归 | 数据中仅 ~0.5% 含 pack-year，样本量不足 |
| 因果解耦建模（PPD × Years） | 定量字段被脱敏，正交投影架构无训练数据 |
| USPSTF 资格判定精度评估 | 依赖 pack-year + quit years，均不可用 |
| PubMedBERT 微调吸烟分类 | 成本收益不匹配，regex 已足够 |

## 3.3 评测口径

- **Researcher-reviewed subset**：由研究者（而非临床专家）对 50-100 条样本进行人工核验
- 论文中**不得**称之为"expert gold standard"或"clinical gold standard"
- 正确口径：**"研究者复核的小规模验证集，用于 sanity check 而非精度宣称"**
- 主要指标：4 分类 Accuracy、各类别 Precision/Recall/F1、coverage rate

## 3.4 在全文中的定位

- **第三章方法**：作为 §3.4 "患者属性推断模块"的一部分，说明脱敏约束与 fallback 策略设计
- **第五章实验**：作为 §5.3 "模块性能解耦分析"中的子节，篇幅控制在 1-2 页
- **不单独设表**：将 E1.1 结果并入系统整体评估表中的一行，避免喧宾夺主
- **核心叙事**：模块1不是贡献点，而是"在数据受限条件下的工程折衷"，为系统闭环提供保守的默认风险分级

---

# 4. 模块2最终实验重构方案

## 4.1 方法定义（锁定版）

### "我们的方法"（Method-PubMedBERT）

**论文中的命名建议**：`Method-PubMedBERT (Ours)`

"我们的方法"不是一个裸模型，而是一个**任务特定的方法框架**，包含以下不可分割的设计决策：

| 组件 | 具体设计 | 对应消融实验 |
|------|---------|------------|
| 输入预处理 | Section-aware parsing（限定 FINDINGS/IMPRESSION） | Phase 4: section-aware vs full-text |
| 特征构造 | Mention-centered text window（以结节 mention 为中心截取局部上下文） | A1: window vs full-text（+38~63pp） |
| 标签策略 | Silver label weak supervision（含 explicit + silver + weak 三级标签） | A3: explicit-only vs all silver |
| 主干模型 | PubMedBERT（microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext） | 主结果对比 |
| 数值处理 | 共享正则尺寸解析器（Shared Size Parser） | Phase 5.1a 口径修正 |

**关键论点**：方法框架中的每一个组件都有对应的消融实验或设计论证。mention-centered window 是最核心的设计贡献（消融证明带来 38-63pp 增益），PubMedBERT 是可替换的主干。

### Vanilla PubMedBERT

**定义**：在 **full-text**（整篇报告文本）上直接微调 PubMedBERT，不使用 mention-centered window，不使用 section-aware preprocessing，其他训练设置（learning rate、epochs、batch size）保持一致。

**目的**：证明"我们的方法"的性能增益来自方法框架设计（mention-centered window + section-aware），而非 PubMedBERT 本身的能力。

**与现有消融的关系**：Phase 5 消融 A1（mention window vs full-text）已在 TF-IDF+LR 上验证了 window 设计的价值。Vanilla PubMedBERT 实验是将同一消融扩展到 PubMedBERT 主干上，补齐完整的对比证据链。

### PubMedBERT-CRF 评估

**结论：不建议补跑，从 baseline 列表中移除。**

理由：
1. 当前三个任务（density 5分类、has_size 二分类、location 9分类）均为**分类任务**，输入是 mention-centered 短文本窗口，输出是单标签分类
2. CRF 层的核心价值在于**序列标注**任务中建模标签间转移依赖（如 BIO 标签约束），对分类任务无增益
3. 在 3090 资源约束下，CRF 训练耗时增加但无预期收益
4. 如果未来改为序列标注范式（如 NER），可在后续工作中引入

## 4.2 Baseline 定案

### 最终锁定的对比方法列表

| # | 方法 | 论文中命名 | 类型 | 状态 | 补跑需求 |
|---|------|----------|------|------|---------|
| 1 | Regex Baseline | Regex | 规则基线 | ✅ 已有 Phase 4/5 结果 | 仅需表格重组 |
| 2 | TF-IDF + LR | TF-IDF + LR | ML 基线 | ✅ 已有 Phase 5 结果 | 仅需表格重组 |
| 3 | TF-IDF + SVM | TF-IDF + SVM | ML 基线 | ✅ 已有 Phase 5 结果 | 仅需表格重组 |
| 4 | Vanilla PubMedBERT (full-text) | Vanilla PubMedBERT | 对照消融 | ❌ **必须补跑** | 新实验 |
| 5 | Method-PubMedBERT (我们的方法) | Method-PubMedBERT (Ours) | 主模型 | ✅ 已有 Phase 5 结果 | 仅需表格重组 |

### 从列表中移除的方法及理由

| 方法 | 移除理由 |
|------|---------|
| PubMedBERT-CRF | 任务为分类而非序列标注，CRF 无增益（见 §4.1） |
| DyGIE++ | 关系抽取框架，当前任务是独立属性分类，不涉及实体间关系 |
| SpERT | 同上，且在 3090 上部署和复现成本较高 |
| BiLSTM-CRF | 序列标注方法，与当前分类范式不匹配 |
| LLM few-shot | 模块2的核心叙事是弱监督微调的价值，few-shot 对比放在模块3更合适 |
| 多 PLM 横向刷表 | 资源受限，且与论文核心论点（方法框架 vs 裸模型）关系不大 |

### Vanilla PubMedBERT 补跑方案

**训练配置**（与 Method-PubMedBERT 保持一致，仅改变输入）：
- 模型：microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- 输入：**full-text**（截断到 max_seq_length=512），不使用 mention-centered window
- max_seq_length：512（full-text 需要更长序列）
- learning_rate：2e-5
- epochs：5
- batch_size：根据 512 长度调整（预估 batch_size=8-16，3090 24GB 可承受）
- 数据：与 Phase 5 完全相同的 train/val/test 切分
- 任务：density_category / has_size / location_lobe 三个任务分别训练
- **预估耗时**：每任务约 60-90 分钟（序列更长），共约 3-4.5 小时

## 4.3 表格设计

### 表 M2-1：模块2主结果表（Silver 测试集）

**位置**：正文 §5.2

| 方法 | density Acc | density Macro F1 | has_size Acc | has_size F1 | location Acc | location Macro F1 | 平均训练耗时 |
|------|-----------|-----------------|-------------|------------|-------------|------------------|------------|
| Regex | — | — | — | — | — | — | — |
| TF-IDF+LR | — | — | — | — | — | — | — |
| TF-IDF+SVM | — | — | — | — | — | — | — |
| Vanilla PubMedBERT | — | — | — | — | — | — | — |
| **Method-PubMedBERT (Ours)** | — | — | — | — | — | — | — |

**注释**：Regex 在 Silver 集上为 1.0 是 Silver Label 循环性所致（标签由 Regex 生成），并非真实完美。

### 表 M2-2：模块2 Gold 评测结果（N=62）

**位置**：正文 §5.4

| 方法 | density Macro F1 | has_size F1 | size_mm MAE | location Macro F1 |
|------|-----------------|------------|------------|------------------|
| Regex / Silver | — | — | — | — |
| TF-IDF+LR | — | — | — | — |
| TF-IDF+SVM | — | — | — | — |
| Vanilla PubMedBERT | — | — | — | — |
| **Method-PubMedBERT (Ours)** | — | — | — | — |

**注释**：size_mm 列所有方法共享同一正则解析模块，结果一致。Vanilla PubMedBERT 在 Gold 集上需补跑评估。

### 表 M2-3：消融实验表

**位置**：正文 §5.2 或附录

| 消融变量 | density Macro F1 | has_size F1 | location Macro F1 | Delta (vs Ours) |
|---------|-----------------|------------|------------------|----------------|
| Ours（完整方法） | baseline | baseline | baseline | — |
| A1: Full-text input (去除 mention-centered window) | — | — | — | — |
| A2: 去除 section-aware preprocessing | — | — | — | — |
| A3: Explicit-only silver labels | — | — | — | — |
| A4: 去除 exam name 特征 | — | — | — | — |

**说明**：A1 已有 Phase 5 LR 结果，需补 PubMedBERT 版本（即 Vanilla PubMedBERT 实验）。A2 需确认是否等价于 full-text regex 实验。A3、A4 已有结果。

### 表 M2-4：参数讨论表

**位置**：正文 §5.2 或附录

#### P1: Mention-centered Window Size（max_seq_length）

控制以结节 mention 为中心截取的文本窗口长度（token 数）。

| max_seq_length | density Macro F1 | has_size F1 | location Macro F1 |
|---------------|-----------------|------------|------------------|
| 64 | — | — | — |
| 96 | — | — | — |
| 128 (当前默认) | — | — | — |
| 160 | — | — | — |
| 192 | — | — | — |

**状态**：❌ **必须补跑**。基于 TF-IDF+LR 即可（LR 使用对应长度截取的文本），无需 PubMedBERT。
**预估耗时**：5 settings × 3 tasks × ~2min = ~30 分钟 CPU。

#### P2: Section Strategy（Section 融合策略）

控制从报告中选取哪些段落作为候选文本来源。

| Section 设置 | density Macro F1 | has_size F1 | location Macro F1 |
|-------------|-----------------|------------|------------------|
| FINDINGS only | — | — | — |
| IMPRESSION only | — | — | — |
| FINDINGS + IMPRESSION (当前默认) | — | — | — |
| FINDINGS + IMPRESSION + TECHNIQUE | — | — | — |
| Full text (no section filter) | — | — | — |

**状态**：❌ **必须补跑**。基于 TF-IDF+LR 即可。
**预估耗时**：5 settings × 3 tasks × ~2min = ~30 分钟 CPU。

#### P3: Silver Label Quality Gate（训练数据质量分层）

控制训练集中包含哪些质量层级的 silver label。当前实现中 `label_quality` 字段为离散三级分层（`explicit` / `silver` / `weak`），由 `build_datasets.py` 中的 `_infer_label_quality()` 函数根据 density_text / size_text / location_text 是否存在来判定。**不存在连续置信度阈值**，因此本参数讨论基于离散质量组合。

| Quality Gate 设置 | 包含的 label_quality 层级 | density Macro F1 | has_size F1 | location Macro F1 |
|------------------|------------------------|-----------------|------------|------------------|
| explicit only | `explicit` | — | — | — |
| explicit + silver | `explicit` + `silver` | — | — | — |
| all (当前默认) | `explicit` + `silver` + `weak` | — | — | — |
| all + regex-agree filter | 全部，但仅保留与 Regex 预测一致的样本 | — | — | — |
| all + high-confidence filter | 全部，但仅保留 PubMedBERT 预测置信度 ≥ 0.95 的样本（需先跑一轮推理获取置信度） | — | — | — |

**状态**：⚠️ 部分已有（A3 消融覆盖了 `explicit only` vs `all` 的两端），需补充中间 3 个梯度。
**预估耗时**：3 new settings × 3 tasks × ~2min = ~20 分钟 CPU。

> **实现依赖说明**：前 3 个设置可直接基于现有 `label_quality` 字段实现。第 4 个设置（regex-agree filter）需在数据构建阶段增加 Regex 预测一致性校验逻辑。第 5 个设置（high-confidence filter）需先完成一轮 PubMedBERT 推理以获取置信度分数，属于执行前需确认的实现依赖项。

## 4.4 补跑建议汇总

| 实验 | 优先级 | 预估耗时（3090） | 理由 |
|------|--------|----------------|------|
| Vanilla PubMedBERT (3 tasks) | **P0 必须** | ~4 小时 | 缺少此行则无法证明方法框架的价值 |
| P1 Window Size (5 settings × 3 tasks, LR) | **P1 必须** | ~30 分钟 | 参数讨论表核心项 |
| P2 Section Strategy (5 settings × 3 tasks, LR) | **P1 必须** | ~30 分钟 | 参数讨论表核心项 |
| P3 Quality Gate 中间梯度 (3 new settings × 3 tasks, LR) | **P1 必须** | ~20 分钟 | 补充 A3 消融的中间值，P3 第 4-5 设置有实现依赖（见 §4.3） |
| Vanilla PubMedBERT Gold 评测 (已有 62 Gold 样本) | **P0 必须** | ~10 分钟（推理） | Gold 表需要 Vanilla PubMedBERT 行 |

## 4.5 可直接复用的已有结果

| 已有结果 | 来源 | 复用方式 |
|---------|------|---------|
| Regex/LR/SVM/PubMedBERT Silver 主结果 | Phase 5 | 直接填入表 M2-1 |
| Regex/LR/SVM/PubMedBERT Gold 结果 | Phase 5.1 | 直接填入表 M2-2 |
| A1 Window vs Full-text (LR) | Phase 5 消融 | 填入表 M2-3 A1 行 |
| A3 Explicit-only vs All (LR) | Phase 5 消融 | 填入表 M2-3 A3 行 |
| A4 Exam name (LR) | Phase 5 消融 | 填入表 M2-3 A4 行 |
| max_features/ngram_range/C 参数讨论 | Phase 5 | **移入附录**（ML baseline 参数，非核心方法参数） |
| Phase 4 section-aware vs full-text | Phase 4 | 填入表 M2-3 A2 行或引用 |
| Phase 5 错误分析 | Phase 5 | 直接复用于论文 §5.5 |
| Phase 5.1 Silver Ceiling 分析 | Phase 5.1 | 直接复用于论文 §5.4 |

---

# 5. 模块3最终实验重构方案

## 5.1 图谱智能体 MVP 定义

当前模块3仅实现了 `lung_rads_engine.py`——一个 flat IF-THEN 规则引擎（约 420 行 Python），实现了 Lung-RADS v2022 的决策树逻辑。这与开题报告中"基于逻辑图谱约束的智能体推理"存在显著差距。

**最小可行图谱智能体原型（MVP）必须包含以下 7 个能力**：

| # | 能力 | 描述 | 与 flat engine 的区别 |
|---|------|------|---------------------|
| 1 | 显式指南决策图（CDSG） | 将 Lung-RADS / Fleischner 指南编码为有向图：state nodes（状态节点）+ conditional edges（条件转移边） | flat engine 用嵌套 if-else，无显式图结构 |
| 2 | Graph Executor | 在决策图上执行路径遍历，根据输入事实逐步推进状态转移 | flat engine 直接跳到结论，无中间状态 |
| 3 | Guideline Anchor | 每个决策节点关联具体指南条款引用（如 "Lung-RADS v2022 §4A: solid nodule ≥8mm"） | flat engine 已有 guideline_anchor 字段，可复用 |
| 4 | Hard Constraint Validation | 布尔逻辑严格匹配：size 阈值、density 枚举、age 范围等确定性条件 | flat engine 已有，需迁移到图节点 |
| 5 | Soft Match / Semantic Match | 对模糊输入（如 "possibly part-solid"、"subcentimeter"）使用语义相似度或 NLI 蕴含判断 | flat engine 完全没有，需新增 |
| 6 | Abstention / Insufficient-data / Conservative Fallback | 当输入事实不足以确定性推理时，显式输出"信息不足，建议保守处理"而非强行给出结论 | flat engine 有 uncertainty_note 但无正式 abstention 机制 |
| 7 | Reasoning Path 输出 | 输出完整的推理轨迹：经过了哪些节点、每个节点的判断依据、最终结论 | flat engine 有 reasoning_path 列表，需升级为图路径 |

### 技术选型建议

| 组件 | 推荐方案 | 理由 |
|------|---------|------|
| 图结构 | Python dict/dataclass 定义的 DAG，或 NetworkX DiGraph | 轻量、无外部依赖、3090 友好 |
| Graph Executor | 自定义 Python walker（BFS/DFS 遍历） | 决策图规模小（<100 节点），无需图数据库 |
| Soft Match | sentence-transformers（all-MiniLM-L6-v2）计算余弦相似度，或 NLI 模型（cross-encoder/nli-deberta-v3-small） | 小模型，3090 可承受 |
| Abstention | 基于置信度阈值 + missing field 计数的规则 | 简单有效，可解释 |
| 指南编码 | 手工将 Lung-RADS v2022 + Fleischner 2017 编码为 JSON/YAML 格式的图定义文件 | 指南条款有限（<50 条规则），手工编码可控 |

### 3090 资源评估

| 组件 | GPU 需求 | 预估显存 |
|------|---------|---------|
| Graph Executor | CPU only | 0 |
| Hard Constraint | CPU only | 0 |
| Soft Match (sentence-transformers) | GPU 推理 | ~500MB |
| NLI 模型 (DeBERTa-small) | GPU 推理 | ~1GB |
| LLM-only baseline (见 §5.3) | GPU 推理 | 取决于模型选择 |

**结论**：图谱智能体 MVP 本身几乎不消耗 GPU 资源，主要开销在外部范式对比的 LLM baseline。

### MVP 完成标准（Definition of Done）

#### A. 第一轮必须完成（Must-have）

以下 6 项功能构成图谱智能体的最小可运行主干。**全部完成后方可开始模块3实验。**

| # | 功能 | 完成标准（可验证） |
|---|------|-----------------|
| M1 | 显式 CDSG 图结构 | Lung-RADS v2022 的全部决策路径已编码为 JSON/YAML 格式的有向图定义文件，包含 ≥30 个状态节点和 ≥40 条条件转移边，可被 Graph Executor 加载 |
| M2 | Report-Intent Router | 给定 case_bundle，能正确输出 `screening` 或 `incidental` 场景标签，并选择对应指南分支；在 Phase 4 recommendation_eval 数据上路由准确率 ≥ 90% |
| M3 | Graph Executor | 能在 CDSG 上执行完整路径遍历：输入 case_bundle → 逐节点状态转移 → 输出最终推荐节点；对 Phase 4 rule_derived 子集的推荐结果与现有 FlatRule 引擎 100% 一致（hard constraint 下） |
| M4 | Hard Constraint Validation | 所有 Lung-RADS v2022 的确定性规则（size 阈值、density 枚举、age 范围）已迁移到图节点条件中，覆盖现有 `lung_rads_engine.py` 的全部规则分支 |
| M5 | Guideline Anchor Output | 每个推荐结果附带具体指南条款引用（如 "Lung-RADS v2022 §4A"），格式与现有 `guideline_anchor` 字段兼容 |
| M6 | Abstention / Insufficient-data / Conservative Fallback | 当输入事实缺少 ≥2 个关键字段（density + size 同时缺失）时，系统输出 `abstention` 而非强行推荐；abstention 输出包含缺失字段列表和保守建议 |

#### B. 第一轮增强项（Nice-to-have）

以下功能提升系统能力但**不阻塞** Must-have 落地和模块3主实验启动。

| # | 功能 | 说明 |
|---|------|------|
| N1 | Soft Match / Semantic Match | 对模糊输入（如 "possibly part-solid"）使用 sentence-transformers 或 NLI 模型进行语义匹配，扩展 hard constraint 的覆盖范围 |
| N2 | 多跳路径排序（Top-K Path Ranking） | 当存在多条可行推理路径时，按置信度排序输出 Top-K 候选路径 |
| N3 | 自然语言解释生成 | 将推理路径转化为人类可读的中文/英文解释文本，而非仅输出节点 ID 序列 |

> **硬约束**：第一轮执行阶段不得因增强项阻塞 Must-have 功能落地。Must-have 全部通过验收后，方可投入增强项开发。

## 5.2 内部比较（第一层：同体系内部）

### 对比对象

| # | 方法 | 论文命名 | 描述 | 状态 |
|---|------|---------|------|------|
| I-1 | Flat Rule Engine | FlatRule | 当前 lung_rads_engine.py 的直接输出 | ✅ 已有 Phase 4 结果 |
| I-2 | Graph-State Executor | GraphExec | CDSG 图结构 + graph walker，仅 hard constraint | ❌ **必须新建** |
| I-3 | GraphExec + Guideline Anchor | GraphExec+GA | I-2 基础上增加每步指南条款引用 | ❌ **必须新建** |
| I-4 | GraphExec + Soft Match + Abstention | GraphExec+SM+Abs（完整版） | I-3 基础上增加语义软匹配和弃权机制 | ❌ **必须新建** |

### 内部比较的核心论点

- I-1 → I-2：证明显式图结构比 flat if-else 更可维护、更可扩展（主要是工程质量论证）
- I-2 → I-3：证明 guideline anchor 提升可解释性和可审计性
- I-3 → I-4：证明 soft match 处理模糊输入的能力，abstention 提升安全性

**注意**：I-1 和 I-2 在 hard constraint 下的推理结果应该**完全一致**（同一套规则，只是表示形式不同）。差异体现在：
- 可维护性（代码行数、修改成本）
- 可扩展性（新增指南的工作量）
- 推理路径的可读性
- I-4 的 soft match 才会产生**结果层面**的差异

## 5.3 外部比较（第二层：外部范式对比）

### 对比对象

| # | 方法 | 论文命名 | 描述 | 实现方案 | 状态 |
|---|------|---------|------|---------|------|
| E-1 | LLM-only + Guideline Prompt | LLM-Direct | 将指南全文放入 system prompt，直接让 LLM 根据 case_bundle 生成随访建议 | 本地部署 Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct（3090 可跑 4-bit 量化） | ❌ **必须新建** |
| E-2 | RAG + LLM | RAG-LLM | 将指南分块存入向量库，检索相关条款后拼接到 prompt 中让 LLM 生成建议 | sentence-transformers 编码 + FAISS 检索 + 同上 LLM | ❌ **必须新建** |

### 外部比较的核心论点

- Ours vs E-1：证明纯 LLM 在严格指南推理中的不可靠性（幻觉、规则遗漏、不可审计）
- Ours vs E-2：证明 RAG 虽然引入了指南知识，但检索粒度和推理链路仍不如显式图结构可控

### 3090 资源方案

| 模型 | 量化 | 显存占用 | 推理速度 |
|------|------|---------|---------|
| Qwen2.5-7B-Instruct | GPTQ 4-bit | ~5GB | 可接受 |
| Llama-3.1-8B-Instruct | GPTQ 4-bit | ~6GB | 可接受 |
| Mistral-7B-Instruct-v0.3 | GPTQ 4-bit | ~5GB | 可接受 |

**建议选择 Qwen2.5-7B-Instruct**：中文支持好（论文可能需要中文案例），医学推理能力在 7B 级别中较强。

**GraphRAG 评估**：
- 原始 GraphRAG（微软）需要 LLM 构建知识图谱，成本极高
- 在 3090 约束下，建议将 GraphRAG 简化为"指南知识图谱 + 图检索 + LLM 生成"
- 如果资源不足，可将 GraphRAG 降级为"建议补跑"而非"必须补跑"，在论文中以文献对比替代实验对比

## 5.4 模块3实验表格设计

### 表 M3-1：主结果表——随访建议生成质量

**位置**：正文 §5.6

| 方法 | Rec. Accuracy | Guideline Adherence | Path Validity | Abstention Rate | Abstention Correctness | Avg. Reasoning Steps |
|------|--------------|--------------------|--------------|-----------------|-----------------------|---------------------|
| FlatRule (I-1) | — | — | — | N/A | N/A | — |
| GraphExec (I-2) | — | — | — | N/A | N/A | — |
| GraphExec+GA (I-3) | — | — | — | N/A | N/A | — |
| **GraphExec+SM+Abs (I-4, Ours)** | — | — | — | — | — | — |
| LLM-Direct (E-1) | — | — | — | — | — | — |
| RAG-LLM (E-2) | — | — | — | — | — | — |

**指标定义**：
- **Rec. Accuracy**：推荐结果与指南标准答案的一致率（基于 case_bundle 中的 ground truth）
- **Guideline Adherence**：输出的推荐是否引用了正确的指南条款（人工评估或规则校验）
- **Path Validity**：推理路径中每一步是否逻辑自洽（无跳步、无矛盾）
- **Abstention Rate**：系统主动弃权（信息不足）的比例
- **Abstention Correctness**：弃权决策的正确率（确实信息不足时弃权 = 正确）
- **Avg. Reasoning Steps**：平均推理步数

### 表 M3-2：消融实验表

**位置**：正文 §5.6 或附录

| 消融变量 | Rec. Accuracy | Guideline Adherence | Path Validity |
|---------|--------------|--------------------|--------------| 
| 完整版 (I-4) | baseline | baseline | baseline |
| 去除 soft match | — | — | — |
| 去除 abstention | — | — | — |
| 去除 guideline anchor | — | — | — |
| 去除图结构（退化为 FlatRule） | — | — | — |

### 表 M3-3：参数讨论表

**位置**：附录或正文

#### P1: Soft-Match Threshold（语义匹配阈值）

| 阈值 | Rec. Accuracy | False Match Rate | Abstention Rate |
|------|--------------|-----------------|----------------|
| 0.5 | — | — | — |
| 0.6 | — | — | — |
| 0.7 (默认) | — | — | — |
| 0.8 | — | — | — |
| 0.9 | — | — | — |

#### P2: Candidate Path Top-K / Beam Width

| Top-K | Rec. Accuracy | Path Diversity | Avg. Inference Time |
|-------|--------------|---------------|-------------------|
| 1 (greedy) | — | — | — |
| 3 | — | — | — |
| 5 (默认) | — | — | — |
| 10 | — | — | — |
| 全路径 | — | — | — |

#### P3: Abstention Threshold（弃权阈值）

| 阈值 | Abstention Rate | Abstention Correctness | Rec. Accuracy (non-abstained) |
|------|----------------|----------------------|------------------------------|
| 0 (从不弃权) | — | — | — |
| 1 missing field | — | — | — |
| 2 missing fields (默认) | — | — | — |
| 3 missing fields | — | — | — |
| any uncertainty | — | — | — |

### 表 M3-4：Case Study / Path Analysis

**位置**：正文 §5.7

选取 8-12 个代表性案例，覆盖以下场景：

| 案例类型 | 数量 | 展示目的 |
|---------|------|---------|
| 标准 solid nodule ≥8mm | 2 | 展示完整推理路径 |
| Part-solid with ground-glass component | 2 | 展示 soft match 处理模糊密度 |
| 信息不足触发 abstention | 2 | 展示安全弃权机制 |
| 多结节 dominant selection | 2 | 展示多结节合并逻辑 |
| LLM-Direct 幻觉 vs Ours 正确 | 2 | 展示图谱智能体的可靠性优势 |
| 边界案例（size 恰好在阈值上） | 1-2 | 展示 hard constraint 的精确性 |

每个案例展示：输入事实 → 推理路径（图节点序列）→ 最终建议 → 指南锚点

## 5.5 必须补建的功能清单

按优先级排序：

| # | 功能 | 优先级 | 预估工作量 | 依赖 |
|---|------|--------|----------|------|
| F1 | 指南决策图定义文件（Lung-RADS v2022 JSON/YAML） | P0 | 2-3 天 | 无 |
| F2 | Graph Executor（图遍历引擎） | P0 | 2-3 天 | F1 |
| F3 | Hard Constraint Validation（迁移现有规则到图节点） | P0 | 1-2 天 | F2 |
| F4 | Guideline Anchor 关联（每节点绑定指南条款） | P0 | 1 天 | F1 |
| F5 | Reasoning Path 输出（图路径格式） | P0 | 1 天 | F2 |
| F6 | Abstention / Insufficient-data 机制 | P1 | 1-2 天 | F2 |
| F7 | Soft Match（sentence-transformers 或 NLI） | P1 | 2-3 天 | F2 |
| F8 | Report-Intent Router（筛查 vs 偶发路由） | P1 | 1 天 | F1 |
| F9 | Fleischner 2017 指南决策图（第二套指南） | P2 | 2-3 天 | F1, F2 |
| F10 | LLM-Direct baseline 实现 | P1 | 1-2 天 | Qwen2.5-7B 部署 |
| F11 | RAG-LLM baseline 实现 | P2 | 2-3 天 | F10 + FAISS |
| F12 | 模块3评测框架（指标计算脚本） | P0 | 2 天 | F2 |

**总预估工作量**：P0 约 10-12 天，P1 约 7-10 天，P2 约 7-9 天。合计约 24-31 天。

## 5.6 模块3评测数据

### 评测集构建

模块3的评测需要"case_bundle → 正确随访建议"的 ground truth。构建方案：

| 评测集 | 来源 | 规模 | 构建方式 |
|--------|------|------|---------|
| Rule-derived test set | Phase 4 recommendation_eval 中的 rule_derived 子集 | ~150 条 | 已有 case_bundle + Lung-RADS 规则推导的标准答案 |
| Gold recommendation set | Phase 5.1 Gold 62 条样本 | 62 条 | 需人工补标 recommendation ground truth |
| Synthetic edge cases | 手工构造的边界案例 | 20-30 条 | 覆盖 abstention、soft match、多结节等场景 |

**注意**：模块3的评测质量高度依赖 ground truth 的可靠性。Rule-derived set 的标准答案来自规则推导，本质上对 FlatRule 和 GraphExec 有利（因为它们使用相同规则）。需要在论文中说明这一局限性。

---

# 6. 全局补跑清单与执行优先级

## 6.1 必须补跑（不做则论文不完整）

| # | 实验/功能 | 所属模块 | 预估耗时 | 依赖 | 理由 |
|---|---------|---------|---------|------|------|
| R1 | Vanilla PubMedBERT (full-text, 3 tasks) | M2 | 4 小时 GPU | 无 | 缺少此行则无法证明方法框架 vs 裸模型的价值 |
| R2 | Vanilla PubMedBERT Gold 评测 | M2 | 10 分钟 GPU | R1 | Gold 表需要 Vanilla PubMedBERT 行 |
| R3 | P1 Window Size 参数讨论 (5 settings × 3 tasks, LR) | M2 | 30 分钟 CPU | 无 | 参数讨论表核心项，论文要求至少 3 参数 |
| R4 | P2 Section Strategy 参数讨论 (5 settings × 3 tasks, LR) | M2 | 30 分钟 CPU | 无 | 参数讨论表核心项 |
| R5 | 指南决策图定义 (Lung-RADS v2022 CDSG) | M3 | 2-3 天 | 无 | 模块3一切实验的基础 |
| R6 | Graph Executor 实现 | M3 | 2-3 天 | R5 | 模块3核心组件 |
| R7 | Hard Constraint + Guideline Anchor 迁移 | M3 | 2 天 | R6 | 从 flat engine 迁移规则到图节点 |
| R8 | Reasoning Path 图路径输出 | M3 | 1 天 | R6 | 推理可解释性的基础 |
| R9 | Abstention 机制 | M3 | 1-2 天 | R6 | 安全弃权是图谱智能体的核心差异化能力 |
| R10 | Soft Match 实现 | M3 | 2-3 天 | R6 | 处理模糊输入的核心能力 |
| R11 | 模块3评测框架 | M3 | 2 天 | R6 | 所有模块3实验的评测基础 |
| R12 | LLM-Direct baseline (Qwen2.5-7B) | M3 | 1-2 天 | 无 | 外部范式对比必做项 |
| R13 | 模块3内部比较实验 (I-1 ~ I-4) | M3 | 2-3 天 | R5-R11 | 主结果表 |
| R14 | 模块3外部比较实验 (E-1) | M3 | 1 天 | R12, R11 | 主结果表 |
| R15 | 模块3消融实验 | M3 | 1-2 天 | R13 | 消融表 |
| R16 | 模块3参数讨论 (P1-P3, 各 5 值) | M3 | 2-3 天 | R13 | 参数讨论表 |
| R17 | Case Study 选取与分析 (8-12 cases) | M3 | 2 天 | R13, R14 | 案例分析 |

**模块2必须补跑总耗时**：约 5-6 小时（主要是 GPU 时间）
**模块3必须补建+补跑总耗时**：约 20-25 天（主要是开发时间）

## 6.2 建议补跑（做了更好，不做可以用文字说明替代）

| # | 实验/功能 | 所属模块 | 预估耗时 | 理由 |
|---|---------|---------|---------|------|
| S1 | P3 Silver Label 置信度阈值中间梯度 (3 settings × 3 tasks, LR) | M2 | 20 分钟 CPU | 补充 A3 消融的中间值，使参数讨论更完整 |
| S2 | RAG-LLM baseline (E-2) | M3 | 2-3 天 | 外部范式对比的第二个 baseline，增强说服力 |
| S3 | Fleischner 2017 指南决策图 | M3 | 2-3 天 | 第二套指南支持，展示系统可扩展性 |
| S4 | Report-Intent Router 实现 | M3 | 1 天 | 筛查 vs 偶发路由，系统完整性 |
| S5 | 模块1 Researcher-reviewed subset 构建与评测 | M1 | 1-2 天 | 模块1的 sanity check 实验 |
| S6 | Density 多密度表述处理优化 | M2 | 1-2 天 | 针对 Gold 评测中 unclear 低召回率的改进 |

## 6.3 不需要补跑，仅需重组表格

| # | 内容 | 来源 | 重组方式 |
|---|------|------|---------|
| T1 | Regex/LR/SVM/PubMedBERT Silver 主结果 | Phase 5 | 填入表 M2-1 |
| T2 | Regex/LR/SVM/PubMedBERT Gold 结果 | Phase 5.1 | 填入表 M2-2 |
| T3 | A1/A3/A4 消融结果 | Phase 5 | 填入表 M2-3 |
| T4 | Phase 4 section-aware vs full-text | Phase 4 | 填入表 M2-3 A2 行 |
| T5 | Phase 4 smoking fallback ablation | Phase 4 | 填入模块1相关表格 |
| T6 | Phase 4 cue_only vs structured_rule | Phase 4 | 填入模块3 FlatRule baseline 行 |
| T7 | Phase 5 错误分析 | Phase 5 | 直接复用于论文 §5.5 |
| T8 | Phase 5.1 Silver Ceiling 分析 | Phase 5.1 | 直接复用于论文 §5.4 |
| T9 | ML baseline 参数讨论 (max_features/ngram/C) | Phase 5 | 移入附录 |

## 6.4 暂时不做

| # | 内容 | 理由 |
|---|------|------|
| N1 | 模块1因果解耦建模 (ClinicalBERT + 正交投影) | MIMIC 脱敏率 97.9%，无训练数据 |
| N2 | Clinical-Longformer + CRF 序列标注 | 任务为分类而非序列标注，且 Longformer 在 3090 上资源紧张 |
| N3 | 多 PLM 横向对比 (BioBERT/ClinicalBERT/GatorTron 等) | 资源受限，且与核心论点无关 |
| N4 | LLM few-shot 抽取对比 (模块2) | 模块2核心叙事是弱监督微调，few-shot 对比放在模块3 |
| N5 | 多模态 (DICOM 影像 + 文本) | 超出当前项目范围 |
| N6 | 结节共指消解 | 需要跨 section 合并，工作量大且非核心贡献 |
| N7 | 纵向时序对比 (同一患者多次报告) | 需要时序对齐，数据准备复杂 |
| N8 | GraphRAG (微软原版) | 需要 LLM 构建知识图谱，成本极高，3090 不可行 |

## 6.5 执行顺序建议

### Phase A：模块2补跑（约 1 天）
1. R1: Vanilla PubMedBERT 训练（3 tasks，~4h GPU）
2. R2: Vanilla PubMedBERT Gold 评测（~10min）
3. R3: Window Size 参数讨论（~30min CPU）
4. R4: Section Strategy 参数讨论（~30min CPU）
5. T1-T9: 表格重组（~2h 整理）

### Phase B：模块3基础设施（约 8-10 天）
1. R5: 指南决策图定义
2. R6: Graph Executor 实现
3. R7: Hard Constraint + Guideline Anchor 迁移
4. R8: Reasoning Path 输出
5. R11: 评测框架
6. R9: Abstention 机制
7. R10: Soft Match 实现

### Phase C：模块3实验（约 8-10 天）
1. R12: LLM-Direct baseline 部署
2. R13: 内部比较实验
3. R14: 外部比较实验
4. R15: 消融实验
5. R16: 参数讨论
6. R17: Case Study

### Phase D：建议补跑（约 5-7 天，视时间决定）
1. S2: RAG-LLM baseline
2. S5: 模块1 researcher-reviewed subset
3. S1: P3 置信度阈值中间梯度
4. S4: Report-Intent Router
5. S3: Fleischner 2017 指南图
6. S6: Density 多密度优化

---

# 7. 论文图表与附录组织建议

## 7.1 正文表格目录

| 表号 | 内容 | 对应章节 | 数据来源 | 状态 |
|------|------|---------|---------|------|
| 表 1 | 核心 Schema 字段定义摘要 | §3 方法 | schema_design.md | 已有，需精简 |
| 表 2 | 数据集统计（mentions/subjects/split） | §4 数据集 | Phase 5 数据 | 已有，需整理 |
| 表 3 | 模块2主结果表（Silver，4 baselines + 1 Ours × 3 任务） | §5.2 | Phase 5 + Vanilla PubMedBERT 补跑 | **需补 Vanilla PubMedBERT 行** |
| 表 4 | 模块2 Gold 评测结果（N=62） | §5.4 | Phase 5.1 + Vanilla PubMedBERT 补跑 | **需补 Vanilla PubMedBERT 行** |
| 表 5 | Silver Ceiling 现象量化（Silver vs Gold 性能对比） | §5.4 | Phase 5 + 5.1 | 已有，需重组 |
| 表 6 | 模块2消融实验 | §5.2 | Phase 5 消融 + Vanilla PubMedBERT | **需补 PubMedBERT 版 A1** |
| 表 7 | 模块3主结果表（6 方法 × 6 指标） | §5.6 | **全部需新做** | ❌ 待补 |
| 表 8 | 模块3消融实验 | §5.6 | **全部需新做** | ❌ 待补 |
| 表 9 | 典型误差案例分类表 | §5.5 | Phase 5 错误分析 | 已有，需精选 |

## 7.2 正文图目录

| 图号 | 内容 | 对应章节 | 状态 |
|------|------|---------|------|
| 图 1 | 神经符号双层系统总体架构图 | §3.1 | **需新画**（含三模块 + router） |
| 图 2 | 临床决策状态图（CDSG）示例 | §3.5 | **需新画**（模块3核心） |
| 图 3 | Mention-centered Window 设计示意图 | §3.3 | **需新画** |
| 图 4 | 共享尺寸解析模块数据流图 | §3.5 | **需新画** |
| 图 5 | Silver vs Gold 性能落差对比图（柱状图） | §5.4 | **需新画**（数据已有） |
| 图 6 | Density 混淆矩阵（Gold 集） | §5.5 | **需新画**（数据已有） |
| 图 7 | 模块3推理路径可视化（Case Study） | §5.7 | **需新画**（依赖模块3实现） |
| 图 8 | 参数敏感性曲线（Window Size / Soft-Match Threshold） | §5.2/5.6 | **需新画**（依赖参数讨论实验） |

## 7.3 附录放置建议

| 附录 | 内容 | 来源 |
|------|------|------|
| 附录 A | 完整 JSON Schema 定义与示例 | schema_design.md + schema_examples.md |
| 附录 B | Phase 3 Baseline 正则表达式规则清单 | phase3_baseline_design.md |
| 附录 C | Lung-RADS v2022 规则逻辑映射树（完整版） | 需从 lung_rads_engine.py 整理 |
| 附录 D | 62 条 Gold 评测完整明细 | Phase 5.1 数据 |
| 附录 E | ML Baseline 参数讨论（max_features/ngram/C） | Phase 5 参数讨论 |
| 附录 F | 模块2完整参数讨论表（Window Size / Section Strategy / Confidence） | 补跑后整理 |
| 附录 G | 模块3完整参数讨论表（Soft-Match / Top-K / Abstention） | 补跑后整理 |
| 附录 H | CDSG 完整图定义（所有节点和边） | 模块3实现后整理 |

## 7.4 已有报告中可复用的图表

| 已有报告 | 可复用内容 | 复用方式 |
|---------|----------|---------|
| phase5_main_results.md | Silver 主结果数据 | 填入表 3 |
| phase5_1_gold_eval.md | Gold 评测数据 + 混淆矩阵 | 填入表 4、图 6 |
| phase5_error_analysis.md | 错误案例分类 | 填入表 9 |
| final_experiment_summary.md | 核心结论 | 论文讨论章节引用 |
| thesis_claim_guardrails.md | 写作口径约束 | 论文写作时参考 |
| phase4_first_results.md | Phase 4 对比数据 | 填入表 6 A2 行 |

## 7.5 必须新做的图表

| 图表 | 依赖 | 优先级 |
|------|------|--------|
| 系统架构图（图 1） | 无 | P0（可立即画） |
| CDSG 示例图（图 2） | R5 指南决策图定义 | P0 |
| Mention-centered Window 示意图（图 3） | 无 | P0（可立即画） |
| 模块3主结果表（表 7） | R13, R14 | P0 |
| 模块3消融表（表 8） | R15 | P0 |
| 推理路径可视化（图 7） | R13 | P1 |
| 参数敏感性曲线（图 8） | R3, R4, R16 | P1 |

---

# 8. 最终结论

## 逐条回答核心问题

### 1. 模块1最终如何处理？

**降级为辅助属性模块**。仅保留 coarse-grained smoking status 4 分类（current/former/never/unknown）的 sanity check 实验，使用 researcher-reviewed subset（非 expert gold）。不再尝试 pack-year 精确回归或因果解耦建模。在论文中作为 §3.4 的子节和 §5.3 的子节，篇幅控制在 1-2 页，不单独设主表。

### 2. 模块2最终 baseline 是否锁定？

**已锁定，共 4 个 baseline + 1 个我们的方法**：
1. Regex（规则基线）
2. TF-IDF + LR（ML 基线）
3. TF-IDF + SVM（ML 基线）
4. Vanilla PubMedBERT（full-text 微调，对照消融）
5. Method-PubMedBERT (Ours)（我们的方法）

PubMedBERT-CRF 已移除（任务为分类非序列标注）。DyGIE++/SpERT 已移除（非属性分类任务）。后续执行阶段不允许再增减 baseline。

### 3. 模块2必须补跑哪些？

- **Vanilla PubMedBERT**（3 tasks Silver + Gold 评测）：~4.5 小时 GPU
- **P1 Window Size 参数讨论**（5 settings × 3 tasks, LR）：~30 分钟 CPU
- **P2 Section Strategy 参数讨论**（5 settings × 3 tasks, LR）：~30 分钟 CPU
- 合计约 **1 天**即可完成

### 4. 模块3必须补哪些功能？

必须从零构建的核心功能（P0）：
1. 指南决策图定义文件（Lung-RADS v2022 CDSG）
2. Graph Executor（图遍历引擎）
3. Hard Constraint Validation（迁移现有规则）
4. Guideline Anchor 关联
5. Reasoning Path 图路径输出
6. 评测框架

必须新增的差异化能力（P1）：
7. Abstention / Insufficient-data 机制
8. Soft Match（sentence-transformers 或 NLI）
9. LLM-Direct baseline（Qwen2.5-7B 部署）

### 5. 模块3必须做哪些实验？

- **主结果表**：6 方法（I-1~I-4 + E-1 + E-2）× 6 指标
- **消融表**：4 个消融变量
- **参数讨论表**：3 个参数（soft-match threshold / top-K / abstention threshold）× 各 5 个值
- **Case Study**：8-12 个代表性案例的推理路径分析
- 合计约 **20-25 天**开发+实验时间

### 6. 现在下一步最先做什么？

**Phase A：模块2补跑**（约 1 天）——这是最快能完成、最低风险的工作：
1. 训练 Vanilla PubMedBERT（3 tasks）
2. 跑 Vanilla PubMedBERT Gold 评测
3. 跑 P1/P2 参数讨论
4. 重组所有已有结果到最终表格

完成 Phase A 后，模块2的实验部分即可视为**完整**，可以开始写论文模块2相关章节。

### 7. 现在绝对不要做什么？

1. **不要**开始写模块3的代码——先完成模块2补跑，锁定模块2结果
2. **不要**尝试模块1的因果解耦建模——数据不支持，已定案降级
3. **不要**引入新的 PLM（BioBERT/ClinicalBERT/GatorTron）做横向对比——资源受限且偏离核心论点
4. **不要**尝试 LLM few-shot 做模块2的信息抽取对比——放在模块3更合适
5. **不要**修改已有的 Phase 5/5.1 实验结果或重新训练已有模型——结果已锁定
6. **不要**在没有完成 CDSG 图定义的情况下就开始写 Graph Executor——先设计后实现
7. **不要**在模块3基础设施未就绪时就尝试跑模块3实验——会浪费时间

---

> **本文件为方案三最终实验重构计划的定案版。后续执行阶段以本文件为唯一权威参考。任何偏离本计划的决策需要明确记录理由。**

