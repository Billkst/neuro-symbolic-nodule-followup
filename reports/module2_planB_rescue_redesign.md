# 模块2 Plan B 激进挽救重构方案

> 日期：2026-04-19  
> 阶段：模块2 A2.5 strict 聚合后的 Plan B 设计阶段  
> 目标：重构模块2方法与评测设计，最大化 Ours 成为主表最强方法的可能性  
> 约束：不伪造结果，不暗改已有结果，不靠不诚实口径包装；允许在学术上合理、可答辩的前提下重定义任务、方法结构、baseline（基线方法）、消融和参数讨论。

## 0. 当前判断

当前模块2不能继续按 A2.5 主表进入论文定稿。`density` 任务的正式主表结果异常差，且 `MWS-CFE (Ours)` 在 `density`、`has_size` 和 `location` 三个主指标上均不是最强方法。这个问题不是表头美化或正文解释能解决的，必须进入 Plan B 激进挽救阶段。

当前 A2.5 产物仍然有价值，但只能作为诊断依据和部分重跑资产：

- 正式主表：`outputs/phaseA2/tables/a2_5_main_table.csv`
- 质量门控表：`outputs/phaseA2/tables/a2_quality_gate_summary.csv`
- 聚合方式表：`outputs/phaseA2/tables/a3_aggregation_summary.csv`
- 参数 P1 表：`outputs/phaseA2/tables/p1_max_length_summary.csv`
- 参数 P3 表：`outputs/phaseA2/tables/p3_section_input_summary.csv`
- 审计报告：`reports/module2_final_table_audit.md`

## 1. 诊断当前主表为什么失败

### 1.1 `density` 主表为什么异常差

当前 `density` 主表低到不可接受的直接原因是训练分布与正式评测分布严重错位，尤其是 `unclear` 类。

当前 A2.5 `density` G2 训练集：

| Split | N | solid | part_solid | ground_glass | calcified | unclear |
|---|---:|---:|---:|---:|---:|---:|
| `density_train_ws_g2` | 36,038 | 7,385 | 1,091 | 18,898 | 7,907 | 757 |
| `phase5/density_test` | 42,057 | 1,090 | 185 | 2,976 | 1,325 | 36,481 |

也就是说：

- G2 训练集中 `unclear` 只有 757 / 36,038，约 2.1%。
- Phase5 正式 test 中 `unclear` 有 36,481 / 42,057，约 86.7%。

这个分布错位足以解释为什么所有方法的 `Density Acc.` 都只有约 12% 到 13%。模型主要学到的是显式密度子类型分类，但正式 test 大部分样本本质上是 `unclear / no-evidence`。当前评测把“没有明确密度证据”和“具体密度子类型”混在一个五分类任务中，导致模型必须同时解决两个不同问题：

1. 有没有明确 density 证据。
2. 有明确证据时属于哪一种 density subtype。

这不是一个干净的单阶段分类任务。

### 1.2 异常差主要是任务定义和标签空间问题，不是单纯模型问题

当前问题的主因排序如下：

1. **任务定义问题**：`density` 被建成 `solid / part_solid / ground_glass / calcified / unclear` 单阶段五分类，但 `unclear` 既包含真正语义不明确，也包含没有证据、非目标描述、混合多结节、报告级泛化和候选抽取噪声。
2. **标签空间问题**：`unclear` 与四个显式子类型不是同一语义层级。前者是证据状态，后者是影像学子类型。
3. **`unclear / no-evidence` 污染问题**：正式 test 被 `unclear` 主导，训练集却几乎没有对应比例的 `unclear`，导致正式主表同时惩罚 evidence detection（证据检测）和 subtype classification（子类型分类）。
4. **模型问题**：模型确实有问题，但不是唯一主因。当前 `MWS-CFE` 的 `ws_confidence` 设计没有真正进入 loss（损失函数），`ConfidenceWeightedTrainer` 接口有 `sample_weights`，但 `compute_loss` 实际只用了 class weight（类别权重），没有用样本置信度权重。

这一判断有真实结果支持。以 seed 42 为例：

- `MWS-CFE` 在 WS test 上 `density` Macro-F1（宏平均 F1）为 0.7524，但在 Phase5 test 上只有 0.2516。
- `Vanilla PubMedBERT` 在 WS test 上 `density` Macro-F1 为 0.8155，但在 Phase5 test 上为 0.5164。
- 两者在 Phase5 test 上的 `unclear` F1 都约为 0.009，说明 `unclear` 类几乎没有被正确建模。

因此当前 `density` 失败不能被粉饰成“模型略弱”。它是任务定义、评测口径、标签空间和训练目标共同失败。

### 1.3 为什么当前 Ours 没能赢主表

当前主表关键数值如下：

| Method | Density Macro-F1 | Has-size F1 | Location Macro-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 17.64 ± 0.00 | 87.83 ± 0.00 | 95.95 ± 0.00 |
| TF-IDF + SVM | 18.52 ± 0.00 | 87.53 ± 0.00 | 96.73 ± 0.00 |
| Vanilla PubMedBERT | 43.76 ± 6.92 | 82.11 ± 1.28 | 97.13 ± 0.83 |
| MWS-CFE (Ours) | 23.17 ± 4.47 | 81.75 ± 1.26 | 96.63 ± 0.33 |

当前 Ours 没有赢，原因很直接：

1. **Ours 与 Vanilla 的结构差异不足**：两者都是同一 BiomedBERT / PubMedBERT backbone（主干预训练模型）上的 hard-label classifier（硬标签分类器）。Ours 主要多了 class weight，但没有真正实现 confidence-aware training（置信度感知训练）。
2. **质量门控主配置不适合 density**：当前主表用 G2。A2 quality-gate 结果显示 density 的 G3 Macro-F1 为 `39.84 ± 3.60`，明显高于 G2 的 `23.17 ± 4.47`。但不能事后把 G3 直接替换成主结果，除非重新设计并通过 validation（验证集）锁定 gate，否则会产生 test 后选择偏差。
3. **加权聚合不是当前瓶颈**：A3 aggregation 显示 density weighted vote 为 `23.17 ± 4.47`，uniform vote 为 `23.24 ± 8.34`，差异极小。
4. **`has_size` 是强词面任务**：TF-IDF + LR/SVM 在 `has_size` 上明显强于神经模型，说明尺寸有无检测主要依赖数字、单位和简单窗口，当前 Ours 没有把符号规则优势并入最终方法。
5. **`location` 已经接近天花板**：多方法 Macro-F1 都在 95% 以上，Ours 只有小幅落后，但这类任务很难支撑“方法显著最强”的主结论。

### 1.4 当前 4 个 baseline 为什么不够

当前只有 4 个方法：

1. TF-IDF + LR
2. TF-IDF + SVM
3. Vanilla PubMedBERT
4. MWS-CFE (Ours)

这个面板不够，原因如下：

- 缺少 Regex / cue-only（正则或线索词）baseline，无法证明规则线索在 density/size/location 中的上限或下限。
- 缺少 lightweight neural baseline（轻量神经基线），例如 MLP（多层感知机）或 fastText 风格平均词向量模型，导致传统 ML 与 PLM 之间断层。
- 缺少第二个 PLM baseline（预训练语言模型基线），无法证明结果不是某一个 backbone 的偶然表现。不过当前本地只看到 `biomedbert_base` 和 `biomedbert_base_safe`，第二个 PLM 若无本地缓存，需要额外下载，不应作为最小抢救路径的硬依赖。
- 缺少 two-stage baseline（两阶段基线），如果 Plan B 把 density 改成两阶段任务，baseline 也必须同步两阶段化。

### 1.5 当前消融与参数讨论为什么不达标

当前消融不是标准论文里的 `Full vs w/o ...` 格式，而是工程实验表：

- A2 quality-gate：G1/G2/G3/G4/G5
- A3 aggregation：weighted vs uniform

这些能说明一些趋势，但不像标准消融。它们没有明确回答 Full model 由哪些组件构成，也没有逐项去掉组件来证明每个组件的贡献。

当前参数讨论也不完整：

- 已有 P1：`max_seq_length`
- 已有 P3：`section/input strategy`
- 缺少至少一个正式 P2 参数。
- 目前只有表，没有正式图。
- P1/P3 都是 density-only，不能直接支撑整个模块2方法的参数鲁棒性。

因此，当前消融和参数讨论都不能直接作为毕业论文最终实验设计。

## 2. 新的正式主表方案

### 2.1 `density` 任务必须重定义

必须重定义。继续使用单阶段五分类会把 `unclear / no-evidence` 与显式密度子类型混在一起，既不符合临床语义，也不利于模型公平比较。

新的 density 主任务应改成 two-stage density（两阶段密度抽取）：

**Stage 1：Explicit-density detection（显式密度证据检测）**

- 输入：候选 nodule mention（结节提及）及其上下文。
- 输出：`explicit_density` vs `unclear_or_no_evidence`。
- 指标：F1、AUPRC（精确率-召回率曲线下面积）、AUROC（受试者工作特征曲线下面积）、Precision / Recall（精确率 / 召回率）。
- 目的：判断该候选是否有足够证据进入 subtype classification。

**Stage 2：Density subtype classification（显式子类型分类）**

- 输入：Stage 1 判定为显式密度的样本，或 gold/silver 中非 `unclear` 的 explicit subset（显式子集）。
- 输出：`solid / part_solid / ground_glass / calcified`。
- 指标：Macro-F1、Balanced Accuracy（平衡准确率）、per-class F1（逐类 F1）。
- 目的：只在有明确证据的样本上比较子类型识别能力。

### 2.2 `unclear / no-evidence` 应从主表核心口径中拆出

应该拆出，而且这是 Plan B 的核心。

学术正当性：

- `unclear / no-evidence` 表示证据状态，不是 density subtype。
- 放射报告里没有写密度并不等价于密度本身属于“unclear subtype”。
- 临床抽取系统更合理的流程是先判断证据是否存在，再在证据存在时抽取结构化类别。
- gold evaluation（人工金标评估）显示 density silver-gold agreement（一致率）只有 72.6%，且 `unclear` recall（召回率）只有 28.6%，说明 `unclear` 本身是高噪声区域。

### 2.3 推荐主表定义

推荐新的正文主表采用下面结构：

| Method | Stage 1 Explicit F1 | Stage 1 AUPRC | Stage 2 Density Macro-F1 | Has-size F1 | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Regex / cue-only |  |  |  |  |  |
| TF-IDF + LR |  |  |  |  |  |
| TF-IDF + SVM |  |  |  |  |  |
| MLP / fastText-style |  |  |  |  |  |
| Vanilla PubMedBERT |  |  |  |  |  |
| MWS-CFE (Ours) |  |  |  |  |  |

如果篇幅允许，另加一个 appendix table（附录表）报告旧单阶段五分类结果，明确说明这是 legacy monolithic density（旧单阶段密度任务）诊断，不作为 Plan B 主结论。

最有学术正当性、也最有机会体现 Ours 价值的定义是：

1. density 主评价改为 two-stage。
2. `unclear / no-evidence` 只作为 Stage 1 的 negative class（负类）。
3. Stage 2 只比较 explicit subset 上的 subtype classification。
4. gate、section strategy 和阈值必须通过 validation set 选择，不能用 test 后挑选。

### 2.4 新主表至少包含多少 baseline

新主表至少需要 6 个方法：

1. Regex / cue-only
2. TF-IDF + LR
3. TF-IDF + SVM
4. MLP 或 fastText-style
5. Vanilla PubMedBERT
6. MWS-CFE (Ours)

第二个 PLM baseline 可作为增强项，但不是最小挽救路径的硬要求，因为当前本地模型目录只有 `biomedbert_base` 和 `biomedbert_base_safe`。如果没有本地缓存，临时联网下载第二个 PLM 会增加失败风险。

## 3. Ours 的方法升级方案

### 3.1 当前 Ours 的关键瓶颈

当前 Ours 的关键瓶颈不是“名字不够强”，而是方法结构没有真正落到 MWS-CFE 应有的差异化能力上：

1. **confidence-aware training 没有真正生效**：代码中 `ConfidenceWeightedTrainer` 没有把 `ws_confidence` 乘到 per-sample loss（逐样本损失）上。
2. **density 单阶段建模错误**：`unclear` 与 subtype 混成一个五分类任务。
3. **主配置 gate 不适合 density**：G3 明显优于 G2，但当前没有 validation-locked gate selection（验证集锁定门控选择）。
4. **输入策略弱**：主配置只用 `mention_text`，但已有 P3 显示 `impression` 的 density Macro-F1 为 `31.27 ± 7.21`，高于 `mention_text / main` 的 `23.17 ± 4.47`。
5. **符号规则没有成为 Ours 的正式组件**：`has_size` 和一部分 `location` 本质上高度符号化，当前 Ours 没有把 rule / section / candidate quality 作为最终方法的可见部件。

### 3.2 必要重构与优先级

| 优先级 | 改动 | 类型 | 预期收益 | 成本 | 是否必要 |
|---:|---|---|---|---|---|
| P0 | 修复 confidence-aware loss，真正使用 `ws_confidence` | 训练目标 | 高 | 低 | 必要 |
| P0 | density 改成 two-stage | 任务建模 | 很高 | 中 | 必要 |
| P0 | `unclear / no-evidence` 拆成 Stage 1 negative | 标签设计 | 很高 | 中 | 必要 |
| P1 | gate 通过 validation 选择，不再固定 G2 | 主导样本选择 | 高 | 中 | 必要 |
| P1 | Stage 2 只在 explicit subset 训练和评估 | 样本选择 | 高 | 中 | 必要 |
| P1 | section strategy 加入 `mention + impression` 或 validation-selected input | 输入策略 | 中到高 | 中 | 必要 |
| P2 | focal loss（聚焦损失）或 class-balanced focal loss | 训练目标 | 中 | 中 | 可选增强 |
| P2 | temperature calibration（温度校准）和阈值选择 | 决策阈值 | 中 | 中 | 可选增强 |
| P3 | 更换第二个 PLM backbone | 模型骨干 | 不确定 | 高 | 非最小路径 |

### 3.3 最可能显著提升 Ours 主表表现的改动

最可能有效的组合是：

1. **Two-stage density**：先 explicit detection，再 subtype classification。
2. **真正 confidence-aware training**：`loss_i = CE_i * class_weight_y * ws_confidence_i`，并记录是否启用。
3. **Validation-selected quality gate**：在 validation 上选择 G1/G2/G3/G4/G5，而不是固定 G2 或 test 后挑选。
4. **Explicit subset training**：Stage 2 不让 `unclear` 参与 subtype 学习。
5. **Section-aware input**：至少比较 `mention_text`、`impression`、`findings_impression`，在 validation 上确定 Full 的输入策略。

### 3.4 不值得优先做的 cosmetic 改动

以下改动只能改善表面，不足以挽救模块2：

- 只把 `Has_size` 改名为 `Has-size`。
- 删除主表 `N` 列。
- 把 G3 事后写成 density 主结果，但不说明选择规则。
- 把旧单阶段 density 换一个更好看的表头。
- 只写“弱监督更稳定”但不补方法级证据。
- 增加效率表，试图转移 Ours 不是最强的问题。

## 4. 重新设计 baseline 面板

### 4.1 新主表建议保留与新增的方法

| 方法 | 是否保留 | 是否重跑 | 说明 |
|---|---|---|---|
| Regex / cue-only | 新增 | 必须新跑 | 最高性价比新增 baseline，能解释密度线索词和尺寸规则的强度 |
| TF-IDF + LR | 保留 | 必须按 Plan B 重跑 | 当前结果可诊断，但不能直接复用到 two-stage |
| TF-IDF + SVM | 保留 | 必须按 Plan B 重跑 | 当前结果可诊断，但不能直接复用到 two-stage |
| MLP / fastText-style | 新增 | 必须新跑 | 补上传统 ML 与 PLM 之间的中间档 baseline |
| Vanilla PubMedBERT | 保留 | 必须按 Plan B 重跑或至少加载 checkpoint 重评估 | 当前是最强神经 baseline，必须保留 |
| MWS-CFE (Ours) | 保留并升级 | 必须重训 | 当前版本不是最终 Ours |
| 第二个 PLM | 可选 | 视本地模型而定 | 当前未发现本地第二 PLM checkpoint，不作为最小矩阵硬要求 |

### 4.2 哪个 baseline 最值得补

最值得补的是 **Regex / cue-only two-stage baseline**。

原因：

- 成本最低。
- 对 density Stage 1 和 size 任务非常有解释力。
- 论文答辩时可以回答“简单规则能做到什么程度”。
- 如果 Ours 能超过 cue-only，说明不是只靠关键词；如果不能超过，也能及时暴露方法不值得继续包装。

第二性价比是 **MLP / fastText-style baseline**，因为它能补齐 TF-IDF 与 PLM 之间的模型复杂度层级。

## 5. 把消融重构成标准论文格式

### 5.1 Full model 定义

Plan B 的 Full model 应定义为：

**MWS-CFE two-stage model：**

- 多源弱监督标签。
- Validation-selected quality gate。
- Weighted aggregation（加权聚合）。
- Confidence-aware training。
- Section-aware input strategy。
- Stage 1 explicit-density detection。
- Stage 2 explicit density subtype classification。
- 对 `has_size` 和 `location` 保留同一 weak-supervision protocol，并允许符号 fallback 作为 Ours 框架的一部分，但必须在方法节写清楚。

### 5.2 标准消融表设计

正文消融表建议如下：

| Variant | Stage 1 Explicit F1 | Stage 1 AUPRC | Stage 2 Density Macro-F1 | Has-size F1 | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Full |  |  |  |  |  |
| w/o quality gate |  |  |  |  |  |
| w/o weighted aggregation |  |  |  |  |  |
| w/o confidence-aware training |  |  |  |  |  |
| w/o section strategy |  |  |  |  |  |
| w/o multi-source supervision |  |  |  |  |  |

### 5.3 变体对应关系

| 变体 | 定义 | 当前资产能否复用 | 是否必须补跑 |
|---|---|---|---|
| Full | Plan B 完整 two-stage MWS-CFE | 不能直接复用 | 必须补跑 |
| w/o quality gate | 使用 G1 或无 gate，不做质量筛选 | A2 表可作 legacy 参考 | 必须按 Plan B 补跑 |
| w/o weighted aggregation | 使用 uniform vote | A3 表可作 legacy 参考 | 必须按 Plan B 补跑 |
| w/o confidence-aware training | 关闭 `ws_confidence` loss 权重 | 当前代码近似这个状态 | 必须按修复后代码补跑 |
| w/o section strategy | 固定 `mention_text`，不使用 section-aware input | P3 表可作 legacy 参考 | 必须按 Plan B 补跑 |
| w/o multi-source supervision | 只用 explicit 或 single-source labels | 旧 Phase5 ablation 不可直接复用 | 必须补跑 |

当前已有 A2/A3/P3 结果可以作为“为什么选择这些消融项”的依据，但不应直接拼成最终标准消融表。否则 Full 与 w/o 的任务定义不一致，答辩时站不住。

## 6. 参数讨论补成正式 3 参数方案

### 6.1 P1：`max_seq_length`

参数取值：

- 64
- 96
- 128
- 160
- 192

已有结果：

- 可复用为 legacy 诊断。
- 旧结果显示 192 的 density Macro-F1 最高，为 `24.77 ± 6.10`，但仍是旧单阶段任务。

Plan B 要求：

- 需要在 two-stage density 上重跑或至少重评估。
- 图形建议：折线图。横轴为 `max_seq_length`，纵轴为 Stage 2 Density Macro-F1；可加 Stage 1 F1 的第二条线。
- 位置建议：附录为主，正文只写趋势。

### 6.2 P2：`quality_gate`

参数取值：

- G1
- G2
- G3
- G4
- G5

已有结果：

- 可复用为 legacy 诊断。
- 旧结果中 density G3 最强：`39.84 ± 3.60`。
- location G1 最强：`99.31 ± 0.22`。
- G5 对 location 明显退化：`67.49 ± 2.11`。

Plan B 要求：

- 必须在 validation 上确定 Full 使用的 gate。
- 不能按 test 最优挑选。
- 图形建议：分组柱状图或折线图。每个 gate 展示 Stage 1 F1 和 Stage 2 Macro-F1。
- 位置建议：正文，因为质量门控是 MWS-CFE 的核心部件。

### 6.3 P3：`section/input strategy`

参数取值：

- `mention_text`
- `findings`
- `impression`
- `findings_impression`
- `full_text`

已有结果：

- 可复用为 legacy 诊断。
- 旧结果显示 `impression` 的 density Macro-F1 最高：`31.27 ± 7.21`。
- `full_text` Accuracy 最高但 Macro-F1 不突出，说明长文本可能强化多数类或噪声。

Plan B 要求：

- 在 two-stage density 上重跑。
- 图形建议：柱状图。横轴为 input strategy，纵轴为 Stage 2 Density Macro-F1；另可用浅色柱展示 Stage 1 Explicit F1。
- 位置建议：正文或附录均可。若 Full 使用 section-aware input，则正文必须展示 P3；否则放附录。

### 6.4 可选 P4：`confidence_threshold`

如果时间允许，新增 P4：

- 0.0
- 0.3
- 0.5
- 0.7
- 0.9

用途：

- 控制 Stage 1 进入 Stage 2 的置信度阈值。
- 支撑 precision-recall trade-off（精确率-召回率权衡）。

图形建议：

- 折线图，展示 threshold 对 Precision、Recall、F1 的影响。

位置建议：

- 附录。除非 P4 对最终 Full 选择非常关键，否则不放正文。

## 7. 最小可行重跑矩阵

### 7.1 优先级

重跑顺序必须是：

1. 先让 Ours 变强。
2. 再修复主表。
3. 再修复消融。
4. 最后修复参数讨论和图。

### 7.2 最小矩阵

| 阶段 | 实验 | 方法 | Seeds | 操作 | 目的 |
|---|---|---|---:|---|---|
| M0 | 构建 two-stage density 数据与评测脚本 | 全部 | - | 新增脚本，不训练 | 修复任务定义 |
| M1 | Ours two-stage density Full | MWS-CFE | 5 | 必须重训 | 先确认 Ours 是否有救 |
| M2 | Vanilla two-stage density | PubMedBERT | 5 | 必须重训或加载 checkpoint 重评估后决定 | 最强神经 baseline |
| M3 | Regex / cue-only two-stage | Rule | 1 | 新跑 | 补最重要 baseline |
| M4 | TF-IDF + LR/SVM two-stage | ML | 5 或确定性 1 | 新跑 | 保留当前传统 baseline |
| M5 | MLP / fastText-style | ML | 5 | 新跑 | 补第 5 个以上 baseline |
| M6 | Ours ablation: w/o confidence-aware | MWS-CFE | 5 | 必须重训 | 证明修复 loss 有贡献 |
| M7 | Ours ablation: w/o section strategy | MWS-CFE | 5 | 必须重训 | 证明输入策略贡献 |
| M8 | Ours ablation: w/o quality gate | MWS-CFE | 5 | 必须重训 | 证明质量门控贡献 |
| M9 | Ours ablation: w/o weighted aggregation | MWS-CFE | 5 | 必须重训 | 证明聚合贡献 |
| M10 | P1/P2/P3 参数图 | Ours | 视预算 | 部分重跑 | 补正式参数讨论 |

### 7.3 可直接复用、只重评估、必须重训

**可以直接复用：**

- 当前 A2.5 主表作为失败诊断。
- A2 quality-gate 表作为 legacy evidence。
- A3 aggregation 表作为 legacy evidence。
- P1/P3 表作为参数方向依据。
- gold/silver agreement 作为任务重定义依据。

**只需重评估：**

- 若现有 saved checkpoint 可加载，可在 explicit subset 上重评估 `Vanilla PubMedBERT` 与旧 `MWS-CFE`，作为过渡诊断。
- 但因为当前结果 JSON 没有保存逐样本预测，不能只靠 JSON 重新切指标；需要加载模型重新 predict（预测）。

**必须重训：**

- Plan B Full Ours。
- 修复 confidence-aware loss 后的 Ours。
- two-stage Vanilla baseline。
- two-stage TF-IDF / MLP baseline。
- 标准 `Full vs w/o ...` 消融。

### 7.4 时间预算

保守估计：

- two-stage 数据构建与评测脚本：0.5 天。
- Ours density Full 5 seeds：约 0.5 到 1 小时 GPU 时间，外加排错。
- Vanilla density 5 seeds：约 0.5 到 1 小时 GPU 时间。
- TF-IDF / Regex / MLP：约 0.5 小时到 1 小时 CPU 时间。
- 4 个关键 Ours 消融 × 5 seeds：约 2 到 4 小时 GPU 时间。
- P1/P2/P3 最小图：若只补关键点，约 2 到 4 小时；若完整 5 seeds 全矩阵，约 0.5 到 1 天。

最小抢救版可以在 1 到 2 天内完成；完整稳健版需要 2 到 3 天。

## 8. 最终判断

### 8.1 模块2是否还有机会

有机会，但前提是必须做方法级和评测级重构。

最有希望的路径不是继续修饰当前单阶段五分类主表，而是：

1. 把 density 改成 two-stage。
2. 从主表核心口径中拆出 `unclear / no-evidence`。
3. 修复 Ours 的 confidence-aware training。
4. 用 validation 选择 gate 和 input strategy。
5. 扩展 baseline 面板到至少 6 个方法。
6. 用标准 `Full vs w/o ...` 消融证明每个组件的必要性。

### 8.2 如果只做表格重排是否足够

不够。只做表格重排已经不足以挽救模块2。

当前 Ours 在主表三个主指标上都不是最强，且 density Acc. 异常低。继续用文字包装会非常容易被导师或答辩委员指出：

- 为什么 `Density Acc.` 只有 12% 左右。
- 为什么 Ours 不如 Vanilla PubMedBERT。
- 为什么 has-size 不如 TF-IDF。
- 为什么消融不是标准 `Full vs w/o ...`。

这些问题不能靠文字解释解决，必须补方法和实验。

### 8.3 下一步是否必须暂停模块3

必须暂停模块3，先集中修复模块2。

原因：

- 模块2是肺结节随访建议生成系统的信息抽取基座。
- 如果模块2主表不成立，模块3即使完成，也会建立在不可靠的结构化输入上。
- 当前最大论文风险不是模块3缺少功能，而是模块2核心实验无法支撑方法贡献。

结论：立即进入 Plan B 激进挽救，优先完成 two-stage density、Ours 方法修复、扩展 baseline、标准消融和 3 参数图。模块3只能在模块2 Plan B 主表出现可答辩结果后再恢复。
