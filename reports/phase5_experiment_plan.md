# Phase 5 实验计划：模块二放射报告信息抽取主模型实验

## 1. 实验目标

本阶段目标是围绕放射报告中的肺结节关键信息，建立可复现实验框架，并完成 3 个主任务的统一建模与评测。

- **density_category**：输入为以结节 mention 为中心截取的文本窗口，输出为密度类别标签。评测指标以 `macro_f1` 为主，同时报告 `accuracy`、各类 `F1` 与混淆矩阵。
- **size_mm**：输入为 mention-centered 文本窗口，首先输出是否存在明确尺寸信息的二分类结果（`has_size`），再对存在尺寸的样本输出 size 数值回归结果。评测指标包括 `accuracy`、`precision`、`recall`、`f1`，以及回归部分的 `MAE`、`exact match rate`、`within tolerance rate`。
- **location_lobe**：输入为 mention-centered 文本窗口，输出为结节所属肺叶位置标签。评测指标以 `macro_f1` 为主，同时报告 `accuracy`、各类 `F1` 与混淆矩阵。

3 个任务共同构成模块二主模型实验的核心评测对象，目标是在规则方法之上建立可扩展、可比较、可落地的监督学习基线与主模型方案。

## 2. 数据来源与构建

数据来源为 **MIMIC-IV radiology notes**。本阶段仅使用放射报告文本，不引入额外结构化标签作为模型输入。

数据构建遵循以下原则：

- **过滤条件**：优先筛选 Chest CT 报告，并要求文本中存在非否定的结节 mention，避免将否定、排除或历史描述误当作当前有效样本。
- **mention-centered text window 设计**：以结节 mention 为中心截取局部上下文，而不是直接使用整篇 full text。这样可以降低噪声、缩短序列长度，并增强对局部修饰词的聚焦能力。
- **silver label 生成逻辑**：优先使用规则抽取到的显式字段构造标签；当字段表达不完全显式但可由上下文稳定推断时，生成 silver label；若仅存在弱线索，则标记为较低置信度样本。
- **标签质量分层**：
  - `explicit`：报告中存在明确、直接、可定位的字段表达；
  - `silver`：通过规则和上下文可较可靠推导；
  - `weak`：存在弱提示但噪声较大，仅适合作为扩展样本或消融对照。

结合 Phase 4 baseline 结果，当前候选报告约 631 个，涉及约 295 个 unique subjects。标签分布明显不均衡：

- `density_category` 中 `unclear` 约占 86.6%，`part_solid` 仅约 6 个样本；
- `location_lobe` 中 `null/no_location` 占比约 58.5%；
- `size_mm` 的显式抽取率约为 40.9%。

因此，Phase 5 必须把类别不平衡、标签置信度与评测口径统一作为基础设施的一部分来处理。

## 3. 数据切分原则

为避免同一患者的多份报告同时进入训练与测试，数据切分采用 **subject_id 去重切分**。

- `train`：70%
- `val`：15%
- `test`：15%
- 随机种子固定为 `seed=42`

该切分策略能够减少患者级信息泄漏，使评测结果更接近真实泛化能力。所有任务共享同一切分逻辑，便于横向对比不同模型与不同任务的表现。

## 4. 模型选择与理由

### 4.1 Rule baseline (regex)

规则基线复用 Phase 4 的 `section_aware_regex` 思路，作为最低性能基线。

选择理由：

- 已有实现基础，复用成本最低；
- 具备较强可解释性，便于分析错误类型；
- 可为后续 ML 与神经模型提供明确参照；
- 在字段显式时通常 precision 较高，适合充当高精度低召回下界。

### 4.2 Lightweight ML baseline

轻量级机器学习基线包括：

- `TF-IDF + Logistic Regression`
- `TF-IDF + Linear SVM`

选择理由：

- 训练速度快，CPU 即可运行；
- 在小样本、短文本、多类别文本分类中通常有稳定表现；
- 作为非深度学习对照，可以区分“预训练语言模型收益”与“简单词袋模型即可解决”的边界。

### 4.3 Main neural model (PubMedBERT)

主模型采用 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`。

选择理由：

- 约 110M 参数，规模适中；
- 已在医学文本上完成预训练，更适合放射报告语言风格；
- 对局部上下文中的修饰词、比较词、部位词与数值表达更敏感；
- 在单卡 3090 环境下能够稳定完成微调实验。

## 5. 为什么这套方案适合单卡 3090

该方案针对单卡 3090 做了有意识的资源约束设计：

- PubMedBERT base 模型权重约 440MB，`fp16` 训练下显存通常可控制在 4GB 以内；
- mention-centered window 采用短序列设计，`max_seq_length=128` 时配合 `batch_size=16` 基本无压力；
- 以单任务方式分别训练 `density_category`、`size_mm`、`location_lobe`，避免多任务训练带来的显存竞争与调参复杂度；
- 每个任务预计训练约 3 到 10 个 epoch，单次实验通常可控制在约 10 分钟量级；
- 不需要多卡、不需要 DeepSpeed，也不依赖复杂分布式训练框架。

## 6. 对比实验设计

每个主任务都采用统一对比框架：

- `regex` 基线
- `Lightweight ML baseline`
- `PubMedBERT` 主模型

统一评测口径包括：

- 分类任务统一报告 `accuracy`、主指标 `macro_f1` 或 `f1`、各类别 `F1` 与混淆矩阵；
- `size_mm` 额外报告回归 `MAE` 与容差命中指标；
- 所有结果在相同数据切分与相同标签集上比较，保证结论可解释。

## 7. 消融实验设计

- **A1: section-aware window vs full-text window**  
  检验 mention-centered / section-aware 局部窗口是否优于整篇文本输入。

- **A2: with vs without chest CT exam filtering**  
  检验仅保留 Chest CT 是否能够减少标签噪声、提升模型稳定性。

- **A3: explicit-only subset vs all silver labels**  
  检验高质量小样本与更大规模 silver 数据之间的收益权衡。

其中，A3 对 `part_solid` 等极少数类别尤为重要，因为这些类别样本极少，若只保留 `explicit`，可能进一步恶化类别学习能力。

## 8. 参数讨论实验设计

- **P1: max_seq_length: 64 / 128 / 256**  
  检验序列长度对性能与显存的影响，确认 128 是否是性能与效率平衡点。

- **P2: learning_rate: 2e-5 / 3e-5 / 5e-5**  
  检验 PubMedBERT 微调在小样本医学文本任务中的学习率敏感性。

- **P3: window context: mention-only / mention+1sent / mention+2sent**  
  检验上下文范围对位置、密度和尺寸信息识别的影响，评估局部上下文是否足够。

参数讨论实验优先在开发集上进行，避免在测试集上过度调参。

## 9. 3090 约束下的设计决策

### 为什么没有选择更大的模型

更大的医学语言模型虽然可能提升上限，但在当前样本规模下未必带来稳定收益，反而会增加显存占用、训练时间与过拟合风险。对当前数据规模而言，PubMedBERT base 已经是更合理的容量选择。

### 为什么没有做多任务大一统模型

`density_category`、`size_mm` 与 `location_lobe` 的标签空间、难度结构与噪声来源并不一致。先采用单任务方案有利于清晰定位瓶颈，并减少任务间梯度干扰。待单任务收益明确后，再考虑多任务整合更稳妥。

### 为什么先做模块二，而不是 smoking / recommendation 主模型

模块二中的放射报告字段是后续随访建议生成的核心结构化依据，也是神经符号推理链条中最直接的上游输入。若这些基础字段提取不稳定，后续 recommendation 生成质量将受到直接限制，因此优先级更高。

### 如何压缩训练时间

- 统一使用 mention-centered 短窗口；
- 固定 `max_seq_length=128` 作为默认设置；
- 使用 `fp16`；
- 优先单任务训练；
- 先跑规则与轻量基线，快速验证数据可学性；
- 只对最有希望的组合展开小规模参数搜索。

### 每个任务的预估训练耗时与显存占用

在单卡 3090 条件下，基于当前配置的粗略预估如下：

- `density_category`：约 8 到 12 分钟，显存约 3GB 到 4GB；
- `size_mm` 分类部分：约 6 到 10 分钟，显存约 3GB 到 4GB；
- `location_lobe`：约 8 到 12 分钟，显存约 3GB 到 4GB；
- 轻量级 `TF-IDF` 基线：通常为分钟级，且可在 CPU 上完成。

整体来看，该设计在 3090 资源约束下兼顾了实验可行性、结果可比较性与后续扩展空间，适合作为模块二主模型实验的第一版正式方案。
