# Phase A2 正式执行方案 v2

> 日期：2026-04-12
> 适用范围：模块2（放射学报告结构化信息抽取）正式补跑与成表
> 方法名：**MWS-CFE**（Multi-source Weak Supervision for Clinical Fact Extraction，多源弱监督临床事实抽取）
> 硬件约束：**单张 RTX 3090 24GB**
> 唯一目标：在统一 multi-source weak supervision 框架下完成模块2正式主表、公平比较、核心消融与时间预算重排

---

# 1. 执行摘要

旧版 A2 必须重写，原因不是局部修补，而是实验逻辑本身已经失效：

1. 旧版主表仍把旧 Phase 5 的 single-source Regex silver 结果直接拿来和新版 MWS-CFE 混排，**公平性不成立**。
2. 旧版仍按单 seed 思路组织结果，和导师明确要求的 **5-seeds + mean ± std** 冲突。
3. 旧版主表指标不完整，缺少 `Params (M)`、三任务双指标、`Inference Time (ms/sample)` 等正式论文主表需要的字段。
4. 旧版把 researcher-reviewed 小样本继续放在主线叙事里，默认成了 gold-like 证据，但当前条件下**不能再这样写**。
5. 旧版资源配置仍以 `train_batch_size=32` 为默认中心，和当前要求的 **aggressive but realistic search** 不一致。

新版 A2 的核心变化如下：

1. **正式主表只保留统一 multi-source WS 下的 4 个可训练比较对象**：`TF-IDF + LR`、`TF-IDF + SVM`、`Vanilla PubMedBERT`、`MWS-CFE (Ours)`。
2. **Regex 退出正文公平主表**，降级为历史 teacher / rule reference，仅保留在附录或补充说明。
3. **所有 trainable 方法统一使用同一套 multi-source WS 数据、同一 subject split、同一默认 gate（G2）、同一任务定义**。
4. **正式主表全部改为 5-seeds 汇总**；Regex 为确定性方法，不做 seed，但不进入公平主表。
5. **researcher-reviewed subset 改名并降级**为 `researcher-reviewed sanity subset`，只放附录，不再承担正文主结论。
6. **A2 执行顺序改为“先锁主表，再做核心消融，再做扩展分析”**，避免 GPU 时间被非关键实验提前占满。

---

# 2. 模块2正式主表设计

## 2.1 正文主表

正文主表只回答一个问题：在**统一 multi-source weak supervision** 下，4 个正式比较对象谁更强、代价多大。

### 主表行

仅保留以下 4 行：

1. `TF-IDF + LR`
2. `TF-IDF + SVM`
3. `Vanilla PubMedBERT`
4. `MWS-CFE (Ours)`

### 主表列

正文主表固定为：

| Method | Params (M) | Density Acc | Density Macro-F1 | Has_size Acc | Has_size F1 | Location Acc | Location Macro-F1 | Inference Time (ms/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 主表统计规则

1. `Density Acc`、`Density Macro-F1`、`Has_size Acc`、`Has_size F1`、`Location Acc`、`Location Macro-F1`：
   - 对 5 个 seed 的 test 结果汇总为 `mean ± std`
2. `Params (M)`：
   - 定义为**模块2三任务整体部署参数量**，即 density / has_size / location 三个任务模型参数之和
   - 对 TF-IDF 类方法，按每个任务最终分类器 `coef_ + intercept_` 的参数量累计后换算为百万
   - 对 PubMedBERT / MWS-CFE，按 3 个 task-specific checkpoint 总参数量累计
3. `Inference Time (ms/sample)`：
   - 定义为**生成单个 mention 的三任务完整输出**所需总延迟
   - 即 `density + has_size + location` 三个 task-specific 模型顺序推理后的总 per-sample latency
   - trainable 方法报告 5 seeds 的 `mean ± std`

### 主表不再出现的内容

以下内容不再进入正文主表：

1. Regex
2. researcher-reviewed subset
3. single-source 旧 silver 结果
4. 单 seed 数值

## 2.2 效率表

效率表与主表分离，避免正文主表被工程指标挤爆。

### 效率表建议为 task-granular 长表

| Method | Task | Params (M) | Train Samples | Train Time / seed (min) | Peak GPU Memory (GB) | Throughput (samples/s) | Inference Time (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|

说明：

1. `Task` 为 `density / has_size / location`
2. `Train Time / seed`、`Peak GPU Memory`、`Throughput`、`Inference Time` 对 trainable 方法统一按 5-seeds 报 `mean ± std`
3. 对 CPU-only 的 TF-IDF 方法：
   - `Peak GPU Memory = 0`
   - `Throughput` 与 `Inference Time` 仍测，但注明为 CPU 路径
4. Regex 不进入正文效率表；若需要，可在附录加一个 deterministic rule reference 表

## 2.3 researcher-reviewed subset 的处理方式

当前这批样本**不能再称 gold**。

正式写法固定为以下二选一：

1. `researcher-reviewed sanity subset`
2. `manual audit subset`

处理原则：

1. **退出正文主表与主结论**
2. **不参与模型选择**
3. **不作为公平比较的核心证据**
4. 仅在附录中给出 sanity-check 性质的小表或误差分析
5. 如果要报告，仍建议沿用 5-seeds 训练后的最终模型做评测，但只用于“方向一致性检查”，不用于“真实性能定论”

---

# 3. 正式公平比较框架

## 3.1 统一 multi-source WS 下的比较对象

新版 A2 的公平性来自“**同数据源、同 split、同 gate、同任务定义**”，不是来自“大家都叫 baseline”。

### 统一约束

所有 trainable 方法统一使用：

1. **同一套 multi-source WS 数据**：基于 Phase A1 已生成的 WS 产物
2. **同一套 subject-level split**
3. **同一默认主配置 gate**：`G2`
4. **同一任务集合**：
   - `density_category`
   - `has_size`
   - `location_lobe`
5. **同一默认输入配置**：
   - 主结果默认 `mention_text`
6. **同一评测口径**：
   - 在统一 Phase 5 test split 上出正式主表

### 4 个正式比较对象的定义

| 方法 | 训练数据 | 训练策略 | 角色 |
|---|---|---|---|
| TF-IDF + LR | multi-source WS, G2 | 传统线性分类 | 轻量级 ML baseline |
| TF-IDF + SVM | multi-source WS, G2 | 传统线性分类 | 强线性 baseline |
| Vanilla PubMedBERT | multi-source WS, G2 | plain hard-label fine-tuning | 神经 baseline |
| MWS-CFE (Ours) | multi-source WS, G2 | full MWS-CFE 训练配置 | 正式方法 |

### Vanilla PubMedBERT 与 MWS-CFE 的区分

为了避免“同一个 BERT 改个名字”，两者在 runbook 中应这样定义：

1. `Vanilla PubMedBERT`：
   - 使用**同一套 multi-source WS 聚合标签**
   - 但按 plain hard-label supervised fine-tuning 训练
   - 不承担 MWS-CFE 的完整方法性设定
2. `MWS-CFE (Ours)`：
   - 使用相同的 multi-source WS 来源
   - 但保留 MWS-CFE 的完整方法身份：multi-source LF、aggregation、quality gate、以及质量感知训练配置

这样做的意义是：

1. 保证**数据来源公平**
2. 同时仍能证明“方法配置本身”是否带来增益

## 3.2 Regex 的新定位

Regex 不再是正文公平主表的正式比较对象。

它的新定位是：

1. 历史 teacher
2. 规则参考上界/下界说明
3. 附录中的 deterministic rule reference
4. single-source 旧体系的来源说明

Regex 可以保留，但只能出现在：

1. 附录历史表
2. 方法部分对旧 pipeline 的回顾
3. supervision-source ablation 的背景说明

**不能再出现在新版正文主表里与 4 个 trainable 方法同排。**

## 3.3 single-source 旧体系的降级处理

旧 Phase 5 的 single-source 体系不删，但必须降级：

1. 旧 Phase 5 主结果：
   - 仅作“历史参考”
   - 不作为新版 A2 正式主结果
2. 旧 explicit-only vs all silver：
   - 不再作为新版正式 ablation
   - 只作旧体系局限说明
3. 旧 exam_name ablation：
   - 降级为附录背景证据
   - 不再占据新版 A2 正文资源

---

# 4. 5-seeds 实验矩阵

## 4.1 seed 方案

统一采用同一组 seeds：

`[13, 42, 87, 3407, 31415]`

要求：

1. 所有 trainable 方法都使用这同一组 seed
2. 所有任务都使用这同一组 seed
3. `random_state` 与深度学习 seed 保持一一对应

## 4.2 哪些方法必须做 5-seeds

| 方法 | 是否 5-seeds | 说明 |
|---|---|---|
| Regex | 否 | 确定性方法，不做 seed |
| TF-IDF + LR | 是 | `random_state=seed` |
| TF-IDF + SVM | 是 | `random_state=seed`；若结果稳定到 std≈0，仍按 5-seeds 记账 |
| Vanilla PubMedBERT | 是 | 标准 5-seeds |
| MWS-CFE | 是 | 标准 5-seeds |

## 4.3 主表运行矩阵

### trainable 方法总 run 数

| 方法 | 任务数 | seeds | 总 runs | 设备 |
|---|---:|---:|---:|---|
| TF-IDF + LR | 3 | 5 | 15 | CPU |
| TF-IDF + SVM | 3 | 5 | 15 | CPU |
| Vanilla PubMedBERT | 3 | 5 | 15 | GPU |
| MWS-CFE | 3 | 5 | 15 | GPU |
| **合计** | - | - | **60** | 30 CPU + 30 GPU |

## 4.4 每个任务如何展开

对每个方法都按以下三任务展开：

1. `density_category`
2. `has_size`
3. `location_lobe`

每个 task 都单独训练 5 次，然后在 test split 上汇总：

1. `accuracy`
2. 主指标（`macro_f1` 或 `f1`）
3. 效率指标

## 4.5 如何汇总成 mean ± std

对 trainable 方法，所有正式表格统一采用：

`mean ± std`

计算规则：

1. 先拿 5 个 seed 在同一 test split 上的指标
2. 计算样本均值
3. 计算样本标准差
4. 表中统一保留 2 位小数（百分比）或 3 位小数（延迟/时间）

Regex 因为不进正文主表，所以不存在“Regex 要不要补 std”的问题。

---

# 5. 消融与参数讨论重排

新版 A2 不再做“所有轴都对所有任务全展开”的爆炸式矩阵，而是按**核心正文**与**扩展补充**两层重排。

## 5.1 五组消融

### 核心原则

1. **消融只围绕正式方法线展开**
2. **优先证明弱监督升级是否成立**
3. **优先使用 5-seeds**
4. **对低信息量任务做裁剪**，避免 GPU 时间浪费在几乎必然平的曲线上

### 新版五组消融保留方式

| 组别 | 保留方式 | Backbone | 任务 | 变体 | seeds | 优先级 |
|---|---|---|---|---|---:|---|
| A1 supervision-source | 保留并升级 | Vanilla PubMedBERT | 3 任务 | single-source Regex silver vs multi-source WS(G2) | 5 | **核心** |
| A2 quality-gate | 保留并升级 | MWS-CFE | density + location 为正式；size 仅做诊断性补充 | G1 / G2 / G3 / G4 / G5 | 5 | **核心** |
| A3 aggregation | 保留并升级 | MWS-CFE | density + location 为正式；size 不单独补跑 | weighted vs uniform | 5 | **核心** |
| A4 input-scope | 保留但后置 | MWS-CFE | 3 任务 | mention_text vs full_text | 5 | 扩展 |
| A5 section-scope | 保留但后置 | MWS-CFE | density + location 为正式；size 可附录 | findings+impression vs findings_only vs impression_only | 5 | 扩展 |

### 为什么这样裁剪

1. `size` 在 Phase A1 中 LF 冲突率为 **0.00%**，因此：
   - `aggregation` 对 size 的信息量很低
   - `quality gate` 的 G1/G2/G4 与 G3/G5 也出现明显塌缩
2. `density` 和 `location` 更能反映：
   - 多源信号冲突
   - gate 过滤收益
   - aggregation 是否真在去噪

### 被移除出正式五组消融的旧内容

以下旧内容不再作为新版正式五组消融之一：

1. `explicit-only vs all silver`
2. `exam_name feature`

原因：

1. 它们属于旧 single-source pipeline 语境
2. 和新版 A2 需要回答的“multi-source WS 是否成立”不是同一个问题

## 5.2 三组参数讨论

新版参数讨论是 **A2-Core 的正式组成部分**，不是挂名存在，也不是默认后移到扩展波次。执行定义固定为：

1. `P1 max_seq_length`
2. `P2 quality-gate sensitivity`
3. `P3 section/input strategy`

其中，P2 复用 A2 结果完成成表，P1 与 P3 作为 **A2-Core 内独立执行的最小正式参数讨论** 单独补跑。

| 组别 | 处理方式 | 是否新增 GPU 运行 |
|---|---|---|
| P1 max_seq_length | 保留，聚焦 `density` 主任务，比较 `64 / 96 / 128 / 160 / 192`，其中 `128` 为默认主配置，`192` 为上界探索值 | **是** |
| P2 quality-gate sensitivity | 正式纳入 A2-Core，但执行上直接复用 A2 的 5-seeds 结果成表 | 否 |
| P3 section/input strategy | 改为 `density-only` 的独立最小正式版，比较 5 个 section/input strategy 设置，使用 5-seeds | **是** |

### P1 的 5 个值

P1 固定比较以下 5 个长度设置：

1. `64`
2. `96`
3. `128`（默认主配置，主表复用）
4. `160`
5. `192`（上界探索值）

这样做的目的不是扩张任务范围，而是满足参数讨论必须具备 5 个设置的正式要求，同时保持讨论仍然聚焦在 `density` 单任务。

### P3 的最小正式版定义

P3 不再依赖 A4/A5。当前阶段直接定义为：**仅在 `density` 任务上、使用 5-seeds、比较 5 个 section/input strategy 设置。**

正式比较的 5 个设置为：

1. `mention_text + findings_only`
2. `mention_text + impression_only`
3. `mention_text + findings+impression`
4. `full_text`
5. `mention_text + unfiltered WS`（当前仓库里最接近理想 `all_clinical_sections` 的可运行替代设置）

关于第 5 个设置，需要在文档中明确：

1. 当前仓库没有一个与理想 `all_clinical_sections` 完全同名、完全等价的现成开关
2. 当前最接近、且可直接执行的替代设置是：**不做 section 过滤，直接在现有 WS 数据上使用 `mention_text`**
3. 这个替代设置代表“保留当前 Phase A1 / A2 流水线中所有已进入 WS 构建的可用 section”，但**不等于**“对整份报告的所有临床 section 做无差别 raw context 注入”

### 参数讨论的执行顺序

A2-Core 中参数讨论的执行顺序固定为：

1. 先完成正式主表
2. 再完成核心消融 A1-A3
3. 然后完成 P1-P3
4. 最后统一成表与报告

---

# 6. 复用与重跑边界

## 6.1 可复用

以下内容可以直接复用，不需要重跑：

1. **Phase A0 方法定义与 MWS-CFE 命名**
2. **Phase A1 多源弱监督基础设施**
   - LF 实现
   - aggregation 逻辑
   - gate 定义
   - WS 数据文件
3. **split 相关资产**
   - Phase 5 split manifest
   - Phase A2 split verification
4. **LF / gate 统计**
   - coverage
   - conflict rate
   - gate retention
5. **旧单 seed 训练耗时**
   - 只用于时间预算先验
   - 不用于正式结果表
6. **Regex 历史输出**
   - 仅作 teacher reference / appendix reference

## 6.2 必须重跑

以下内容必须重跑，不能直接沿用旧结果：

1. `TF-IDF + LR` 在 multi-source WS(G2) 下的 5-seeds 三任务结果
2. `TF-IDF + SVM` 在 multi-source WS(G2) 下的 5-seeds 三任务结果
3. `Vanilla PubMedBERT` 在 multi-source WS(G2) 下的 5-seeds 三任务结果
4. `MWS-CFE` 在 multi-source WS(G2) 下的 5-seeds 三任务结果
5. 所有将进入正文的效率指标
6. 所有将进入正文的 core ablations（A1-A3）

## 6.3 降级到附录

以下结果可以保留，但只能降级到附录或补充分析：

1. 旧 Phase 5 single-source 主表
2. Regex 与 old silver 的对照
3. 旧 `explicit-only vs all silver`
4. 旧 `exam_name` 结果
5. researcher-reviewed subset 的历史结果

## 6.4 暂不执行

以下内容本阶段明确不做：

1. 模块3任何正式实现
2. 模块2任务集合扩展到 3 个正式主任务之外
3. 将 reviewed subset 再包装成 gold mainline
4. 为了“表格更满”而额外补的大规模无效 sweep

---

# 7. 资源配置与 aggressive search

## 7.1 原则

新版 A2 不再默认 `train_batch_size=32`，但也不做不现实的极端配置。

唯一合理做法是：

1. **先做短时 batch preflight**
2. 锁定 `mention_text` 与 `full_text` 两套统一配置
3. 再启动 5-seeds 正式矩阵

## 7.2 是否建议先做 batch search 再跑全矩阵

**是，必须先做。**

推荐做法：

1. 只用 `seed=42`
2. 每个候选 batch 只跑 **1 epoch profiling**
3. 记录：
   - peak GPU memory
   - step time
   - samples/s
   - dataloader 是否饿死 GPU
4. 锁定最终 batch 后，主表和后续消融全部沿用

## 7.3 mention_text 的 batch 搜索建议

### 搜索梯度

优先按下面顺序递增搜索：

`64 -> 96 -> 128 -> 160`

### 推荐起始候选

| 任务规模 | 起始 batch | 备选 batch | grad_accum |
|---|---:|---:|---:|
| density G2 / G1 / G4 | 128 | 96 / 160 | 1 |
| size G2 / G1 / G4 | 96 | 64 / 128 | 1 |
| location G2 / G1 / G4 | 96 | 64 / 128 | 1 |
| 小数据量 runs（如 G3 / G5） | 128 | 160 | 1 |

### 落地建议

1. 如果 `128` 下 peak memory 仍明显低于 20GB，density 可继续尝试 `160`
2. 对 size / location，不建议把 `160` 作为默认正式配置；它更适合作为搜索上界
3. 如果出现不稳定或吞吐不升反降，直接回退到 `96`

## 7.4 full_text 的 batch 搜索建议

### 搜索梯度

优先按下面顺序递增搜索：

`32 -> 48 -> 64`

### 推荐起始候选

| 任务 | 起始 batch | 备选 batch | grad_accum |
|---|---:|---:|---:|
| density full_text | 64 | 48 / 32 | 1 |
| has_size full_text | 48 | 32 / 64 | 1 |
| location full_text | 48 | 32 / 64 | 1 |

### 落地建议

1. full_text 只在 A4 中使用，不应反过来拖慢主表
2. 若 `64` 的吞吐优势不明显，优先锁 `48`
3. 只有在 peak memory 和 step time 同时友好时，才使用 `64`

## 7.5 workers / pin_memory / persistent_workers / prefetch_factor

建议按规模分层：

| 场景 | num_workers | pin_memory | persistent_workers | prefetch_factor |
|---|---:|---|---|---:|
| mention_text 大规模正式 runs | 8 | true | true | 4 |
| full_text 正式 runs | 6 | true | true | 2 |
| 小规模 runs（G3/G5 或 debug） | 4 | true | true | 2 |

补充说明：

1. 当前训练脚本已经支持：
   - `dataloader_pin_memory=True`
   - `dataloader_persistent_workers=True`
   - `dataloader_prefetch_factor`
2. 64 核 CPU 足以支撑 `8` 个 worker，但不建议盲目拉到 `16+`
3. 如果 profiling 发现 dataloader 仍是瓶颈，再从 `8` 提到 `10`；否则维持 `8`

## 7.6 并行策略

### GPU 侧

新版 A2 默认采用：

**一次只跑 1 个 transformer 训练任务**

原因：

1. aggressive batch 的目标是提升单 run 吞吐，不是堆双开
2. 单卡 3090 下，双开虽然理论可行，但更容易：
   - 触发 OOM
   - 干扰 batch search 结果
   - 拉长单 run wall-clock

### CPU 侧

以下工作可以和 GPU 训练并行：

1. TF-IDF + LR / SVM 训练
2. 表格汇总脚本
3. uniform WS / section-filtered WS 预处理

### 结论

**GPU 串行，CPU 并行**，这是新版 A2 在单张 3090 上最稳妥的执行策略。

---

# 8. 新的时间预估

以下预算建立在两个前提上：

1. 先完成 1 轮 aggressive batch preflight
2. 正式运行时采用锁定后的统一 batch 配置，而不是继续边跑边试

## 8.1 正式主表

### 预热与锁配置

| 项目 | 预算 |
|---|---:|
| batch preflight（mention_text + full_text） | 1.0-1.5h |
| CPU baselines 5-seeds 全部跑完 | 0.3-0.5h |

### GPU 主表

按单 seed 估计：

| 方法 | density / seed | has_size / seed | location / seed | 三任务合计 / seed |
|---|---:|---:|---:|---:|
| Vanilla PubMedBERT | 8-10min | 18-22min | 14-18min | 40-50min |
| MWS-CFE | 9-12min | 20-24min | 16-20min | 45-56min |

因此主表总预算为：

| 项目 | 预算 |
|---|---:|
| Vanilla PubMedBERT（15 GPU runs） | 3.5-4.5h |
| MWS-CFE（15 GPU runs） | 4.0-5.0h |
| 主表效率统计与整理 | 0.5-1.0h |
| **正式主表总计** | **9.3-12.5h** |

## 8.2 消融

### Core ablations（建议进入正文）

| 组别 | 额外 GPU 预算 | 说明 |
|---|---:|---|
| A1 supervision-source | 5-6h | 只需补跑 single-source 旧数据上的 5-seed Vanilla PubMedBERT；multi-source 结果可复用主表 |
| A2 quality-gate | 9-11h | 以 density + location 为正式主轴；G2 可复用主表 |
| A3 aggregation | 2.5-4h | 以 density + location 为主；weighted 主结果复用主表 |
| **Core ablations 小计** | **16.5-21h** | - |

### Extended ablations（主表锁定后再做）

| 组别 | 额外 GPU 预算 | 说明 |
|---|---:|---|
| A4 input-scope | 5-6h | full_text 仅作为补充分析，不先于主表 |
| A5 section-scope | 8-10h | 以 density + location 为主，size 可降级附录 |
| **Extended ablations 小计** | **13-16h** | - |

## 8.3 参数讨论

| 组别 | 额外 GPU 预算 | 说明 |
|---|---:|---|
| P1 max_seq_length | 3.5-5h | 以 density 为主，`64 / 96 / 128 / 160 / 192`，其中 `128` 复用主表，`192` 为新增上界探索值 |
| P2 quality-gate sensitivity | 0h | 直接复用 A2，作为 A2-Core 内正式参数讨论成表 |
| P3 section/input strategy | 4-5h | `density-only`，5 个 section/input strategy 设置，5-seeds，独立于 A4/A5 执行 |
| 成表与报告 | 1-1.5h | 汇总主表、A1-A3、P1-P3，并完成 A2-Core 的正式报告输出 |
| **参数讨论与报告小计** | **8.5-11.5h** | - |

## 8.4 总工期

### 若只做到可进入正文主线的 A2-Core

`正式主表 + A1 + A2 + A3 + P1 + P2 + P3 + 成表与报告`

预算：

**34.3-45.0h**

即：

1. 连续运行约 **2-3 天**
2. 按白天监控、夜间续跑的现实节奏，约 **4-5 个日历日**

### 若完成完整版 A2-Full

`A2-Core + A4 + A5`

预算：

**47.3-61.0h**

即：

1. 连续运行约 **3-4 天**
2. 按现实监控节奏，约 **6-7 个日历日**

### 明确结论

在单张 3090 上：

1. **新版 A2 主表不是做不到，而是必须先裁剪优先级**
2. **A2-Core 已显式包含正式主表、A1-A3、P1-P3 与成表报告，可以直接执行**
3. **A2-Full 也可执行，但它只是在 A2-Core 之上补 A4/A5，不应在主表未锁前全部同时展开**

---

# 9. 最终结论

## 9.1 新版 A2 的正式主表到底怎么长

正式主表固定为：

| Method | Params (M) | Density Acc | Density Macro-F1 | Has_size Acc | Has_size F1 | Location Acc | Location Macro-F1 | Inference Time (ms/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

只保留 4 个正式行：

1. `TF-IDF + LR`
2. `TF-IDF + SVM`
3. `Vanilla PubMedBERT`
4. `MWS-CFE (Ours)`

全部按 **5-seeds, mean ± std** 汇总。

## 9.2 哪些模型进入正式公平比较

进入新版正式公平比较的只有：

1. `TF-IDF + LR`
2. `TF-IDF + SVM`
3. `Vanilla PubMedBERT`
4. `MWS-CFE (Ours)`

统一前提是：

1. 同一 multi-source WS 数据
2. 同一 split
3. 同一默认 gate（G2）
4. 同一任务定义

Regex 不进入主表，只做附录 reference。

## 9.3 5-seeds 怎么做

统一 seeds：

`[13, 42, 87, 3407, 31415]`

执行方式：

1. LR / SVM：每任务 5 个 `random_state`
2. Vanilla PubMedBERT：每任务 5 个 seed
3. MWS-CFE：每任务 5 个 seed
4. 主表总计 60 个 train runs，其中 GPU runs 为 30 个

## 9.4 researcher-reviewed subset 怎么处理

它不再叫 gold。

正式处理方式：

1. 改名为 `researcher-reviewed sanity subset`
2. 退出正文主表
3. 退出主结论
4. 仅放附录做 sanity-check 或误差分析

## 9.5 现在下一步最先做什么

**第一步不是直接开跑 5-seeds，而是先做 1 轮 aggressive batch preflight。**

具体就是：

1. 锁 `mention_text` 的 batch（从 `64 -> 96 -> 128 -> 160` 搜）
2. 锁 `full_text` 的 batch（从 `32 -> 48 -> 64` 搜）
3. 锁 `num_workers / pin_memory / persistent_workers / prefetch_factor`
4. 然后按 **A2-Core 正式顺序** 启动：正式主表 -> A1-A3 -> P1-P3 -> 成表与报告

这一步完成后，新版 A2 就可以进入真正执行阶段。
