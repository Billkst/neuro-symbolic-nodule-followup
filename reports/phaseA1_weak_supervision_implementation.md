# Phase A1：弱监督基础设施实现报告

> 生成日期：2026-04-10
> 状态：**已完成**
> 前置文件：`reports/phaseA0_module2_weak_supervision_redesign.md`

---

## 1. 实现文件清单

### 核心代码 (`src/weak_supervision/`)

| 文件 | 行数 | 功能 |
|------|------|------|
| `base.py` | 72 | 基础类型定义：`LFOutput`, `AggregatedLabel`, `GateResult`, `ABSTAIN` |
| `aggregation.py` | 53 | 加权多数投票聚合器 `weighted_majority_vote()` |
| `quality_gate.py` | 55 | Quality Gate G1-G5 定义 + `evaluate_gate()` + `filter_by_gate()` |
| `labeling_functions/__init__.py` | 14 | LF 注册表，导出 `ALL_LFS`, `DENSITY_LFS`, `SIZE_LFS`, `LOCATION_LFS` |
| `labeling_functions/density_lfs.py` | 264 | Density 5 个 LF：D1-D5 |
| `labeling_functions/size_lfs.py` | 180 | Size 5 个 LF：S1-S5 |
| `labeling_functions/location_lfs.py` | 200 | Location 5 个 LF：L1-L5 |

### 脚本 (`scripts/phaseA1/`)

| 文件 | 功能 |
|------|------|
| `build_ws_datasets.py` | 端到端 pipeline：读取 Phase 5 数据 → 运行 LF → 聚合 → Gate → 导出 + 统计 |

### 测试 (`tests/weak_supervision/`)

| 文件 | 测试数 | 覆盖范围 |
|------|--------|---------|
| `test_labeling_functions.py` | 45 | 15 个 LF 的正例/负例/边界/空输入 |
| `test_aggregation_gate.py` | 17 | 聚合器 + Gate 的全路径覆盖 |

**总计 62 个测试，全部通过。**

---

## 2. 三类任务的 Labeling Functions

### 2.1 Density (LF-D1 ~ LF-D5)

| LF | 名称 | 信号源 | 覆盖率 | 标签分布 |
|----|------|--------|--------|---------|
| LF-D1 | Keyword-Exact | 精确关键词（part_solid > ground_glass > calcified > solid） | 12.71% | GG 13090, Cal 7090, Sol 4899, PS 583 |
| LF-D2 | Keyword-Fuzzy | 非标准同义词（subsolid, hazy, frosted, attenuating） | 0.63% | GG 553, PS 458, Sol 240, Cal 22 |
| LF-D3 | Negation-Aware | 否定词检测 → 输出 unclear | 0.89% | unclear 1800 |
| LF-D4 | Multi-Density | 多密度共现 → 输出 unclear | 0.77% | unclear 1552 |
| LF-D5 | Impression-Cue | IMPRESSION 段落交叉线索 | 9.59% | GG 11258, Sol 4338, Cal 2974, PS 800 |

**冲突率**：2.04%（4,120 条记录中至少 2 个 LF 给出不同标签）

### 2.2 Size (LF-S1 ~ LF-S5)

| LF | 名称 | 信号源 | 覆盖率 | 标签分布 |
|----|------|--------|--------|---------|
| LF-S1 | Regex-Standard | 标准尺寸模式（N mm, NxN mm, N cm） | 31.75% | true 64,128 |
| LF-S2 | Regex-Tolerant | 容错模式（7mm, 4mmm, measuring N） | 5.09% | true 10,282 |
| LF-S3 | Numeric-Context | 数字 + 尺寸上下文词 | 13.47% | true 27,199 |
| LF-S4 | Subcentimeter-Cue | 定性尺寸词（subcentimeter, tiny, large） | 4.67% | true 9,432 |
| LF-S5 | No-Size-Negative | 无数字 + 无尺寸词 → false | 45.98% | false 92,865 |

**冲突率**：0.00%（size 是二分类，LF 之间高度一致）

### 2.3 Location (LF-L1 ~ LF-L5)

| LF | 名称 | 信号源 | 覆盖率 | 标签分布 |
|----|------|--------|--------|---------|
| LF-L1 | Lobe-Exact | 精确叶位匹配 | 38.25% | RUL 18894, RLL 14895, LUL 11544, LLL 12361, RML 7705, bilateral 4957, lingula 1592, unclear 4883 |
| LF-L2 | Multi-Lobe | 多叶位检测 → bilateral | 2.88% | bilateral 4614, 其他 1200 |
| LF-L3 | Bilateral-Keyword | 双肺关键词 | 3.87% | bilateral 7818 |
| LF-L4 | Laterality-Inference | 侧别推断 → unclear | 2.96% | unclear 5978 |
| LF-L5 | Context-Window | 上下文窗口位置继承 | 28.58% | RUL 14K, RLL 10K, LUL 8K, LLL 7K, ... |

**冲突率**：2.75%（5,554 条记录）

---

## 3. 聚合方案

采用 A0 锁定的**加权多数投票**（Weighted Majority Vote）。

权重基于 A0 计划的 fallback 方案（Gold 集太小，使用预设权重）：
- 精确匹配类（D1/S1/L1）：权重 1.0
- 扩展/否定类（D2/D3/S2/L2/L3）：权重 0.75-0.9
- 上下文/推断类（D5/S3/S4/L4/L5）：权重 0.7-0.85

聚合逻辑：
1. 过滤 ABSTAIN 的 LF
2. 按 label 分组，计算加权得分
3. 选得分最高的 label 为聚合标签
4. 置信度 = winner_score / total_score
5. 全部 ABSTAIN → 输出 ABSTAIN（不参与训练）

---

## 4. Quality Gate

| Gate | 条件 | Density 保留量 | Size 保留量 | Location 保留量 |
|------|------|--------------|------------|----------------|
| G1 (No Gate) | 所有非 ABSTAIN | 40,023 (19.8%) | 168,433 (83.4%) | 136,687 (67.7%) |
| G2 (Conf ≥ 0.7) | 聚合置信度 ≥ 0.7 | 36,038 (17.8%) | 168,433 (83.4%) | 131,233 (65.0%) |
| G3 (Cov ≥ 2) | 至少 2 个 LF 非 ABSTAIN | 8,734 (4.3%) | 26,133 (12.9%) | 17,456 (8.6%) |
| G4 (Agr ≥ 0.8) | LF 一致性 ≥ 0.8 | 35,907 (17.8%) | 168,433 (83.4%) | 131,130 (64.9%) |
| G5 (Strict) | Conf ≥ 0.8 AND Cov ≥ 2 | 4,618 (2.3%) | 26,133 (12.9%) | 11,899 (5.9%) |

---

## 5. 新产物输出路径

```
outputs/phaseA1/
├── ws_summary.json                    # 全局统计摘要
├── density/
│   ├── ws_train.jsonl                 # 完整 WS 记录（含 LF 详情）
│   ├── ws_val.jsonl
│   ├── ws_test.jsonl
│   ├── density_train_ws.jsonl         # 训练用（非 ABSTAIN，兼容 Phase 5 格式）
│   ├── density_val_ws.jsonl
│   ├── density_test_ws.jsonl
│   ├── density_train_ws_g1.jsonl      # G1 gate 筛选后
│   ├── density_train_ws_g2.jsonl
│   ├── density_train_ws_g3.jsonl
│   ├── density_train_ws_g4.jsonl
│   ├── density_train_ws_g5.jsonl
│   └── lf_stats.json                 # LF 统计
├── size/
│   ├── (同上结构)
│   └── lf_stats.json
└── location/
    ├── (同上结构)
    └── lf_stats.json
```

每条训练记录包含：`sample_id`, `note_id`, `subject_id`, `mention_text`, `full_text`, `ws_confidence`, `lf_coverage`, `gate_level`, `passed_gates` + 任务标签字段。与 Phase 5 `LazyMentionDataset` 兼容。

---

## 6. LF 统计关键发现

### 6.1 多源性验证

LF 之间确实形成了多源弱监督，而非换皮单 teacher：

1. **Density**：D1（精确匹配）和 D5（IMPRESSION 交叉线索）的 pairwise overlap 仅 3.33%，agreement rate 74.43%——说明 D5 提供了独立于 D1 的信号，且两者在重叠区域有 25.57% 的分歧
2. **Density**：D3（否定检测）和 D4（多密度检测）覆盖了 D1 无法处理的错误模式，直接对应 Gold 评测中的两大错误源
3. **Size**：冲突率为 0%，因为 size 是二分类且 LF 信号高度一致。但 S5（负信号）覆盖了 45.98% 的样本，与 S1（31.75%）形成互补
4. **Location**：L5（上下文窗口）覆盖 28.58%，与 L1（38.25%）的 overlap 仅 4.39%，agreement rate 87.72%——上下文窗口确实提供了额外的位置信息

### 6.2 覆盖率分析

| 任务 | 联合覆盖率（非 ABSTAIN） | 旧 Regex 单源覆盖率 | 增量 |
|------|------------------------|-------------------|------|
| Density | 19.8% (40,023/201,947) | 12.7% (D1 alone) | +7.1pp |
| Size | 83.4% (168,433/201,947) | 31.8% (S1 alone) | +51.6pp |
| Location | 67.7% (136,687/201,947) | 38.3% (L1 alone) | +29.4pp |

### 6.3 Quality Gate 效果

- G5（最严格）在 density 上仅保留 4,618 条（2.3%），但这些是多 LF 高置信度一致的样本
- Size 任务在 G1-G4 之间几乎无差异（168K），因为 LF 冲突率为 0
- Location 的 G3（Cov ≥ 2）大幅缩减到 17K（8.6%），说明大部分 location 样本只有 1 个 LF 覆盖

### 6.4 Density 的 ABSTAIN 率偏高

Density 有 80.2% 的样本被标记为 ABSTAIN（无 LF 能判断密度）。这与现有数据一致——原始 Regex 也只能提取 12.7% 的密度标签。这不是 bug，而是反映了大量 mention 确实不包含密度信息。

---

## 7. 当前局限

1. **LF-D3 覆盖率偏低**（0.89%）：否定模式在放射学报告中相对少见，但在 Gold 评测的错误案例中占比很高。D3 的价值更多体现在 Gold 集上的精度提升，而非大规模覆盖
2. **LF 权重为预设值**：A0 计划建议在 Gold 集上校准权重，但 Gold 集仅 62 条，统计量不足。当前使用 fallback 预设权重（精确类 1.0 / 扩展类 0.75-0.9 / 推断类 0.7-0.85）
3. **Size 任务 LF 冲突率为 0**：说明 size 的 5 个 LF 之间没有真正的分歧信号。这在消融实验中需要讨论——size 任务的多源弱监督增益可能主要来自覆盖率提升而非标签去噪
4. **Density ABSTAIN 率 80%**：大量 mention 不含密度信息，导致可训练样本仅 40K。A2 阶段需要评估这是否足够

---

## 8. 是否可以进入 Phase A2

**可以。** 所有 A1 前置条件已满足：

| 条件 | 状态 |
|------|------|
| 3 个主任务的 LF 已实现 | ✅ 15 个 LF |
| 聚合器已实现并可运行 | ✅ 加权多数投票 |
| Quality Gate 已实现并可运行 | ✅ G1-G5 |
| 已导出新的 weak supervision 数据产物 | ✅ 3 任务 × 3 splits + gate 筛选 |
| 已输出 LF 统计 | ✅ coverage/conflict/overlap/gate |
| 已补测试并通过 | ✅ 62 tests |
| 已完成 smoke test | ✅ 小样本 + 全量 |
| 已写实现报告 | ✅ 本文件 |

**Phase A2 可以立即启动。** 建议优先级：
1. 训练 Vanilla PubMedBERT（使用旧 Phase 5 数据，无依赖）
2. 训练 MWS-CFE（使用 `density_train_ws.jsonl` / `size_train_ws.jsonl` / `location_train_ws.jsonl`）
3. Gold 评测对比
4. 消融实验（A-lf, A-agg, A-qg 为核心新增消融）
