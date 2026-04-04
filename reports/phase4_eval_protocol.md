# Phase 4 评测协议

## 1. 评测总览

Phase 4 建立了固定口径的评测框架，用于衡量 Phase 3 baseline 的实际表现。所有评测基于 CPU-only 的 regex baseline，不涉及任何 GPU 训练。

### 评测子集

| 子集名称 | 样本量 | 来源 | 用途 |
|---------|--------|------|------|
| radiology_explicit_eval | 500 | 胸部CT报告（含显式结节字段） | 评测 radiology extractor |
| smoking_explicit_eval | 500 | 出院记录（含显式吸烟线索） | 评测 smoking extractor |
| recommendation_eval | 464 | 混合：显式cue + 规则推导 + insufficient | 评测 recommendation baseline |
| case_study_set | 16 | 覆盖所有结节类型的案例 | 论文案例分析 |

### 对比实验

| 对比组 | Baseline | Variant | 目的 |
|--------|----------|---------|------|
| Radiology ablation | full_text_regex | section_aware_regex | 证明 section 解析的价值 |
| Smoking comparison | social_history_only | social_history_plus_fallback | 证明 Phase 3.1 fallback 修订的收益 |
| Recommendation comparison | cue_only | structured_rule | 证明结构化规则推理优于直接抓 cue |

## 2. 评测子集构建逻辑

### 2.1 radiology_explicit_eval

**构建条件：**
- 来源：MIMIC-IV radiology.csv.gz + radiology_detail.csv.gz
- 过滤：exam_name 匹配 Chest CT → 报告文本含非否定结节关键词
- 显式字段检测：正则匹配报告文本中是否显式出现 size（mm/cm）、density（solid/ground-glass/part-solid/calcified）、location（lobe名称）、change（stable/new/increased/decreased）、recommendation cue（recommend/follow-up/suggested）
- 入选条件：至少 1 个显式字段存在（min_explicit_fields=1）
- 采样：按显式字段数量降序排列，取 top 1500 后随机采样 500（seed=42）

**标签类型：** Silver label — 基于文本中显式出现的关键词，非人工标注。

### 2.2 smoking_explicit_eval

**构建条件：**
- 来源：MIMIC-IV discharge.csv.gz
- 过滤：文本中包含吸烟状态线索（smoker/former smoker/never smoker/quit smoking/tobacco use/pack-year/ppd）
- 区分：
  - `has_status_cue`：含状态关键词
  - `has_quantitative_cue`：含定量数据（数字+pack-year/ppd/years smoked）
  - `cue_source`：social_history（来自 Social History 段落）或 full_text_fallback（全文搜索）
- 采样：定量样本优先（占 1/3 配额），其余为状态样本（seed=42）

**标签类型：** Silver label — 基于文本中显式出现的吸烟线索。

### 2.3 recommendation_eval

**构建条件：**
- 来源：radiology_explicit_eval 的提取结果
- 三类样本：
  - `explicit_cue`：报告中有显式 recommendation cue 的样本
  - `rule_derived`：有 size_mm 但无 recommendation cue，可通过 Lung-RADS 规则推导
  - `insufficient_data`：缺少关键字段，无法推导
- 采样：三类各占 1/3 配额（seed=42）

**标签类型：**
- explicit_cue 子集：Silver label（报告中的 recommendation 文本）
- rule_derived 子集：Silver label（Lung-RADS 规则推导）
- insufficient_data 子集：无标签（用于测试系统的 graceful degradation）

### 2.4 case_study_set

**构建条件：**
- 来源：radiology_explicit_eval 的提取结果
- 覆盖目标（12 个维度）：solid、ground_glass、part_solid、multiple_nodules、size_missing、smoking_unknown、recommendation_cue_present、recommendation_cue_absent、high_confidence、low_confidence、change_present、calcified
- 每个维度至少选 1 个代表案例，剩余配额随机填充至 16 个

**标签类型：** 无标签 — 用于定性分析。

## 3. 指标体系

### 3.1 可严肃解释的指标

以下指标基于明确的计算逻辑，可直接用于论文：

| 指标 | 含义 | 可解释性 |
|------|------|---------|
| schema_valid_rate | 输出通过 JSON Schema 校验的比例 | 工程质量指标，100% 为基本要求 |
| nodule_detect_rate | 检测到至少 1 个结节的报告比例 | 系统召回能力的上界 |
| size_mm_extract_rate | 成功提取 size_mm 的结节比例 | 关键字段覆盖率 |
| density_category_extract_rate | 成功提取密度类别的结节比例 | 关键字段覆盖率 |
| location_lobe_extract_rate | 成功提取位置的结节比例 | 关键字段覆盖率 |
| non_unknown_rate (smoking) | 非 unknown 的吸烟状态比例 | fallback 策略的有效性 |
| guideline_anchor_presence_rate | 有指南锚点的推荐比例 | 规则引擎的可解释性 |

### 3.2 仅作工程参考的指标

| 指标 | 原因 |
|------|------|
| confidence_distribution | 基于启发式规则（size+density+location 三字段计数），非概率校准 |
| evidence_quality (smoking) | 基于规则判定，非人工审核 |
| explicit_*_exact_rate | 当前无 gold label，explicit subset 的"正确答案"本身也是 silver |
| recommendation_cue_extract_rate | cue 提取依赖关键词匹配，召回率未知 |

### 3.3 不能当主实验的指标

| 指标 | 原因 |
|------|------|
| smoking 全量 unknown_rate | 97.9% 的 Social History 被脱敏，unknown 不代表提取失败 |
| pack_year_parse_rate | MIMIC 中仅 ~0.5% 含 pack-year 数据，样本量不足以支撑统计结论 |
| explicit_cue_agreement_rate | cue_only baseline 的 recommendation_action 是原始文本，与结构化输出不可直接比较 |

## 4. 当前评测局限性

1. **无 gold label**：所有"正确答案"均为 silver standard（基于文本中显式出现的关键词），precision 和 recall 的绝对值不可信，只能用于对比实验
2. **结节共指消解缺失**：FINDINGS 和 IMPRESSION 中同一结节可能被重复计数，导致 nodule_count 偏高
3. **密度分类偏保守**：86.6% 的结节密度为 unclear，说明 regex 覆盖率不足
4. **纵向对比缺失**：change_status 依赖单次报告文本，无法做跨报告时序对齐
5. **demographics 全部缺失**：mimic-iv-3.1.zip 未解压，所有 demographics 字段为 null
6. **吸烟数据极稀疏**：MIMIC 脱敏导致 Social History 大面积缺失，fallback 策略只能部分弥补

## 5. 为什么 smoking 模块不能当主实验

1. **数据质量**：97.9% 的 Social History 被脱敏为 `___`，即使 fallback 也只能从全文中捕获弱信号
2. **evidence_quality 分布**：high 仅 2.8%，medium 8.4%，low 70.6%，none 18.2%
3. **定量字段极稀疏**：pack_year_parse_rate 仅 10.8%，ppd_parse_rate 仅 15.7%
4. **无法验证准确性**：没有 gold label，无法判断提取的吸烟状态是否正确
5. **定位**：smoking 模块是辅助线（为 Lung-RADS 引擎提供 risk level），不是主要贡献

## 6. 可复现性保证

| 要素 | 值 |
|------|-----|
| random_seed | 42 |
| pipeline_version | 0.2.0 |
| eval_version | phase4_v1 |
| data_version | mimic-iv-note-2.2 |
| rules_version | lung_rads_v2022_minimal_0.1 |
| extractor_version | regex_baseline_0.1 |
| manifest 生成逻辑 | src/eval/manifest_builder.py |
| manifest 文件 | outputs/phase4/manifests/*.json |
| 一键复现命令 | `python -u scripts/run_phase4_benchmark.py` |
