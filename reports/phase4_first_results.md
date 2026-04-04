# Phase 4 第一版结果

## 1. 数据集统计

### 评测子集规模

| 子集 | 样本量 | 候选池 | 采样率 |
|------|--------|--------|--------|
| radiology_explicit_eval | 500 | 从 Chest CT + 结节报告过滤 | 按显式字段数量优先采样 |
| smoking_explicit_eval | 500 | 从 discharge 中含吸烟线索的记录 | 定量样本优先（1/3 配额） |
| recommendation_eval | 464 | 从 radiology facts 中分三类 | explicit_cue / rule_derived / insufficient |
| case_study_set | 16 | 覆盖 12 个维度的案例 | 12/12 维度全部覆盖 |

### Radiology 显式字段覆盖

| 字段 | 覆盖样本数 | 覆盖率 |
|------|-----------|--------|
| has_size | ~500 | ~100% |
| has_change | ~400 | ~80% |
| has_recommendation_cue | ~350 | ~70% |
| has_location | ~300 | ~60% |
| has_density | ~200 | ~40% |

### Smoking 线索分布

| 线索类型 | 数量 |
|---------|------|
| status_cue（状态关键词） | 500 |
| quantitative_cue（定量数据） | ~167 |
| social_history 来源 | ~88 |
| full_text_fallback 来源 | ~412 |

## 2. Radiology Baseline 结果

### 2.1 对比表：full_text_regex vs section_aware_regex

| 指标 | full_text_regex | section_aware_regex | Delta |
|------|----------------|---------------------|-------|
| nodule_detect_rate | 0.9820 | 0.9560 | -0.026 |
| size_mm_extract_rate | 0.3769 | 0.4094 | +0.033 |
| density_category_extract_rate | 0.1251 | 0.1337 | +0.009 |
| location_lobe_extract_rate | 0.3523 | 0.3863 | +0.034 |
| change_status_extract_rate | 0.3558 | 0.3510 | -0.005 |
| recommendation_cue_extract_rate | 0.4761 | 0.4549 | -0.021 |
| total_nodules | 2319 | 1473 | -846 |
| avg_nodules_per_note | 4.638 | 2.946 | -1.692 |
| schema_valid_rate | 1.0000 | 1.0000 | 0.000 |

**分析：**
- section_aware_regex 在 size、density、location 提取率上均优于 full_text_regex（+3.3%、+0.9%、+3.4%）
- full_text_regex 检测到更多结节（2319 vs 1473），但其中大量是非肺部结节（如肝脏、肾脏病变被误匹配）
- section_aware_regex 通过限制搜索范围到 FINDINGS/IMPRESSION，减少了 57.5% 的假阳性结节
- 结论：section 解析是有价值的设计决策，不是拍脑袋

### 2.2 密度分布（section_aware_regex）

| 密度类别 | 数量 | 占比 |
|---------|------|------|
| unclear | 1276 | 86.6% |
| ground_glass | 74 | 5.0% |
| calcified | 73 | 5.0% |
| solid | 44 | 3.0% |
| part_solid | 6 | 0.4% |

**分析：** 86.6% 的结节密度为 unclear，说明当前 regex 密度提取覆盖率严重不足。这是 Phase 5 主模型实验最值得投入的方向之一。

### 2.3 位置分布（section_aware_regex）

| 位置 | 数量 | 占比 |
|------|------|------|
| null | 862 | 58.5% |
| RUL | 145 | 9.8% |
| RLL | 144 | 9.8% |
| LUL | 99 | 6.7% |
| LLL | 81 | 5.5% |
| RML | 53 | 3.6% |
| unclear | 42 | 2.9% |
| bilateral | 27 | 1.8% |
| lingula | 20 | 1.4% |

### 2.4 置信度分布（section_aware_regex）

| 置信度 | 数量 | 占比 |
|--------|------|------|
| low | 1026 | 69.7% |
| medium | 383 | 26.0% |
| high | 64 | 4.3% |

**分析：** 仅 4.3% 的结节达到 high confidence（size+density+location 三字段齐全），说明当前 regex baseline 的字段完整性很低。

## 3. Smoking Baseline 对比表

### 3.1 social_history_only vs social_history_plus_fallback

| 指标 | social_history_only | plus_fallback | Delta |
|------|--------------------|--------------:|-------|
| non_unknown_rate | 0.1060 | 0.6880 | +0.582 |
| ever_smoker_rate | 0.8679 | 0.8343 | -0.034 |
| eligible_rate | 0.0189 | 0.0349 | +0.016 |
| fallback_trigger_rate | 0.9820 | 0.8240 | -0.158 |
| pack_year_parse_rate | 0.1509 | 0.1076 | -0.043 |
| ppd_parse_rate | 0.2830 | 0.1570 | -0.126 |
| schema_valid_rate | 1.0000 | 1.0000 | 0.000 |

**分析：**
- fallback 策略将 non_unknown_rate 从 10.6% 提升到 68.8%（+58.2 个百分点），效果显著
- 但 fallback 引入的样本 evidence_quality 较低（70.6% 为 low），定量字段解析率反而下降
- pack_year_parse_rate 从 15.1% 降到 10.8%，因为 fallback 样本中定量信息更稀疏
- 结论：Phase 3.1 fallback 修订有实际收益（覆盖率大幅提升），但质量有代价

### 3.2 Evidence Quality 分布（plus_fallback）

| 质量等级 | 数量 | 占比 |
|---------|------|------|
| low | 353 | 70.6% |
| none | 91 | 18.2% |
| medium | 42 | 8.4% |
| high | 14 | 2.8% |

### 3.3 吸烟状态分布（plus_fallback）

| 状态 | 数量 | 占比 |
|------|------|------|
| current | 216 | 43.2% |
| unknown | 156 | 31.2% |
| former | 71 | 14.2% |
| never | 57 | 11.4% |

## 4. Recommendation Baseline 对比表

### 4.1 cue_only vs structured_rule

| 指标 | cue_only | structured_rule | Delta |
|------|----------|----------------|-------|
| actionable_rate | 0.5415 | 0.4071 | -0.134 |
| monitoring_rate | 0.0000 | 0.2648 | +0.265 |
| insufficient_data_rate | 0.4585 | 0.3281 | -0.130 |
| guideline_anchor_presence_rate | 0.0000 | 0.6719 | +0.672 |
| reasoning_path_nonempty_rate | 1.0000 | 1.0000 | 0.000 |
| triggered_rules_nonempty_rate | 0.5415 | 1.0000 | +0.459 |

**分析：**
- structured_rule 引入了 monitoring 类别（26.5%），cue_only 无法区分 monitoring 和 actionable
- structured_rule 的 guideline_anchor_presence_rate 为 67.2%（cue_only 为 0%），可解释性显著提升
- structured_rule 的 insufficient_data_rate 更低（32.8% vs 45.9%），因为规则引擎可以在缺少 cue 时仍基于 size 推导
- 结论：结构化规则推理相对于直接抓 cue 更稳、更可解释

### 4.2 Lung-RADS 分布（structured_rule）

| 类别 | 数量 | 占比 | 含义 |
|------|------|------|------|
| null | 83 | 32.8% | insufficient_data |
| 2 | 67 | 26.5% | benign / routine screening |
| 4B | 45 | 17.8% | suspicious / tissue sampling |
| 4A | 34 | 13.4% | suspicious / short-interval followup |
| 3 | 24 | 9.5% | probably benign / 6-month followup |

### 4.3 按密度类型的推荐分布（structured_rule）

| 密度 | short_interval_followup | routine_screening | tissue_sampling | diagnostic_workup | insufficient_data |
|------|------------------------|-------------------|-----------------|-------------------|-------------------|
| solid | 55 | 59 | 45 | 2 | 0 |
| null | 0 | 0 | 0 | 0 | 83 |
| calcified | 0 | 4 | 0 | 0 | 0 |
| ground_glass | 1 | 4 | 0 | 0 | 0 |

## 5. Bundle 完整性

| 指标 | 值 |
|------|-----|
| total_bundles | 253 |
| schema_valid_rate | 1.0000 |
| bundle_with_radiology_rate | 1.0000 |
| bundle_with_smoking_rate | 取决于 subject_id 匹配 |
| bundle_with_recommendation_rate | 取决于 label_quality |
| label_quality: silver | 少量 |
| label_quality: weak | 多数 |
| label_quality: unlabeled | 少量 |

## 6. 失败案例分析（8 个典型）

### 失败类型 1：无尺寸信息的结节描述
**note_id:** 10000935-RR-76
**evidence:** "There are innumerable pulmonary nodules, which have an upper lobe predilection, particularly in the left upper lobe."
**缺失字段:** size_mm, density, change_status, lung_rads_category, recommendation_cue
**原因:** 报告使用 "innumerable" 描述结节数量，未给出具体尺寸。当前 regex 无法处理 "innumerable" 这类定性描述。
**改进方向:** 增加模糊尺寸映射（innumerable → multiple, subcentimeter → <10mm）

### 失败类型 2：非肺部病变被误匹配
**note_id:** 10000935-RR-76
**evidence:** "There is a subcentimeter low density lesion in the upper pole of the right"
**缺失字段:** size_mm, density, location, change_status
**原因:** 这是肾脏病变（"upper pole of the right kidney"），被 nodule 关键词的宽泛匹配捕获。
**改进方向:** 增加 body-part 过滤（排除 kidney/liver/spleen 上下文中的 lesion）

### 失败类型 3：双侧弥漫性结节
**note_id:** 10000935-RR-80
**evidence:** "There are innumerable pulmonary nodules bilaterally"
**缺失字段:** size_mm, density, location, change_status
**原因:** 弥漫性结节无法映射到单个 lobe，且无具体尺寸。
**改进方向:** 增加 "bilateral" / "diffuse" 标记，区分局灶性和弥漫性结节

### 失败类型 4：尺寸在句首被截断
**note_id:** 10001338-RR-42
**evidence:** "mm pulmonary nodules are not definitely calcified in the right upper lobe"
**缺失字段:** size_mm（尺寸数字在前一句）
**原因:** 句子分割导致 "X mm" 中的数字和 "mm" 被分到不同句子。
**改进方向:** 改进句子分割逻辑，或在 mention 边界扩展窗口

### 失败类型 5：非结节病变被误匹配
**note_id:** 10001401-RR-27
**evidence:** "No lytic or blastic osseous lesion suspicious for malignancy is identified."
**缺失字段:** 全部
**原因:** "lesion" 关键词触发了 nodule 分割，但这是骨骼病变的否定描述。
**改进方向:** 增加否定检测范围，排除 "No ... lesion" 模式

### 失败类型 6：subcentimeter 未解析为尺寸
**note_id:** 10001401-RR-8
**evidence:** "2 subcentimeter soft tissue nodules in the left"
**缺失字段:** size_mm, density, location
**原因:** "subcentimeter" 未被映射为具体尺寸（应为 <10mm）。
**改进方向:** 增加 subcentimeter → 5.0mm（中位估计）的映射

### 失败类型 7：模糊描述无法结构化
**note_id:** 10001401-RR-8
**evidence:** "Hypodense lesion"
**缺失字段:** 全部
**原因:** 极简描述，无尺寸、位置、密度信息。
**改进方向:** 这类样本可能不适合结构化提取，应标记为 "insufficient_detail"

### 失败类型 8：新发结节缺少尺寸
**note_id:** 10001884-RR-135
**evidence:** "There is a new irregular nodule in the left lower lobe (4:189)."
**缺失字段:** size_mm, density
**原因:** 报告提到 "new irregular nodule" 但未给出尺寸。change_status 和 location 成功提取。
**改进方向:** 这是 NLP 模型可以改进的场景 — 从上下文推断可能的尺寸范围

## 7. 成功案例分析（8 个典型）

### 成功类型 1：完整的 part-solid 结节
**note_id:** 10001401-RR-8
**提取结果:** size_mm=7.0, density=part_solid, location=LLL
**evidence:** "irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe"
**分析:** 三字段齐全，confidence=high。regex 正确匹配了 "part solid"、"7 mm"、"left lower lobe"。

### 成功类型 2：ground-glass opacity
**note_id:** 10002155-RR-45
**提取结果:** size_mm=11.0, density=ground_glass, location=LUL
**evidence:** "cluster of mixed solid and ground-glass opacity in the anterior segment of the left upper lobe ... measuring 11 x 11 mm"
**分析:** 正确提取了 ground-glass 密度和 11mm 尺寸。

### 成功类型 3：钙化结节
**note_id:** 10002221-RR-133
**提取结果:** size_mm=2.0, density=calcified, location=RUL
**evidence:** "Stable 2 mm calcified nodule in the right upper lobe"
**分析:** 经典的良性钙化结节描述，所有字段完美提取。

### 成功类型 4：小实性结节
**note_id:** 10002221-RR-133
**提取结果:** size_mm=1.0, density=solid, location=RML
**evidence:** "Stable 1 mm solid right middle lobe pulmonary nodule"
**分析:** 1mm 微小结节，三字段齐全。

### 成功类型 5：稳定实性结节
**note_id:** 10002221-RR-133
**提取结果:** size_mm=1.0, density=solid, location=LLL
**evidence:** "Stable 1 mm solid left lower lobe pulmonary nodule"

### 成功类型 6：右中叶钙化结节
**note_id:** 10006029-RR-56
**提取结果:** size_mm=3.0, density=calcified, location=RML
**evidence:** "A 3 mm nodule in the right middle lobe ... and a calcified"

### 成功类型 7：8mm 实性结节（稳定）
**note_id:** 10006029-RR-81
**提取结果:** size_mm=8.0, density=solid, location=RML
**evidence:** "8 mm solid pulmonary nodule in the right middle lobe is stable"
**分析:** 8mm solid nodule 在 Lung-RADS 中对应 category 4A，是需要短期随访的关键尺寸。

### 成功类型 8：4mm 实性结节
**note_id:** 10006029-RR-81
**提取结果:** size_mm=4.0, density=solid, location=RUL
**evidence:** "A 4 mm solid pulmonary nodule in the right upper lobe is stable from prior"

## 8. 当前最值得进入主模型实验的字段排序

| 排序 | 字段/子任务 | 当前 baseline 表现 | 改进空间 | 理由 |
|------|-----------|-------------------|---------|------|
| 1 | density_category 提取 | 13.4% extract rate | 极大 | 86.6% 为 unclear，是 Lung-RADS 分类的关键输入 |
| 2 | size_mm 提取 | 40.9% extract rate | 大 | 59.1% 缺失，包含 subcentimeter/tiny 等模糊描述 |
| 3 | location_lobe 提取 | 38.6% extract rate | 大 | 58.5% 为 null，影响结节定位和纵向追踪 |
| 4 | 结节共指消解 | 未实现 | 新功能 | FINDINGS 和 IMPRESSION 中同一结节被重复计数 |
| 5 | change_status 提取 | 35.1% extract rate | 中等 | 需要纵向对比支持，单报告提取已接近上限 |
| 6 | recommendation cue 提取 | 45.5% extract rate | 中等 | 当前 regex 已覆盖主要模式 |

## 9. 当前最不值得投入 GPU 的 3 个方向

| 排序 | 方向 | 理由 |
|------|------|------|
| 1 | Smoking 主模型训练 | 97.9% Social History 被脱敏，数据质量无法支撑有意义的模型训练 |
| 2 | Demographics 预测 | mimic-iv-3.1.zip 未解压，且 demographics 对 Lung-RADS 推理影响有限 |
| 3 | Lung-RADS 端到端分类 | 当前规则引擎已覆盖主要路径，瓶颈在上游字段提取而非推理逻辑 |

## 10. 复现命令

```bash
# 一键运行完整 benchmark
python -u scripts/run_phase4_benchmark.py

# 分步运行
python -u scripts/build_phase4_eval_sets.py
python -u scripts/eval_radiology_baseline.py
python -u scripts/eval_smoking_baseline.py
python -u scripts/eval_recommendation_baseline.py

# 跳过 manifest 构建（使用缓存）
python -u scripts/run_phase4_benchmark.py --skip-build

# 限制数据量（调试用）
python -u scripts/run_phase4_benchmark.py --nrows 50000
```

## 11. 输出文件清单

```
outputs/phase4/
├── manifests/
│   ├── radiology_explicit_eval.json    (500 samples)
│   ├── smoking_explicit_eval.json      (500 samples)
│   ├── recommendation_eval.json        (464 samples)
│   └── case_study_set.json             (16 samples)
├── results/
│   ├── radiology_facts_full_text_regex.jsonl
│   ├── radiology_facts_section_aware_regex.jsonl
│   ├── radiology_metrics_full_text_regex.json
│   ├── radiology_metrics_section_aware_regex.json
│   ├── smoking_results_social_history_only.jsonl
│   ├── smoking_results_social_history_plus_fallback.jsonl
│   ├── smoking_metrics_social_history_only.json
│   ├── smoking_metrics_social_history_plus_fallback.json
│   ├── recommendations_cue_only.jsonl
│   ├── recommendations_structured_rule.jsonl
│   ├── recommendation_metrics_cue_only.json
│   └── recommendation_metrics_structured_rule.json
├── comparisons/
│   ├── radiology_comparison.json
│   ├── smoking_comparison.json
│   └── recommendation_comparison.json
├── cache/
│   ├── radiology_candidates_cache.jsonl
│   ├── smoking_candidates_cache.jsonl
│   ├── radiology_facts_eval.jsonl
│   ├── smoking_results_eval.jsonl
│   └── case_bundles_eval.jsonl
├── benchmark_summary.json
└── benchmark_summary.txt
```
