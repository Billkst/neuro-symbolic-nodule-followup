# Phase 3 Smoke Test 报告

> 日期：2026-04-04
> 运行环境：conda activate follow-up
> 数据：MIMIC-IV-Note 2.2 (radiology.csv.gz / radiology_detail.csv.gz / discharge.csv.gz)

---

## 1. 运行命令

### 端到端 demo（推荐）
```bash
conda run -n follow-up python -u scripts/run_phase3_demo.py
```

### 分步运行
```bash
# Step 1: 导出候选报告
conda run -n follow-up python -u scripts/export_radiology_candidates.py --limit 500

# Step 2: 运行 radiology baseline
conda run -n follow-up python -u scripts/run_radiology_baseline.py --nrows 100000 --limit 500

# Step 3: 运行 smoking baseline
conda run -n follow-up python -u scripts/run_smoking_baseline.py --nrows 50000

# Step 4: 组装 case bundles
conda run -n follow-up python -u scripts/build_case_bundles.py

# Step 5: 运行 recommendation baseline
conda run -n follow-up python -u scripts/run_recommendation_baseline.py
```

---

## 2. 实际输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase3_demo_output.json` | 端到端 demo 综合输出（含样例 + 统计） |
| `outputs/radiology_candidates.jsonl` | 候选报告（含 sections） |
| `outputs/radiology_facts.jsonl` | radiology 抽取结果 |
| `outputs/smoking_results.jsonl` | smoking 抽取结果 |
| `outputs/case_bundles.jsonl` | 组装后的 case bundles |
| `outputs/recommendations.jsonl` | 规则推理结果 |
| `logs/run_phase3_demo.log` | demo 运行日志 |

---

## 3. Demo 运行摘要

```
radiology_candidates: 50
radiology_facts_extracted: 50, schema_valid: 50, schema_invalid: 0
smoking_extracted: 116, schema_valid: 116, schema_invalid: 0
case_bundles_assembled: 25, schema_valid: 25, schema_invalid: 0
recommendations_generated: 25, schema_valid: 25, schema_invalid: 0
validation_error_count: 0
```

耗时：
| 步骤 | 耗时 |
|------|------|
| 数据加载 (100K rad + 200K detail + 50K discharge) | 10.16s |
| 过滤候选 | 0.45s |
| Radiology 抽取 (50 reports) | 0.22s |
| Smoking 抽取 (116 notes) | 0.22s |
| Bundle 组装 (25 bundles) | 0.14s |
| Recommendation 推理 (25 cases) | 0.03s |

---

## 4. Radiology Extraction 样例（10 例）

### 样例 1: 多结节 + 部分有尺寸
```
note_id: 10000935-RR-76
exam_name: CT CHEST W/CONTRAST
modality: CT
nodule_count: 3
nodules:
  #1: size=null, density=unclear, location=LUL, change=null, confidence=low
  #2: size=6.0mm, density=unclear, location=null, change=null, confidence=low
  #3: size=null, density=unclear, location=null, change=null, confidence=low
```

### 样例 2: CTA 单结节
```
note_id: 10000935-RR-80
exam_name: CTA CHEST W&W/O C&RECONS, NON-CORONARY
modality: CTA
nodule_count: 1
nodules:
  #1: size=null, density=unclear, location=null, change=null, confidence=low
```

### 样例 3: 钙化结节 + 不明结节
```
note_id: 10001338-RR-42
exam_name: CTA CHEST W&W/O C&RECONS, NON-CORONARY
modality: CTA
nodule_count: 2
nodules:
  #1: size=null, density=calcified, location=null, change=null, confidence=low
  #2: size=null, density=unclear, location=RUL, change=null, confidence=low
```

### 样例 4: 多结节 + 稳定 + 有尺寸
```
note_id: 10001401-RR-27
exam_name: CTA CHEST
modality: CTA
nodule_count: 4
nodules:
  #1: size=9.0mm, density=calcified, location=null, change=stable, confidence=medium
  #2: size=10.0mm, density=unclear, location=RML, change=null, confidence=medium
  #3: size=null, density=unclear, location=null, change=null, confidence=low
  #4: size=7.0mm, density=unclear, location=null, change=null, confidence=low
```

### 样例 5: 7 结节 + 多密度类型
```
note_id: 10001401-RR-8
exam_name: CT CHEST W/CONTRAST
modality: CT
nodule_count: 7
nodules:
  #1: size=8.0mm, density=calcified, location=null, change=decreased, confidence=medium
  #2: size=null, density=unclear, location=null, change=null, confidence=low
  #3: size=null, density=unclear, location=null, change=null, confidence=low
  #4: size=null, density=part_solid, location=null, change=null, confidence=low
  #5: size=null, density=solid, location=RUL, change=null, confidence=medium
  #6: size=7.0mm, density=part_solid, location=LLL, change=null, confidence=high
  #7: size=3.0mm, density=unclear, location=LLL, change=null, confidence=medium
```

### 样例 6: 增长 + 新发结节
```
note_id: 10001884-RR-135
exam_name: CT CHEST W/O CONTRAST
modality: CT
nodule_count: 3
nodules:
  #1: size=4.0mm, density=unclear, location=RML, change=increased, confidence=medium
  #2: size=null, density=unclear, location=LLL, change=new, confidence=low
  #3: size=null, density=unclear, location=null, change=null, confidence=low
```

### 样例 7: 稳定单结节
```
note_id: 10001884-RR-60
exam_name: CT CHEST W/O CONTRAST
modality: CT
nodule_count: 1
nodules:
  #1: size=null, density=unclear, location=RUL, change=stable, confidence=low
```

### 样例 8: 多结节 + 小尺寸
```
note_id: 10001884-RR-71
exam_name: CT CHEST W/O CONTRAST
modality: CT
nodule_count: 4
nodules:
  #1: size=null, density=unclear, location=null, change=null, confidence=low
  #2: size=null, density=unclear, location=RML, change=null, confidence=low
  #3: size=3.0mm, density=unclear, location=unclear, change=null, confidence=low
  #4: size=null, density=unclear, location=null, change=stable, confidence=low
```

### 样例 9: 多稳定 + 新发
```
note_id: 10001884-RR-77
exam_name: CTA CHEST W&W/O C&RECONS, NON-CORONARY
modality: CTA
nodule_count: 4
nodules:
  #1: size=null, density=unclear, location=RML, change=stable, confidence=low
  #2: size=null, density=unclear, location=null, change=stable, confidence=low
  #3: size=null, density=unclear, location=unclear, change=new, confidence=low
  #4: size=null, density=unclear, location=RML, change=stable, confidence=low
```

### 样例 10: 双结节
```
note_id: 10001884-RR-81
exam_name: CT CHEST W/O CONTRAST
modality: CT
nodule_count: 2
nodules:
  #1: size=null, density=unclear, location=RML, change=null, confidence=low
  #2: size=null, density=unclear, location=null, change=stable, confidence=low
```

---

## 5. Smoking Extraction 样例（5 例）

### 样例 1: 脱敏导致 unknown
```
note_id: 10000935-DS-18
subject_id: 10000935
source_section: Social History
smoking_status_norm: unknown
evidence_quality: none
missing_flags: [smoking_status_raw, pack_year_value, pack_year_text, ppd_value, ppd_text,
                years_smoked_value, years_smoked_text, quit_years_value, quit_years_text,
                evidence_span, ever_smoker_flag]
```

### 样例 2: 脱敏导致 unknown
```
note_id: 10001338-DS-6
subject_id: 10001338
source_section: Social History
smoking_status_norm: unknown
evidence_quality: none
missing_flags: [同上]
```

### 样例 3: Full-text fallback 成功 — former smoker (quit smoking)
```
note_id: 10000032-DS-21
subject_id: 10000032
source_section: full_text_fallback
smoking_status_norm: former_smoker
smoking_status_raw: "quit smoking"
pack_year_value: null
evidence_quality: low
evidence_span: "quit smoking"
data_quality_notes: "Social History missing or de-identified; used full-text fallback."
```

### 样例 4: Full-text fallback 成功 — former smoker (history of smoking)
```
note_id: 10000032-DS-22
subject_id: 10000032
source_section: full_text_fallback
smoking_status_norm: former_smoker
smoking_status_raw: "history of smoking"
pack_year_value: null
evidence_quality: low
evidence_span: "history of smoking"
data_quality_notes: "Social History missing or de-identified; used full-text fallback."
```

### 样例 5: Full-text fallback 成功 — never smoker
```
note_id: 10000980-DS-21
subject_id: 10000980
source_section: full_text_fallback
smoking_status_norm: never_smoker
smoking_status_raw: "non-smoker"
pack_year_value: null
evidence_quality: low
evidence_span: "non-smoker"
data_quality_notes: "Social History missing or de-identified; used full-text fallback."
```

### 样例 6: Full-text fallback 失败 — 有烟草句子但无法判定状态
```
note_id: 10000826-DS-17
subject_id: 10000826
source_section: full_text_fallback
smoking_status_norm: unknown
evidence_quality: none
data_quality_notes: "Social History missing or de-identified; used full-text fallback."
```

### 样例 7: Full-text fallback 失败 — 全文无烟草线索
```
note_id: 10000935-DS-19
subject_id: 10000935
source_section: Social History
smoking_status_norm: unknown
evidence_quality: none
data_quality_notes: "Social History appears de-identified."
```

> **Phase 3.1 更新**：新增 full-text fallback 策略后，在 20K discharge 样本中：
> - Social History 路径成功抽取: ~193 条
> - Full-text fallback 成功抽取: ~1,210 条（覆盖率提升约 6 倍）
> - Fallback 仍然失败: ~1,914 条（找到烟草句子但无法判定状态，或全文无线索）
> - Fallback 来源的 evidence_quality 上限为 `low`，不会高估证据强度
> - PPD 歧义保护在 fallback 路径中仍然有效
> - 这仍然是弱监督 baseline，不是高精度 eligibility extractor

---

## 6. Recommendation 样例（5 例）

### 样例 1: 短期随访 (Lung-RADS 3)
```json
{
  "case_id": "CASE-10000935-001",
  "recommendation_level": "short_interval_followup",
  "recommendation_action": "建议 6 个月后复查 LDCT。",
  "followup_interval": "6_months",
  "lung_rads_category": "3",
  "guideline_source": "Lung-RADS_v2022",
  "triggered_rules": ["solid_6_to_7_category_3"],
  "missing_information": ["density_category"],
  "uncertainty_note": "density_category 缺失或不明确，按 solid 路径保守处理。"
}
```

### 样例 2: 数据不足
```json
{
  "case_id": "CASE-10001338-001",
  "recommendation_level": "insufficient_data",
  "recommendation_action": "缺少足够结构化事实，当前只能返回 insufficient_data。",
  "followup_interval": null,
  "lung_rads_category": null,
  "guideline_source": "Lung-RADS_v2022",
  "triggered_rules": ["fallback_insufficient_data"],
  "missing_information": ["nodule_size"],
  "uncertainty_note": "关键尺寸信息缺失，无法稳定进入 Lung-RADS 最小规则集。"
}
```

### 样例 3: 可疑结节 (Lung-RADS 4A)
```json
{
  "case_id": "CASE-10001401-001",
  "recommendation_level": "short_interval_followup",
  "recommendation_action": "建议 3 个月后复查 LDCT，必要时补充 PET-CT。",
  "followup_interval": "3_months",
  "lung_rads_category": "4A",
  "guideline_source": "Lung-RADS_v2022",
  "triggered_rules": ["solid_8_to_14_category_4A"],
  "missing_information": ["density_category"],
  "uncertainty_note": "density_category 缺失或不明确，按 solid 路径保守处理。"
}
```

### 样例 4: 增长上调 (Lung-RADS 3)
```json
{
  "case_id": "CASE-10001884-001",
  "recommendation_level": "short_interval_followup",
  "recommendation_action": "建议 6 个月后复查 LDCT。",
  "followup_interval": "6_months",
  "lung_rads_category": "3",
  "guideline_source": "Lung-RADS_v2022",
  "triggered_rules": ["solid_lt6_category_2", "growth_modifier_upgrade"],
  "missing_information": ["density_category"],
  "uncertainty_note": "density_category 缺失或不明确，按 solid 路径保守处理。 存在增长证据，类别上调一级。"
}
```

### 样例 5: 良性钙化 (Lung-RADS 2)
```json
{
  "case_id": "CASE-10001919-001",
  "recommendation_level": "routine_screening",
  "recommendation_action": "继续 annual LDCT 筛查，12 个月后复查。",
  "followup_interval": "12_months",
  "lung_rads_category": "2",
  "guideline_source": "Lung-RADS_v2022",
  "triggered_rules": ["benign_density_category_2"],
  "missing_information": []
}
```

---

## 7. Case Bundle 样例（3 例）

### 样例 1
```
case_id: CASE-10000935-001
subject_id: 10000935
label_quality: weak
radiology_facts_count: 2
smoking_eligibility: present (但 evidence_quality=none)
has_recommendation_output: false
provenance.radiology_note_ids: [10000935-RR-76, 10000935-RR-80]
provenance.discharge_note_id: 10000935-DS-18
```

### 样例 2
```
case_id: CASE-10001338-001
subject_id: 10001338
label_quality: weak
radiology_facts_count: 1
smoking_eligibility: present (但 evidence_quality=none)
has_recommendation_output: false
provenance.radiology_note_ids: [10001338-RR-42]
provenance.discharge_note_id: 10001338-DS-6
```

### 样例 3
```
case_id: CASE-10001401-001
subject_id: 10001401
label_quality: weak
radiology_facts_count: 2
smoking_eligibility: present (但 evidence_quality=none)
has_recommendation_output: false
provenance.radiology_note_ids: [10001401-RR-27, 10001401-RR-8]
provenance.discharge_note_id: 10001401-DS-17
```

---

## 8. 失败案例分析

### 8.1 Radiology 抽取失败（2 例）

#### 失败 1: "innumerable pulmonary nodules" — 无法抽取尺寸
```
note_id: 10000935-RR-76, nodule #1
evidence_span: "There are innumerable pulmonary nodules, which have an upper lobe predilection, particularly in the left upper lobe."
问题: 报告描述"无数结节"但未给出具体尺寸，regex 无法抽取 size_mm
missing_flags: [size_mm, size_text, density_text, change_status, change_text, lung_rads_category, recommendation_cue]
影响: 下游 recommendation 只能输出 insufficient_data 或保守估计
```

#### 失败 2: "subcentimeter low density lesion" — 非标准描述
```
note_id: 10000935-RR-76, nodule #3
evidence_span: "There is a subcentimeter low density lesion in the upper pole of the right"
问题: "subcentimeter" 未被解析为具体尺寸（应为 <10mm），"low density" 未映射到标准密度类别
missing_flags: [size_mm, size_text, density_text, location_lobe, location_text, change_status, change_text, lung_rads_category, recommendation_cue]
改进方向: 增加 "subcentimeter" -> size_mm=9.9 (保守上界) 的规则
```

### 8.2 Smoking 抽取失败（2 例）

#### 失败 1: Social History 完全脱敏
```
note_id: 10000935-DS-18
问题: Social History section 内容为 "___"（MIMIC 脱敏），无法抽取任何吸烟信息
evidence_quality: none
影响: 该患者的 case bundle 中 smoking_eligibility 全部为 null/unknown
```

#### 失败 2: 无 Social History section
```
note_id: 10000935-DS-19
问题: discharge note 中未找到 Social History section header
evidence_quality: none
影响: 同上
```

### 8.3 Recommendation 失败（2 例）

#### 失败 1: 尺寸缺失导致 insufficient_data
```
case_id: CASE-10001338-001
问题: 该患者唯一的 radiology fact 中所有结节 size_mm 均为 null
triggered_rules: [fallback_insufficient_data]
uncertainty_note: "关键尺寸信息缺失，无法稳定进入 Lung-RADS 最小规则集。"
改进方向: 增加基于 density/location 的 fallback 规则路径
```

#### 失败 2: density 缺失导致保守处理
```
case_id: CASE-10001884-001
问题: density_category 为 unclear，引擎按 solid 路径保守处理
uncertainty_note: "density_category 缺失或不明确，按 solid 路径保守处理。"
影响: 可能高估风险等级（ground-glass 结节被当作 solid 处理）
```

---

## 9. Schema 校验结果

| Schema | 校验数量 | 通过 | 失败 |
|--------|---------|------|------|
| radiology_fact_schema.json | 50 | 50 | 0 |
| smoking_eligibility_schema.json | 116 | 116 | 0 |
| case_bundle_schema.json | 25 | 25 | 0 |
| recommendation_schema.json | 25 | 25 | 0 |

---

## 10. 已知局限与后续改进

| 问题 | 当前状态 | 改进方向 |
|------|---------|---------|
| density 大量为 unclear | regex 未覆盖所有密度表达 | 扩展密度关键词库，增加上下文窗口 |
| size_mm 缺失率高 | 仅匹配显式 "X mm/cm" | 增加 "subcentimeter"、"tiny"、"small" 等模糊尺寸映射 |
| smoking 几乎全部 unknown | MIMIC 脱敏导致 | 扩大采样范围，或接入其他数据源 |
| label_quality 全部为 weak | smoking evidence 不足 + confidence 偏低 | 提升抽取质量后重新评估 |
| demographics 全部为 null | mimic-iv-3.1.zip 未解压 | 调用已实现的 load_patients_from_zip 接口 |
| 结节共指消解缺失 | FINDINGS 和 IMPRESSION 中同一结节未合并 | 后续可加基于 size+location 的去重逻辑 |
