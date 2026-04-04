# Phase 3 Baseline 设计文档

> 日期：2026-04-04
> 阶段：Phase 3 — 最小可运行 baseline
> 目标：构建端到端原型闭环 `radiology report → radiology facts → case bundle → rule-based recommendation`

---

## 1. 实现范围

本阶段交付一个最小闭环 baseline 系统，包含 5 个模块：

| 模块 | 实现方式 | 核心产物 |
|------|---------|---------|
| Part A: 数据读取与预处理 | pandas + regex section parser | `src/data/`, `src/parsers/` |
| Part B: Radiology baseline | 规则/regex 驱动 | `src/extractors/radiology_extractor.py` |
| Part C: Smoking baseline | 规则/regex 驱动（弱监督辅助） | `src/extractors/smoking_extractor.py` |
| Part D: Case bundle assembler | subject_id 主键组装 | `src/assemblers/case_bundle_assembler.py` |
| Part E: Rule-based recommendation | Lung-RADS v2022 最小规则子集 | `src/rules/lung_rads_engine.py` |

---

## 2. 字段支持状态

### 2.1 Radiology Fact Schema

| 字段 | 支持状态 | 说明 |
|------|---------|------|
| note_id | ✅ 稳定 | 从 DataFrame 构造，格式 `{subject_id}-RR-{note_seq}` |
| subject_id | ✅ 稳定 | 直接从数据读取 |
| exam_name | ✅ 稳定 | 从 radiology_detail 关联 |
| modality | ✅ 稳定 | 基于 exam_name 规则映射（CT/LDCT/CTA/X-ray/PET_CT/MRI/other） |
| body_site | ✅ 稳定 | 基于 exam_name 规则映射 |
| report_text | ✅ 稳定 | 原始全文保留 |
| sections | ⚠️ 中等 | regex section parser，覆盖 INDICATION/TECHNIQUE/COMPARISON/FINDINGS/IMPRESSION |
| nodule_count | ⚠️ 中等 | 基于 mention 分割计数，多结节场景可能不精确 |
| nodules[].size_mm | ⚠️ 中等 | 支持 mm/cm 换算、2D/3D 取最大值、连字符格式 |
| nodules[].size_text | ⚠️ 中等 | 保留原始文本 |
| nodules[].density_category | ⚠️ 中等 | 支持 solid/part_solid/ground_glass/calcified/unclear |
| nodules[].density_text | ⚠️ 中等 | 保留原始密度描述 |
| nodules[].location_lobe | ⚠️ 中等 | 支持 RUL/RML/RLL/LUL/LLL/lingula/bilateral/unclear |
| nodules[].location_text | ⚠️ 中等 | 保留原始位置描述 |
| nodules[].count_type | ⚠️ 中等 | single/multiple/unclear |
| nodules[].change_status | ❌ 弱 | 依赖纵向对比信息，很多报告缺乏 prior imaging |
| nodules[].change_text | ⚠️ 中等 | 保留原始变化描述 |
| nodules[].calcification | ⚠️ 中等 | 关键词触发，排除 non-calcified |
| nodules[].spiculation | ⚠️ 中等 | 关键词触发 |
| nodules[].lobulation | ⚠️ 中等 | 关键词触发 |
| nodules[].cavitation | ⚠️ 中等 | 关键词触发 |
| nodules[].perifissural | ⚠️ 中等 | 关键词触发 |
| nodules[].lung_rads_category | ⚠️ 中等 | 仅当报告显式提及时抽取 |
| nodules[].recommendation_cue | ⚠️ 中等 | 关键词触发 + 句子抽取 |
| nodules[].evidence_span | ✅ 稳定 | 保留 mention 原文 |
| nodules[].confidence | ✅ 稳定 | 基于 size/density/location 三要素评分 |
| nodules[].missing_flags | ✅ 稳定 | 自动收集 null 字段 |

### 2.2 Smoking Eligibility Schema

| 字段 | 支持状态 | 说明 |
|------|---------|------|
| subject_id / note_id | ✅ 稳定 | 直接传入 |
| source_section | ✅ 稳定 | `Social History` / `full_text_fallback` / `null` |
| smoking_status_raw | ⚠️ 中等 | Social History 原文片段或全文 fallback 片段 |
| smoking_status_norm | ⚠️ 中等 | current_smoker/former_smoker/never_smoker/unknown |
| ever_smoker_flag | ⚠️ 中等 | 从 status_norm 推导 |
| pack_year_text / pack_year_value | ❌ 弱 | 数据中仅 ~0.5% 含 pack-year |
| ppd_text / ppd_value | ❌ 弱 | 含 PPD 歧义保护（排除 postpartum/TB） |
| years_smoked_text / years_smoked_value | ❌ 弱 | 极少出现 |
| quit_years_text / quit_years_value | ❌ 弱 | 极少出现 |
| eligible_for_high_risk_screening | ⚠️ 中等 | 大量输出 unknown（数据不足） |
| eligibility_reason | ⚠️ 中等 | 解释判定依据 |
| evidence_quality | ✅ 稳定 | high/medium/low/none 四级评估；fallback 来源上限为 low |
| missing_flags | ✅ 稳定 | 自动收集 |

> **Phase 3.1 更新**：Smoking baseline 新增 full-text fallback 策略。
> 当 Social History 段不存在或内容被脱敏（如 `___`）时，自动在全文中搜索含烟草语义的句子进行抽取。
> 该 fallback 保留 PPD 歧义保护（排除 postpartum/TB 上下文），且 evidence_quality 上限为 `low`。
> 在 20K discharge 样本中，fallback 额外捕获 ~1,210 条 smoking 信息（对比 Social History 路径的 ~193 条），覆盖率提升约 6 倍。
> 这仍然是弱监督 baseline，不是高精度 eligibility extractor。

### 2.3 Recommendation Schema

| 字段 | 支持状态 | 说明 |
|------|---------|------|
| case_id | ✅ 稳定 | 从 case bundle 传入 |
| recommendation_level | ✅ 稳定 | 7 级枚举全覆盖 |
| recommendation_action | ✅ 稳定 | 人类可读建议文本 |
| followup_interval | ✅ 稳定 | immediate/1-24_months/none/unspecified |
| followup_modality | ✅ 稳定 | LDCT/CT_with_contrast/PET_CT/biopsy 等 |
| lung_rads_category | ✅ 稳定 | 0-4X/S 全覆盖 |
| guideline_source | ✅ 稳定 | 当前固定 Lung-RADS_v2022 |
| guideline_anchor | ✅ 稳定 | 引用具体规则段落 |
| reasoning_path | ✅ 稳定 | 有序推理步骤 |
| triggered_rules | ✅ 稳定 | 触发的规则 ID 列表 |
| input_facts_used | ✅ 稳定 | 消费的输入事实子集 |
| missing_information | ✅ 稳定 | 缺失信息列表 |
| uncertainty_note | ✅ 稳定 | 不确定性说明 |
| output_type | ✅ 稳定 | 固定 rule_based |
| generation_metadata | ✅ 稳定 | 版本 + 时间戳 |

### 2.4 Case Bundle Schema

| 字段 | 支持状态 | 说明 |
|------|---------|------|
| case_id | ✅ 稳定 | CASE-{subject_id}-001 |
| subject_id | ✅ 稳定 | 主键 |
| demographics | ❌ 占位 | 当前全部为 null + missing_flags（mimic-iv-3.1.zip 未解压） |
| radiology_facts | ✅ 稳定 | 数组，minItems=1 |
| smoking_eligibility | ⚠️ 中等 | 可为 null |
| recommendation_target | ⚠️ 中等 | ground_truth 从 recommendation_cue 提取 |
| provenance | ✅ 稳定 | note_ids + 版本信息 |
| split | ⚠️ 占位 | 当前全部 unlabeled |
| label_quality | ✅ 稳定 | silver/weak/unlabeled 三级分类 |

---

## 3. 暂不支持的字段与原因

| 字段/功能 | 原因 |
|-----------|------|
| demographics (age/sex/race/insurance) | mimic-iv-3.1.zip 为 10GB，当前阶段未解压；已实现 `load_patients_from_zip` 接口，后续可接入 |
| split (train/val/test) | 需要先完成 silver-standard 评测集构建，当前全部标记为 unlabeled |
| 多指南支持 (Fleischner/Chinese) | 当前仅实现 Lung-RADS v2022 最小子集，后续可扩展 |
| 精确 pack-year 计算 | discharge Social History 97.9% 被脱敏，定量信息极稀疏 |
| 纵向对比 (change_status 精确判定) | 需要同一患者多次报告的时序对齐，当前仅做单报告内关键词匹配 |
| 结节共指消解 | 同一结节在 FINDINGS 和 IMPRESSION 中的不同描述未做合并 |

---

## 4. 为什么这么收缩

1. **数据现实约束**：discharge Social History 大面积脱敏，smoking 只能做弱监督辅助，不能作为主实验
2. **无 gold label**：没有人工标注的结构化抽取结果，无法做精确评测，只能用 rule-consistency / silver-standard / case study
3. **阶段目标明确**：Phase 3 的目标是"端到端可执行的原型闭环"，不是训练复杂主模型
4. **先跑通再优化**：先保证 schema 合法、pipeline 可运行、输出可审查，再逐步提升抽取质量

---

## 5. 技术选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 抽取方法 | regex/rule-based | Phase 3 禁止上 BERT/LLM，规则 baseline 可解释、可调试 |
| 推理引擎 | IF-THEN 规则 | 不需要复杂图推理，Lung-RADS 本身就是决策树结构 |
| 数据格式 | JSONL | 流式处理友好，单行可独立校验 |
| Schema 校验 | jsonschema (Draft 2020-12) | 支持 $ref 解析，与已有 schema 兼容 |
| 测试框架 | pytest | 项目标准 |

---

## 6. 管线架构

```
radiology.csv.gz + radiology_detail.csv.gz
    │
    ├── filter_chest_ct (exam_name 过滤)
    ├── filter_nodule_reports (关键词过滤)
    ├── parse_sections (section 切分)
    │
    ▼
extract_radiology_facts (regex baseline)
    │
    ▼                          discharge.csv.gz
case_bundle_assembler  ◄────── extract_smoking_eligibility
    │
    ▼
generate_recommendation (Lung-RADS v2022 rules)
    │
    ▼
recommendation output (JSON)
```

---

## 7. 后续方向

1. 接入 demographics（从 mimic-iv-3.1.zip 读取 patients/admissions）
2. 构建 silver-standard 评测集（人工抽样 + 规则一致性检查）
3. 扩展规则引擎（Fleischner 2017 + 中国指南 2021/2024）
4. 结节共指消解（跨 section 合并同一结节的不同描述）
5. 纵向对比支持（同一患者多次报告的时序对齐）
