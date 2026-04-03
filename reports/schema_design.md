# 肺结节随访建议生成系统 Schema 设计文档

> 文档目的：为模块一（B2 因果解耦）、模块二（B1 事实库抽象）、模块三（B3 神经符号推理）建立统一的数据契约（data contract），减少字段反复改名、接口反复返工、评测口径不一致的问题。
> 
> 适用范围：`schemas/radiology_fact_schema.json`、`schemas/smoking_eligibility_schema.json`、`schemas/recommendation_schema.json`、`schemas/case_bundle_schema.json`。
> 
> 数据依据：`reports/data_audit.md` 中已核实统计与样例；当前项目 readiness 结论为“(b) 可以启动部分 baseline 工作，但缺少关键标注”。

## 1. 为什么需要这 4 个 Schema

当前项目不是单一模型任务，而是三个模块串联的闭环系统：B1 从放射报告抽事实，B2 从出院小结抽吸烟史与高危筛查资格，B3 再基于结构化事实生成随访建议。如果三个模块各自定义字段、命名、缺失值策略、枚举值范围，就会导致下游频繁适配上游输出，研究迭代时出现“字段刚改完又要回滚”的返工。

这种返工在本项目中尤其严重，因为上游数据本身已经存在明显不完整性：`radiology.csv.gz` 有 2,321,355 行，`discharge.csv.gz` 有 331,793 行；结节相关 radiology 报告约 190K（采样估计），但吸烟定量字段极稀疏，`pack-year` 仅约 0.5%，且 `Social History` 在采样中有 97.9% 被脱敏为 `___`。如果接口不先稳定，下游模块就会把“上游缺失”误当成“模型失败”，造成实验解释混乱。

这 4 个 Schema 的分工如下：

- `radiology_fact_schema`：定义单份放射报告的结构化事实输出，是 B1 的主产物。
- `smoking_eligibility_schema`：定义单份出院小结中的吸烟史、高危筛查资格与证据质量，是 B2 的主产物。
- `recommendation_schema`：定义 B3 规则推理后的随访建议、规则触发路径、缺失信息与不确定性说明。
- `case_bundle_schema`：把前 3 者与病例级元数据统一封装成实验单位（unified experiment unit），用于训练、验证、对比与错误分析。

如果没有 `case_bundle_schema`，项目会出现两个常见问题：

- B1 输出按 report 粒度组织，B2 输出按 discharge note 粒度组织，B3 却按 patient/case 粒度推理，粒度不一致。
- 研究脚本会各自实现一套拼接逻辑，导致同一个 `subject_id` 在不同实验里的样本构造规则不一致。

因此，Schema 的核心价值不是“把 JSON 写漂亮”，而是把以下约束固定下来：

- 字段名字固定，避免 `size` / `diameter_mm` / `nodule_size` 多套命名并存。
- 枚举值固定，避免 `ggo`、`ground glass`、`ground_glass` 混用。
- 缺失表达固定，避免有人用 `null`、有人用 `unknown`、有人直接删字段。
- 证据来源固定，便于区分“真实数据里没有”与“规则暂时没抽出来”。
- 实验单位固定，便于后续做 `train/val/test` 切分、silver-standard 评测和案例复盘。

在本项目流水线中，主数据流应理解为：

`radiology report -> radiology_fact -> case_bundle -> recommendation`

其中吸烟史路径不是替代主链，而是并行补充链：

`discharge note -> smoking_eligibility -> case_bundle`

最终两条链在 `case_bundle` 汇合，供 B3 做联合推理。也就是说，`recommendation` 不是直接读取原始文本，而是读取已经结构化后的病例事实，这样才能让规则推理保持可解释、可审计、可回放。

## 2. 各 Schema 字段详解

说明：本节只覆盖“运行时实例字段”，不展开 JSON Schema 自身的定义字段，如 `$schema`、`$id`、`title`、`description`。表中“是否必填”以实例层 required 为准；“当前可靠度”表示在当前数据条件与当前项目状态下，这个字段能否稳定填出，而不是理论上是否有定义。

### 2.1 `radiology_fact_schema`

| 字段名 | 类型 | 是否必填 | 数据来源 | 当前可靠度 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `note_id` | `string` | 是 | 真实抽取 | ✅ 高 | radiology note 主键，MIMIC 风格如 `10046097-RR-34`。 |
| `subject_id` | `integer` | 是 | 真实抽取 | ✅ 高 | 患者主键，用于跨 radiology / discharge / demographics 关联。 |
| `exam_name` | `string` | 是 | 真实抽取 | ✅ 高 | 检查名称，主要来自 `radiology_detail.csv.gz` 中 `exam_name`。 |
| `modality` | `string` | 是 | 规范化映射 | ⚠️ 中 | 由 `exam_name` 或报告头部归一到 `CT`、`LDCT`、`CTA` 等。 |
| `body_site` | `string` | 是 | 规范化映射 | ⚠️ 中 | 由检查名或正文归一到 `chest`、`abdomen` 等。 |
| `report_text` | `string` | 是 | 真实抽取 | ✅ 高 | 原始放射报告全文，是后续抽取与审计的根证据。 |
| `sections` | `object` | 是 | 规范化映射 | ⚠️ 中 | 由规则切分出的 section 容器，便于针对 `FINDINGS` / `IMPRESSION` 定位。 |
| `sections.indication` | `string/null` | 否 | 规范化映射 | ⚠️ 中 | `INDICATION` 段；采样中覆盖率较高，但 section parser 仍需验证。 |
| `sections.technique` | `string/null` | 否 | 规范化映射 | ⚠️ 中 | `TECHNIQUE` 段；用于理解是否低剂量筛查 CT。 |
| `sections.comparison` | `string/null` | 否 | 规范化映射 | ⚠️ 中 | `COMPARISON` 段；后续判断 `change_status` 的关键证据。 |
| `sections.findings` | `string/null` | 否 | 规范化映射 | ⚠️ 中 | `FINDINGS` 段；结节大小、密度、位置主要在此抽取。 |
| `sections.impression` | `string/null` | 否 | 规范化映射 | ⚠️ 中 | `IMPRESSION` 段；Lung-RADS 与 recommendation cue 常见于此。 |
| `nodule_count` | `integer/null` | 是 | 规则推导 | ⚠️ 中 | 报告级结节数量，来自 mention 聚合，不保证与真实病灶数完全一致。 |
| `nodules` | `array<object>` | 是 | 规范化映射 | ⚠️ 中 | 结节级事实数组，是 B1 的核心输出容器。 |
| `nodules[].nodule_id_in_report` | `integer` | 是 | 元数据 | ✅ 高 | 在单报告内的 1-based 顺序编号，便于引用与调试。 |
| `nodules[].size_mm` | `number/null` | 是 | 规范化映射 | ⚠️ 中 | 将 `4 mm`、`1 cm` 等统一换算到毫米。 |
| `nodules[].size_text` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 原始大小表达，便于人工复核单位与边界值。 |
| `nodules[].density_category` | `string/null` | 是 | 规范化映射 | ⚠️ 中 | 归一到 `solid`、`part_solid`、`ground_glass` 等。 |
| `nodules[].density_text` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 原始密度用语，如 `part solid`、`ground-glass`。 |
| `nodules[].location_lobe` | `string/null` | 是 | 规范化映射 | ⚠️ 中 | 归一到 `RUL`、`RML`、`RLL`、`LUL`、`LLL`、`lingula`。 |
| `nodules[].location_text` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 原始部位表达，如 `right middle lobe`。 |
| `nodules[].count_type` | `string` | 是 | 规则推导 | ⚠️ 中 | 区分 `single`、`multiple`、`unclear`，用于多结节规则。 |
| `nodules[].change_status` | `string/null` | 是 | 规则推导 | ❌ 低 | 需要依赖 prior imaging 对比，很多报告没有足够纵向信息。 |
| `nodules[].change_text` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 支撑 `stable`、`new`、`increased` 等判断的原文证据。 |
| `nodules[].calcification` | `boolean/null` | 是 | 规则推导 | ⚠️ 中 | 基于显式关键词触发；当文本明确时较可靠，但总体未系统验证。 |
| `nodules[].spiculation` | `boolean/null` | 是 | 规则推导 | ⚠️ 中 | 毛刺征是重要危险信号，但出现率低且缺标准标注。 |
| `nodules[].lobulation` | `boolean/null` | 是 | 规则推导 | ❌ 低 | 术语更稀疏，且与形态学描述混用，容易漏检。 |
| `nodules[].cavitation` | `boolean/null` | 是 | 规则推导 | ❌ 低 | 稀有字段，当前更适合保留接口，不适合当核心 baseline 指标。 |
| `nodules[].perifissural` | `boolean/null` | 是 | 规则推导 | ❌ 低 | 依赖专门描述词，普通结节报告中覆盖有限。 |
| `nodules[].lung_rads_category` | `string/null` | 是 | 真实抽取 | ❌ 低 | 仅在 screening CT 中更常见，不是所有 chest CT 都会写。 |
| `nodules[].recommendation_cue` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 从原文复制的随访提示，可作为 silver-standard 参照。 |
| `nodules[].evidence_span` | `string` | 是 | 真实抽取 | ⚠️ 中 | 触发抽取的精确文本片段，是错误分析的核心锚点。 |
| `nodules[].confidence` | `string` | 是 | 元数据 | 🔲 未验证 | 由抽取器生成的置信度标签，目前尚无校准验证。 |
| `nodules[].missing_flags` | `array<string>` | 是 | 元数据 | ✅ 高 | 明示该结节下哪些字段缺失，便于区分“没提到”与“没抽出”。 |
| `extraction_metadata` | `object` | 是 | 元数据 | ✅ 高 | 抽取系统运行信息容器。 |
| `extraction_metadata.extractor_version` | `string` | 是 | 元数据 | ✅ 高 | 抽取 pipeline 版本。 |
| `extraction_metadata.extraction_timestamp` | `string` | 是 | 元数据 | ✅ 高 | 结构化结果生成时间。 |
| `extraction_metadata.model_name` | `string` | 是 | 元数据 | ✅ 高 | 使用的模型或规则系统名称。 |

### 2.2 `smoking_eligibility_schema`

| 字段名 | 类型 | 是否必填 | 数据来源 | 当前可靠度 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `subject_id` | `integer` | 是 | 真实抽取 | ✅ 高 | 与 radiology、demographics 的公共连接键。 |
| `note_id` | `string` | 是 | 真实抽取 | ✅ 高 | discharge note 主键，格式如 `10014967-DS-11`。 |
| `source_section` | `string/null` | 是 | 规范化映射 | ⚠️ 中 | 吸烟证据所在 section，优先 `Social History`，否则回退到 `HPI` 等。 |
| `smoking_status_raw` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 原始吸烟文本片段，很多情况下来自非 `Social History` 段。 |
| `smoking_status_norm` | `string/null` | 是 | 规范化映射 | ⚠️ 中 | 归一到 `current_smoker`、`former_smoker`、`never_smoker` 等。 |
| `pack_year_value` | `number/null` | 是 | 规范化映射 | ❌ 低 | 定量包年数；采样仅约 0.5% 记录出现，无法作为稳定主特征。 |
| `pack_year_text` | `string/null` | 是 | 真实抽取 | ❌ 低 | 原始包年表达，如 `46 pack year`，极度稀疏。 |
| `ppd_value` | `number/null` | 是 | 规范化映射 | ❌ 低 | `packs per day` 数值；采样约 2.5%，且与 `Postpartum Day` 存歧义。 |
| `ppd_text` | `string/null` | 是 | 真实抽取 | ❌ 低 | 原始 `PPD` 文本片段，需做歧义消解。 |
| `years_smoked_value` | `number/null` | 是 | 规范化映射 | ❌ 低 | 吸烟年限，出院小结中不稳定且常为模糊表达。 |
| `years_smoked_text` | `string/null` | 是 | 真实抽取 | ❌ 低 | 原始年限表述，如 `30+ years`。 |
| `quit_years_value` | `number/null` | 是 | 规范化映射 | ❌ 低 | 戒烟距今年数；常见表达模糊，如 `a couple of years ago`。 |
| `quit_years_text` | `string/null` | 是 | 真实抽取 | ❌ 低 | 原始戒烟时点表述。 |
| `evidence_span` | `string/null` | 是 | 真实抽取 | ⚠️ 中 | 支撑吸烟状态或高危资格判断的精确文本。 |
| `ever_smoker_flag` | `boolean/null` | 是 | 规则推导 | ⚠️ 中 | 当前最可行的 baseline 输出之一，优于强求完整定量史。 |
| `eligible_for_high_risk_screening` | `string` | 是 | 规则推导 | ❌ 低 | 受脱敏与年龄未接入影响，当前大多数样本应为 `unknown`。 |
| `eligibility_criteria_applied` | `string/null` | 是 | 规则推导 | ❌ 低 | 使用 `USPSTF_2021`、`Chinese_2021` 等规则框架，但上游证据常不全。 |
| `eligibility_reason` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 解释为何 `eligible`、`not_eligible` 或 `unknown`，对调试很重要。 |
| `evidence_quality` | `string` | 是 | 规则推导 | ⚠️ 中 | 对证据显式度、完整性、脱敏程度做质量评估。 |
| `extraction_metadata` | `object` | 是 | 元数据 | ✅ 高 | 抽取运行信息容器。 |
| `extraction_metadata.extractor_version` | `string` | 是 | 元数据 | ✅ 高 | 吸烟抽取器版本。 |
| `extraction_metadata.extraction_timestamp` | `string` | 是 | 元数据 | ✅ 高 | 抽取时间戳。 |
| `extraction_metadata.model_name` | `string` | 是 | 元数据 | ✅ 高 | 规则系统或模型名称。 |
| `missing_flags` | `array<string>` | 是 | 元数据 | ✅ 高 | 显示缺失字段列表，适合记录 `pack_year_value` 等空缺。 |
| `data_quality_notes` | `string/null` | 是 | 元数据 | ⚠️ 中 | 记录脱敏、冲突叙述、section 识别失败等质量问题。 |

### 2.3 `recommendation_schema`

| 字段名 | 类型 | 是否必填 | 数据来源 | 当前可靠度 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `case_id` | `string` | 是 | 元数据 | ✅ 高 | 推荐结果实例 ID，通常由病例 ID 与推理上下文组成。 |
| `recommendation_level` | `string` | 是 | 规则推导 | ⚠️ 中 | 归一到 `routine_screening`、`diagnostic_workup` 等离散建议层级。 |
| `recommendation_action` | `string` | 是 | 规则推导 | ⚠️ 中 | 面向人读的动作描述，如复查 CT、PET-CT、活检等。 |
| `followup_interval` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 归一到 `3_months`、`6_months`、`12_months` 等。 |
| `followup_modality` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 建议的检查或处理方式，如 `LDCT`、`PET_CT`、`biopsy`。 |
| `lung_rads_category` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 若推理走 Lung-RADS 路径，则输出相应 category；否则可空。 |
| `guideline_source` | `string` | 是 | 规则推导 | ✅ 高 | 标记推理主要依据 `Lung-RADS_v2022`、`Fleischner_2017` 等。 |
| `guideline_anchor` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 具体引用规则、表格或条款锚点，便于审计。 |
| `reasoning_path` | `array<string>` | 是 | 规则推导 | ⚠️ 中 | 有序推理链，是神经符号可解释性的关键输出。 |
| `triggered_rules` | `array<string>` | 是 | 规则推导 | ✅ 高 | 被触发的内部规则 ID 列表。 |
| `input_facts_used` | `object` | 是 | 元数据 | ⚠️ 中 | 记录本次决策实际消费的输入事实子集。 |
| `input_facts_used.nodule_size_mm` | `number/null` | 否 | 元数据 | ⚠️ 中 | 从上游 radiology facts 复制进入决策上下文的尺寸值。 |
| `input_facts_used.nodule_density` | `string/null` | 否 | 元数据 | ⚠️ 中 | 从上游复制的密度类别。 |
| `input_facts_used.nodule_count` | `integer/null` | 否 | 元数据 | ⚠️ 中 | 参与规则判断的结节数量。 |
| `input_facts_used.change_status` | `string/null` | 否 | 元数据 | ❌ 低 | 受纵向比较缺失影响，常为空或不稳定。 |
| `input_facts_used.patient_risk_level` | `string/null` | 否 | 元数据 | ❌ 低 | 依赖年龄与吸烟史联合确定，当前上游不足。 |
| `input_facts_used.smoking_eligible` | `string/null` | 否 | 元数据 | ❌ 低 | 直接受 B2 `eligible_for_high_risk_screening` 的低可用性限制。 |
| `missing_information` | `array<string>` | 是 | 规则推导 | ✅ 高 | 明确指出哪些缺失事实阻碍了更精确建议。 |
| `uncertainty_note` | `string/null` | 是 | 规则推导 | ⚠️ 中 | 解释 fallback、信息不足或多指南冲突。 |
| `output_type` | `string` | 是 | 元数据 | ✅ 高 | 标记结果是 `rule_based`、`silver_label` 等。 |
| `generation_metadata` | `object` | 是 | 元数据 | ✅ 高 | 推荐生成运行信息容器。 |
| `generation_metadata.engine_version` | `string` | 是 | 元数据 | ✅ 高 | 规则引擎版本。 |
| `generation_metadata.generation_timestamp` | `string` | 是 | 元数据 | ✅ 高 | 推荐生成时间。 |
| `generation_metadata.rules_version` | `string` | 是 | 元数据 | ✅ 高 | 规则库版本。 |

### 2.4 `case_bundle_schema`

| 字段名 | 类型 | 是否必填 | 数据来源 | 当前可靠度 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `case_id` | `string` | 是 | 元数据 | ✅ 高 | 统一病例 ID，是实验、评测、案例分析的主索引。 |
| `subject_id` | `integer` | 是 | 真实抽取 | ✅ 高 | 患者主键，连接 radiology、discharge、demographics。 |
| `demographics` | `object` | 是 | 人口统计 | ⚠️ 中 | 患者人口统计容器，数据源在 `mimic-iv-3.1.zip`。 |
| `demographics.age` | `integer/null` | 是 | 人口统计 | ❌ 低 | 来自 `patients.csv.gz` 的年龄信息，数据存在但尚未正式抽取接入。 |
| `demographics.sex` | `string/null` | 是 | 人口统计 | ⚠️ 中 | 同样来自 `patients.csv.gz`，技术上较容易接入。 |
| `demographics.race` | `string/null` | 是 | 人口统计 | ❌ 低 | 需要 `admissions.csv.gz`，当前未解压接入，且事件级映射仍待定。 |
| `demographics.insurance` | `string/null` | 是 | 人口统计 | ❌ 低 | 来自 `admissions.csv.gz`，当前也未接入。 |
| `demographics.source` | `string/null` | 是 | 元数据 | 🔲 未验证 | 记录 demographics 来源文件，如 `patients.csv.gz` 或 `admissions.csv.gz`。 |
| `demographics.missing_flags` | `array<string>` | 是 | 元数据 | ✅ 高 | 标记年龄、race、insurance 等人口统计缺失项。 |
| `radiology_facts` | `array<object>` | 是 | 规范化映射 | ⚠️ 中 | 一个病例中链接的一份或多份 radiology facts。 |
| `smoking_eligibility` | `object/null` | 是 | 规则推导 | ❌ 低 | 由 B2 生成；当前多为弱监督、缺失或 `unknown`。 |
| `recommendation_target` | `object` | 是 | 规则推导 | ⚠️ 中 | 真实标签与模型/规则输出的统一容器。 |
| `recommendation_target.ground_truth_action` | `string/null` | 是 | 真实抽取 | ❌ 低 | 当前没有系统 gold label，只能尝试从报告 recommendation cue 中弱抽取。 |
| `recommendation_target.ground_truth_source` | `string/null` | 是 | 元数据 | ⚠️ 中 | 标记 `extracted_from_report`、`silver_label`、`expert_annotation` 或 `none`。 |
| `recommendation_target.ground_truth_interval` | `string/null` | 是 | 规范化映射 | ❌ 低 | 需从原文 recommendation cue 归一到标准时间间隔，当前缺稳定标签。 |
| `recommendation_target.recommendation_output` | `object/null` | 是 | 规则推导 | ⚠️ 中 | B3 生成的结构化推荐结果。 |
| `provenance` | `object` | 是 | 元数据 | ✅ 高 | 病例组装来源追踪容器。 |
| `provenance.radiology_note_ids` | `array<string>` | 是 | 元数据 | ✅ 高 | 参与该病例组装的 radiology note 列表。 |
| `provenance.discharge_note_id` | `string/null` | 是 | 元数据 | ⚠️ 中 | 参与吸烟史抽取的 discharge note ID；无匹配时为空。 |
| `provenance.data_version` | `string` | 是 | 元数据 | ✅ 高 | 底层数据版本，如 `mimic-iv-note-2.2`。 |
| `provenance.extraction_date` | `string` | 是 | 元数据 | ✅ 高 | 该病例 bundle 组装或导出的日期。 |
| `provenance.pipeline_version` | `string` | 是 | 元数据 | ✅ 高 | 病例组装 pipeline 版本。 |
| `split` | `string` | 是 | 元数据 | ✅ 高 | 数据切分标记：`train`、`val`、`test`、`unlabeled`。 |
| `label_quality` | `string` | 是 | 元数据 | ✅ 高 | 标签质量层级：`gold`、`silver`、`weak`、`unlabeled`。 |
| `case_notes` | `string/null` | 是 | 元数据 | ⚠️ 中 | 记录链接歧义、人工复核、排除原因等自由文本说明。 |

## 3. 字段来源分类

说明：本节按“运行时实例字段”逐一归类；为避免遗漏，容器字段也纳入分类。由于用户要求仅分 3 类，所有系统追踪字段（如 version、timestamp、missing flags）统一放入“规则生成的字段”一类中的“系统元数据/追踪字段”子组。

### 3.1 来自真实数据的字段（directly from MIMIC/LIDC）

- `radiology_fact.note_id`、`radiology_fact.subject_id`、`radiology_fact.exam_name`、`radiology_fact.report_text`
- `radiology_fact.nodules[].size_text`、`radiology_fact.nodules[].density_text`、`radiology_fact.nodules[].location_text`、`radiology_fact.nodules[].change_text`
- `radiology_fact.nodules[].recommendation_cue`、`radiology_fact.nodules[].evidence_span`
- `radiology_fact.nodules[].lung_rads_category`（注意：来自真实报告文本，但覆盖只在部分 screening CT 中较高）
- `smoking_eligibility.subject_id`、`smoking_eligibility.note_id`
- `smoking_eligibility.smoking_status_raw`、`smoking_eligibility.pack_year_text`、`smoking_eligibility.ppd_text`
- `smoking_eligibility.years_smoked_text`、`smoking_eligibility.quit_years_text`、`smoking_eligibility.evidence_span`
- `case_bundle.subject_id`
- `case_bundle.demographics`、`case_bundle.demographics.age`、`case_bundle.demographics.sex`
- `case_bundle.demographics.race`、`case_bundle.demographics.insurance`
- `case_bundle.recommendation_target.ground_truth_action`

这些字段的共同特点是：理论上它们都应该能在原始表、原始文本或原始人口统计表中找到证据，不依赖规则系统“生成”。但“来自真实数据”不等于“现在就高覆盖”，例如 `pack_year_text` 来自真实数据，可仍然极稀缺。

### 3.2 来自规范化处理的字段（NLP extraction + normalization）

- `radiology_fact.modality`、`radiology_fact.body_site`、`radiology_fact.sections`
- `radiology_fact.sections.indication`、`radiology_fact.sections.technique`、`radiology_fact.sections.comparison`
- `radiology_fact.sections.findings`、`radiology_fact.sections.impression`
- `radiology_fact.nodules`、`radiology_fact.nodule_count`
- `radiology_fact.nodules[].size_mm`、`radiology_fact.nodules[].density_category`、`radiology_fact.nodules[].location_lobe`
- `smoking_eligibility.source_section`、`smoking_eligibility.smoking_status_norm`
- `smoking_eligibility.pack_year_value`、`smoking_eligibility.ppd_value`
- `smoking_eligibility.years_smoked_value`、`smoking_eligibility.quit_years_value`
- `case_bundle.radiology_facts`
- `case_bundle.recommendation_target.ground_truth_interval`

这类字段的共同特点是：原始证据是文本，但真正进入 Schema 的值已经被归一化。例如 `1 cm` 被转换为 `10` 毫米，`ground-glass opacity` 被映射到 `ground_glass`，`right upper lobe` 被映射到 `RUL`。它们是最适合立即启动 baseline 的字段，因为上游语料充足、文本模式相对稳定。

### 3.3 来自规则生成的字段（rule engine output）

#### 3.3.1 规则/判断输出字段

- `radiology_fact.nodules[].count_type`
- `radiology_fact.nodules[].change_status`
- `radiology_fact.nodules[].calcification`、`radiology_fact.nodules[].spiculation`
- `radiology_fact.nodules[].lobulation`、`radiology_fact.nodules[].cavitation`、`radiology_fact.nodules[].perifissural`
- `smoking_eligibility.ever_smoker_flag`
- `smoking_eligibility.eligible_for_high_risk_screening`
- `smoking_eligibility.eligibility_criteria_applied`
- `smoking_eligibility.eligibility_reason`
- `smoking_eligibility.evidence_quality`
- `recommendation.recommendation_level`、`recommendation.recommendation_action`
- `recommendation.followup_interval`、`recommendation.followup_modality`
- `recommendation.lung_rads_category`、`recommendation.guideline_source`
- `recommendation.guideline_anchor`、`recommendation.reasoning_path`
- `recommendation.triggered_rules`、`recommendation.missing_information`
- `recommendation.uncertainty_note`
- `case_bundle.smoking_eligibility`
- `case_bundle.recommendation_target`
- `case_bundle.recommendation_target.recommendation_output`

#### 3.3.2 系统元数据 / 追踪字段

- `radiology_fact.nodules[].nodule_id_in_report`、`radiology_fact.nodules[].confidence`、`radiology_fact.nodules[].missing_flags`
- `radiology_fact.extraction_metadata`、`radiology_fact.extraction_metadata.extractor_version`
- `radiology_fact.extraction_metadata.extraction_timestamp`、`radiology_fact.extraction_metadata.model_name`
- `smoking_eligibility.extraction_metadata`、`smoking_eligibility.extraction_metadata.extractor_version`
- `smoking_eligibility.extraction_metadata.extraction_timestamp`、`smoking_eligibility.extraction_metadata.model_name`
- `smoking_eligibility.missing_flags`、`smoking_eligibility.data_quality_notes`
- `recommendation.case_id`、`recommendation.input_facts_used`
- `recommendation.input_facts_used.nodule_size_mm`、`recommendation.input_facts_used.nodule_density`
- `recommendation.input_facts_used.nodule_count`、`recommendation.input_facts_used.change_status`
- `recommendation.input_facts_used.patient_risk_level`、`recommendation.input_facts_used.smoking_eligible`
- `recommendation.output_type`、`recommendation.generation_metadata`
- `recommendation.generation_metadata.engine_version`
- `recommendation.generation_metadata.generation_timestamp`
- `recommendation.generation_metadata.rules_version`
- `case_bundle.case_id`、`case_bundle.demographics.source`、`case_bundle.demographics.missing_flags`
- `case_bundle.recommendation_target.ground_truth_source`
- `case_bundle.provenance`、`case_bundle.provenance.radiology_note_ids`
- `case_bundle.provenance.discharge_note_id`、`case_bundle.provenance.data_version`
- `case_bundle.provenance.extraction_date`、`case_bundle.provenance.pipeline_version`
- `case_bundle.split`、`case_bundle.label_quality`、`case_bundle.case_notes`

这部分字段的职责不是“提供新的临床事实”，而是让每一次抽取、拼接、推理都能被追踪和复现。没有这些字段，实验表面上也能跑，但一旦结果异常，就无法判断是数据版本变了、规则版本变了、还是上游字段缺失导致的。

## 4. 当前可靠度与高风险字段

当前最需要明确标红的不是“字段是否定义了”，而是“字段在当前数据条件下能否稳定被填出”。以下字段属于高风险字段，应该在 baseline 设计、loss 设计、评测口径和错误分析中单独处理。

| 高风险字段 | 风险级别 | 风险原因 | 对系统的影响 | 建议处理方式 |
| --- | --- | --- | --- | --- |
| `pack_year_value` | 极高 | 出院小结采样中仅约 0.5% 出现，且多不在 `Social History` 主段 | 无法稳定支撑高危筛查定量判断 | baseline 只把它当可选加分特征，不作为必需输入 |
| `eligible_for_high_risk_screening` | 极高 | `Social History` 97.9% 脱敏，年龄尚未接入，导致大量样本只能判 `unknown` | B2 直接影响 B3 的筛查路径选择 | 先允许 `unknown` 成为合法主输出，不强行二分类 |
| `demographics.age` | 高 | 数据在 `mimic-iv-3.1.zip` 内已审计，但尚未抽取接入 pipeline | 无法完整判断 USPSTF / 中国筛查指南年龄阈值 | 将其列为 `case_bundle` 的 pending 关键字段 |
| `demographics.race` | 高 | 需要 `admissions.csv.gz` 且当前未解压接入 | 影响人群分层分析与公平性分析，不影响最小规则闭环 | 先保留接口与 `missing_flags`，不作为 baseline 前置条件 |
| `lung_rads_category` | 高 | 只在 screening CT 报告里更常见，并非所有 chest CT 都写 | 影响能否直接套用 Lung-RADS 分支 | 同时保留 Lung-RADS 路径与 Fleischner 路径 |
| `change_status` | 高 | 需要 prior imaging comparison；很多报告无明确 comparison 或描述含糊 | 影响增长/稳定性判断，进而影响随访间隔 | 允许 `unclear/null`，不要把缺失当成负例 |
| `ppd_value` | 中高 | 采样约 2.5%，且 `PPD` 与 `Postpartum Day` 歧义严重 | 会污染 smoking intensity 推断 | baseline 仅在上下文明确为 smoking 时使用 |
| `ground_truth_action` | 高 | 当前无成体系 gold label，只能从报告 recommendation cue 弱抽取 | 端到端评估易高估或低估真实性能 | 采用 silver-standard + spot-check，而不是假装有 gold |
| `ground_truth_interval` | 高 | 需要把自由文本建议统一映射到标准间隔 | recommendation 评测容易因归一化不一致失真 | 单独维护 interval normalization 规则 |

特别说明如下：

- `pack_year_value` 不是“模型做不好”，而是“源数据大多没有”；它在 schema 中保留是为了捕捉少量高价值样本，而不是期待全量覆盖。
- `eligible_for_high_risk_screening` 当前大多数记录应被视为信息不足下的 `unknown`，这是一种正确输出，不应被误判为错误。
- `demographics.age` 与 `demographics.race` 的问题不在数据源本身，而在 `mimic-iv-3.1.zip` 尚未抽取接入，属于 pipeline readiness 问题。
- `lung_rads_category` 的稀疏性来自任务场景差异：screening CT 会写，普通 diagnostic chest CT 往往不会写。
- `change_status` 看似简单，实则高度依赖纵向就诊上下文；若缺 prior exam，最好显式输出 `unclear` 或放入 `missing_information`。

## 5. 三个模块如何共用这些 Schema

项目中的三个模块不是各自独立产出论文图表，而是围绕统一病例单位协同工作。

- 模块一（B2 因果解耦）：消费 discharge text，输出 `smoking_eligibility_schema`。
- 模块二（B1 事实库抽象）：消费 radiology text，输出 `radiology_fact_schema`。
- 模块三（B3 神经符号推理）：消费 `radiology_fact` + `smoking_eligibility`，输出 `recommendation_schema`。
- `case_bundle_schema`：负责把同一 `subject_id` 下可关联的 radiology、discharge、demographics、recommendation target 绑成统一实验单位。

这里的关键不是“都用 JSON”，而是“都用同一套字段意义”。例如：

- B1 只要输出 `size_mm`、`density_category`、`location_lobe`，B3 就不需要再读原始放射文本。
- B2 只要输出 `smoking_status_norm`、`ever_smoker_flag`、`eligible_for_high_risk_screening`，B3 就不需要重复写吸烟识别逻辑。
- `case_bundle` 把 `label_quality`、`split`、`provenance` 固定下来，训练脚本和评测脚本才能共享同一病例集合。

建议采用如下统一数据流：

```text
radiology.csv.gz(text)
        |
        v
  [B1 放射报告信息抽取]
        |
        v
radiology_fact_schema.json
        |
        |
        |                         discharge.csv.gz(text)
        |                                 |
        |                                 v
        |                       [B2 吸烟史/高危判定]
        |                                 |
        |                                 v
        |                    smoking_eligibility_schema.json
        |                                 |
        +---------------+-----------------+
                        |
                        v
                 case_bundle_schema.json
                        |
                        v
              [B3 神经符号规则推理]
                        |
                        v
              recommendation_schema.json
```

从工程角度看，`case_bundle_schema` 是真正的接口稳定器：

- 对 B1 来说，它把“报告级事实”升格为“病例级样本”。
- 对 B2 来说，它允许吸烟信息为空，但必须明确为空的原因。
- 对 B3 来说，它提供统一输入视图，减少脚本中散落的 join 与 if-else。
- 对评测来说，它把 `recommendation_target` 和 `recommendation_output` 放到同一个对象中，便于做 case-level 对比。

## 6. 当前数据条件下长期为空的字段

以下字段在当前数据条件下，即使 schema 已定义，也很可能在大多数记录中长期为空、为 `null`、或只能填入 `unknown`。这不是 schema 设计错误，而是对真实数据约束的诚实表达。

- `pack_year_value`、`ppd_value`、`years_smoked_value`、`quit_years_value`：主要受出院小结脱敏与定量表达稀疏影响，难以形成稳定覆盖。
- `pack_year_text`、`ppd_text`、`years_smoked_text`、`quit_years_text`：原始文本本身就少，即便保留接口，也只能捕捉少数高价值样本。
- `demographics.race`、`demographics.insurance`：来源于尚未接入的 `mimic-iv-3.1.zip` 内表，短期内大概率为空。
- `lung_rads_category`：只在 screening CT 报告中常见，普通诊断型 chest CT 通常不会显式给出。
- `change_text`、`change_status`：需要纵向比较或明确 comparison 语句，没有 prior imaging 时天然缺失。
- `recommendation_target.ground_truth_action`、`recommendation_target.ground_truth_interval`：当前没有 gold labels，短期内只能在少量含 recommendation cue 的报告中弱构造。
- `input_facts_used.patient_risk_level`、`input_facts_used.smoking_eligible`：受 B2 与 demographics 接入限制，会随上游缺失而长期为空。

这些字段在建模上不应被简单视为“坏样本”。更合理的策略是：

- 把它们设计成合法缺失字段，而不是强制填默认值。
- 用 `missing_flags`、`missing_information`、`uncertainty_note` 记录缺失原因。
- 在 baseline 评测中，把“可评字段覆盖率”与“字段值准确率”拆开汇报。
- 在错误分析中区分“源数据没有”与“规则/模型没抽出来”。

## 7. Baseline I/O Contract

本节定义三个模块当前阶段的最小可运行输入输出契约。这里的重点不是追求最终最优模型，而是保证三个模块都能在同一套 schema 下形成可串联、可评测、可复现的 baseline。

### 7.1 模块二 Baseline（放射报告信息抽取）

说明：项目命名上该模块对应 B1 事实库抽象，但此处按用户要求保留“模块二”标题；实际职责是 radiology report 信息抽取。

- 输入文件：`radiology.csv.gz`。
- 输入字段：`text`。
- 过滤条件：仅保留 `exam_name` 含 `CT` 且含 `CHEST` 的记录；`exam_name` 可由 `radiology_detail.csv.gz` 提供。
- 输入粒度：单份 radiology report。
- 输出格式：逐条输出满足 `radiology_fact_schema.json` 的 JSON 对象。
- 最小可运行 baseline：基于 `regex` / `rule-based` 抽取 `size_mm`、`density_category`、`location_lobe`。
- 推荐的最小 section 使用策略：优先 `FINDINGS`，其次 `IMPRESSION`，`COMPARISON` 仅用于辅助 `change_status`。
- 不应强求的字段：`lung_rads_category`、`change_status`、`perifissural`、`cavitation`。
- 当前评估方式：人工抽检 50 条 + 与报告中的 recommendation cue / 显式 category 做 silver-standard 对比。
- 当前可接受输出下界：即使只有一个结节被抽出，也必须保留 `evidence_span`、`missing_flags`、`extraction_metadata`。

建议的最小 I/O 断言：

```text
输入:
  来源 = radiology.csv.gz
  粒度 = 单份 radiology note
  必需原始字段 = {note_id, subject_id, text}
  过滤条件 = exam_name contains "CT" and "CHEST"

输出:
  单条结果需满足 radiology_fact_schema.json
  最小输出字段 = {note_id, subject_id, exam_name, modality, body_site, report_text, sections, nodule_count, nodules, extraction_metadata}
  最小结节字段 = {nodule_id_in_report, size_mm, density_category, location_lobe, evidence_span, confidence, missing_flags}
```

### 7.2 模块一 Baseline（吸烟史与高危判定）

说明：项目命名上该模块对应 B2 因果解耦；此处按用户要求保留“模块一”标题。

- 输入文件：`discharge.csv.gz`。
- 输入字段：`text`。
- section 策略：优先读取 `Social History`，若为 `___` 或缺失，则回退到 `HPI`、`Assessment`、`Family History` 等段落。
- 输入粒度：单份 discharge note。
- 输出格式：逐条输出满足 `smoking_eligibility_schema.json` 的 JSON 对象。
- 最小可运行 baseline：基于 `regex` 抽取 `pack-year`、`PPD`、吸烟状态关键词。
- 当前实际上限：稳定做到 `smoking_status_norm` + `ever_smoker_flag`；`pack_year_value` 覆盖率极低，不应视为 baseline 主任务。
- `eligible_for_high_risk_screening` 的默认策略：当年龄或定量吸烟史不足时输出 `unknown`，而不是强行二选一。
- 当前评估方式：人工抽检 30 条，重点核查 `current/former/never` 与 `ever_smoker_flag` 是否合理。
- 当前输出重点：比起“填满所有定量字段”，更重要的是把缺失与脱敏原因写进 `evidence_quality`、`missing_flags`、`data_quality_notes`。

建议的最小 I/O 断言：

```text
输入:
  来源 = discharge.csv.gz
  粒度 = 单份 discharge note
  必需原始字段 = {note_id, subject_id, text}
  优先 sections = Social History
  回退 sections = HPI, Assessment, Family History

输出:
  单条结果需满足 smoking_eligibility_schema.json
  最小有效字段 = {subject_id, note_id, source_section, smoking_status_raw, smoking_status_norm, evidence_span, ever_smoker_flag, eligible_for_high_risk_screening, evidence_quality, extraction_metadata, missing_flags}
  当前较强字段 = {smoking_status_norm, ever_smoker_flag}
  当前较弱字段 = {pack_year_value, ppd_value, years_smoked_value, quit_years_value}
```

### 7.3 模块三 Baseline（随访建议生成）

- 输入文件：`case_bundle_schema.json` 实例集合。
- 输入字段：至少需要 `radiology_facts` + `smoking_eligibility`；若 `smoking_eligibility` 缺失，也要能输出带不确定性的 recommendation。
- 输入粒度：单个 `case_bundle`。
- 输出格式：逐条输出满足 `recommendation_schema.json` 的 JSON 对象。
- 最小可运行 baseline：基于 Lung-RADS + Fleischner 的 `IF-THEN rule engine`。
- 前置依赖：B1 / B2 的输出先完成 schema 对齐，再由 bundle assembler 组装成 `case_bundle`。
- 规则优先级建议：若存在明确 `lung_rads_category`，优先走 Lung-RADS；否则回退到基于 `size_mm`、`density_category`、`change_status` 的 Fleischner / 中国指南规则。
- 当前评估方式：与报告中的 `recommendation_cue` 对比 + 指南一致性检查；不假设存在 gold recommendation labels。
- 当前输出重点：必须显式输出 `missing_information`、`uncertainty_note`、`triggered_rules`，否则无法判断是规则错还是上游字段缺。

建议的最小 I/O 断言：

```text
输入:
  来源 = case_bundle_schema.json records
  粒度 = 单个 unified case bundle
  必需结构化输入 = {case_id, radiology_facts, smoking_eligibility, provenance, label_quality}
  关键事实字段 = {size_mm, density_category, location_lobe, count_type, change_status, eligible_for_high_risk_screening}

输出:
  单条结果需满足 recommendation_schema.json
  最小字段 = {case_id, recommendation_level, recommendation_action, followup_interval, followup_modality, guideline_source, reasoning_path, triggered_rules, input_facts_used, missing_information, output_type, generation_metadata}
  优先保留的可解释字段 = {guideline_anchor, uncertainty_note}
```

## 结语：为什么这份 Schema 文档现在就要定下来

当前项目的真正瓶颈不是“没有足够多的 JSON 字段”，而是上游真实数据存在结构性缺失、三模块又必须串成闭环。如果不先把字段语义、来源分类、缺失策略、baseline I/O contract 固定下来，后续每推进一个模块，都会重新改一次接口。

因此，这 4 个 Schema 的设计原则应保持不变：

- 先保证闭环与可解释，再追求字段覆盖率。
- 先诚实表达缺失，再讨论更复杂模型。
- 先做 silver-standard + spot-check 评测，再谈端到端监督学习。
- 让 `case_bundle_schema` 成为唯一统一实验单位，避免多套样本构造逻辑并存。

只要这套 Schema 稳定，项目即使在当前“吸烟史严重受限、人口统计待接入、gold label 缺失”的条件下，仍然可以启动一条可信的 baseline 路线：B1 用规则抽 radiology facts，B2 输出粗粒度 smoking eligibility，B3 结合指南做可解释 recommendation，并通过 `case_bundle` 完成统一组装、评测与案例复盘。
