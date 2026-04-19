# 模块3图谱智能体随访建议生成执行方案

> 日期：2026-04-19  
> 当前阶段：模块2正式补实验基本闭环后，模块3正式实现前的执行方案设计  
> 文档目标：锁定模块3的系统边界、图谱智能体架构、接口协议、实验设计、baseline 设计和落地顺序，避免后续退化为 flat rule engine（平铺规则引擎）。

---

## 0. 执行结论

1. 模块3应正式定位为“基于指南图谱约束的神经符号随访建议生成智能体”，不是单纯规则分类器。
2. 模块3主输入应来自模块2的结构化放射学事实，包括 `density_category`、`size_mm`、`location_lobe`、`confidence`、`evidence_span`、`change_status` 等字段。
3. 模块1不再承担筛查入口主判断，只保留为辅助属性源；筛查 / 偶发分流必须由模块3前置 `report-intent router`（报告意图路由器）处理。
4. 已有 `src/rules/lung_rads_engine.py` 更适合作为最小规则 baseline 或 MVP 参照，不能代表最终模块3。
5. 模块3必须显式构建 Clinical Decision State Graph, CDSG（临床决策状态图），并通过 graph executor（图执行器）在图上执行受约束路径推理。
6. hard constraints（硬约束）优先处理尺寸、密度、数量、增长、良性形态等确定性条件；soft match（软匹配）只在字段缺失、术语不一致或语义蕴含场景中补充。
7. 最终输出必须包含建议结论、图谱路径、触发条件、输入证据 span、指南锚点、缺失信息和 fallback / abstention（拒答或降级）原因。
8. 实验必须同时比较内部递进版本和外部范式 baseline，不能只报告一个完整系统结果。
9. 模块3现在可以进入正式实现阶段；下一步应优先做模块3基础设施，而不是继续补模块2大规模训练。
10. 正式实现前需要冻结模块2推理导出协议，避免模块3后期反复适配字段。

---

## 1. 模块3的正式系统定位

### 1.1 模块3职责

模块3负责把上游结构化事实转化为循证肺结节随访建议报告。它不是重新抽取影像事实，也不是简单复述放射科报告，而是在指南知识图谱上完成以下任务：

1. 判断当前报告属于 screening（筛查场景）还是 incidental finding（偶发发现）场景。
2. 根据场景选择 Lung-RADS、Fleischner、中国肺癌筛查与临床指南等可计算指南子图。
3. 消费模块2输出的肺结节事实，包括尺寸、密度、位置、数量、变化状态、置信度和证据 span。
4. 在 CDSG 上执行图路径推理，得到可审计的终态建议。
5. 对缺失、冲突、低置信度输入进行 abstention / fallback，而不是强行生成。
6. 输出可解释、可追踪、可引用依据的随访建议报告。

### 1.2 与模块1、模块2的边界

| 模块 | 当前定位 | 向模块3提供什么 | 不再承担什么 |
|---|---|---|---|
| 模块1 | 辅助属性模块 | 年龄、吸烟史、包年、戒烟年限、家族史、职业暴露等可选风险属性 | 不再主导筛查 / 偶发分流，不再作为当前主战场 |
| 模块2 | 放射学报告结构化信息抽取模块 | 结节级事实、字段级置信度、证据 span、缺失标记、模型版本 | 不生成最终随访建议，不承担指南路径推理 |
| 模块3 | 图谱智能体随访建议生成模块 | 产出最终循证报告、路径解释、指南锚点和缺失信息 | 不重新训练模块2，不退化为 flat rule engine |

### 1.3 模块1弱化后为何仍然闭环

模块1弱化不会破坏系统闭环，原因是原闭环中的“准入判定”职责被拆成两层：

1. `report-intent router` 先根据检查类型、报告目的、文本提示和患者风险属性判断当前报告是筛查还是偶发场景。
2. 模块1只在需要高危筛查语义时提供辅助属性，例如吸烟史和包年证据。
3. 若模块1信息缺失，模块3仍可进入 Fleischner 偶发结节路径，或输出缺失风险信息的保守报告。
4. 因此整体链路仍然是：非结构化报告 -> 模块2结构化事实 -> 模块3图谱推理 -> 循证随访报告。

---

## 2. 模块3总体架构

### 2.1 组件串联

```text
模块2 Radiology Fact
        +
模块1辅助风险属性
        |
        v
report-intent router
        |
        v
指南知识表示层 / CDSG
        |
        v
graph executor
        |
        +--> hard constraints
        |
        +--> soft match
        |
        +--> abstention / fallback
        |
        v
证据追踪与解释生成
        |
        v
最终随访建议报告
```

### 2.2 `report-intent router`（报告意图路由器）

**输入**

1. `exam_name`、`modality`、`body_site`。
2. `sections.indication`、`sections.technique`、`sections.impression`。
3. 模块2结节事实中的 `lung_rads_category`、`recommendation_cue`、`evidence_span`。
4. 模块1辅助属性：年龄、吸烟史、包年、戒烟年限、家族史、职业暴露。

**输出**

1. `intent_type`：`screening`、`incidental`、`diagnostic_oncology`、`uncertain`。
2. `guideline_family`：`Lung-RADS_v2022`、`Fleischner_2017`、`Chinese_Screening_2021`、`Chinese_Clinical_2024`、`composite`。
3. `router_confidence`：高 / 中 / 低或连续分数。
4. `router_evidence`：触发分流的原文片段和字段。

**串联方式**

router 先决定主图谱入口。若明确为筛查，优先进入 Lung-RADS / 中国筛查路径；若是偶发 CT 发现，优先进入 Fleischner 路径；若存在已知转移癌、感染、术后等诊断性上下文，进入 `diagnostic_oncology` 或 `uncertain` 并触发人工复核提示。

### 2.3 图谱 / 指南知识表示层

**输入**

1. 指南原文或人工整理后的指南条款。
2. 节点定义：临床状态、风险层级、结节类型、建议动作。
3. 边定义：从状态到状态的条件逻辑。

**输出**

1. `guideline_graph.json`：可版本化的图谱定义。
2. 节点表：`node_id`、`node_type`、`risk_level`、`action_desc`、`guideline_anchor`。
3. 边表：`edge_id`、`source`、`target`、`condition_expr`、`priority`、`anchor_text`。
4. 校验报告：可达性、互斥性、完备性、冲突规则清单。

**建议图谱来源**

1. 筛查路径：Lung-RADS v2022 和中国肺癌筛查指南。
2. 偶发路径：Fleischner 2017。
3. 中文论文叙事：中华医学会肺癌临床诊疗指南 2024 作为本地化临床依据。

### 2.4 graph executor（图执行器）

**输入**

1. router 输出的 `guideline_family` 和入口节点。
2. 标准化 case bundle。
3. 图谱节点和边。

**输出**

1. `trajectory`：按顺序记录的节点和边。
2. `terminal_node`：终态建议节点。
3. `matched_conditions`：已满足条件。
4. `unmatched_conditions`：未满足或无法判断条件。
5. `executor_status`：`success`、`abstain`、`fallback`、`conflict`。

**执行原则**

1. 每一步只允许沿图谱合法出边转移。
2. 出边条件先经过 hard constraints 判定。
3. hard constraints 无唯一结果时，再调用 soft match。
4. 多条边同时满足时使用优先级和更高风险优先原则，并记录冲突。
5. 无边可走时不强行给结论，进入 abstention / fallback。

### 2.5 hard constraints（硬约束层）

**输入**

1. 标准化事实：`size_mm`、`density_category`、`count_type`、`change_status`、`calcification`、`spiculation`、`perifissural` 等。
2. 条件表达式：例如 `density == solid AND size_mm >= 8 AND size_mm < 15`。

**输出**

1. 布尔匹配结果。
2. 触发条件列表。
3. 必需但缺失的字段列表。
4. 条件对应的证据 span。

**职责**

硬约束层保证关键数值和类别阈值不被语言模型改写。例如 6 mm、8 mm、15 mm、30 mm 等阈值只能由结构化字段触发，不能由生成文本自由解释。

### 2.6 soft match（软匹配层）

**输入**

1. 未完全满足的条件。
2. 原始报告 span、模块1风险属性文本、推荐 cue。
3. 医学同义词表和指南术语映射。

**输出**

1. `soft_match_decision`：`entailed`、`contradicted`、`unknown`。
2. `soft_match_score`。
3. 检索到的医学语义依据。
4. 是否允许修正硬匹配结果。

**适用边界**

soft match 只能处理语义鸿沟，不能越过数值硬约束。例如“父亲死于肺部占位”可辅助匹配“肺癌家族史”，但不能把“约 6 mm”改成“8 mm”。

### 2.7 abstention / fallback 机制

**触发条件**

1. `size_mm` 缺失且没有可用的报告显式建议。
2. `density_category` 缺失且当前图谱路径必须依赖密度。
3. router 无法判断筛查 / 偶发场景。
4. 多条高风险路径冲突且无法通过优先级消解。
5. 模块2字段置信度低，且证据 span 不支持关键条件。

**输出**

1. `recommendation_level = insufficient_data` 或 `multidisciplinary_review`。
2. 缺失字段列表。
3. 建议补录的信息。
4. 已能确认的低风险或高风险事实。
5. 不给出确定随访间隔的原因。

### 2.8 证据追踪与解释生成

**输入**

1. `trajectory`。
2. 每条边的条件、指南锚点和原文 anchor。
3. 模块2字段的 `evidence_span`、`confidence`、`missing_flags`。

**输出**

1. `reasoning_path`：图谱路径。
2. `input_facts_used`：实际消费的字段。
3. `evidence_trace`：字段 -> 原文 span -> 条件 -> 指南边。
4. `uncertainty_note`：不确定性解释。

### 2.9 最终随访建议报告生成

**输入**

1. 终态节点的 `action_desc`。
2. 证据追踪对象。
3. 缺失信息和 fallback 状态。

**输出**

结构化 JSON 和面向医生的中文报告文本。核心结论必须来自图谱终态节点或 fallback 模板，不能由自由生成模型直接决定。

---

## 3. 与开题报告的对应关系

### 3.1 一一对应

| 开题报告模块3设计 | 当前方案对应组件 | 说明 |
|---|---|---|
| 临床决策状态图 CDSG | 指南知识表示层 | 用节点和边形式化指南状态与转移条件 |
| Agent 在图谱上受限随机游走 | graph executor | 在线推理只能沿合法边转移 |
| 符号逻辑硬匹配 | hard constraints | 用布尔条件处理尺寸、密度、数量、增长等字段 |
| 语义蕴含软匹配 | soft match | 在语义鸿沟或风险属性文本不一致时补充判定 |
| 缺失信息警报 | abstention / fallback | 低置信度、缺失关键字段或冲突路径时停止自动结论 |
| 基于轨迹的循证报告生成 | 证据追踪与报告生成 | 输出建议、推理链条、指南锚点和证据 span |
| 图谱自洽性校验 | 图谱校验报告 | 检查可达性、互斥性、完备性和冲突规则 |

### 3.2 还原部分

1. 保留“指南形式化为有向图”的核心设定。
2. 保留“节点代表临床决策状态、边代表条件逻辑”的形式化方法。
3. 保留“硬匹配优先、软匹配补充”的符号-语义协同推理。
4. 保留“推理轨迹驱动报告生成”的零幻觉结论机制。
5. 保留“指南锚点 + 输入证据”的白盒解释逻辑。

### 3.3 增强部分

1. 增加 `report-intent router`，把筛查 / 偶发分流从模块1迁移到模块3入口。
2. 增加明确的 `screening`、`incidental`、`diagnostic_oncology`、`uncertain` 四类意图。
3. 增加模块2字段置信度和证据 span 的强制消费。
4. 增加 abstention / fallback 的显式工程协议。
5. 增加内部递进实验版本，避免只展示最终系统。
6. 增加外部范式 baseline，证明图谱智能体相对平铺规则和文本生成方案的价值。

### 3.4 增强为何不改变原始主线

这些增强没有改变“图谱约束 + 神经符号推理 + 循证报告”的主线，只是把原开题方案中较抽象的 Agent 推理过程拆成可实现、可评测、可复现的工程组件。`report-intent router` 属于图谱入口选择器，不替代 CDSG；fallback 属于安全边界，不替代推理；soft match 属于语义补偿，不替代硬规则。

---

## 4. 模块3输入输出协议

### 4.1 模块2到模块3的接口字段

当前仓库已有 `schemas/radiology_fact_schema.json` 和 `schemas/case_bundle_schema.json`，可作为初始接口依据。模块3正式实现前应冻结以下字段：

| 字段 | 来源 | 模块3用途 |
|---|---|---|
| `case_id` | case bundle | 推荐实例主键 |
| `subject_id` | case bundle / radiology fact | 患者级聚合 |
| `exam_name` | radiology fact | router 判断检查目的和路径 |
| `modality` | radiology fact | 判断 LDCT、CT、CTA、PET-CT 等路径 |
| `body_site` | radiology fact | 排除非胸部报告或复合报告 |
| `sections.indication` | radiology fact | 判断筛查、诊断、肿瘤随访等意图 |
| `sections.technique` | radiology fact | 识别 low-dose protocol 等筛查线索 |
| `sections.findings` | radiology fact | 证据回溯 |
| `sections.impression` | radiology fact | recommendation cue 和意图判断 |
| `nodule_count` | radiology fact | 多发结节路径 |
| `nodules[].size_mm` | nodule fact | 尺寸阈值硬约束 |
| `nodules[].size_text` | nodule fact | 报告证据展示 |
| `nodules[].density_category` | nodule fact | 实性、部分实性、磨玻璃、良性密度路径 |
| `nodules[].density_text` | nodule fact | 密度证据展示 |
| `nodules[].location_lobe` | nodule fact | 报告完整性和病例解释 |
| `nodules[].location_text` | nodule fact | 位置证据展示 |
| `nodules[].count_type` | nodule fact | 单发 / 多发路径 |
| `nodules[].change_status` | nodule fact | 新发、稳定、增大、缩小等路径修正 |
| `nodules[].change_text` | nodule fact | 变化证据展示 |
| `nodules[].calcification` | nodule fact | 良性钙化 hard constraint |
| `nodules[].spiculation` | nodule fact | 高危形态修饰 |
| `nodules[].lobulation` | nodule fact | 高危形态修饰 |
| `nodules[].perifissural` | nodule fact | 良性 perifissural 结节路径 |
| `nodules[].lung_rads_category` | nodule fact | 报告已有分级的辅助证据 |
| `nodules[].recommendation_cue` | nodule fact | silver label、cue baseline、软参照 |
| `nodules[].evidence_span` | nodule fact | 字段证据追踪 |
| `nodules[].confidence` | nodule fact | 低置信度降级或人工复核 |
| `nodules[].missing_flags` | nodule fact | 缺失字段与 fallback |
| `extraction_metadata.model_name` | radiology fact | 追踪模块2版本 |

### 4.2 模块1到模块3的辅助字段

模块1若保留，应只提供辅助风险属性，不直接决定最终随访动作：

| 字段 | 用途 |
|---|---|
| `age` | 筛查资格和指南适用性判断 |
| `sex` | 报告完整性，不作为核心路径主条件 |
| `ever_smoker_flag` | 高危筛查属性 |
| `pack_years` | 筛查资格阈值 |
| `quit_years` | 筛查资格阈值 |
| `family_history_lung_cancer` | 高危属性和 soft match |
| `occupational_exposure` | 高危属性和 soft match |
| `evidence_span` | 风险属性解释 |
| `evidence_quality` | 低置信度风险属性降级 |

### 4.3 模块3输出报告结构模板

建议在现有 `schemas/recommendation_schema.json` 基础上扩展为图谱智能体输出：

```json
{
  "case_id": "CASE-10001338-001",
  "intent": {
    "intent_type": "incidental",
    "guideline_family": "Fleischner_2017",
    "router_confidence": "high",
    "router_evidence": ["CT CHEST", "incidental pulmonary nodules"]
  },
  "recommendation": {
    "recommendation_level": "routine_screening",
    "recommendation_action": "建议 12 个月后复查低剂量胸部 CT；若稳定，可根据风险因素决定是否继续随访。",
    "followup_interval": "12_months",
    "followup_modality": "LDCT",
    "lung_rads_category": null,
    "guideline_source": "Fleischner_2017",
    "guideline_anchor": "Fleischner 2017 solid nodule pathway"
  },
  "graph_trace": {
    "entry_node": "INCIDENTAL_START",
    "terminal_node": "INCIDENTAL_SOLID_LT6_HIGH_RISK",
    "trajectory": [
      {
        "edge_id": "E_INC_SOLID",
        "condition": "density_category == solid",
        "matched": true,
        "evidence_span": "solid pulmonary nodule"
      }
    ]
  },
  "input_facts_used": {
    "nodule_size_mm": 4.0,
    "nodule_density": "solid",
    "nodule_count": 1,
    "change_status": null,
    "patient_risk_level": "high_risk",
    "smoking_eligible": "eligible"
  },
  "evidence_trace": [
    {
      "field": "size_mm",
      "value": 4.0,
      "confidence": "high",
      "evidence_span": "4 mm pulmonary nodule",
      "used_by_edge": "E_SIZE_LT6"
    }
  ],
  "missing_information": [],
  "uncertainty_note": null,
  "abstention": {
    "status": "none",
    "reason": null
  },
  "generation_metadata": {
    "engine_version": "cdsg_agent_0.1",
    "graph_version": "guideline_graph_0.1",
    "generation_timestamp": "2026-04-19T00:00:00Z"
  }
}
```

面向医生的报告文本建议固定为以下段落：

1. 病例摘要：检查类型、主要结节事实、关键风险属性。
2. 建议结论：复查间隔、复查方式、是否需要进一步诊断。
3. 推理依据：图谱路径和触发条件。
4. 原文证据：模块2 evidence span。
5. 指南依据：指南来源、条款或表格锚点。
6. 不确定性与需补充信息：缺失字段、低置信度字段、人工复核建议。

### 4.4 仍需核对的文件和字段

正式编码前需要再核对以下文件：

1. `schemas/radiology_fact_schema.json`：确认模块2最终推理导出是否仍使用 `size_mm`、`density_category`、`location_lobe`、`confidence`、`evidence_span`。
2. `schemas/case_bundle_schema.json`：确认模块3入口是否统一使用 case bundle。
3. `schemas/recommendation_schema.json`：确认是否扩展 `output_type` 以支持 `graph_agent`、`graph_agent_with_soft_match`。
4. `outputs/phaseA2/tables/a2_5_manifest_report.json`：确认模块2 A2.5 正式闭环状态和主配置名称。
5. 模块2最终 inference 导出脚本：当前仓库尚未看到明确的“把 A2.5 模型预测写成 radiology_fact_schema”的正式脚本，后续应先补一个薄适配层，而不是重跑训练。

---

## 5. 模块3实验设计

### 5.1 模块3内部比较

内部比较应展示同一图谱智能体体系的递进价值：

| 版本 | 名称 | 目的 |
|---|---|---|
| M3-V0 | cue-only extraction | 只复用报告中显式 recommendation cue，作为最低语义 baseline |
| M3-V1 | flat structured rule | 使用现有 `lung_rads_engine` 类似平铺规则，不建图 |
| M3-V2 | graph hard-only | CDSG + hard constraints，不启用 soft match |
| M3-V3 | graph hard + abstention | 增加缺失和冲突时的安全拒答 |
| M3-V4 | graph hard + soft match | 增加语义蕴含补偿 |
| M3-V5 | full graph agent report | 增加证据追踪、指南锚点和完整报告生成 |

### 5.2 同体系内部递进对比

必须至少跑以下内部递进：

1. `flat structured rule` vs `graph hard-only`：证明图谱路径不是代码形式换皮，而是带来路径一致性和可追踪性。
2. `graph hard-only` vs `graph hard + abstention`：证明安全降级降低错误确定建议。
3. `graph hard + abstention` vs `graph hard + soft match`：证明 soft match 提升覆盖率但不破坏规则正确性。
4. `graph hard + soft match` vs `full graph agent report`：证明报告完整性和解释可追溯性提升。

### 5.3 外部范式比较

外部 baseline 不应全部依赖大模型，建议分层设置：

| 类型 | baseline | 是否必须 |
|---|---|---|
| 文本 cue baseline | 从报告 recommendation cue 抽取建议 | 必须 |
| 平铺规则 baseline | 当前 `src/rules/lung_rads_engine.py` | 必须 |
| 表格 ML baseline | TF-IDF / LR 或简单 classifier 预测建议类别 | 可后置 |
| LLM zero-shot | 直接根据报告生成建议 | 可后置 |
| LLM with guideline prompt | 把指南摘要放入 prompt 后生成 | 可后置 |
| RAG-LLM | 检索指南片段后生成 | 可后置 |

必须先跑 cue-only、flat structured rule、graph hard-only、full graph agent 四个版本。LLM 类 baseline 可作为附加实验，避免网络、成本和隐私问题拖慢主线。

### 5.4 评估报告质量

模块3输出不是单标签，因此评估应覆盖结构、路径、建议和解释：

1. 结构合法性：JSON Schema 通过率。
2. 场景分流正确性：router accuracy / macro-F1。
3. 规则正确性：边条件匹配是否符合指南。
4. 路径一致性：图谱轨迹是否从合法入口到合法终态。
5. 建议正确性：随访间隔、方式、风险等级与 gold / silver 标注一致。
6. 报告完整性：是否包含摘要、建议、依据、证据、缺失信息。
7. 解释可追溯性：每个关键结论是否能追溯到 evidence span 和 guideline anchor。
8. 安全性：低置信度或缺失关键字段时是否 abstain。

### 5.5 指标定义

| 评估目标 | 指标 |
|---|---|
| 规则正确性 | edge condition accuracy、threshold violation rate、rule agreement rate |
| 路径一致性 | valid trajectory rate、terminal node validity、conflict rate、dead-end rate |
| 建议正确性 | recommendation level accuracy、interval accuracy、modality accuracy、category accuracy、macro-F1 |
| 报告完整性 | required section completion rate、schema valid rate、missing required report block rate |
| 解释可追溯性 | evidence coverage rate、guideline anchor presence rate、fact-to-edge trace rate |
| 安全降级 | abstention precision、unsafe recommendation rate、low-confidence override rate |
| 覆盖能力 | actionable rate、insufficient data rate、soft match activation rate |

### 5.6 数据集划分建议

1. `rule-unit set`：人工构造或从真实样本抽出的规则单元测试，覆盖每条关键边。
2. `case-level silver set`：由现有 recommendation cue 和规则可推导样本形成。
3. `manual audit subset`：小规模人工审查样本，用于报告质量和边界情况验证。
4. `stress set`：低置信度、缺尺寸、多发、肿瘤转移、感染、术后、矛盾描述等困难样本。

---

## 6. baseline 设计

### 6.1 内部 baseline

1. `cue_only`：只抽取报告原文建议 cue。
2. `structured_rule`：现有平铺规则版本。
3. `graph_hard_only`：图谱执行器 + 硬约束。
4. `graph_hard_abstain`：图谱执行器 + 硬约束 + abstention。
5. `graph_soft_match`：加入 soft match。
6. `graph_agent_full`：完整报告生成版本。

必须跑：`cue_only`、`structured_rule`、`graph_hard_only`、`graph_agent_full`。  
可以后置：`graph_hard_abstain`、`graph_soft_match` 的细粒度消融，但建议尽早预留接口。

### 6.2 外部 baseline

1. `zero_shot_llm`：直接输入病例事实，要求生成建议。
2. `guideline_prompt_llm`：输入病例事实和指南摘要。
3. `rag_llm`：检索指南条款后生成。
4. `classifier_recommendation`：用结构化字段直接预测建议类别。

必须跑：无，外部 LLM baseline 可后置。  
建议优先跑：`guideline_prompt_llm` 的小样本 sanity check，用于论文讨论“无约束生成”的风险。  
可不跑：成本过高或涉及联网的商业闭源大模型批量实验。

### 6.3 避免后期补实验的策略

1. 先冻结版本矩阵，所有实验输出统一写入 `outputs/module3/`。
2. 每个版本都输出同一 schema，便于统一成表。
3. 先实现 evaluation harness（评估框架），再补 full graph agent。
4. 每次新增组件必须能在同一评估集上回放。
5. 把 LLM baseline 明确写为可选附加，而不是主结论依赖项。

---

## 7. 模块3代码落地顺序

### 7.1 阶段一：基础设施

1. 新建 `src/module3/` 包。
2. 定义图谱 schema：`schemas/guideline_graph_schema.json`。
3. 定义模块3输出 schema：扩展或新增 `schemas/graph_agent_recommendation_schema.json`。
4. 建立 `data/guidelines/structured/` 或 `configs/module3/guideline_graphs/` 存放图谱 JSON。
5. 实现 graph validator：检查可达性、死端、冲突条件、重复边。
6. 实现统一日志规范，按项目要求实时写入 `logs/module3_*.log`。

### 7.2 阶段二：核心能力

1. 实现 `report_intent_router.py`。
2. 实现 condition parser / evaluator。
3. 实现 `graph_executor.py`。
4. 实现 dominant nodule selector（主导结节选择器），支持最大尺寸、最高风险、增长优先。
5. 实现 hard constraints。
6. 实现 abstention / fallback。
7. 实现 evidence tracer。

### 7.3 阶段三：增强项

1. 实现 soft match 的同义词表版本。
2. 预留 NLI 接口，但先不依赖外部大模型。
3. 增加 guideline anchor 的原文回溯。
4. 增加图谱可视化导出，例如 DOT / Mermaid。
5. 增加 case study 报告渲染。

### 7.4 阶段四：实验与成表

1. 构建 `outputs/module3/eval_sets/`。
2. 实现 `scripts/module3/run_graph_agent.py`。
3. 实现 `scripts/module3/eval_graph_agent.py`。
4. 跑内部版本矩阵。
5. 输出 `module3_main_table.csv`、`module3_ablation_table.csv`、`module3_case_study.jsonl`。
6. 写论文结果解释：强调路径一致性、可追溯性和安全降级，而不只强调 accuracy。

---

## 8. 风险点与最小可行版本

### 8.1 must-have

1. `report-intent router`。
2. 至少两个指南子图：筛查图谱和偶发图谱。
3. graph executor。
4. hard constraints。
5. abstention / fallback。
6. evidence trace。
7. 统一输出 schema。
8. 内部 baseline：cue-only、structured rule、graph hard-only、full graph agent。
9. 规则单元测试和 case-level 评估。

### 8.2 enhancement

1. NLI soft match。
2. RAG-LLM 外部 baseline。
3. 图谱自动构建中的 Judge Agent。
4. Neo4j 图数据库落地。
5. D3.js 图谱可视化。
6. 多指南冲突调和策略。
7. 医生交互式补录界面。

### 8.3 主要风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 模块2最终导出协议未冻结 | 模块3反复适配字段 | 先锁 `radiology_fact_schema` 和 case bundle |
| 图谱过大导致实现延期 | 主线无法成表 | 先做覆盖核心肺结节路径的 MVP 图谱 |
| soft match 引入不稳定性 | 破坏规则正确性 | soft match 只能补语义，不可改数值阈值 |
| LLM baseline 拖慢进度 | 延误模块3主线 | LLM baseline 后置，不作为必需主结论 |
| 缺少人工 gold 建议 | 建议正确性难评估 | 用 rule-unit set + silver cue set + manual audit subset 三层评估 |
| 退化成 flat rule engine | 与开题报告不一致 | 强制保留图谱节点、边、轨迹、validator 和路径指标 |

### 8.4 最小可行实现

最小可行模块3不需要先接入 Neo4j 或大模型，但必须满足以下形态：

1. 图谱以 JSON 存储，包含节点、边、条件、指南锚点。
2. router 能把病例分到 screening / incidental / uncertain。
3. executor 在图上走路径，而不是直接调用一组 if-else 返回建议。
4. hard constraints 能覆盖尺寸、密度、数量、变化状态、良性特征。
5. fallback 能处理缺尺寸、缺密度、低置信度和意图不明。
6. 输出报告包含建议、路径、证据 span、指南锚点和缺失信息。
7. 评估能比较 cue-only、structured rule、graph hard-only、full graph agent。

这已经保留开题报告“图谱智能体系统”的主线，同时把实现风险控制在可毕业答辩的范围内。

---

## 9. 下一步判断

模块3已经可以进入正式实现阶段。下一步应先实现模块3基础设施，包括图谱 schema、router、graph executor、hard constraints、fallback 和评估 harness，而不是继续补模块2大规模训练。

模块2当前只需要补一个薄接口层：把 A2.5 最终模型或现有 schema 输出稳定转换为模块3 case bundle。除非发现 `incomplete_entries` 或主表闭环有新的硬错误，不应重新占用主线去补跑模块2训练。
