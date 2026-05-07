# 模块3初始实验设计

> 日期：2026-05-07  
> 约束：本文件只给实验设计初稿；未训练、未运行 GPU、未启动模块3实验。

## 1. 任务定义确认

模块3的任务不是普通推荐分类，也不是从报告自由生成建议文本，而是构建“神经符号临床推理智能体”。系统输入为模块1/2产出的结构化临床事实，核心过程是在指南知识图谱/临床决策状态图（CDSG）上进行受约束路径推理，最终用模板槽位填充输出标准化循证报告。

核心原则：

1. 结构化事实优先，尤其是 `size_mm`、`density_category`、`location_lobe`、`change_status`、`evidence_span`。
2. 数值和枚举条件必须 hard match，禁止 LLM 改写阈值。
3. 只有在硬匹配失败且存在语义鸿沟时才触发 soft match。
4. 最终建议必须来自指南终态节点或 fallback 模板。
5. 所有输出必须可追溯到输入证据和指南锚点。

## 2. 最终输出 schema 草案

建议在现有 `schemas/recommendation_schema.json` 基础上扩展为 `graph_agent_recommendation` 兼容格式。

```json
{
  "case_id": "CASE-10001401-001",
  "recommendation": {
    "recommendation_level": "short_interval_followup",
    "recommendation_action": "建议 3 个月后复查 LDCT，必要时补充 PET-CT。",
    "followup_interval": "3_months",
    "followup_modality": "LDCT"
  },
  "risk": {
    "lung_rads_category": "4A",
    "risk_category": "suspicious",
    "guideline_family": "Lung-RADS_v2022"
  },
  "reasoning_path": [
    {
      "node_id": "LR_START",
      "edge_id": "E_SOLID",
      "condition": "density_category == solid",
      "match_type": "hard",
      "matched": true,
      "evidence_span": "solid right upper lobe pulmonary nodule"
    }
  ],
  "guideline_anchor": [
    {
      "anchor_id": "Lung-RADS_v2022_4A_solid_8_14",
      "source": "Lung-RADS_v2022",
      "condition": "solid nodule >= 8 mm and < 15 mm"
    }
  ],
  "missing_info": [],
  "confidence": {
    "overall": "medium",
    "fact_quality": "predicted",
    "evidence_quality": "span_grounded",
    "soft_match_score": null
  },
  "safety": {
    "abstention_status": "none",
    "harmful_risk_flag": false,
    "under_followup_flag": false
  },
  "input_facts_used": {
    "nodule_size_mm": 10.0,
    "nodule_density": "solid",
    "nodule_count": 1,
    "change_status": null,
    "patient_risk_level": "unknown"
  },
  "generation_metadata": {
    "engine_version": "cdsg_agent_0.1",
    "graph_version": "lung_rads_cdsg_0.1",
    "output_type": "graph_agent"
  }
}
```

必要字段：

| 字段 | 含义 | 当前仓库基础 |
|---|---|---|
| `recommendation` | 标准化随访建议 | 已有 `recommendation_schema`。 |
| `risk` / `lung_rads_category` | 风险分层或 Lung-RADS 分类 | 已有 flat rule 输出。 |
| `reasoning_path` | 图谱节点/边轨迹和触发条件 | 当前只有字符串列表，需要升级为结构化轨迹。 |
| `guideline_anchor` | 指南条款锚点 | 当前有字符串锚点，需要细化为条款 ID。 |
| `missing_info` | 阻塞或降低置信度的缺失字段 | 当前有 `missing_information`。 |
| `confidence` / `evidence_quality` | 输入事实质量和软匹配质量 | 当前散落在 nodule `confidence` 与 smoking `evidence_quality`。 |

## 3. 主实验方法分组

| 组别 | 方法名 | 作用 | 当前可行性 |
|---|---|---|---|
| G0 | Guideline Rule Reference | 使用人工/规则从 oracle facts 推导标准答案，作为 reference 或银标准 | 部分可行；现有 `lung_rads_engine.py` 可作 Lung-RADS reference。 |
| B1 | Text-only PLM Baseline | 输入原始报告文本，预测 recommendation category/action | 模块2已有 PLM 基础，但模块3标签缺失，需先构建标签。 |
| B2 | Structured-only Rule Engine | 输入 `case_bundle`，运行现有 flat Lung-RADS rule engine | 已有，Phase 4 已跑 structured_rule。 |
| B3 | LLM-only Prompt Baseline | 将 case facts 和指南提示输入本地 LLM，直接生成结构化建议 | 当前不执行；需确认模型和 GPU 策略。 |
| B4 | RAG / Guideline Retrieval Baseline | 检索指南条款后让 LLM 输出建议 | 当前不执行；需先有指南 chunk 和本地 LLM。 |
| Ours | MWS-CFE / CDSG-constrained Neuro-symbolic Agent | 模块2结构化事实 + CDSG + hard match + semantic fallback + abstention | 目标方法；必须新建 graph schema、图定义和 executor。 |

建议内部递进版本：

| 版本 | 名称 | 目的 |
|---|---|---|
| M3-V0 | FlatRule | 现有 `lung_rads_engine.py`，作为可运行下界。 |
| M3-V1 | CDSG-Hard | Lung-RADS 图定义 + hard constraint executor；结果应与 FlatRule 在 hard-only 子集 100% 一致。 |
| M3-V2 | CDSG-Hard+Anchor | 在每个节点/边输出结构化 guideline anchor。 |
| M3-V3 | CDSG-Hard+Abstention | 对缺失/冲突/低置信度事实显式 abstain 或 conservative fallback。 |
| M3-V4 | CDSG+SemanticFallback | 只在语义鸿沟条件上启用 soft match。 |

## 4. 主指标

| 指标 | 定义 | 评估对象 |
|---|---|---|
| Recommendation Accuracy / Macro-F1 | 推荐类别、随访间隔或 Lung-RADS category 与 reference/gold 一致性 | 主结果。 |
| Guideline Concordance | 输出建议是否符合指定指南条款 | 规则校验 + 人工抽样。 |
| Path Validity | 推理路径是否沿合法节点/边转移，是否无跳步、无矛盾 | CDSG 方法核心指标。 |
| Harmful Recommendation Rate | 输出比 reference 更激进且可能造成伤害的比例 | 安全指标。 |
| Under-follow-up Rate | 输出比 reference 更保守/随访不足的比例 | 临床安全关键指标。 |
| Evidence Grounding Accuracy | `input_facts_used` 和 `reasoning_path.evidence_span` 是否支持触发条件 | 证据锚定指标。 |
| Valid Schema Rate | 输出是否满足 schema | 工程可靠性指标。 |
| Abstention Correctness | 信息不足时是否正确拒答/提示补充 | fallback 质量指标。 |
| Missing-info Accuracy | 输出缺失字段是否确实是路径阻塞字段 | 缺失信息质量。 |

## 5. 双层评估协议

### 5.1 Oracle Structured Facts

输入使用人工校验或强银标准结构化事实，评估 CDSG 本身的决策逻辑。

候选来源：

1. `reports/schema_examples.md` 中的手工映射案例。
2. `outputs/phase4/manifests/recommendation_eval.json` 的 `rule_derived` 子集。
3. 后续人工抽样标注的 100-200 条 case-level gold facts。

目的：隔离上游抽取误差，验证图谱推理、路径有效性和指南一致性。

### 5.2 Predicted Module2 Facts

输入使用模块2预测事实，评估端到端链路。

现有基础：

1. `outputs/phase4/cache/radiology_facts_eval.jsonl`：500 条 regex/section-aware 事实。
2. `outputs/phase4/cache/case_bundles_eval.jsonl`：253 条 case bundle。
3. `outputs/phase5_1/gold_eval_metrics.json`：模块2 gold eval 62 条，密度 macro-F1 约 0.7003，size ±1mm 覆盖率约 0.9074。

当前缺口：没有把 Phase A2/Phase 5 模型预测正式导出为 `radiology_fact_schema` 的模块3入口适配层。

### 5.3 Predicted Facts + Semantic Fallback

在 predicted facts 上启用 soft match，只允许处理以下场景：

1. 高危准入语义：如“heavy smoker”“父亲死于肺部占位”与指南条件之间的蕴含。
2. 术语映射：如 “subsolid”“mixed attenuation” 与 `part_solid` / `ground_glass` 的映射。
3. 模糊证据：如 “possibly part-solid” 进入低置信度候选，不直接覆盖 hard value。

禁止项：

1. 不能把 `6 mm` 软匹配成 `8 mm`。
2. 不能在缺失 size 时让 LLM 估计尺寸。
3. 不能让 LLM 直接决定最终随访间隔。

## 6. 数据与标签设计

| 数据层 | 当前状态 | 模块3用途 | 缺口 |
|---|---|---|---|
| Phase 4 recommendation_eval | 464 个 note-level 样本，含 explicit_cue/rule_derived/insufficient | 初始弱评估集 | 运行时聚合后只有 253 个 case bundle；不是 gold。 |
| Phase 4 case_bundles_eval | 253 条 case bundle | FlatRule/CDSG 输入 | `split` 仍为 unlabeled，缺 case-level gold label。 |
| Phase 5 datasets | 模块2提及级 train/val/test，subject disjoint | predicted facts 上游基础 | 不是模块3推荐标签。 |
| Phase 5.1 gold eval | 62 条人工评估模块2字段 | 可抽样构建 oracle facts | 数量小，尚无随访建议 gold。 |

建议先构建三个模块3集合：

1. `module3_oracle_small_gold`：100-200 条人工校验 case，覆盖 Lung-RADS 边界和 incidental 小结节。
2. `module3_rule_derived_silver`：从 Phase 4/5 中抽取事实完整、规则可确定的 case。
3. `module3_predicted_facts_eval`：模块2预测事实导出为 `case_bundle` 后的端到端评估集。

## 7. 开始写代码前的最小前置条件

可以开始模块3基础设施代码，但不应直接写训练脚本或启动 GPU 实验。建议第一批代码只做：

1. `schemas/guideline_graph_schema.json`
2. `configs/module3/guideline_graphs/lung_rads_v2022.json`
3. `src/module3/graph_executor.py`
4. `src/module3/hard_constraints.py`
5. `src/module3/report_renderer.py`
6. `tests/module3/test_lung_rads_cdsg_equivalence.py`

第一批验收标准：

1. CDSG hard-only 输出与现有 FlatRule 在 Phase 4 rule-derived 子集上等价。
2. 输出满足扩展 recommendation schema。
3. 所有结果包含结构化 reasoning path、guideline anchor、missing info。
4. 缺失 size 或关键条件冲突时明确 abstain，不生成确定建议。
