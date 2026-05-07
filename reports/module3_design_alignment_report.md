# 模块3设计对齐审计报告

> 日期：2026-05-07  
> 范围：仅做文档、代码、数据静态审计与设计对齐；未训练、未启动 GPU、未运行实验。

## 1. 审计材料

本次审计覆盖以下模块3相关材料：

| 类别 | 文件 | 结论 |
|---|---|---|
| 开题报告 | `docs/刘俊希-开题报告.docx` | 模块3正式定位的最高优先级来源。 |
| 项目理解稿 | `docs/项目三内容理解稿.md` | 与开题报告一致，强调 CDSG、硬匹配、软匹配、轨迹报告。 |
| 模块3执行计划 | `docs/module3_graph_agent_execution_plan.md` | 已把开题报告抽象设计拆成 router、guideline graph、graph executor、soft match、abstention 等工程组件。 |
| 最终实验重构计划 | `reports/final_experiment_redesign_plan.md` | 明确指出当前仅有 flat rule engine，必须升级为最小图谱智能体。 |
| Phase 3 baseline 文档 | `reports/phase3_baseline_design.md`、`reports/phase3_smoke_test.md` | 记录了可运行的端到端 baseline，但定位是原型闭环。 |
| 论文草稿 | `reports/thesis_chapter3_draft.md` | 已写成神经感知 + 符号规则推理路线，但尚未充分体现 CDSG 智能体。 |
| deep-research-report | 未发现同名或等价文档 | 仅发现 `outputs/phase5/results/deep_analysis/*`，内容属于模块2深度分析，不是模块3调研报告。 |

## 2. 模块3在开题报告中的正式定位

开题报告中，模块3是“基于因果解耦与神经符号推理的随访建议生成智能体”，不是普通分类任务，也不是自由文本生成任务。其正式任务定义应理解为：

1. 输入来自上游模块的结构化临床事实，包括吸烟史/高危准入属性、结节大小、密度、位置、数量、变化状态、证据 span 与置信度。
2. 系统核心是神经符号临床推理智能体，在临床决策状态图（Clinical Decision State Graph, CDSG）上执行受约束路径推理。
3. 指南被形式化为确定性状态节点和逻辑转移边；Agent 只能沿合法边转移。
4. 推理策略是“符号硬匹配 + 语义软匹配”：明确数值和分类字段优先用布尔逻辑；硬匹配失败且存在语义鸿沟时，再用向量检索、自然语言推理或冻结 LLM 做软判定。
5. 最终建议不由 LLM 自由生成，而由完整决策轨迹驱动模板槽位填充，输出 `recommendation`、`reasoning_path`、`guideline_anchor`、`missing_info` 等循证报告字段。

开题报告要解决的三个核心问题是：

| 困境 | 开题报告要求 | 对模块3的含义 |
|---|---|---|
| 准入判定难 | 处理“重度吸烟”和“吸烟指数>=30包年”之间的定性-定量语义鸿沟 | 模块1风险属性可作为辅助输入；语义软匹配只能补足高危准入语义，不能替代数值硬规则。 |
| 信息提取难 | 依赖事实库和高保真抽取，避免数值幻觉 | 模块3必须消费结构化事实和证据 span，不得重新从文本中自由生成 size/density。 |
| 决策合规难 | CDSG 把指南转成可执行图结构 | 模块3核心贡献应是图谱约束推理，而不是 flat IF-THEN 分类器。 |

## 3. 当前仓库与开题报告一致的部分

| 一致点 | 当前实现/产物 | 对齐说明 |
|---|---|---|
| 结构化事实作为决策输入 | `schemas/radiology_fact_schema.json`、`schemas/case_bundle_schema.json`、`src/assemblers/case_bundle_assembler.py` | 已形成 `radiology report -> radiology_fact -> case_bundle -> recommendation` 的接口链。 |
| 避免自由文本直接决策 | `src/rules/lung_rads_engine.py` | 当前建议由规则引擎产生，不是 LLM 自由生成。 |
| 明确数值/密度硬规则 | `src/rules/lung_rads_engine.py` | 已实现 Lung-RADS v2022 最小规则子集，覆盖 solid、part_solid、ground_glass、calcified 等路径。 |
| 输出指南锚点与推理路径 | `schemas/recommendation_schema.json`、Phase 4 `recommendations_structured_rule.jsonl` | 结构化规则输出包含 `guideline_anchor`、`reasoning_path`、`triggered_rules`、`missing_information`。 |
| 缺失信息显式输出 | `lung_rads_engine.py` 的 `insufficient_data` 与 `missing_information` | 对 `size_mm` 缺失已有保守退回；对 `density_category` 缺失有保守 solid fallback。 |
| 最小闭环可运行 | `scripts/run_phase3_demo.py`、`reports/phase3_smoke_test.md` | Phase 3 demo 已验证 25 个 case bundle 和 25 条 recommendation schema 全部合法。 |
| Phase 4 推荐 baseline | `scripts/eval_recommendation_baseline.py`、`outputs/phase4/results/*recommendation*` | 已有 cue-only 与 structured-rule 两类 baseline 结果。 |

## 4. 当前仓库与开题报告的偏差

| 偏差 | 当前状态 | 风险 |
|---|---|---|
| 缺少显式 CDSG | 未发现 `guideline_graph.json`、CDSG schema 或图定义文件 | 不能证明“在图谱上受约束推理”，只能证明 flat rule baseline。 |
| 缺少 graph executor | 未发现 graph walker / executor 实现 | 无法输出真实节点-边轨迹，也无法做 path validity 指标。 |
| 缺少 soft match | 未发现向量检索、NLI、冻结 LLM 语义蕴含组件 | 尚不能解决开题报告强调的语义鸿沟。 |
| 缺少 report-intent router | 当前默认 Lung-RADS v2022 | 无法区分 screening、incidental、diagnostic oncology 场景；Fleischner/中国指南路径未落地。 |
| 缺少多指南图谱 | 仅有 Lung-RADS 最小规则子集 | 与开题报告“指南知识图谱/临床决策图”相比覆盖不足。 |
| 论文草稿降级过度 | `thesis_chapter3_draft.md` 主要写“神经感知层 + 决策树规则引擎” | 叙事会把模块3误写成普通规则系统，削弱开题报告创新点。 |
| 缺少模块3专用数据 split | `case_bundle.split` 当前多为 `unlabeled`；Phase 5 split 是模块2提及级 split | 无法直接报告模块3 train/val/test 推荐生成指标。 |
| 缺少随访建议 gold label | Phase 4 只有 explicit cue、rule-derived、insufficient 三类弱评估样本 | recommendation accuracy、harmful rate 等高价值指标需要 gold 或强银标准。 |
| LLM/RAG baseline 未实现 | 文档计划存在，代码不存在 | 外部范式对比暂不能执行。 |

## 5. 必须保留的设计

以下设计是开题报告主线，不能在模块3实现中删除或长期省略：

1. `case_bundle` 作为模块3唯一统一输入，包含结构化事实、证据 span、置信度和缺失字段。
2. CDSG 或等价的显式指南图结构：节点代表临床状态/推荐终态，边代表条件逻辑。
3. 符号硬匹配优先：`size_mm`、`density_category`、`change_status`、`count_type` 等明确事实必须由布尔条件触发。
4. 软匹配只处理语义鸿沟，不能覆盖或改写数值阈值。
5. 完整推理轨迹：输出节点、边、触发条件、输入证据、未满足条件。
6. 指南锚点：每个终态建议和关键转移都要可回溯到 Lung-RADS/Fleischner/中国指南条款。
7. 模板槽位填充生成报告：最终建议和解释来自图谱终态与轨迹，不由 LLM 自由决定。
8. `missing_info` 和 abstention：信息不足时应保守输出，不强行给随访间隔。

## 6. 可降级为 MVP 的设计

| 开题报告完整设计 | MVP 可接受降级 | 原因 |
|---|---|---|
| LLM 自动解析指南构建图谱 | 先手工编码 Lung-RADS v2022 图定义 | 医疗指南图谱自动构建风险高；手工图更可控。 |
| Neo4j 图数据库 | JSON/YAML 图定义 + Python executor | 模块3第一轮图规模小，图数据库不是必要条件。 |
| Lung-RADS + Fleischner + 中国指南 | 第一轮只做 Lung-RADS v2022 CDSG | 先保证 hard constraint 与现有 flat rule 100% 对齐。 |
| 向量检索 + NLI + 冻结 LLM 全部上线 | 第一轮 soft match 可用同义词表/规则 mock，NLI 放增强项 | 不阻塞图谱主干验收。 |
| LLM-only 与 RAG-LLM 完整外部对比 | 可先设计协议，待本地模型条件确认后再补跑 | 当前用户明确要求不跑 GPU、不写实验脚本。 |
| 专家 gold 全量标注 | 先用 Phase 4 rule-derived + explicit cue 作为银标准，抽样人工复核 | 现有仓库没有模块3 gold 标签。 |

## 7. 当前结论

当前仓库已经具备模块3前置资产：结构化事实 schema、case bundle、Lung-RADS flat rule baseline、Phase 4 弱评估集、推荐 schema 和基础指标。但它还没有达到开题报告定义的“神经符号临床推理智能体”完成态。下一步可以开始写模块3基础设施代码，但不应直接进入训练或 GPU 实验；首要工作应是 CDSG 图 schema、Lung-RADS 图定义、graph executor、hard-rule 等价性测试、abstention 协议和模块3评估标签构建。
