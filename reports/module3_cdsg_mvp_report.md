# Module 3 CDSG MVP Run Report

> 日期：2026-05-07
> 范围：M3-1 CDSG MVP 基础设施落地。未训练、未运行 GPU、未接入 LLM/NLI/RAG。

## 1. 本轮目标

本轮把已有 flat Lung-RADS rule baseline 升级为可追踪的 CDSG（Clinical Decision State Graph）hard-rule executor。实现目标不是自由文本生成，而是在图节点和图边约束下，根据结构化事实输出终态模板推荐、实际推理路径、指南锚点、缺失信息、证据质量和 abstention 原因。

## 2. 新增基础设施

| 产物 | 作用 |
|---|---|
| `schemas/module3_case_bundle_schema.json` | 定义 Module3 CDSG executor 的标准输入；兼容现有 Phase4 `case_bundles_eval.jsonl`。 |
| `schemas/module3_recommendation_schema.json` | 定义标准输出，包含 `recommendation`、`lung_rads_category`、`reasoning_path`、`guideline_anchor`、`missing_info`、`evidence_quality`、`decision_path`、`visited_nodes`、`abstention_reason`。 |
| `outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json` | Lung-RADS v2022 MVP CDSG 图定义，包含节点、边、hard conditions、终态推荐、abstention 终态和 guideline anchors。 |
| `src/rules/cdsg_executor.py` | 通用 CDSG 图执行器；不调用模型，不生成自由文本，只执行 hard-rule 图路径并渲染终态模板。 |
| `scripts/phaseA3/run_cdsg_executor.py` | 批量读取 Phase4 case bundle，输出 CDSG recommendations 和 summary。 |
| `scripts/phaseA3/compare_cdsg_with_flat_lung_rads.py` | 对比 CDSG executor 与旧 flat Lung-RADS baseline。 |
| `tests/test_cdsg_executor.py` | CDSG 单元测试，覆盖 solid、part-solid、ground-glass、缺失 size、缺失 density、schema 和 guideline anchor。 |

## 3. CDSG 图覆盖范围

当前图覆盖的是现有 flat Lung-RADS baseline 已支持的最小确定性规则范围：

1. `solid` 非新发阈值：`<6 mm -> 2`，`6-7 mm -> 3`，`8-14 mm -> 4A`，`>=15 mm -> 4B`。
2. `solid` 新发阈值：`<4 mm -> 2`，`4-7 mm -> 3`，`8-14 mm -> 4A`，`>=15 mm -> 4B`。
3. `part_solid`：新发 -> `3`；非新发 `<6 mm -> 2`；`>=6 mm` 按 solid component 分支，缺失 solid component 时保守 `4A`。
4. `ground_glass`：`<30 mm -> 2`，`>=30 mm -> 3`，新发与非新发分别有锚点。
5. `calcified` / `fat_containing` 良性密度 -> `2`。
6. modifier：稳定至少 2 年降级到 `2`；增长上调一级；缩小维持当前类别。
7. dominant nodule 选择：优先选择最高终态严重度，其次选择更大 size。

明确不覆盖：

1. soft match / NLI / LLM / RAG。
2. Lung-RADS 4X、S、完整准入筛查语义。
3. Fleischner、中国指南、多指南冲突裁决。
4. 对缺失 density 的自由补全。M3-1 按开题报告“关键事实缺失不得胡乱生成”的原则，对 missing/unclear density 显式 abstain。

## 4. Phase4 批量运行结果

输入：`outputs/phase4/cache/case_bundles_eval.jsonl`
输出：`outputs/phaseA3/recommendations/cdsg_phase4_recommendations.jsonl`

| 指标 | 结果 |
|---|---:|
| 输入 case 数 | 253 |
| 成功输出 recommendations | 253 |
| 运行错误 | 0 |
| Module3 recommendation schema valid rate | 1.000000 |
| guideline anchor 非空率 | 1.000000 |
| reasoning path 非空率 | 1.000000 |
| abstention count | 205 |
| abstention rate | 0.810277 |

CDSG 输出分布：

| 类别 | 数量 |
|---|---:|
| insufficient_data | 205 |
| routine_screening | 34 |
| short_interval_followup | 12 |
| tissue_sampling | 2 |

Lung-RADS 分布：

| Lung-RADS | 数量 |
|---|---:|
| `2` | 34 |
| `3` | 8 |
| `4A` | 4 |
| `4B` | 2 |
| `null` | 205 |

Abstention 原因：

| 原因 | 数量 |
|---|---:|
| `missing_nodule_density` | 122 |
| `missing_nodule_size` | 76 |
| `no_structured_nodule` | 7 |

## 5. 与 flat Lung-RADS baseline 的一致性

对比脚本：`scripts/phaseA3/compare_cdsg_with_flat_lung_rads.py`
输出：

1. `outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_comparison.csv`
2. `outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_mismatches.csv`

总体一致性：

| 指标 | 结果 |
|---|---:|
| total cases | 253 |
| exact match count | 109 |
| exact match rate | 0.430830 |
| recommendation match count | 109 |
| recommendation match rate | 0.430830 |
| category match count | 109 |
| category match rate | 0.430830 |
| CDSG schema valid rate | 1.000000 |

Hard-rule 可执行子集：

| 指标 | 结果 |
|---|---:|
| CDSG hard-evaluable count | 48 |
| hard-evaluable recommendation match rate | 0.541667 |
| hard-evaluable category match rate | 0.541667 |
| strict hard-comparable count | 20 |
| strict hard-comparable recommendation match rate | 1.000000 |
| strict hard-comparable category match rate | 1.000000 |

说明：

1. `hard-evaluable` 表示 CDSG 最终不是 abstention。
2. `strict hard-comparable` 表示 CDSG 不是 abstention，且旧 flat baseline 的 dominant path 没有依赖 `density_category` 缺失时按 `solid` 回退。
3. strict hard-comparable 子集 100% 等价，说明已覆盖的确定性图边/终态模板与 flat rule 在明确事实上保持一致。

## 6. Mismatch 分析

Mismatch 共 144 条，原因分布：

| mismatch reason | 数量 |
|---|---:|
| `case_bundle_density_missing_or_unclear__cdsg_abstains__flat_solid_fallback` | 122 |
| `dominant_selection_delta_from_flat_density_fallback` | 22 |

结论：

1. 主要 mismatch 不是 CDSG 图定义缺失，而是 Phase4 case bundle 的 `density_category` 大量缺失或为 `unclear`。
2. 旧 flat baseline 的简化逻辑会把 missing/unclear density 保守当作 `solid` 继续给出 Lung-RADS category。
3. M3-1 CDSG 按开题报告约束，不把关键 density 事实自由补全为 `solid`，因此在 122 条 case 上 abstain。
4. 另外 22 条 case 中，flat baseline 选择了“density 缺失但按 solid 回退”的更高危 dominant nodule；CDSG 选择了明确 density 的可执行结节，因此出现 dominant selection 差异。

## 7. Schema 与测试

已执行：

```bash
conda run -n follow-up python -m py_compile src/rules/cdsg_executor.py scripts/phaseA3/run_cdsg_executor.py scripts/phaseA3/compare_cdsg_with_flat_lung_rads.py
conda run -n follow-up pytest -q tests/test_cdsg_executor.py tests/rules/test_lung_rads_engine.py
```

结果：

1. 新增 Python 文件语法检查通过。
2. `21 passed, 2 warnings in 0.21s`。
3. 253 条 Phase4 case bundle 对 `module3_case_bundle_schema.json` 校验全部通过。
4. 253 条 CDSG recommendation 对 `module3_recommendation_schema.json` 校验全部通过。

## 8. 是否足够进入 M3-2

可以进入 M3-2 数据集构建，但不建议直接进入正式实验或训练。

M3-2 可以基于本轮产物做：

1. 构建 `strict_hard_comparable` 子集，作为 CDSG hard-rule 等价性 sanity set。
2. 构建 `missing_density_abstention` 子集，作为模块2 density 改进和 semantic fallback 的目标集。
3. 构建 `missing_size_abstention` 子集，作为高保真 size extraction 的目标集。
4. 将 `visited_nodes`、`matched_edges`、`guideline_anchor`、`abstention_reason` 写入数据集字段，支持后续 path validity 和 evidence grounding 评估。

进入 M3-2 前的关键决策：

1. 是否保留当前严格 CDSG 策略：missing/unclear density 一律 abstain。
2. 是否额外提供一个 `flat_compatibility_mode`，仅用于复现实验 baseline，不作为最终临床推理策略。
3. 是否优先修复模块2 density extraction，以减少 122 条 density abstention。
