# Module 3 Abstention / Conflict / Candidate Aggregation Audit

## 审计范围

本轮审计使用 M3-2.6 anchor-based Module 2 prediction exports，以及当前
`module3_ready_case_bundles_v2.jsonl` / `module3_strong_silver_v2.jsonl`。

未运行训练、GPU 推理、LLM、NLI 或 RAG。

## 当前 Abstention 来源

当前 CDSG strong-silver v2 汇总如下：

| Item | Count |
|---|---:|
| Total cases | 253 |
| Actionable | 76 |
| Abstention | 177 |
| missing_nodule_density | 104 |
| missing_nodule_size | 66 |
| no_structured_nodule | 7 |

因此，177 条 abstention 已经不是 prediction rows 缺失导致，而是恢复出的事实仍不满足 CDSG hard-rule 的严格准入条件。

## Missing Density 诊断

在 104 条 `missing_nodule_density` case 中：

| Diagnostic | Count |
|---|---:|
| Any density_stage1 explicit prediction | 7 |
| Any density_stage2 subtype prediction | 104 |
| Stage2 subtype present but stage2_applicable=false / stage1 non-explicit | 97 |
| Any candidate with density | 8 |
| Any candidate with size | 104 |
| Root cause: stage1 no explicit density, stage2 subtype not applicable | 97 |
| Root cause: size and density split across candidates | 7 |

解释：

- Stage2 对每条 anchor mention 都会给出 subtype，但 97/104 个 missing-density case 没有通过 stage1 explicit-density gating。按开题报告中的硬匹配设计，这些 stage2 subtype 不应直接提升为 CDSG 事实。
- 7 个 case 有 explicit density 证据，但 size 和 density 分散在不同 candidate 上，因此 CDSG 选中的 dominant candidate 仍缺 density。
- 少数 case 内确实有 density candidate，但这不足以支持无 provenance、无风险标注的 case-level 合并。

## Missing Size 诊断

在 66 条 `missing_nodule_size` case 中：

| Diagnostic | Count |
|---|---:|
| Any size_mm prediction | 1 |
| Any candidate with size | 0 |
| Root cause: no size_mm prediction | 65 |
| Root cause: size_mm prediction not written to candidate | 1 |

解释：

- size 瓶颈是当前 Module 2 export 中真实缺少 `size_mm`，不是 CDSG executor 行为问题。
- `has_size` probability 不能转换成 `size_mm`；这样做会违反“不凭空填值”的约束。
- aggregation simulation 没有降低 `missing_nodule_size`。

## No Structured Nodule 诊断

7 条 `no_structured_nodule` case 在 anchor prediction set 中均没有 Phase4-aligned pulmonary prediction cue。除非后续 extraction pass 或人工审查恢复有效 pulmonary nodule evidence，否则应继续 abstain。

## Conflict 诊断

Adapter v2 conflict 汇总如下：

| Item | Count |
|---|---:|
| Total conflict rows | 425 |
| Conflict cases | 150 |
| Conflict case-notes | 280 |
| location_lobe conflicts | 270 |
| size_mm conflicts | 130 |
| density_category conflicts | 25 |
| Aggregation-resolvable rows | 275 |
| Manual-review rows | 150 |
| High-risk rows | 25 |
| Medium-risk rows | 400 |

解释：

- 大多数 conflict 来自 location 或 size。location 对当前 CDSG 路径影响较低，但会影响 evidence grounding。
- size conflict 多数可以通过 largest-size dominant-nodule policy 解释，但 16 行包含多个 size value，建议人工抽查。
- 25 行 density conflict 为 high risk，因为 density subtype 会改变 CDSG 路径，不能静默解决。

## Aggregation Simulation

| Strategy | Actionable | Delta | Missing density | Delta | Missing size | High-risk candidates |
|---|---:|---:|---:|---:|---:|---:|
| Original v2 | 76 | 0 | 104 | 0 | 66 | 0 |
| same-case union | 84 | +8 | 96 | -8 | 66 | 4 |
| same-note union | 80 | +4 | 100 | -4 | 66 | 0 |
| dominant size-first + density-nearest | 84 | +8 | 96 | -8 | 66 | 4 |
| confidence-gated same-note union | 79 | +3 | 101 | -3 | 66 | 0 |

解释：

- same-case union 和 density-nearest 恢复 case 最多，但恢复出的候选中包含 high-risk cross-note / cross-mention union。
- same-note union 更安全，但只恢复 4 个 case。
- confidence-gated same-note union 最安全，只恢复 3 个 case，收益不足以显著改变数据集。
- 没有任何 aggregation strategy 能降低 `missing_nodule_size`。

## 建议

不建议把 same-case union 或 dominant size-first + density-nearest 纳入正式 v2 数据构建。它们适合作为诊断，但对 hard-rule labels 来说风险过高。

最安全的可选升级是 conservative same-note、confidence-gated aggregation variant，并明确标注为 simulation 或 v2.1，且需要先做人工抽查。当前 M3-3 hard-match deterministic evaluation 应使用官方 v2 strong-silver labels，并把 abstention 作为一等结果报告。

## M3-3 准入判断

M3-3 只能以 hard-match deterministic evaluation 形式推进，评估范围应包括：

- 76 条 actionable strong-silver cases；
- 253 条全量 case，并显式报告 abstention；
- abstention / error analysis 表。

当前不应进入 learned-model main experiment。仍需要小规模人工 gold / audit set，尤其覆盖：

- density stage1 false negatives；
- density conflicts；
- size-density split candidates；
- 仅通过 aggregation simulation 恢复的 case。
