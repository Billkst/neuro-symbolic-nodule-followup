# 模块2 Plan B 全盘重构执行文档

日期：2026-04-19

状态：模块3暂停；模块2进入方法级与评测级重构执行阶段。

## 0. 执行结论

当前模块2不再沿用旧 A2.5 单阶段 density 五分类主表作为论文定稿口径。Plan B 的正式闭环已经固定为：

1. density 改为 two-stage density。
2. Ours 修复为真正使用 `ws_confidence` 的 confidence-aware training。
3. baseline 面板扩展到 6 类方法。
4. 消融改为 `Full vs w/o ...` 标准格式。
5. 参数讨论固定为 P1/P2/P3，并输出图。
6. 旧单阶段 density 五分类只保留为诊断表或附录，不再作为主表结论来源。

## 1. 新模块2任务定义

### 1.1 Density：two-stage density

Stage 1 是二分类：

| 类别 | 定义 |
|---|---|
| `explicit_density` | 文本中存在可归入正式 density subtype 的明确证据 |
| `unclear_or_no_evidence` | `unclear`、无密度证据、LF abstain、或无法答辩为明确 subtype 的样本 |

Stage 2 只在 explicit subset 上做 subtype classification：

| 类别 |
|---|
| `solid` |
| `part_solid` |
| `ground_glass` |
| `calcified` |

### 1.2 数据构造

新增脚本：`scripts/phaseA2/build_planb_density_two_stage.py`。

正式构造规则：

1. 训练集从 `outputs/phaseA1/density/ws_train.jsonl` 的完整 WS 记录构造，而不是只从旧的 non-ABSTAIN 训练文件构造。
2. Stage 1 训练：
   - `ws_label in {solid, part_solid, ground_glass, calcified}` -> `explicit_density`
   - 其他标签、`unclear`、`__ABSTAIN__` -> `unclear_or_no_evidence`
   - `__ABSTAIN__` 负类样本保留，默认 `ws_confidence=0.5`，避免负类 loss 被 0 权重完全抹掉。
3. Stage 2 训练：
   - 只保留 `ws_label` 属于 4 个 explicit subtype 的样本。
4. 验证/测试：
   - 从 `outputs/phase5/datasets/density_val.jsonl` 与 `density_test.jsonl` 派生。
   - Stage 1 使用全量样本。
   - Stage 2 只使用 explicit subset。
5. G1-G5 gate：
   - explicit 正类按原 `passed_gates` 过滤。
   - Stage 1 负类在所有 gate 下保留，因为它们是 `unclear_or_no_evidence` 判别所需的正式负类。

默认输出：

```bash
outputs/phaseA2_planB/density_stage1/
outputs/phaseA2_planB/density_stage2/
outputs/phaseA2_planB/density_two_stage_summary.json
```

### 1.3 Density 指标

Stage 1 正式指标：

| 指标 | 用途 |
|---|---|
| AUPRC | 主指标，适合 explicit density 不均衡检测 |
| AUROC | 辅助判别指标 |
| Precision / Recall / F1 | 解释 explicit 检出质量 |
| Accuracy | 只作辅助，不作为主结论 |

Stage 2 正式指标：

| 指标 | 用途 |
|---|---|
| Macro-F1 | 主指标，衡量 subtype 均衡性能 |
| Accuracy | 辅助 |
| per-class F1 | 附录或错误分析 |

### 1.4 Has_size

`has_size` 保留当前二分类定义，不作为 Plan B 的主要重构风险点。

正式指标：

1. F1 为主指标。
2. Precision/Recall/Accuracy 为辅助指标。

处理原则：

1. 不为了让 Ours 变强而改写 `has_size`。
2. 正文可以说明该任务词面模式强，传统 baseline 可能很强。
3. Plan B 主线只保证它与 two-stage density、location 在同一主表并列。

### 1.5 Location

`location` 保留当前任务定义，主要做论文表达和公平评估整理。

正式指标：

1. Macro-F1 为主指标。
2. Accuracy 与 per-class F1 为辅助指标。

处理原则：

1. 继续使用 `no_location` fallback 口径，保证与现有 MWS-CFE 和 baseline 一致。
2. 不把 location 改成新任务，以免拖慢 Plan B 主线。

## 2. Ours 方法缺陷修复

### 2.1 已确认的问题

旧 `ConfidenceWeightedTrainer.compute_loss()` 中虽然保留了 `sample_weights` 字段，但实际 loss 是：

```python
loss = per_sample_loss.mean()
```

这意味着 `ws_confidence` 没有进入 per-sample loss，confidence-aware training 没有真正生效。

### 2.2 已落地修复

修改文件：

1. `scripts/phaseA2/train_mws_cfe_common.py`
2. `src/phase5/evaluation/metrics.py`

新 loss 定义：

```text
CE_i = CrossEntropy(logits_i, y_i; optional class_weight)
w_i = ws_confidence_i
Loss = sum_i(w_i * CE_i) / max(sum_i(w_i), 1e-8)
```

可选 focal loss：

```text
FocalCE_i = (1 - p_t)^gamma * CE_i
Loss = sum_i(w_i * FocalCE_i) / max(sum_i(w_i), 1e-8)
```

默认正式 Full 使用 `loss_type=ce`，focal 只作为后续增强，不作为第一波硬依赖。

### 2.3 修复验证方式

最低验证：

```bash
conda run -n follow-up python -m py_compile \
  scripts/phaseA2/train_mws_cfe_common.py \
  src/phase5/evaluation/metrics.py
```

功能验证：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage1.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage1 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
  --output-dir outputs/phaseA2_planB \
  --gate g2 \
  --tag smoke_conf_weight_seed42 \
  --seed 42 \
  --epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128
```

结果 JSON 中必须出现：

```json
"method_components": {
  "confidence_weighting": true,
  "sample_weight_field": "ws_confidence"
}
```

### 2.4 必要重构与增强项

| 优先级 | 项目 | 状态 | 说明 |
|---|---|---|---|
| P0 | `ws_confidence` 进入 per-sample loss | 已实现 | 必要修复 |
| P0 | explicit subset training for Stage 2 | 已实现数据与训练入口 | 必要重构 |
| P0 | two-stage density 数据构造 | 已实现 | 必要重构 |
| P1 | validation-selected quality gate | 命令矩阵固定 | 必要增强，先跑 G1-G5 再选 |
| P1 | section-aware input strategy | 已实现字段与默认输入 | 必要增强 |
| P2 | focal / class-balanced focal | 已提供参数 | 可选增强，不包装成主方法 |

## 3. Baseline 面板

正式主表至少包含：

| 方法 | 状态 | 实现 |
|---|---|---|
| Regex / cue-only | 新增 Plan B 入口 | `scripts/phaseA2/train_planb_baselines.py --methods regex_cue` |
| TF-IDF + LR | 可复用并扩展 | `train_planb_baselines.py --methods tfidf_lr` |
| TF-IDF + SVM | 可复用并扩展 | `train_planb_baselines.py --methods tfidf_svm` |
| MLP 或 fastText-style | 新增 | `train_planb_baselines.py --methods tfidf_mlp` |
| Vanilla PubMedBERT | 新增 two-stage wrappers | `train_vanilla_density_stage1.py`, `train_vanilla_density_stage2.py` |
| MWS-CFE (Ours) | 新增 two-stage wrappers | `train_mws_density_stage1.py`, `train_mws_density_stage2.py` |

第二个 PLM 当前不作为第一阶段硬依赖。原因：本地首要瓶颈是 two-stage 任务定义、Ours 修复和 6 类 baseline 闭环；新增第二 PLM 会显著增加 GPU 成本，但不直接修复旧 density 任务污染问题。

## 4. 标准消融表

Full model 正式定义：

```text
two-stage density
+ multi-source WS
+ weighted LF aggregation
+ G2 quality gate
+ class-balanced CE
+ confidence-aware sample loss
+ section_aware_text input
+ max_seq_length=128
```

标准 ablation variants：

| Variant | 实现方式 | 是否必须补跑 |
|---|---|---|
| Full | `tag=planb_full` | 必须 |
| w/o quality gate | G1 训练，`tag=ab_wo_quality_gate` | 必须 |
| w/o weighted aggregation | 先用 uniform vote 重建 WS，再 two-stage 构造 | 必须 |
| w/o confidence-aware training | 加 `--no-confidence-weight` | 必须 |
| w/o section strategy | `--input-field full_text` | 必须 |
| w/o multi-source supervision | `--source-mode single_lf --single-lf-name LF-D1` | 必须 |

输出：

```bash
outputs/phaseA2_planB/tables/planb_ablation_table.csv
outputs/phaseA2_planB/tables/planb_ablation_table.json
```

## 5. 参数讨论：3 参数 + 图

### P1 max_seq_length

取值：

```text
64, 96, 128, 160, 192
```

正文图：Stage 1 AUPRC 与 Stage 2 Macro-F1 随长度变化。

### P2 quality_gate

取值：

```text
G1, G2, G3, G4, G5
```

正文图：Stage 1 AUPRC 与 Stage 2 Macro-F1 随 gate 变化。

### P3 section/input strategy

取值：

```text
mention_text
section_aware_text
findings_text
impression_text
findings_impression_text
full_text
```

正文图：优先放 Stage 2 Macro-F1；Stage 1 AUPRC 可放正文或附录，取决于结果是否支持主论点。

绘图脚本：

```bash
conda run -n follow-up python -u scripts/phaseA2/plot_planb_parameters.py \
  --summary-csv outputs/phaseA2_planB/tables/planb_parameter_summary.csv \
  --output-dir outputs/phaseA2_planB/figures
```

该脚本直接输出 SVG，不依赖 `matplotlib`。

## 6. 新主表方案

Stage 1 与 Stage 2 density 放同一张主表，但分列展示：

| Method | Density-S1 AUPRC | Density-S1 AUROC | Density-S1 F1 | Density-S2 Macro-F1 | Has-size F1 | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Regex / cue-only |  |  |  |  |  |  |
| TF-IDF + LR |  |  |  |  |  |  |
| TF-IDF + SVM |  |  |  |  |  |  |
| TF-IDF + MLP |  |  |  |  |  |  |
| Vanilla PubMedBERT |  |  |  |  |  |  |
| MWS-CFE (Ours) |  |  |  |  |  |  |

正文表：

1. Plan B 主表：6 methods × density two-stage + has_size + location。
2. Plan B 标准消融表：Full vs w/o variants。
3. 参数图：P1/P2/P3，至少 Stage 2 Macro-F1 图；Stage 1 AUPRC 视结果进入正文或附录。

附录表：

1. 旧单阶段 density 五分类诊断表。
2. per-class F1。
3. 完整效率表。
4. 参数讨论完整 CSV。

## 7. 最小可执行重跑矩阵

### Wave 0：构建数据与语法检查

```bash
conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --output-dir outputs/phaseA2_planB \
  --source-mode weighted \
  --log logs/build_planb_density_two_stage.log

conda run -n follow-up python -m py_compile \
  src/phase5/evaluation/metrics.py \
  scripts/phaseA2/train_mws_cfe_common.py \
  scripts/phaseA2/build_planb_density_two_stage.py \
  scripts/phaseA2/train_planb_baselines.py \
  scripts/phaseA2/aggregate_planb_results.py \
  scripts/phaseA2/plot_planb_parameters.py
```

CPU 预算：5-15 分钟。

### Wave 1：先救 Ours

对每个 seed 跑 Stage 1 和 Stage 2：

```bash
SEED=42
conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage1.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage1 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
  --output-dir outputs/phaseA2_planB \
  --gate g2 \
  --tag planb_full_seed${SEED} \
  --seed ${SEED}

conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage2.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage2 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
  --output-dir outputs/phaseA2_planB \
  --gate g2 \
  --tag planb_full_seed${SEED} \
  --seed ${SEED}
```

GPU 预算：Stage 1 约 1-2 小时/seed；Stage 2 约 0.3-1 小时/seed，实际取决于显卡与 batch。

### Wave 2：补 baseline 面板

```bash
conda run -n follow-up python -u scripts/phaseA2/train_planb_baselines.py \
  --methods regex_cue,tfidf_lr,tfidf_svm,tfidf_mlp \
  --tasks density_stage1,density_stage2,size,location \
  --gate g2 \
  --tag planb_full_seed42 \
  --seed 42 \
  --output-dir outputs/phaseA2_planB
```

Vanilla PubMedBERT：

```bash
SEED=42
conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage1.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage1 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
  --output-dir outputs/phaseA2_planB \
  --gate g2 \
  --tag planb_full_seed${SEED} \
  --seed ${SEED}

conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage2.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage2 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
  --output-dir outputs/phaseA2_planB \
  --gate g2 \
  --tag planb_full_seed${SEED} \
  --seed ${SEED}
```

CPU baseline 预算：约 0.5-2 小时/5 seeds，MLP 可能最慢。

### Wave 3：标准消融

必须补跑：

1. `ab_wo_quality_gate`：G1。
2. `ab_wo_weighted_aggregation`：先 uniform WS，再构造 two-stage。
3. `ab_wo_confidence`：`--no-confidence-weight`。
4. `ab_wo_section`：`--input-field full_text`。
5. `ab_wo_multisource`：single LF 数据。

关键命令：

```bash
conda run -n follow-up python -u scripts/phaseA1/build_ws_datasets.py \
  --tasks density \
  --uniform-weights \
  --output-dir outputs/phaseA2/ws_uniform \
  --log logs/build_ws_uniform_density.log

conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --ws-source-dir outputs/phaseA2/ws_uniform/density \
  --output-dir outputs/phaseA2_planB_uniform \
  --source-mode uniform \
  --log logs/build_planb_density_two_stage_uniform.log

conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --output-dir outputs/phaseA2_planB_single_lf \
  --source-mode single_lf \
  --single-lf-name LF-D1 \
  --log logs/build_planb_density_two_stage_single_lf.log
```

### Wave 4：参数图

1. P1/P2/P3 按 tag 补跑。
2. 聚合。
3. 绘图。

```bash
conda run -n follow-up python -u scripts/phaseA2/aggregate_planb_results.py \
  --results-dir outputs/phaseA2_planB/results \
  --output-dir outputs/phaseA2_planB/tables \
  --main-tag planb_full

conda run -n follow-up python -u scripts/phaseA2/plot_planb_parameters.py \
  --summary-csv outputs/phaseA2_planB/tables/planb_parameter_summary.csv \
  --output-dir outputs/phaseA2_planB/figures
```

## 8. 新增/修改文件清单

新增：

1. `docs/module2_planB_full_reconstruction_execution.md`
2. `scripts/phaseA2/build_planb_density_two_stage.py`
3. `scripts/phaseA2/train_mws_density_stage1.py`
4. `scripts/phaseA2/train_mws_density_stage2.py`
5. `scripts/phaseA2/train_vanilla_density_stage1.py`
6. `scripts/phaseA2/train_vanilla_density_stage2.py`
7. `scripts/phaseA2/train_planb_baselines.py`
8. `scripts/phaseA2/aggregate_planb_results.py`
9. `scripts/phaseA2/plot_planb_parameters.py`

修改：

1. `scripts/phaseA2/train_mws_cfe_common.py`
2. `src/phase5/evaluation/metrics.py`

## 9. 当前判断

当前已经可以进入 Plan B 全盘重构的正式补跑阶段。下一步必须全面集中修复模块2，不应继续推进模块3。

仍需注意：

1. 当前没有伪造任何结果。
2. 是否能让 MWS-CFE 成为最终最强方法，必须由补跑结果决定。
3. 如果 two-stage + confidence-aware 后仍无法超过 Vanilla PubMedBERT，论文必须诚实呈现，而不能通过表格包装制造结论。
