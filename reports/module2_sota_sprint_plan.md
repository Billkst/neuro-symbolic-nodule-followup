# 模块2 learned-model SOTA 冲刺方案

> 日期：2026-04-21  
> 阶段：Plan B 结果之后的 learned-model SOTA 冲刺  
> 目标：在不把 cue-only 纳入公平主表的前提下，让 Ours 在 learned-model 比较中达到或超过当前最佳方法  
> 原则：最短时间、最大收益；先 smoke test，再决定 full 5 seeds；不做大而全重构。

## 1. 当前公平比较口径

cue-only 不再纳入公平主表比较。它只作为 deterministic label-construction reference，用于说明当前 silver / constructed label 与显式规则线索高度一致。正文 learned-model 主表只比较 TF-IDF + LR、TF-IDF + SVM、TF-IDF + MLP、Vanilla PubMedBERT、MWS-CFE / hybrid Ours。

当前 Ours 的缺口：

| Metric | Current Ours | Best learned model | Gap |
|---|---:|---:|---:|
| Stage 1 AUPRC | 98.82 | 98.82 | Ours 已最强 |
| Stage 1 AUROC | 99.84 | 99.84 | Ours 已最强 |
| Stage 1 F1 | 60.29 | 88.68 | -28.39 |
| Stage 2 Macro-F1 | 91.05 | 91.19 | -0.14 |
| Has-size F1 | 81.93 | 87.82 | -5.89 |
| Location Macro-F1 | 96.90 | 97.39 | -0.49 |

因此冲刺目标不是“继续写结果解释”，而是 learned-model SOTA：优先修 Stage 1 决策层，其次把 Stage 2 推过 Vanilla PubMedBERT，再用 neuro-symbolic hybrid 改造 Has-size / Location。

## 2. 四个问题的最小修复路线

### 2.1 为什么 Stage 1 AUPRC/AUROC 高但 F1 低

Stage 1 的 ranking 指标已经最强，说明模型能把显式密度证据样本排到更高分；F1 低说明默认 argmax / 0.5 决策边界不适合当前类别分布或校准状态。最可能的问题是概率校准和阈值选择，而不是 encoder 表示能力不足。

修复路线：

1. 固定已有 Stage 1 模型，不重训。
2. 在 validation set 上做 Platt scaling。
3. 只在 validation set 上扫描 F1 最优阈值。
4. 将固定校准器和阈值应用到 WS test / Phase5 test。
5. 输出兼容 Plan B aggregator 的新 JSON，并保留旧 0.5 阈值与 tuned F1 的对比。

新增脚本：

`scripts/phaseA2/tune_planb_stage1_threshold.py`

### 2.2 如何把 Stage 2 推过 Vanilla PubMedBERT

Stage 2 与 Vanilla PubMedBERT 只差 0.14 Macro-F1，属于最小缺口。已有参数表显示 G3 和 max_length=192 在 Stage 2 上可能更强，但这些目前是 test 聚合视角，不能直接作为最终定稿依据。冲刺必须用 validation 选配置。

修复路线：

1. 固定保留 section-aware text。
2. 先比较 G2 vs G3。
3. 再比较 max_length 128 vs 192。
4. selection metric 只用 `ws_val_results.macro_f1`。
5. smoke seed 先用 42；如果 G3 或 192 在 validation 上胜出，再补 full 5 seeds。

新增脚本：

`scripts/phaseA2/select_planb_stage2_config.py`

### 2.3 如何把 Has-size 改成更强的 hybrid/neuro-symbolic 方案

Has-size 是典型规则强任务：数字、mm/cm、尺寸上下文是强判别线索。纯神经 classifier 当前比 TF-IDF + LR 低 5.89 F1，不值得优先靠重训硬追。

修复路线：

1. 使用 rule-first：`extract_size(mention_text)` 命中则直接预测 has_size。
2. 对 rule 未命中的样本使用 model fallback。
3. fallback 置信阈值只在 validation set 上选择。
4. 输出 `mws_cfe_size_results_<tag>_seed<seed>.json`，保持主表聚合兼容。

新增脚本：

`scripts/phaseA2/eval_planb_hybrid.py --task size`

### 2.4 如何把 Location 拉到第一

Location 当前只差 0.49 Macro-F1，且位置 cue 规则非常强。纯模型主要短板可能来自显式 lobe cue 没有被硬约束、no_location fallback 与真实部署逻辑不一致。

修复路线：

1. 使用 rule-first：`extract_location(mention_text)` 命中具体 lobe / bilateral / unclear 时直接采用规则结果。
2. rule 未命中时使用 model fallback；fallback 置信阈值在 validation set 上选择。
3. 低置信 fallback 统一为 no_location。
4. 输出 `mws_cfe_location_results_<tag>_seed<seed>.json`，保持主表聚合兼容。

新增脚本：

`scripts/phaseA2/eval_planb_hybrid.py --task location`

## 3. Full 组件取舍

建议保留：

1. section-aware text：消融中去掉后 Stage 1 和 Stage 2 均明显下降，是当前最稳定正向组件。
2. confidence-aware training：Stage 2 有小幅稳定作用，成本低。
3. two-stage density：解决旧 density 单阶段混淆问题，应继续保留。

建议暂时弱化或替换：

1. 默认 0.5 / argmax 决策层：Stage 1 F1 明显低，必须替换为 validation-tuned threshold。
2. 固定 G2 主配置：P2 显示 G3 可能更强，先用 validation 重选。
3. 固定 max_length=128：P1 显示 192 可能改善 Stage 2，先用 validation 重选。
4. Has-size / Location 纯 classifier：这两个字段规则线索强，短期应切到 rule-first + model-fallback hybrid。
5. cue-only 主表行：继续从公平主表排除，只作为 label-construction reference。

## 4. 最小补实验矩阵

### 4.1 Smoke test

只跑 seed 42，目标是快速判断方向，不做论文结论。

| Priority | Task | Candidate | Command type | Selection metric |
|---|---|---|---|---|
| P0 | density_stage1 | existing G2/128 model + Platt + threshold tuning | eval only | val F1 |
| P1 | density_stage2 | G2/128 vs G3/128 | train/eval if missing, then selector | val Macro-F1 |
| P1 | density_stage2 | selected gate with 128 vs 192 | train/eval if missing, then selector | val Macro-F1 |
| P2 | size | rule-first + model-fallback | eval only | val F1 |
| P2 | location | rule-first + model-fallback | eval only | val Macro-F1 |

### 4.2 Full 5 seeds

只对 smoke 明确胜出的配置补 5 seeds：

| Task | Full condition |
|---|---|
| density_stage1 | tuned threshold 在 Phase5 test 上明显提升 F1，且 validation F1 提升稳定 |
| density_stage2 | validation 选出的 gate/max_length 配置优于 G2/128 |
| size | hybrid validation F1 不低于 TF-IDF + LR，并且 Phase5 audit 不退化 |
| location | hybrid validation Macro-F1 不低于 Vanilla PubMedBERT，并且 no_location 策略合理 |

不建议同时展开所有 gate、所有 max_length、所有 input strategy。当前只需要 G2/G3 和 128/192 的最小矩阵。

## 5. 新增脚本职责

| Script | Priority | Solves |
|---|---|---|
| `scripts/phaseA2/tune_planb_stage1_threshold.py` | P0 | Stage 1 AUPRC/AUROC 高但 F1 低；用 validation Platt scaling + threshold tuning 修决策层 |
| `scripts/phaseA2/select_planb_stage2_config.py` | P1 | 用 validation Macro-F1 重选 Stage 2 gate/max_length，避免用 test 选配置 |
| `scripts/phaseA2/eval_planb_hybrid.py` | P2 | Has-size / Location 切到 rule-first + model-fallback hybrid，并输出兼容 JSON |

## 6. 推荐执行顺序

1. 先跑 P0：它不需要重训，收益最大，直接解决 Stage 1 F1 短板。
2. 再跑 P2：size/location hybrid 也是 eval-only，最快可能追上或超过 learned-model SOTA。
3. 最后跑 P1：Stage 2 只差 0.14，但需要训练或至少确认已有 candidate 的 validation 排名，成本高于 P0/P2。

## 7. 明确判断

当前最应该先修 Stage 1 F1。因为 AUPRC/AUROC 已经最强，说明模型排序能力足够；用 validation threshold tuning 和 Platt scaling 最可能在最短时间内释放已有模型能力。

最可能最快达到 learned-model SOTA 的改动是：

1. P0 validation-based Stage 1 threshold tuning。
2. P2 Has-size rule-first + model-fallback。
3. P2 Location rule-first + model-fallback。
4. P1 Stage 2 G2/G3 和 128/192 validation selection。

需要重新跑实验，但只跑最小必要矩阵：P0/P2 先 eval-only smoke，P1 只补缺失的 seed 42 candidate；只有 smoke 明确胜出后，才补 full 5 seeds。
