# Phase A2.5 缺口补齐与 5-seed 汇总方案

> 日期：2026-04-18  
> 适用范围：模块2，放射学报告结构化信息抽取  
> 正式方法名：MWS-CFE（Multi-source Weak Supervision for Clinical Fact Extraction，多源弱监督临床事实抽取）  
> 当前阶段目标：先补正式主表入口与汇总机制，不继续盲跑训练  
> 硬件约束：单张 RTX 3090 24GB

---

## 0. 当前判断

Phase A2 已经从“继续大规模训练”进入 **A2.5：汇总与补齐缺口** 阶段。

原因是：MWS-CFE 主线、A2 quality-gate、A3 aggregation、P1 max_seq_length、P3 section/input strategy 的主要 MWS 结果已经按交接状态完成；当前阻塞点不是继续追加 MWS 训练，而是正式主表仍缺少新版 A2 口径下的三个比较入口，以及一个不依赖旧 Phase 5 逻辑的 5-seed 汇总机制。

本阶段明确不做以下事情：

1. 不继续启动训练。
2. 不沿用 `scripts/phaseA2/build_main_table.py`。
3. 不把旧 Phase 5 baselines 直接包装成新版 A2 公平主表。
4. 不把 researcher-reviewed subset 放回正文主线。
5. 不忽略本地 safetensors 模型目录与离线模式热修复。

---

## 1. 已完成结果盘点

### 1.1 按交接状态确认已完成

当前已完成的 MWS-CFE 结果包括：

1. 正式主表中的 MWS-CFE：
   - `density` x 5 seeds
   - `has_size` x 5 seeds
   - `location` x 5 seeds
2. A2 quality-gate：
   - `density`: g1/g3/g4/g5 x 5 seeds
   - `location`: g1/g3/g4/g5 x 5 seeds
   - g2 复用主表
3. A3 aggregation：
   - `density` uniform x 5 seeds
   - `location` uniform x 5 seeds
4. P1 max_seq_length，density-only：
   - len64/len96/len160/len192 x 5 seeds
   - len128 复用主表
5. P3 section/input strategy，density-only：
   - findings/impression/findings_impression/full_text x 5 seeds
   - mention_text + unfiltered WS 复用主表
6. CPU 预处理：
   - `rebuild_ws_uniform.py`
   - `filter_ws_by_section.py --section findings`
   - `filter_ws_by_section.py --section impression`
   - `filter_ws_by_section.py --section findings_impression`

### 1.2 本地标准文件名需要 manifest 校验

`outputs/phaseA2/results` 中已能看到大量标准命名结果：

```text
mws_cfe_{task}_results_{tag}_seed{seed}.json
```

但新汇总脚本必须做 manifest（清单）校验，而不是默认相信文件齐全。当前标准命名盘点需要特别检查：

1. `location/main_g2` 与 `size/main_g2` 能按标准命名看到 5 seeds。
2. `density/main_g2` 需要确认 seed `3407` 是否以非标准文件名存在、被备份、或需要从对应 checkpoint 重新导出结果 JSON。
3. 这一步是汇总前的文件一致性校验，不等同于要求立刻补跑训练。

---

## 2. 正式主表仍缺的方法

新版 A2 正文主表只应包含统一 multi-source WS（多源弱监督）口径下的 trainable 方法：

1. `TF-IDF + LR`
2. `TF-IDF + SVM`
3. `Vanilla PubMedBERT`
4. `MWS-CFE (Ours)`

当前已完成的是 `MWS-CFE (Ours)`。正式主表仍缺：

1. `Vanilla PubMedBERT` 的新版 A2 训练入口。
2. `TF-IDF + LR` 的新版 A2 训练入口。
3. `TF-IDF + SVM` 的新版 A2 训练入口。
4. A1 supervision-source（监督来源消融）的数据重构与训练入口。

这些缺口都不能用旧 Phase 5 结果替代，因为旧 Phase 5 依赖 single-source / old silver（单源或旧银标）逻辑，和当前统一 multi-source WS 公平主表不一致。

---

## 3. 哪些旧结果可复用，哪些必须重跑

### 3.1 可以复用

可以复用的结果仅限于已经按新版 A2 口径产生的结果：

1. `outputs/phaseA2/results/mws_cfe_*_main_g2_seed*.json`
   - 作为 MWS-CFE 正式主表结果。
   - 前提是 manifest 校验 5 seeds 齐全。
2. `mws_cfe_density_results_aqg_g{1,3,4,5}_seed*.json`
   - 作为 A2 quality-gate 的 density 部分。
3. `mws_cfe_location_results_aqg_g{1,3,4,5}_seed*.json`
   - 作为 A2 quality-gate 的 location 部分。
4. `mws_cfe_density_results_aagg_uniform_seed*.json`
   - 作为 A3 aggregation 的 density uniform 部分。
5. `mws_cfe_location_results_aagg_uniform_seed*.json`
   - 作为 A3 aggregation 的 location uniform 部分。
6. `mws_cfe_density_results_p1_len{64,96,160,192}_seed*.json`
   - 作为 P1 max_seq_length 的补充长度。
7. `mws_cfe_density_results_p3_{findings,impression,findings_impression,fulltext}_seed*.json`
   - 作为 P3 section/input strategy。
8. `outputs/phaseA2/ws_uniform` 与 `outputs/phaseA2/ws_findings*`
   - 作为已完成的 CPU 预处理产物。

### 3.2 不能复用为正式主表

以下结果不能进入新版 A2 正式主表：

1. `outputs/phase5/results/baselines_summary.json`
2. 旧 `ml_lr` / `ml_svm` 结果。
3. 旧 Phase 5 `pubmedbert_*_results.json`。
4. 旧 gold sanity-check 结果。
5. researcher-reviewed subset 的指标。
6. 任何 single-source Regex teacher 结果。

这些结果最多可作为历史背景或附录说明，不能作为新版 A2 公平主表的数值来源。

### 3.3 必须按新版 A2 重跑的内容

必须补跑或首次运行的是：

1. `Vanilla PubMedBERT` x 3 tasks x 5 seeds。
2. `TF-IDF + LR` x 3 tasks x 5 seeds。
3. `TF-IDF + SVM` x 3 tasks x 5 seeds。
4. A1 supervision-source 的选定 source variants（监督来源变体），具体范围见第 7 节。

---

## 4. Vanilla PubMedBERT 新版 A2 入口设计

### 4.1 方法定义

`Vanilla PubMedBERT` 在新版 A2 中定义为：

1. 使用和 MWS-CFE 相同的 Phase A1 multi-source WS 聚合标签。
2. 默认使用 `G2` gate 的训练集。
3. 默认输入字段为 `mention_text`。
4. 使用 PubMedBERT / BiomedBERT backbone（主干模型）进行 plain hard-label fine-tuning（普通硬标签微调）。
5. 不使用 MWS-CFE 的质量感知训练配置，例如 `ws_confidence` 权重。
6. 评测仍输出：
   - `ws_val_results`
   - `ws_test_results`
   - `phase5_test_results`

### 4.2 建议新增入口

建议新增：

```text
scripts/phaseA2/train_vanilla_pubmedbert_common.py
scripts/phaseA2/train_vanilla_density.py
scripts/phaseA2/train_vanilla_size.py
scripts/phaseA2/train_vanilla_location.py
```

### 4.3 复用与改造点

应优先复用 `scripts/phaseA2/train_mws_cfe_common.py` 的新版 A2 数据与评测框架，而不是直接复用 `scripts/phase5/train_pubmedbert_common.py`。

需要保留：

1. `--seed`
2. `--gate`
3. `--ws-data-dir`
4. `--phase5-data-dir`
5. `--output-dir`
6. `--input-field`
7. `--tag`
8. `evaluate_on_phase5_test`
9. location 的 `no_location` fallback 逻辑
10. 本地 safetensors 模型目录与 `use_safetensors=True`

需要去掉或禁用：

1. `ws_confidence` 作为 sample weight。
2. MWS-CFE 专属的质量感知损失解释。
3. 任何旧 Phase 5 single-source 数据路径。

### 4.4 输出命名

建议输出：

```text
outputs/phaseA2/results/vanilla_pubmedbert_density_results_main_g2_seed42.json
outputs/phaseA2/results/vanilla_pubmedbert_size_results_main_g2_seed42.json
outputs/phaseA2/results/vanilla_pubmedbert_location_results_main_g2_seed42.json
```

其中 `task=size` 在表中显示为 `has_size`。

---

## 5. TF-IDF + LR / SVM 新版 A2 入口设计

### 5.1 方法定义

`TF-IDF + LR` 与 `TF-IDF + SVM` 是 CPU 传统机器学习 baseline（基线方法），但必须使用新版 A2 数据：

1. 训练集：`outputs/phaseA1/{task}/{task}_train_ws_g2.jsonl`
2. 验证集：`outputs/phaseA1/{task}/{task}_val_ws.jsonl`
3. WS 测试集：`outputs/phaseA1/{task}/{task}_test_ws.jsonl`
4. 正式测试集：`outputs/phase5/datasets/{task}_test.jsonl`
5. 输入字段：默认 `mention_text`
6. 任务定义与标签集合：和 MWS-CFE 保持一致

### 5.2 建议新增入口

建议新增一个统一入口：

```text
scripts/phaseA2/train_tfidf_baselines.py
```

参数建议：

```text
--method {tfidf_lr,tfidf_svm}
--task {density,size,location}
--seed 42
--gate g2
--ws-data-dir outputs/phaseA1/{task}
--phase5-data-dir outputs/phase5/datasets
--output-dir outputs/phaseA2
--input-field mention_text
--tag main_g2_seed42
```

### 5.3 复用与禁止事项

可以复用 `scripts/phase5/run_baselines.py` 中的以下实现思想：

1. `TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)`
2. `LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)`
3. `LinearSVC(class_weight="balanced", max_iter=5000, C=1.0)`
4. `evaluate_density`
5. `evaluate_location`
6. `evaluate_size_detection`

但禁止复用旧数据路径和旧 summary：

1. 不读取 `outputs/phase5/datasets/*_train.jsonl` 作为训练集。
2. 不读取 `outputs/phase5/results/baselines_summary.json`。
3. 不把旧 `ml_lr` / `ml_svm` JSON 改名放进 A2。

### 5.4 输出命名

建议输出：

```text
outputs/phaseA2/results/tfidf_lr_density_results_main_g2_seed42.json
outputs/phaseA2/results/tfidf_svm_density_results_main_g2_seed42.json
```

每个 JSON 至少包含：

1. `method`
2. `task`
3. `seed`
4. `gate`
5. `tag`
6. `input_field`
7. `train_samples`
8. `val_samples`
9. `ws_val_results`
10. `ws_test_results`
11. `phase5_test_results`
12. `train_time_seconds`
13. `eval_time_seconds`
14. `model_config`

---

## 6. A1 supervision-source 落地方案

### 6.1 不再使用旧 single-source Phase 5

A1 supervision-source（监督来源消融）不能写成：

```text
single-source = Phase 5 old Vanilla
multi-source = MWS-CFE main
```

这个写法已经不符合当前新版 A2 口径，因为它混入了旧 single-source / old silver 体系。

### 6.2 正确落地方式

正确方式是在当前 Phase A1 multi-source WS 框架内部做 source ablation（来源消融）：

1. 输入原始 WS 文件：
   - `outputs/phaseA1/{task}/ws_train.jsonl`
   - `outputs/phaseA1/{task}/ws_val.jsonl`
   - `outputs/phaseA1/{task}/ws_test.jsonl`
2. 读取其中的 `lf_details`。
3. 按指定 source 保留部分 labeling functions（标注函数）。
4. 对保留的 LF outputs 重新做 aggregation（聚合）。
5. 重新计算：
   - `ws_label`
   - `ws_confidence`
   - `lf_coverage`
   - `lf_agreement`
   - `gate_level`
   - `passed_gates`
6. 输出 source-specific WS 数据目录。
7. 使用现有 MWS-CFE 训练入口通过 `--ws-data-dir` 训练。

建议新增：

```text
scripts/phaseA2/rebuild_ws_by_source.py
```

输出目录建议：

```text
outputs/phaseA2/ws_source/{source_name}/{task}/
```

### 6.3 source 维度

当前 LF 名称天然可作为 source：

| Task | Source |
|---|---|
| density | LF-D1, LF-D2, LF-D3, LF-D4, LF-D5 |
| size | LF-S1, LF-S2, LF-S3, LF-S4, LF-S5 |
| location | LF-L1, LF-L2, LF-L3, LF-L4, LF-L5 |

正式 A1 表建议采用长表：

| Task | Supervision Source | Train Samples | Gate Retention | Phase5 Test Acc | Phase5 Test Main Metric |
|---|---|---:|---:|---:|---:|

其中 multi-source weighted G2 直接复用 MWS-CFE 主表。

### 6.4 工作量控制

完整 single-LF A1 为：

```text
15 source variants x 5 seeds = 75 GPU runs
```

这对单张 RTX 3090 来说成本偏高。建议分两级：

1. **正式最小版**：
   - 每个任务跑覆盖率最高或临床意义最核心的 1-2 个 single-source variants。
   - multi-source G2 复用主表。
2. **完整附录版**：
   - 15 个 LF source variants 全部跑齐。

无论采用哪一级，都必须由 `rebuild_ws_by_source.py` 生成当前 A2 口径的数据，不能复用旧 Phase 5。

---

## 7. 新 5-seed 汇总脚本设计

### 7.1 新脚本

不要继续维护旧 `build_main_table.py`。建议新增：

```text
scripts/phaseA2/aggregate_a2_5_results.py
```

该脚本只读结果，不训练模型。

默认输入：

```text
outputs/phaseA2/results
```

默认输出：

```text
outputs/phaseA2/tables/a2_5_main_table.csv
outputs/phaseA2/tables/a2_5_main_table.json
outputs/phaseA2/tables/a2_5_efficiency_table.csv
outputs/phaseA2/tables/a2_quality_gate_summary.csv
outputs/phaseA2/tables/a3_aggregation_summary.csv
outputs/phaseA2/tables/p1_max_length_summary.csv
outputs/phaseA2/tables/p3_section_input_summary.csv
outputs/phaseA2/tables/a2_5_manifest_report.json
```

### 7.2 读取规则

统一识别：

```text
{method}_{task}_results_{tag}_seed{seed}.json
```

支持的 method：

```text
mws_cfe
vanilla_pubmedbert
tfidf_lr
tfidf_svm
```

支持的 task：

```text
density
size
location
```

表格显示时：

```text
size -> has_size
```

固定 seeds：

```text
[13, 42, 87, 3407, 31415]
```

### 7.3 指标抽取

主表只从 `phase5_test_results` 抽取正式指标：

| Task | 主指标 | 辅助指标 |
---|---|---|
| density | `macro_f1` | `accuracy` |
| size / has_size | `f1` | `accuracy`, `precision`, `recall` |
| location | `macro_f1` | `accuracy` |

WS 验证集和 WS 测试集指标保留到 JSON 汇总中，但不作为正文主表核心列。

### 7.4 mean ± std 规则

对每个 `method / task / tag / metric`：

1. 收集 5 个 seed 的数值。
2. 计算均值。
3. 计算样本标准差。
4. 正式表中以 `mean ± std` 输出。
5. 百分比指标统一乘以 100 后保留两位小数。

如果缺 seed：

1. 默认报错并写入 `a2_5_manifest_report.json`。
2. 只有显式传入 `--allow-partial` 时才生成 draft 表。
3. draft 表必须标记 `N=<实际 seed 数>`，不能冒充 5-seed 正式表。

### 7.5 正式主表

正式主表行：

1. `TF-IDF + LR`
2. `TF-IDF + SVM`
3. `Vanilla PubMedBERT`
4. `MWS-CFE (Ours)`

正式主表列：

| Method | Density Acc | Density Macro-F1 | Has_size Acc | Has_size F1 | Location Acc | Location Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|

如果后续补齐参数量与推理延迟，可扩展为：

| Method | Params (M) | Density Acc | Density Macro-F1 | Has_size Acc | Has_size F1 | Location Acc | Location Macro-F1 | Inference Time (ms/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 7.6 效率表

效率表建议使用长表：

| Method | Task | Train Samples | Train Time / seed | Eval Time / seed | Best Epoch | Peak GPU Memory | Inference Time |
|---|---|---:|---:|---:|---:|---:|---:|

已有 MWS-CFE JSON 可直接读取：

1. `train_samples`
2. `train_time_seconds`
3. `eval_time_seconds`
4. `best_epoch`

Peak GPU Memory 当前主要在日志中出现，应由聚合脚本从对应日志中解析，或在后续入口中写入 JSON。

正式推理延迟如果要答辩使用，建议单独补一个只做 benchmark 的脚本：

```text
scripts/phaseA2/benchmark_a2_inference.py
```

不要把训练期整体 `eval_time_seconds` 直接包装成严格 per-sample inference latency，除非表注明这是 evaluation throughput proxy（评估吞吐代理）。

### 7.7 A2 / A3 / P1 / P3 汇总表

汇总 tag 映射建议如下：

| Table | Tags |
|---|---|
| A2 quality-gate | `main_g2`, `aqg_g1`, `aqg_g3`, `aqg_g4`, `aqg_g5` |
| A3 aggregation | `main_g2`, `aagg_uniform` |
| P1 max_seq_length | `main_g2` as len128, `p1_len64`, `p1_len96`, `p1_len160`, `p1_len192` |
| P3 section/input | `main_g2` as mention_text, `p3_findings`, `p3_impression`, `p3_findings_impression`, `p3_fulltext` |

脚本必须允许每张表声明适用任务范围。例如 P1 / P3 当前是 density-only，不应强行要求 size/location。

---

## 8. 热修复是否纳入正式方案

必须纳入。

当前环境下 GPU 实验依赖以下热修复：

1. `MODEL_NAME` 指向本地 safetensors 模型目录：
   - `outputs/phase5/hf_models/biomedbert_base_safe`
2. `AutoModelForSequenceClassification.from_pretrained(..., use_safetensors=True)`
3. 离线模式：
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
4. 避免当前环境触发 `torch.load` / `.bin` 权重限制。

A2.5 后续入口必须复用这一策略。尤其是 `Vanilla PubMedBERT` 不能回退到在线 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` 下载路径。

新聚合脚本还应检查：

1. JSON 中的 `model_name` 是否记录为本地路径。
2. 如果历史 JSON 中仍写旧 Hugging Face 名称，应在 manifest 中标记为 `metadata_warning`。
3. 是否需要补写 `runtime_environment` 字段，记录离线变量与模型目录。

---

## 9. 剩余工作量估计

### 9.1 必须完成的入口与汇总工作

| 工作 | 类型 | 估计 |
|---|---|---:|
| Vanilla PubMedBERT 新版入口 | 代码 | 0.5-1 天 |
| TF-IDF LR/SVM 新版入口 | 代码 | 0.5 天 |
| A1 supervision-source 数据重构脚本 | 代码 + CPU 预处理 | 0.5-1 天 |
| A2.5 聚合脚本 | 代码 | 0.5-1 天 |
| manifest 校验与表格导出 | CPU | 数分钟 |

### 9.2 必须补跑的实验

| 内容 | Runs | 设备 | 估计 |
|---|---:|---|---:|
| TF-IDF + LR | 15 | CPU | 约 0.5-2 小时 |
| TF-IDF + SVM | 15 | CPU | 约 0.5-2 小时 |
| Vanilla PubMedBERT | 15 | GPU | 约 5-10 GPU 小时，取决于 batch 与 dataloader 配置 |

### 9.3 A1 supervision-source 额外工作

| 方案 | Runs | 说明 |
|---|---:|---|
| 最小版 | 15-30 | 每任务 1-2 个 single-source variants x 5 seeds |
| 完整版 | 75 | 15 个 LF source variants x 5 seeds |

建议先完成主表缺口，再根据剩余时间选择 A1 的最小版或完整版。

---

## 10. 最终成表流程

推荐执行顺序：

1. 新增 `aggregate_a2_5_results.py`，先做只读 manifest dry-run。
2. 对 `outputs/phaseA2/results` 做标准命名和 5-seed 完整性校验。
3. 若发现 MWS-CFE 已完成结果未按标准命名存在，优先定位、重命名或重新导出 JSON，不直接盲跑训练。
4. 新增并检查 `train_tfidf_baselines.py`。
5. 新增并检查 `train_vanilla_pubmedbert_*` 入口。
6. 新增并检查 `rebuild_ws_by_source.py`。
7. 再决定是否运行缺失的 CPU/GPU 补实验。
8. 所有补实验完成后，运行聚合脚本生成：
   - 正式主表
   - 效率表
   - A2 quality-gate 汇总表
   - A3 aggregation 汇总表
   - P1 max_seq_length 汇总表
   - P3 section/input 汇总表
   - A1 supervision-source 汇总表
9. researcher-reviewed subset 只保留为附录 sanity-check，不进入正文主表。
10. 论文或答辩材料中统一声明：所有正文主表 trainable 方法均使用同一 multi-source WS 数据、同一 subject split、同一 G2 主配置与 5-seed mean ± std。

---

## 11. 结论

当前 A2 不应继续优先训练 MWS-CFE 变体。更合理的下一步是：

1. 先补 Vanilla PubMedBERT / TF-IDF + LR / TF-IDF + SVM 的新版 A2 入口。
2. 同步补 A1 supervision-source 的数据重构机制。
3. 新增只读 5-seed 聚合脚本，替代旧 `build_main_table.py`。
4. 用 manifest 先锁定已有结果能否成表，再决定最小必要补跑。

