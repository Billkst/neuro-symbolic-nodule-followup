# Phase A2.5 执行计划

> 日期：2026-04-18  
> 阶段：A2.5 实现与执行入口  
> 目标：补齐新版 A2 公平主表入口，并提供 5-seed 汇总能力  
> 约束：先不继续训练，由人工确认后再执行补跑

---

## 1. 当前已完成

已完成且可复用的结果：

1. `MWS-CFE` 主表结果：
   - `density`
   - `has_size`
   - `location`
2. A2 quality-gate：
   - `density`: G1/G3/G4/G5
   - `location`: G1/G3/G4/G5
   - G2 复用主表
3. A3 aggregation：
   - `density`: uniform
   - `location`: uniform
4. P1 max_seq_length：
   - density-only: 64/96/160/192
   - 128 复用主表
5. P3 section/input strategy：
   - density-only: findings/impression/findings_impression/full_text
   - mention_text 复用主表
6. CPU 预处理：
   - `outputs/phaseA2/ws_uniform`
   - `outputs/phaseA2/ws_findings`
   - `outputs/phaseA2/ws_impression`
   - `outputs/phaseA2/ws_findings_impression`

---

## 2. 当前还缺什么

正式公平主表仍缺新版 A2 口径下的 3 组 baseline：

1. `Vanilla PubMedBERT`
2. `TF-IDF + LR`
3. `TF-IDF + SVM`

A1 supervision-source 也缺入口。它不能再复用旧 Phase 5 single-source 结果，必须从当前 Phase A1 `ws_*.jsonl` 的 `lf_details` 重新构造 source-specific WS 数据。

---

## 3. 新增脚本

### 3.1 Vanilla PubMedBERT

新增文件：

```text
scripts/phaseA2/train_vanilla_pubmedbert_common.py
scripts/phaseA2/train_vanilla_density.py
scripts/phaseA2/train_vanilla_size.py
scripts/phaseA2/train_vanilla_location.py
```

作用：

1. 使用 Phase A1 multi-source WS 数据。
2. 使用统一 subject split 与 G2 gate。
3. 使用本地 safetensors PubMedBERT 模型目录。
4. 使用 plain hard-label fine-tuning，不使用 MWS-CFE 的 confidence weighting。
5. 输出与 MWS-CFE 聚合兼容的 JSON：

```text
outputs/phaseA2/results/vanilla_pubmedbert_{task}_results_main_g2_seed{seed}.json
```

### 3.2 TF-IDF baseline

新增文件：

```text
scripts/phaseA2/train_tfidf_baselines.py
```

作用：

1. 训练 `TF-IDF + LR`。
2. 训练 `TF-IDF + SVM`。
3. 训练数据固定为 Phase A1 multi-source WS。
4. 正式指标输出到 Phase 5 full test split。
5. 输出：

```text
outputs/phaseA2/results/tfidf_lr_{task}_results_main_g2_seed{seed}.json
outputs/phaseA2/results/tfidf_svm_{task}_results_main_g2_seed{seed}.json
```

### 3.3 A1 supervision-source

新增文件：

```text
scripts/phaseA2/rebuild_ws_by_source.py
```

作用：

1. 读取 `outputs/phaseA1/{task}/ws_train.jsonl`、`ws_val.jsonl`、`ws_test.jsonl`。
2. 从 `lf_details` 中保留指定 LF source。
3. 重新做 weighted aggregation。
4. 重新计算 gate。
5. 输出可直接用于 MWS-CFE 训练的 source-specific WS 目录：

```text
outputs/phaseA2/ws_source/{source_name}/{task}/
```

### 3.4 5-seed 汇总

新增文件：

```text
scripts/phaseA2/aggregate_a2_5_results.py
```

作用：

1. 读取 `outputs/phaseA2/results/*.json`。
2. 对 method/task/tag/seed 做 manifest 校验。
3. 对每个方法和任务计算 `mean ± std`。
4. 输出正式主表、效率表、A2/A3/P1/P3 汇总表。
5. 不依赖旧 `build_main_table.py`。

---

## 4. 环境准备

所有命令在项目根目录执行：

```bash
conda activate follow-up
cd /home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p logs
```

GPU 入口使用当前热修复状态：

1. `MODEL_NAME=outputs/phase5/hf_models/biomedbert_base_safe`
2. `use_safetensors=True`
3. 不从 Hugging Face 在线下载模型

---

## 5. 运行顺序

### Step 1：先做只读 manifest 汇总

不启动训练，先看已有结果缺口：

```bash
python -u scripts/phaseA2/aggregate_a2_5_results.py
```

输出：

```text
outputs/phaseA2/tables/a2_5_manifest_report.json
outputs/phaseA2/tables/a2_5_main_table.csv
outputs/phaseA2/tables/a2_5_efficiency_table.csv
outputs/phaseA2/tables/a2_quality_gate_summary.csv
outputs/phaseA2/tables/a3_aggregation_summary.csv
outputs/phaseA2/tables/p1_max_length_summary.csv
outputs/phaseA2/tables/p3_section_input_summary.csv
```

正式成表前可使用 strict 模式：

```bash
python -u scripts/phaseA2/aggregate_a2_5_results.py --strict
```

### Step 2：补 CPU baseline

对每个 seed 执行：

```bash
python -u scripts/phaseA2/train_tfidf_baselines.py \
  --methods tfidf_lr,tfidf_svm \
  --tasks density,size,location \
  --seed 13 \
  --gate g2 \
  --tag main_g2_seed13
```

5 个 seed：

```text
13
42
87
3407
31415
```

### Step 3：补 Vanilla PubMedBERT

对每个 task 和 seed 执行。

Density 示例：

```bash
python -u scripts/phaseA2/train_vanilla_density.py \
  --seed 13 \
  --gate g2 \
  --tag main_g2_seed13 \
  --train-batch-size 160 \
  --eval-batch-size 64 \
  --dataloader-num-workers 8
```

Size 示例：

```bash
python -u scripts/phaseA2/train_vanilla_size.py \
  --seed 13 \
  --gate g2 \
  --tag main_g2_seed13 \
  --train-batch-size 160 \
  --eval-batch-size 64 \
  --dataloader-num-workers 8
```

Location 示例：

```bash
python -u scripts/phaseA2/train_vanilla_location.py \
  --seed 13 \
  --gate g2 \
  --tag main_g2_seed13 \
  --train-batch-size 160 \
  --eval-batch-size 64 \
  --dataloader-num-workers 8
```

如果显存或吞吐不稳定，将 `--train-batch-size` 降到 128 或 96，不改变其他口径。

### Step 4：准备 A1 supervision-source 数据

完整 single-LF 数据重构：

```bash
python -u scripts/phaseA2/rebuild_ws_by_source.py \
  --tasks density,size,location \
  --all-single-sources \
  --gate g2
```

单个 task / 单个 source 示例：

```bash
python -u scripts/phaseA2/rebuild_ws_by_source.py \
  --tasks density \
  --sources LF-D1 \
  --source-name lf_d1 \
  --gate g2
```

之后用现有 MWS-CFE 入口训练 source-specific 数据。例如：

```bash
python -u scripts/phaseA2/train_mws_density.py \
  --seed 13 \
  --gate g2 \
  --tag a1_source_lf_d1_seed13 \
  --ws-data-dir outputs/phaseA2/ws_source/lf_d1/density
```

### Step 5：最终汇总

所有补跑完成后执行：

```bash
python -u scripts/phaseA2/aggregate_a2_5_results.py --strict
```

如果 strict 通过，`outputs/phaseA2/tables` 下的表即可进入论文或答辩材料整理。

---

## 6. 哪些结果复用，哪些重跑

### 6.1 复用

复用当前已经完成的新版 A2 结果：

1. `mws_cfe_*_main_g2_seed*.json`
2. `mws_cfe_*_aqg_g*_seed*.json`
3. `mws_cfe_*_aagg_uniform_seed*.json`
4. `mws_cfe_density_results_p1_len*_seed*.json`
5. `mws_cfe_density_results_p3_*_seed*.json`

### 6.2 必须重跑或首次运行

必须按新版 A2 入口补齐：

1. `tfidf_lr` x 3 tasks x 5 seeds
2. `tfidf_svm` x 3 tasks x 5 seeds
3. `vanilla_pubmedbert` x 3 tasks x 5 seeds
4. A1 supervision-source 的选定 variants

### 6.3 不能复用

不能进入新版公平主表：

1. 旧 Phase 5 `baselines_summary.json`
2. 旧 Phase 5 `ml_lr` / `ml_svm`
3. 旧 Phase 5 `pubmedbert_*_results.json`
4. 旧 gold sanity-check
5. researcher-reviewed subset 主线结果

---

## 7. 最终如何成表

最终成表只使用：

```text
outputs/phaseA2/results
```

聚合脚本按如下规则读取：

```text
{method}_{task}_results_{tag}_seed{seed}.json
```

正式主表使用：

1. `tfidf_lr`
2. `tfidf_svm`
3. `vanilla_pubmedbert`
4. `mws_cfe`

正式主表 task：

1. `density`
2. `size`，表中显示为 `has_size`
3. `location`

正式主表指标：

1. `accuracy`
2. density/location 的 `macro_f1`
3. size 的 `f1`

所有正式表格使用 5 seeds 的：

```text
mean ± std
```

manifest 缺 seed 时，表可以生成 draft，但不能作为正式 5-seed 结果；正式提交前必须运行：

```bash
python -u scripts/phaseA2/aggregate_a2_5_results.py --strict
```

