# 模块2 Plan B 执行 Runbook

日期：2026-04-19

用途：给人工直接执行 Plan B 模块2重构实验。本文档不是规划文档；所有命令均按当前脚本真实 CLI 编写。

当前原则：

1. 模块3继续暂停。
2. 先跑 Wave 0 垂直切片，不直接启动完整 5 seeds。
3. Wave 0 没达到最低判断标准时，立即停下修方法，不继续烧 GPU。
4. 旧单阶段 density 五分类不再进入正文主表。

## 0. 通用执行环境

以下环境块在每个 shell 会话最开始执行一次：

```bash
cd /home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup
mkdir -p logs/planb outputs/phaseA2_planB

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup/outputs/phase5/hf_cache
export HUGGINGFACE_HUB_CACHE=/home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup/outputs/phase5/hf_cache/hub
export TRANSFORMERS_CACHE=/home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup/outputs/phase5/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
```

推荐统一使用：

```bash
conda run -n follow-up python -u <script> ... > logs/planb/<name>.stdout.log 2>&1
```

同时，训练脚本内部还会写入自己的行缓冲日志，例如 `logs/train_mws_density_stage1_<tag>.log`。

## 1. 可执行性核查

### 1.1 已核查脚本 CLI

真实接口来自 `conda run -n follow-up python <script> --help`。

| 脚本 | `--seed` | `--tag` | `--gate` | `--train-batch-size` | `--eval-batch-size` | `--dataloader-num-workers` | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `build_planb_density_two_stage.py` | 否 | 否 | 否 | 否 | 否 | 否 | 只构建数据；支持 `--ws-source-dir`, `--phase5-data-dir`, `--output-dir`, `--source-mode`, `--single-lf-name`, `--negative-confidence`, `--max-rows`, `--log` |
| `train_mws_density_stage1.py` | 是 | 是 | 是 | 是 | 是 | 是 | 继承 MWS 公共训练参数；支持 `--no-confidence-weight`, `--loss-type`, `--focal-gamma` |
| `train_mws_density_stage2.py` | 是 | 是 | 是 | 是 | 是 | 是 | 同上 |
| `train_vanilla_density_stage1.py` | 是 | 是 | 是 | 是 | 是 | 是 | 继承 Vanilla PubMedBERT 训练参数；`--no-confidence-weight` 参数存在但 Vanilla 不使用 confidence weighting |
| `train_vanilla_density_stage2.py` | 是 | 是 | 是 | 是 | 是 | 是 | 同上 |
| `train_planb_baselines.py` | 是 | 是 | 是 | 否 | 否 | 否 | CPU baseline；支持 `--methods`, `--tasks`, `--task-data-dir`, `--input-field`, `--use-confidence-weight`, `--log` |
| `aggregate_planb_results.py` | 否 | 否 | 否 | 否 | 否 | 否 | 聚合；支持 `--results-dir`, `--output-dir`, `--main-tag`, `--expected-seeds`, `--strict` |
| `plot_planb_parameters.py` | 否 | 否 | 否 | 否 | 否 | 否 | 参数图；支持 `--summary-csv`, `--output-dir` |

### 1.2 Two-stage density 数据输出

数据构建命令默认输出到：

```text
outputs/phaseA2_planB/
```

具体文件：

```text
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g1.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g2.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g3.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g4.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g5.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_val_ws.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_test_ws.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_val.jsonl
outputs/phaseA2_planB/density_stage1/density_stage1_test.jsonl

outputs/phaseA2_planB/density_stage2/density_stage2_train_ws.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g1.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g2.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g3.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g4.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g5.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_val_ws.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_test_ws.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_val.jsonl
outputs/phaseA2_planB/density_stage2/density_stage2_test.jsonl

outputs/phaseA2_planB/density_two_stage_summary.json
```

### 1.3 结果文件命名规则

聚合脚本依赖以下命名规则：

```text
<method>_<task>_results_<tag>_seed<seed>.json
```

因此训练时必须把 `--tag` 写成：

```text
<logical_tag>_seed<seed>
```

示例：

```bash
--tag wave0_seed42
--tag planb_full_seed13
--tag ab_wo_confidence_seed3407
--tag p1_len160_seed31415
```

聚合后解析为：

```text
tag=wave0, seed=42
tag=planb_full, seed=13
tag=ab_wo_confidence, seed=3407
tag=p1_len160, seed=31415
```

Stage 1 输出：

```text
outputs/phaseA2_planB/results/regex_cue_density_stage1_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_lr_density_stage1_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_svm_density_stage1_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_mlp_density_stage1_results_<tag>.json
outputs/phaseA2_planB/results/vanilla_pubmedbert_density_stage1_results_<tag>.json
outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_<tag>.json
```

Stage 2 输出：

```text
outputs/phaseA2_planB/results/regex_cue_density_stage2_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_lr_density_stage2_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_svm_density_stage2_results_<tag>.json
outputs/phaseA2_planB/results/tfidf_mlp_density_stage2_results_<tag>.json
outputs/phaseA2_planB/results/vanilla_pubmedbert_density_stage2_results_<tag>.json
outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_<tag>.json
```

PLM 模型目录：

```text
outputs/phaseA2_planB/models/density_stage1_mws_cfe_<tag>/
outputs/phaseA2_planB/models/density_stage2_mws_cfe_<tag>/
outputs/phaseA2_planB/models/density_stage1_vanilla_pubmedbert_<tag>/
outputs/phaseA2_planB/models/density_stage2_vanilla_pubmedbert_<tag>/
```

### 1.4 聚合脚本识别规则

`aggregate_planb_results.py` 识别以下任务：

```text
density_stage1
density_stage2
size
location
```

主表由 `--main-tag` 决定。例如：

```bash
--main-tag wave0
--main-tag planb_full
```

消融表固定识别这些 tag：

```text
planb_full
ab_wo_quality_gate
ab_wo_weighted_aggregation
ab_wo_confidence
ab_wo_section
ab_wo_multisource
```

参数讨论固定识别这些 tag：

```text
P1: p1_len64, p1_len96, planb_full, p1_len160, p1_len192
P2: p2_g1, planb_full, p2_g3, p2_g4, p2_g5
P3: p3_mention_text, planb_full, p3_findings_text, p3_impression_text, p3_findings_impression_text, p3_full_text
```

聚合输出：

```text
outputs/phaseA2_planB/tables/planb_main_table.csv
outputs/phaseA2_planB/tables/planb_main_table.json
outputs/phaseA2_planB/tables/planb_ablation_table.csv
outputs/phaseA2_planB/tables/planb_ablation_table.json
outputs/phaseA2_planB/tables/planb_parameter_summary.csv
outputs/phaseA2_planB/tables/planb_parameter_summary.json
outputs/phaseA2_planB/tables/planb_manifest_report.json
```

## 2. Batch / workers 推荐配置

硬件假设：单张 RTX 3090 24GB。

| 训练类型 | `--train-batch-size` | `--eval-batch-size` | `--dataloader-num-workers` | 备注 |
|---|---:|---:|---:|---|
| MWS Stage 1 | 16 | 64 | 4 | 默认推荐；Stage 1 全量样本较多 |
| MWS Stage 2 | 16 | 64 | 4 | 为公平和稳定先不放大 batch |
| Vanilla Stage 1 | 16 | 64 | 4 | 与 MWS 对齐 |
| Vanilla Stage 2 | 16 | 64 | 4 | 与 MWS 对齐 |
| size/location PLM | 16 | 64 | 4 | 如 GPU 空闲可保持一致 |

OOM 降级顺序：

1. `--train-batch-size 8 --eval-batch-size 32 --gradient-accumulation-steps 2`
2. `--dataloader-num-workers 2`
3. `--max-length 96`
4. 仍 OOM 时使用 `--train-batch-size 4 --eval-batch-size 16 --gradient-accumulation-steps 4`

CUDA assert / index 越界处理：

1. 立即停止当前 wave，不继续排队后续 GPU 任务。
2. 设置 `export CUDA_LAUNCH_BLOCKING=1` 后用同一命令加 `--max-train-samples 512 --max-val-samples 256 --max-test-samples 256` 复现。
3. 检查 `outputs/phaseA2_planB/density_two_stage_summary.json` 中 Stage 2 是否只含 `solid/part_solid/ground_glass/calcified`。
4. 检查命令是否把 Stage 1 的 `--phase5-data-dir` 指到了 `outputs/phaseA2_planB/density_stage1`，Stage 2 指到了 `outputs/phaseA2_planB/density_stage2`。
5. CUDA assert 后重启 Python 进程；不要在同一污染进程里继续训练。

## 3. Wave 0：垂直切片验证

目标：只跑 `seed=42`，只验证 density two-stage rescue 是否真的工作。

Wave 0 方法矩阵：

```text
Regex / cue-only
TF-IDF + LR
TF-IDF + SVM
Vanilla PubMedBERT
MWS-CFE
```

Wave 0 任务矩阵：

```text
density_stage1
density_stage2
```

不跑 `tfidf_mlp`，不跑 `has_size/location`。原因是 Wave 0 是最小救援闭环，不是正式主表。

### 3.1 构建 two-stage density 数据

输入路径：

```text
outputs/phaseA1/density/
outputs/phase5/datasets/
```

输出路径：

```text
outputs/phaseA2_planB/density_stage1/
outputs/phaseA2_planB/density_stage2/
```

命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --ws-source-dir outputs/phaseA1/density \
  --phase5-data-dir outputs/phase5/datasets \
  --output-dir outputs/phaseA2_planB \
  --source-mode weighted \
  --negative-confidence 0.5 \
  --log logs/planb/wave0_build_planb_density_two_stage.log \
  > logs/planb/wave0_build_planb_density_two_stage.stdout.log 2>&1
```

检查：

```bash
test -f outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g2.jsonl
test -f outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g2.jsonl
test -f outputs/phaseA2_planB/density_two_stage_summary.json
wc -l outputs/phaseA2_planB/density_stage1/density_stage1_train_ws_g2.jsonl
wc -l outputs/phaseA2_planB/density_stage2/density_stage2_train_ws_g2.jsonl
```

通过标准：

1. `density_stage1_train_ws_g2.jsonl` 非空。
2. `density_stage2_train_ws_g2.jsonl` 非空。
3. `density_two_stage_summary.json` 存在。

### 3.2 跑 Regex / TF-IDF baselines

输出结果目录：

```text
outputs/phaseA2_planB/results/
```

命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_planb_baselines.py \
  --methods regex_cue,tfidf_lr,tfidf_svm \
  --tasks density_stage1,density_stage2 \
  --seed 42 \
  --gate g2 \
  --output-dir outputs/phaseA2_planB \
  --input-field section_aware_text \
  --tag wave0_seed42 \
  --log logs/planb/wave0_baselines_seed42.log \
  > logs/planb/wave0_baselines_seed42.stdout.log 2>&1
```

应生成 6 个结果文件：

```bash
find outputs/phaseA2_planB/results -maxdepth 1 -name '*_density_stage*_results_wave0_seed42.json' | wc -l
```

其中 baseline 至少应包含：

```bash
test -f outputs/phaseA2_planB/results/regex_cue_density_stage1_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/tfidf_lr_density_stage1_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/tfidf_svm_density_stage1_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/regex_cue_density_stage2_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/tfidf_lr_density_stage2_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/tfidf_svm_density_stage2_results_wave0_seed42.json
```

### 3.3 跑 Vanilla PubMedBERT

Stage 1 命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage1.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage1 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
  --output-dir outputs/phaseA2_planB \
  --seed 42 \
  --gate g2 \
  --tag wave0_seed42 \
  --input-field section_aware_text \
  --max-length 128 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --dataloader-num-workers 4 \
  > logs/planb/wave0_vanilla_density_stage1_seed42.stdout.log 2>&1
```

Stage 2 命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage2.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage2 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
  --output-dir outputs/phaseA2_planB \
  --seed 42 \
  --gate g2 \
  --tag wave0_seed42 \
  --input-field section_aware_text \
  --max-length 128 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --dataloader-num-workers 4 \
  > logs/planb/wave0_vanilla_density_stage2_seed42.stdout.log 2>&1
```

输出模型目录：

```text
outputs/phaseA2_planB/models/density_stage1_vanilla_pubmedbert_wave0_seed42/
outputs/phaseA2_planB/models/density_stage2_vanilla_pubmedbert_wave0_seed42/
```

输出结果：

```bash
test -f outputs/phaseA2_planB/results/vanilla_pubmedbert_density_stage1_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/vanilla_pubmedbert_density_stage2_results_wave0_seed42.json
```

### 3.4 跑 MWS-CFE

Stage 1 命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage1.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage1 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
  --output-dir outputs/phaseA2_planB \
  --seed 42 \
  --gate g2 \
  --tag wave0_seed42 \
  --input-field section_aware_text \
  --max-length 128 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --dataloader-num-workers 4 \
  > logs/planb/wave0_mws_density_stage1_seed42.stdout.log 2>&1
```

Stage 2 命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage2.py \
  --ws-data-dir outputs/phaseA2_planB/density_stage2 \
  --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
  --output-dir outputs/phaseA2_planB \
  --seed 42 \
  --gate g2 \
  --tag wave0_seed42 \
  --input-field section_aware_text \
  --max-length 128 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --dataloader-num-workers 4 \
  > logs/planb/wave0_mws_density_stage2_seed42.stdout.log 2>&1
```

输出模型目录：

```text
outputs/phaseA2_planB/models/density_stage1_mws_cfe_wave0_seed42/
outputs/phaseA2_planB/models/density_stage2_mws_cfe_wave0_seed42/
```

输出结果：

```bash
test -f outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_wave0_seed42.json
test -f outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_wave0_seed42.json
```

额外检查 confidence-aware 是否启用：

```bash
python -c "import json; p='outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_wave0_seed42.json'; d=json.load(open(p)); print(d.get('method_components'))"
python -c "import json; p='outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_wave0_seed42.json'; d=json.load(open(p)); print(d.get('method_components'))"
```

通过标准：输出中应包含：

```text
'confidence_weighting': True
'sample_weight_field': 'ws_confidence'
```

### 3.5 Wave 0 聚合

输出聚合表路径：

```text
outputs/phaseA2_planB/tables_wave0/planb_main_table.csv
outputs/phaseA2_planB/tables_wave0/planb_main_table.json
outputs/phaseA2_planB/tables_wave0/planb_manifest_report.json
```

命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/aggregate_planb_results.py \
  --results-dir outputs/phaseA2_planB/results \
  --output-dir outputs/phaseA2_planB/tables_wave0 \
  --main-tag wave0 \
  --expected-seeds 42 \
  > logs/planb/wave0_aggregate.stdout.log 2>&1
```

检查：

```bash
test -f outputs/phaseA2_planB/tables_wave0/planb_main_table.csv
test -f outputs/phaseA2_planB/tables_wave0/planb_manifest_report.json
python -c "import json; d=json.load(open('outputs/phaseA2_planB/tables_wave0/planb_manifest_report.json')); print('records=', d['records']); print('incomplete=', sum(not r['complete'] for r in d['manifest']))"
```

Wave 0 应至少有 10 个 density 结果文件：

```bash
find outputs/phaseA2_planB/results -maxdepth 1 -name '*_density_stage*_results_wave0_seed42.json' | wc -l
```

通过标准：

1. 结果数量为 `10`。
2. `planb_main_table.csv` 存在。
3. `mws_cfe_density_stage1_results_wave0_seed42.json` 中 `phase5_test_results.auprc` 非空。
4. `mws_cfe_density_stage2_results_wave0_seed42.json` 中 `phase5_test_results.macro_f1` 非空。

### 3.6 Wave 0 先验判断标准

只用 `seed=42`，不能做最终论文结论，但可以决定是否继续烧 5 seeds。

强通过：

1. Stage 1：MWS-CFE 的 AUPRC 高于 Vanilla PubMedBERT，且高于 TF-IDF + LR/SVM。
2. Stage 2：MWS-CFE 的 Macro-F1 高于 Vanilla PubMedBERT，且高于 TF-IDF + LR/SVM。
3. Regex / cue-only 不能明显支配 MWS-CFE，否则说明任务仍然过度词面化。

条件通过：

1. MWS-CFE 在 Stage 1 或 Stage 2 其中一个阶段为最强。
2. 另一个阶段与最强方法差距不超过 3 个百分点。
3. MWS-CFE 明显优于旧单阶段失败口径中的自身表现。

立即停止，不跑 Wave 1：

1. Stage 1 MWS-CFE AUPRC 低于 Vanilla PubMedBERT 超过 5 个百分点，并且也低于 TF-IDF + SVM。
2. Stage 2 MWS-CFE Macro-F1 低于 Vanilla PubMedBERT 超过 5 个百分点，并且也低于 TF-IDF + SVM。
3. MWS-CFE 同时输给 Regex / cue-only 和 TF-IDF + SVM，说明当前方法增量没有出现。
4. 结果文件中 `confidence_weighting` 不是 `True`。
5. Stage 2 类别分布极端塌缩，例如只预测单一 subtype。

如果触发立即停止条件，停止在 Wave 0 聚合后。不要进入 Wave 1。优先回到以下修正点：

1. 检查 `section_aware_text` 是否过长导致 mention 信号被截断。
2. 检查 `negative-confidence` 是否过高或过低。
3. 检查 Stage 2 explicit subset 是否混入非肺结节。
4. 尝试 `--loss-type focal --focal-gamma 2.0` 作为方法增强，而不是继续扩大 seeds。

## 4. Wave 1：正式主表 5 seeds

前提：Wave 0 至少达到条件通过。

正式 seeds：

```bash
export PLANB_SEEDS="13 42 87 3407 31415"
```

正式主表 tag：

```text
planb_full
```

### 4.1 CPU baseline 全任务

任务：

```text
density_stage1,density_stage2,size,location
```

方法：

```text
regex_cue,tfidf_lr,tfidf_svm,tfidf_mlp
```

命令：

```bash
for SEED in 13 42 87 3407 31415; do
  conda run -n follow-up python -u scripts/phaseA2/train_planb_baselines.py \
    --methods regex_cue,tfidf_lr,tfidf_svm,tfidf_mlp \
    --tasks density_stage1,density_stage2,size,location \
    --seed ${SEED} \
    --gate g2 \
    --output-dir outputs/phaseA2_planB \
    --input-field section_aware_text \
    --tag planb_full_seed${SEED} \
    --log logs/planb/wave1_baselines_seed${SEED}.log \
    > logs/planb/wave1_baselines_seed${SEED}.stdout.log 2>&1
done
```

应生成：

```text
4 methods × 4 tasks × 5 seeds = 80 JSON
```

检查：

```bash
find outputs/phaseA2_planB/results -maxdepth 1 -name '*_results_planb_full_seed*.json' | wc -l
```

### 4.2 Vanilla PubMedBERT 5 seeds

Density Stage 1/2：

```bash
for SEED in 13 42 87 3407 31415; do
  conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage1.py \
    --ws-data-dir outputs/phaseA2_planB/density_stage1 \
    --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field section_aware_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_vanilla_density_stage1_seed${SEED}.stdout.log 2>&1

  conda run -n follow-up python -u scripts/phaseA2/train_vanilla_density_stage2.py \
    --ws-data-dir outputs/phaseA2_planB/density_stage2 \
    --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field section_aware_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_vanilla_density_stage2_seed${SEED}.stdout.log 2>&1
done
```

Size/location：

```bash
for SEED in 13 42 87 3407 31415; do
  conda run -n follow-up python -u scripts/phaseA2/train_vanilla_size.py \
    --ws-data-dir outputs/phaseA1/size \
    --phase5-data-dir outputs/phase5/datasets \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field mention_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_vanilla_size_seed${SEED}.stdout.log 2>&1

  conda run -n follow-up python -u scripts/phaseA2/train_vanilla_location.py \
    --ws-data-dir outputs/phaseA1/location \
    --phase5-data-dir outputs/phase5/datasets \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field mention_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_vanilla_location_seed${SEED}.stdout.log 2>&1
done
```

应生成：

```text
4 tasks × 5 seeds = 20 Vanilla JSON
```

### 4.3 MWS-CFE 5 seeds

Density Stage 1/2：

```bash
for SEED in 13 42 87 3407 31415; do
  conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage1.py \
    --ws-data-dir outputs/phaseA2_planB/density_stage1 \
    --phase5-data-dir outputs/phaseA2_planB/density_stage1 \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field section_aware_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_mws_density_stage1_seed${SEED}.stdout.log 2>&1

  conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage2.py \
    --ws-data-dir outputs/phaseA2_planB/density_stage2 \
    --phase5-data-dir outputs/phaseA2_planB/density_stage2 \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field section_aware_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_mws_density_stage2_seed${SEED}.stdout.log 2>&1
done
```

Size/location：

```bash
for SEED in 13 42 87 3407 31415; do
  conda run -n follow-up python -u scripts/phaseA2/train_mws_size.py \
    --ws-data-dir outputs/phaseA1/size \
    --phase5-data-dir outputs/phase5/datasets \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field mention_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_mws_size_seed${SEED}.stdout.log 2>&1

  conda run -n follow-up python -u scripts/phaseA2/train_mws_location.py \
    --ws-data-dir outputs/phaseA1/location \
    --phase5-data-dir outputs/phase5/datasets \
    --output-dir outputs/phaseA2_planB \
    --seed ${SEED} \
    --gate g2 \
    --tag planb_full_seed${SEED} \
    --input-field mention_text \
    --max-length 128 \
    --train-batch-size 16 \
    --eval-batch-size 64 \
    --dataloader-num-workers 4 \
    > logs/planb/wave1_mws_location_seed${SEED}.stdout.log 2>&1
done
```

应生成：

```text
4 tasks × 5 seeds = 20 MWS JSON
```

### 4.4 Wave 1 聚合

命令：

```bash
conda run -n follow-up python -u scripts/phaseA2/aggregate_planb_results.py \
  --results-dir outputs/phaseA2_planB/results \
  --output-dir outputs/phaseA2_planB/tables \
  --main-tag planb_full \
  --expected-seeds 13,42,87,3407,31415 \
  > logs/planb/wave1_aggregate.stdout.log 2>&1
```

正式主表路径：

```text
outputs/phaseA2_planB/tables/planb_main_table.csv
outputs/phaseA2_planB/tables/planb_main_table.json
```

检查：

```bash
test -f outputs/phaseA2_planB/tables/planb_main_table.csv
python -c "import json; d=json.load(open('outputs/phaseA2_planB/tables/planb_manifest_report.json')); rows=[r for r in d['manifest'] if r['scope']=='main_table']; print('main incomplete=', [r for r in rows if not r['complete']])"
```

通过标准：

1. 主表 main entries 无缺失 seeds。
2. `planb_main_table.csv` 中 6 个方法均有 density Stage 1/2 结果。
3. `planb_main_table.csv` 中 `MWS-CFE (Ours)` 的 Stage 1 AUPRC 和 Stage 2 Macro-F1 不为空。

## 5. Wave 2：标准消融

前提：Wave 1 主表值得继续。

消融只跑 MWS-CFE 的 density Stage 1/2。

Full 已由 Wave 1 的 `planb_full` 提供。

### 5.1 构建 w/o weighted aggregation 数据

先构建 uniform WS：

```bash
conda run -n follow-up python -u scripts/phaseA1/build_ws_datasets.py \
  --input-dir outputs/phase5/datasets \
  --output-dir outputs/phaseA2/ws_uniform \
  --tasks density \
  --uniform-weights \
  --log logs/planb/wave2_build_ws_uniform_density.log \
  > logs/planb/wave2_build_ws_uniform_density.stdout.log 2>&1
```

再构建 two-stage：

```bash
conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --ws-source-dir outputs/phaseA2/ws_uniform/density \
  --phase5-data-dir outputs/phase5/datasets \
  --output-dir outputs/phaseA2_planB_uniform \
  --source-mode uniform \
  --negative-confidence 0.5 \
  --log logs/planb/wave2_build_planb_uniform.log \
  > logs/planb/wave2_build_planb_uniform.stdout.log 2>&1
```

### 5.2 构建 w/o multi-source supervision 数据

```bash
conda run -n follow-up python -u scripts/phaseA2/build_planb_density_two_stage.py \
  --ws-source-dir outputs/phaseA1/density \
  --phase5-data-dir outputs/phase5/datasets \
  --output-dir outputs/phaseA2_planB_single_lf \
  --source-mode single_lf \
  --single-lf-name LF-D1 \
  --negative-confidence 0.5 \
  --log logs/planb/wave2_build_planb_single_lf.log \
  > logs/planb/wave2_build_planb_single_lf.stdout.log 2>&1
```

### 5.3 跑 w/o quality gate

```bash
for SEED in 13 42 87 3407 31415; do
  for STAGE in 1 2; do
    conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
      --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --output-dir outputs/phaseA2_planB \
      --seed ${SEED} \
      --gate g1 \
      --tag ab_wo_quality_gate_seed${SEED} \
      --input-field section_aware_text \
      --max-length 128 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --dataloader-num-workers 4 \
      > logs/planb/wave2_ab_wo_quality_gate_stage${STAGE}_seed${SEED}.stdout.log 2>&1
  done
done
```

### 5.4 跑 w/o weighted aggregation

```bash
for SEED in 13 42 87 3407 31415; do
  for STAGE in 1 2; do
    conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
      --ws-data-dir outputs/phaseA2_planB_uniform/density_stage${STAGE} \
      --phase5-data-dir outputs/phaseA2_planB_uniform/density_stage${STAGE} \
      --output-dir outputs/phaseA2_planB \
      --seed ${SEED} \
      --gate g2 \
      --tag ab_wo_weighted_aggregation_seed${SEED} \
      --input-field section_aware_text \
      --max-length 128 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --dataloader-num-workers 4 \
      > logs/planb/wave2_ab_wo_weighted_aggregation_stage${STAGE}_seed${SEED}.stdout.log 2>&1
  done
done
```

### 5.5 跑 w/o confidence-aware training

```bash
for SEED in 13 42 87 3407 31415; do
  for STAGE in 1 2; do
    conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
      --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --output-dir outputs/phaseA2_planB \
      --seed ${SEED} \
      --gate g2 \
      --tag ab_wo_confidence_seed${SEED} \
      --input-field section_aware_text \
      --max-length 128 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --dataloader-num-workers 4 \
      --no-confidence-weight \
      > logs/planb/wave2_ab_wo_confidence_stage${STAGE}_seed${SEED}.stdout.log 2>&1
  done
done
```

### 5.6 跑 w/o section strategy

```bash
for SEED in 13 42 87 3407 31415; do
  for STAGE in 1 2; do
    conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
      --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
      --output-dir outputs/phaseA2_planB \
      --seed ${SEED} \
      --gate g2 \
      --tag ab_wo_section_seed${SEED} \
      --input-field full_text \
      --max-length 128 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --dataloader-num-workers 4 \
      > logs/planb/wave2_ab_wo_section_stage${STAGE}_seed${SEED}.stdout.log 2>&1
  done
done
```

### 5.7 跑 w/o multi-source supervision

```bash
for SEED in 13 42 87 3407 31415; do
  for STAGE in 1 2; do
    conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
      --ws-data-dir outputs/phaseA2_planB_single_lf/density_stage${STAGE} \
      --phase5-data-dir outputs/phaseA2_planB_single_lf/density_stage${STAGE} \
      --output-dir outputs/phaseA2_planB \
      --seed ${SEED} \
      --gate g2 \
      --tag ab_wo_multisource_seed${SEED} \
      --input-field section_aware_text \
      --max-length 128 \
      --train-batch-size 16 \
      --eval-batch-size 64 \
      --dataloader-num-workers 4 \
      > logs/planb/wave2_ab_wo_multisource_stage${STAGE}_seed${SEED}.stdout.log 2>&1
  done
done
```

### 5.8 Wave 2 聚合

```bash
conda run -n follow-up python -u scripts/phaseA2/aggregate_planb_results.py \
  --results-dir outputs/phaseA2_planB/results \
  --output-dir outputs/phaseA2_planB/tables \
  --main-tag planb_full \
  --expected-seeds 13,42,87,3407,31415 \
  > logs/planb/wave2_aggregate.stdout.log 2>&1
```

消融表：

```text
outputs/phaseA2_planB/tables/planb_ablation_table.csv
outputs/phaseA2_planB/tables/planb_ablation_table.json
```

检查：

```bash
test -f outputs/phaseA2_planB/tables/planb_ablation_table.csv
python -c "import json; d=json.load(open('outputs/phaseA2_planB/tables/planb_manifest_report.json')); rows=[r for r in d['manifest'] if r['scope']=='ablation']; print([r for r in rows if not r['complete']])"
```

通过标准：ablation scope 没有缺失 seeds。

## 6. Wave 3：参数讨论与绘图

前提：Wave 2 完成或至少 Full 已稳定。

参数讨论只跑 MWS-CFE 的 density Stage 1/2。

### 6.1 P1 max_seq_length

128 复用 `planb_full`。补跑 64/96/160/192：

```bash
for SEED in 13 42 87 3407 31415; do
  for LEN in 64 96 160 192; do
    for STAGE in 1 2; do
      conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
        --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --output-dir outputs/phaseA2_planB \
        --seed ${SEED} \
        --gate g2 \
        --tag p1_len${LEN}_seed${SEED} \
        --input-field section_aware_text \
        --max-length ${LEN} \
        --train-batch-size 16 \
        --eval-batch-size 64 \
        --dataloader-num-workers 4 \
        > logs/planb/wave3_p1_len${LEN}_stage${STAGE}_seed${SEED}.stdout.log 2>&1
    done
  done
done
```

### 6.2 P2 quality_gate

G2 复用 `planb_full`。补跑 G1/G3/G4/G5：

```bash
for SEED in 13 42 87 3407 31415; do
  for GATE in g1 g3 g4 g5; do
    TAG_GATE=$(echo ${GATE} | tr '[:lower:]' '[:lower:]')
    for STAGE in 1 2; do
      conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
        --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --output-dir outputs/phaseA2_planB \
        --seed ${SEED} \
        --gate ${GATE} \
        --tag p2_${TAG_GATE}_seed${SEED} \
        --input-field section_aware_text \
        --max-length 128 \
        --train-batch-size 16 \
        --eval-batch-size 64 \
        --dataloader-num-workers 4 \
        > logs/planb/wave3_p2_${TAG_GATE}_stage${STAGE}_seed${SEED}.stdout.log 2>&1
    done
  done
done
```

### 6.3 P3 section/input strategy

`section_aware_text` 复用 `planb_full`。补跑其他 input field：

```bash
for SEED in 13 42 87 3407 31415; do
  for FIELD in mention_text findings_text impression_text findings_impression_text full_text; do
    if [ "${FIELD}" = "mention_text" ]; then TAG_FIELD="p3_mention_text"; fi
    if [ "${FIELD}" = "findings_text" ]; then TAG_FIELD="p3_findings_text"; fi
    if [ "${FIELD}" = "impression_text" ]; then TAG_FIELD="p3_impression_text"; fi
    if [ "${FIELD}" = "findings_impression_text" ]; then TAG_FIELD="p3_findings_impression_text"; fi
    if [ "${FIELD}" = "full_text" ]; then TAG_FIELD="p3_full_text"; fi
    for STAGE in 1 2; do
      conda run -n follow-up python -u scripts/phaseA2/train_mws_density_stage${STAGE}.py \
        --ws-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --phase5-data-dir outputs/phaseA2_planB/density_stage${STAGE} \
        --output-dir outputs/phaseA2_planB \
        --seed ${SEED} \
        --gate g2 \
        --tag ${TAG_FIELD}_seed${SEED} \
        --input-field ${FIELD} \
        --max-length 128 \
        --train-batch-size 16 \
        --eval-batch-size 64 \
        --dataloader-num-workers 4 \
        > logs/planb/wave3_${TAG_FIELD}_stage${STAGE}_seed${SEED}.stdout.log 2>&1
    done
  done
done
```

### 6.4 Wave 3 聚合和绘图

聚合：

```bash
conda run -n follow-up python -u scripts/phaseA2/aggregate_planb_results.py \
  --results-dir outputs/phaseA2_planB/results \
  --output-dir outputs/phaseA2_planB/tables \
  --main-tag planb_full \
  --expected-seeds 13,42,87,3407,31415 \
  > logs/planb/wave3_aggregate.stdout.log 2>&1
```

绘图：

```bash
conda run -n follow-up python -u scripts/phaseA2/plot_planb_parameters.py \
  --summary-csv outputs/phaseA2_planB/tables/planb_parameter_summary.csv \
  --output-dir outputs/phaseA2_planB/figures \
  > logs/planb/wave3_plot_parameters.stdout.log 2>&1
```

输出表：

```text
outputs/phaseA2_planB/tables/planb_parameter_summary.csv
outputs/phaseA2_planB/tables/planb_parameter_summary.json
```

输出图：

```text
outputs/phaseA2_planB/figures/p1_max_seq_length_stage_1_auprc.svg
outputs/phaseA2_planB/figures/p1_max_seq_length_stage_2_macro_f1.svg
outputs/phaseA2_planB/figures/p2_quality_gate_stage_1_auprc.svg
outputs/phaseA2_planB/figures/p2_quality_gate_stage_2_macro_f1.svg
outputs/phaseA2_planB/figures/p3_section_input_strategy_stage_1_auprc.svg
outputs/phaseA2_planB/figures/p3_section_input_strategy_stage_2_macro_f1.svg
```

检查：

```bash
test -f outputs/phaseA2_planB/tables/planb_parameter_summary.csv
find outputs/phaseA2_planB/figures -maxdepth 1 -name '*.svg' | wc -l
python -c "import json; d=json.load(open('outputs/phaseA2_planB/tables/planb_manifest_report.json')); rows=[r for r in d['manifest'] if r['scope'].startswith('P')]; print([r for r in rows if not r['complete']])"
```

通过标准：

1. 参数表存在。
2. SVG 图数量至少为 6。
3. P1/P2/P3 manifest 无缺失 seeds。

## 7. 最终执行清单

### 先做：Wave 0 最小闭环

1. 执行通用环境块。
2. 构建 two-stage density 数据：
   - `build_planb_density_two_stage.py`
3. 跑 `seed=42` baseline：
   - `train_planb_baselines.py --methods regex_cue,tfidf_lr,tfidf_svm --tasks density_stage1,density_stage2`
4. 跑 `seed=42` Vanilla：
   - `train_vanilla_density_stage1.py`
   - `train_vanilla_density_stage2.py`
5. 跑 `seed=42` MWS-CFE：
   - `train_mws_density_stage1.py`
   - `train_mws_density_stage2.py`
6. 聚合：
   - `aggregate_planb_results.py --main-tag wave0 --expected-seeds 42`
7. 查看：
   - `outputs/phaseA2_planB/tables_wave0/planb_main_table.csv`

如果 Wave 0 触发停止条件，停在这里，不进入 Wave 1。

### 再做：Wave 1 正式主表

1. 跑 5 seeds CPU baseline 全任务。
2. 跑 5 seeds Vanilla density two-stage + size/location。
3. 跑 5 seeds MWS-CFE density two-stage + size/location。
4. 聚合：
   - `aggregate_planb_results.py --main-tag planb_full --expected-seeds 13,42,87,3407,31415`
5. 查看：
   - `outputs/phaseA2_planB/tables/planb_main_table.csv`

### 然后做：Wave 2 标准消融

1. 构建 uniform WS。
2. 构建 single-LF two-stage 数据。
3. 跑：
   - `ab_wo_quality_gate`
   - `ab_wo_weighted_aggregation`
   - `ab_wo_confidence`
   - `ab_wo_section`
   - `ab_wo_multisource`
4. 聚合并查看：
   - `outputs/phaseA2_planB/tables/planb_ablation_table.csv`

### 最后做：Wave 3 参数讨论与图

1. 跑 P1：`p1_len64`, `p1_len96`, `p1_len160`, `p1_len192`。
2. 跑 P2：`p2_g1`, `p2_g3`, `p2_g4`, `p2_g5`。
3. 跑 P3：`p3_mention_text`, `p3_findings_text`, `p3_impression_text`, `p3_findings_impression_text`, `p3_full_text`。
4. 聚合：
   - `planb_parameter_summary.csv`
5. 绘图：
   - `plot_planb_parameters.py`
6. 查看：
   - `outputs/phaseA2_planB/figures/*.svg`

## 8. 当前可进入阶段

当前可以进入 Plan B Wave 0 垂直切片执行阶段。

下一步必须先跑最小闭环，而不是直接跑完整 5 seeds。只有 Wave 0 至少达到条件通过，才进入 Wave 1。
