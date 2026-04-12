# Phase A2 实验执行手册

> 生成日期：2026-04-12
> 硬件环境：NVIDIA RTX 3090 (24GB) / 64 CPU cores
> Conda 环境：follow-up
> 工作目录：/home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup

---

## 0. 前置核验

### 0.1 Vanilla PubMedBERT 复用条件核验

Phase 5 PubMedBERT 结果直接作为 A2 Vanilla 基线（决策1），核验结果如下：

| 条件 | 状态 | 说明 |
|------|------|------|
| Subject-level split 一致 | ✅ | seed=42, train 23204 / val 4972 / test 4973 subjects, 完全 disjoint |
| 输入形式 = mention_text | ✅ | Phase 5 使用 mention_text, max_length=128 |
| 标签 = 单源 Regex teacher | ✅ | Phase 5 标签由 nodule_extractor.py 的 Regex 规则生成 |
| 训练配置与评测协议一致 | ✅ | model=BiomedBERT, seed=42, lr=2e-5, eval=macro_f1/f1 |

Phase 5 已有结果（Silver test）：
- Density (5类: solid/part_solid/ground_glass/calcified/unclear): accuracy=0.9993, macro_f1=0.9985
- Size (2类: no_size/has_size): accuracy=0.9994, f1=0.9992
- Location (9类: 8 lobes + no_location): accuracy=0.9999, macro_f1=0.9998

**结论：4 个条件全部一致，Phase 5 结果直接复用，不重训。**

### 0.2 Split 一致性核验

已执行，结果为 PASS：

[Start] Phase A2 split consistency verification @ 2026-04-12T12:51:14.605519+00:00
[Phase5] split_manifest loaded: {'train': {'mention_count': 201947, 'subject_count': 23204}, 'val': {'mention_count': 44890, 'subject_count': 4972}, 'test': {'mention_count': 42057, 'subject_count': 4973}}
[Phase5] density/train: 23204 subjects
[Phase5] density/val: 4972 subjects
[Phase5] density/test: 4973 subjects
[Phase5] size/train: 23204 subjects
[Phase5] size/val: 4972 subjects
[Phase5] size/test: 4973 subjects
[Phase5] location/train: 23204 subjects
[Phase5] location/val: 4972 subjects
[Phase5] location/test: 4973 subjects
[Phase5] All tasks: train/val/test subject-disjoint ✓
[WS] density/train: ws=9862 p5=23204 ✓ subset
[WS] density/val: ws=2148 p5=4972 ✓ subset
[WS] density/test: ws=2065 p5=4973 ✓ subset
[WS] size/train: ws=22523 p5=23204 ✓ subset
[WS] size/val: ws=4843 p5=4972 ✓ subset
[WS] size/test: ws=4831 p5=4973 ✓ subset
[WS] location/train: ws=19578 p5=23204 ✓ subset
[WS] location/val: ws=4193 p5=4972 ✓ subset
[WS] location/test: ws=4190 p5=4973 ✓ subset
[WS] density: cross-split disjoint ✓
[WS] size: cross-split disjoint ✓
[WS] location: cross-split disjoint ✓
[Gate] density/g1: 40023 samples, 9862 subjects ✓
[Gate] density/g2: 36038 samples, 9601 subjects ✓
[Gate] density/g3: 8734 samples, 3831 subjects ✓
[Gate] density/g4: 35907 samples, 9595 subjects ✓
[Gate] density/g5: 4618 samples, 2773 subjects ✓
[Gate] size/g1: 168433 samples, 22523 subjects ✓
[Gate] size/g2: 168433 samples, 22523 subjects ✓
[Gate] size/g3: 26133 samples, 9697 subjects ✓
[Gate] size/g4: 168433 samples, 22523 subjects ✓
[Gate] size/g5: 26133 samples, 9697 subjects ✓
[Gate] location/g1: 136687 samples, 19578 subjects ✓
[Gate] location/g2: 131233 samples, 19395 subjects ✓
[Gate] location/g3: 17456 samples, 8219 subjects ✓
[Gate] location/g4: 131130 samples, 19391 subjects ✓
[Gate] location/g5: 11899 samples, 6290 subjects ✓

[Verdict] PASS
[Saved] /home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup/outputs/phaseA2/split_verification.json
[Done] 39.2s

核验要点：
- WS subjects 是 Phase 5 subjects 的严格子集（因 ABSTAIN 导致部分 subject 被过滤）
- train/val/test 之间零 subject 重叠
- 所有 Gate 筛选后的 subjects 均为 train subjects 的子集

---

## 1. 完整运行矩阵

### 1.1 主结果 + Gold

| # | 实验 | 任务 | Gate | 输入 | 训练样本 | 需训练 | 预估时间 |
|---|------|------|------|------|---------|--------|---------|
| M1 | MWS-CFE 主结果 | density | G2 | mention_text | 36,038 | ✅ | ~25min |
| M2 | MWS-CFE 主结果 | size | G2 | mention_text | 168,433 | ✅ | ~70min |
| M3 | MWS-CFE 主结果 | location | G2 | mention_text | 131,233 | ✅ | ~55min |
| G1 | Gold sanity-check | D/S/L | G2 | mention_text | - | 评测 | ~2min |

### 1.2 消融实验（5 组）

#### A-lf: Single-source vs Multi-source
无需额外训练。Single-source = Phase 5 Vanilla（已有），Multi-source = 主结果 M1-M3。

#### A-qg: Quality Gate Strength（与 P3 共享）

| # | 任务 | Gate | 训练样本 | 需训练 | 预估时间 |
|---|------|------|---------|--------|---------|
| Q1 | density | G1 | 40,023 | ✅ | ~28min |
| Q2 | size | G1 | 168,433 | ✅ | ~70min |
| Q3 | location | G1 | 136,687 | ✅ | ~57min |
| Q4 | density | G3 | 8,734 | ✅ | ~8min |
| Q5 | size | G3 | 26,133 | ✅ | ~15min |
| Q6 | location | G3 | 17,456 | ✅ | ~12min |
| Q7 | density | G4 | 35,907 | ✅ | ~25min |
| Q8 | size | G4 | 168,433 | ✅ | ~70min |
| Q9 | location | G4 | 131,130 | ✅ | ~55min |
| Q10 | density | G5 | 4,618 | ✅ | ~5min |
| Q11 | size | G5 | 26,133 | ✅ | ~15min |
| Q12 | location | G5 | 11,899 | ✅ | ~10min |

G2 = 主结果 M1-M3（复用），共 12 个新训练 run。

#### A-window: mention_text vs full_text

| # | 任务 | 输入 | 需训练 | 预估时间 |
|---|------|------|--------|---------|
| W1 | density | full_text | ✅ | ~35min |
| W2 | size | full_text | ✅ | ~90min |
| W3 | location | full_text | ✅ | ~75min |

mention_text = 主结果 M1-M3（复用），共 3 个新训练 run。

#### A-agg: Weighted Vote vs Uniform Vote

| # | 任务 | 聚合方式 | 需训练 | 前置 | 预估时间 |
|---|------|---------|--------|------|---------|
| A1 | density | uniform | ✅ | 数据重生成 | ~25min |
| A2 | size | uniform | ✅ | 数据重生成 | ~70min |
| A3 | location | uniform | ✅ | 数据重生成 | ~55min |

Weighted = 主结果 M1-M3（复用），共 3 个新训练 run + 数据预处理。

#### A-section: Section Strategy

| # | 任务 | Section | 需训练 | 前置 | 预估时间 |
|---|------|---------|--------|------|---------|
| S1 | density | findings_only | ✅ | 数据过滤 | ~20min |
| S2 | size | findings_only | ✅ | 数据过滤 | ~60min |
| S3 | location | findings_only | ✅ | 数据过滤 | ~45min |
| S4 | density | impression_only | ✅ | 数据过滤 | ~15min |
| S5 | size | impression_only | ✅ | 数据过滤 | ~40min |
| S6 | location | impression_only | ✅ | 数据过滤 | ~35min |

findings+impression = 主结果 M1-M3（复用），共 6 个新训练 run + 数据预处理。

### 1.3 参数讨论（3 组）

#### P1: max_seq_length

| # | 任务 | max_length | 需训练 | 预估时间 |
|---|------|-----------|--------|---------|
| P1a | density | 64 | ✅ | ~20min |
| P1b | size | 64 | ✅ | ~55min |
| P1c | location | 64 | ✅ | ~45min |
| P1d | density | 96 | ✅ | ~22min |
| P1e | size | 96 | ✅ | ~60min |
| P1f | location | 96 | ✅ | ~50min |
| P1g | density | 160 | ✅ | ~28min |
| P1h | size | 160 | ✅ | ~75min |
| P1i | location | 160 | ✅ | ~60min |
| P1j | density | 192 | ✅ | ~30min |
| P1k | size | 192 | ✅ | ~80min |
| P1l | location | 192 | ✅ | ~65min |

128 = 主结果 M1-M3（复用），共 12 个新训练 run。

#### P2: Section Strategy
与 A-section + A-window 共享结果：
- findings_only = S1-S3
- impression_only = S4-S6
- findings+impression = M1-M3（主结果）
- full_text = W1-W3（A-window）
无需额外训练。

#### P3: Quality Gate
与 A-qg 完全共享（G1-G5 = Q1-Q12 + M1-M3）。
正文主配置按 G2 出表，density 重点分析 G1/G3/G5。
无需额外训练。

### 1.4 总量统计

| 类别 | 新训练 runs | 复用 runs | 数据预处理 |
|------|-----------|----------|-----------|
| 主结果 | 3 | 0 | 无 |
| Gold 评测 | 0 (评测) | 0 | 无 |
| A-lf | 0 | 3+3 | 无 |
| A-qg / P3 | 12 | 3 | 无 |
| A-window / P2 (full_text) | 3 | 3 | 无 |
| A-agg | 3 | 3 | 需生成 uniform WS 数据 |
| A-section | 6 | 3 | 需按 section 过滤数据 |
| P1 | 12 | 3 | 无 |
| **合计** | **39** | - | 2 项数据预处理 |


---

## 2. 精确执行命令

> 所有命令在项目根目录执行，conda 环境 `follow-up` 已激活。
> 推荐配置：`--train-batch-size 32 --gradient-accumulation-steps 2`（有效 batch=64）。
> 日志实时查看：`tail -f logs/<日志文件名>`

### 2.0 环境准备

```bash
conda activate follow-up
cd /home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs
```

### 2.1 主结果 (M1-M3)

```bash
# M1: Density G2 (训练样本 36,038)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag main_g2 \
    > logs/train_mws_density_main_g2.log 2>&1 &
echo "M1 PID: $!"

# M2: Size G2 (训练样本 168,433)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag main_g2 \
    > logs/train_mws_size_main_g2.log 2>&1 &
echo "M2 PID: $!"

# M3: Location G2 (训练样本 131,233)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag main_g2 \
    > logs/train_mws_location_main_g2.log 2>&1 &
echo "M3 PID: $!"
```

**注意**：M1 (density) 显存占用约 8-10GB，可与 M2 或 M3 并行跑（总占用 ~18GB）。M2+M3 不建议同时跑（总占用可能超 24GB）。

**产物路径**：
- 模型：`outputs/phaseA2/models/{density,size,location}_mws_cfe_main_g2/`
- 结果：`outputs/phaseA2/results/mws_cfe_{density,size,location}_results_main_g2.json`
- 日志：`logs/train_mws_{density,size,location}_main_g2.log`

### 2.2 Gold Sanity-Check (G1)

> 依赖 M1-M3 全部完成

```bash
# G1: Gold 评测 (N=62)
# --tag main_g2 对应训练时的 --tag main_g2，模型目录 = {task}_mws_cfe_main_g2
python -u scripts/phaseA2/run_gold_eval_mws.py \
    --tag main_g2 \
    > logs/gold_eval_mws_main_g2.log 2>&1
```

**产物路径**：`outputs/phaseA2/results/gold_eval_mws_main_g2.json`

### 2.3 消融 A-lf

无需训练。在结果汇总表中直接对比：
- Single-source (Vanilla PubMedBERT) = `outputs/phase5/results/pubmedbert_{task}_results.json`
- Multi-source (MWS-CFE) = `outputs/phaseA2/results/mws_cfe_{task}_results_main_g2.json`

### 2.4 消融 A-qg / 参数讨论 P3 (Q1-Q12)

```bash
# === G1 (No Gate) ===
# Q1: Density G1 (训练样本 40,023)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g1 \
    > logs/train_mws_density_aqg_g1.log 2>&1 &

# Q2: Size G1 (训练样本 168,433)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g1 \
    > logs/train_mws_size_aqg_g1.log 2>&1 &

# Q3: Location G1 (训练样本 136,687)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g1 \
    > logs/train_mws_location_aqg_g1.log 2>&1 &

# === G3 (Cov >= 2) ===
# Q4: Density G3 (训练样本 8,734)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g3 \
    > logs/train_mws_density_aqg_g3.log 2>&1 &

# Q5: Size G3 (训练样本 26,133)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g3 \
    > logs/train_mws_size_aqg_g3.log 2>&1 &

# Q6: Location G3 (训练样本 17,456)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g3 \
    > logs/train_mws_location_aqg_g3.log 2>&1 &

# === G4 (Agr >= 0.8) ===
# Q7: Density G4 (训练样本 35,907)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g4 \
    > logs/train_mws_density_aqg_g4.log 2>&1 &

# Q8: Size G4 (训练样本 168,433)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g4 \
    > logs/train_mws_size_aqg_g4.log 2>&1 &

# Q9: Location G4 (训练样本 131,130)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g4 \
    > logs/train_mws_location_aqg_g4.log 2>&1 &

# === G5 (Strict: Conf >= 0.8 AND Cov >= 2) ===
# Q10: Density G5 (训练样本 4,618)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g5 \
    > logs/train_mws_density_aqg_g5.log 2>&1 &

# Q11: Size G5 (训练样本 26,133)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g5 \
    > logs/train_mws_size_aqg_g5.log 2>&1 &

# Q12: Location G5 (训练样本 11,899)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aqg_g5 \
    > logs/train_mws_location_aqg_g5.log 2>&1 &
```

**G2 = 主结果 M1-M3，直接复用。**

### 2.5 消融 A-window (W1-W3)

```bash
# W1: Density full_text (训练样本 36,038)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --input-field full_text \
    --train-batch-size 16 --gradient-accumulation-steps 4 \
    --tag awin_full_text \
    > logs/train_mws_density_awin_full_text.log 2>&1 &

# W2: Size full_text (训练样本 168,433)
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --input-field full_text \
    --train-batch-size 16 --gradient-accumulation-steps 4 \
    --tag awin_full_text \
    > logs/train_mws_size_awin_full_text.log 2>&1 &

# W3: Location full_text (训练样本 131,233)
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --input-field full_text \
    --train-batch-size 16 --gradient-accumulation-steps 4 \
    --tag awin_full_text \
    > logs/train_mws_location_awin_full_text.log 2>&1 &
```

**注意**：full_text 输入序列更长（max_length=128 会截断），显存占用与 mention_text 相同。但如果需要增大 max_length 以容纳 full_text，则需降低 batch_size。此处保持 max_length=128 以控制变量（仅改变输入字段）。

**mention_text = 主结果 M1-M3，直接复用。**

### 2.6 消融 A-agg (A1-A3)

#### 前置：生成 Uniform 权重 WS 数据

需要修改 `scripts/phaseA1/build_ws_datasets.py` 使所有 LF 权重设为 1.0（uniform vote），重新生成数据到独立目录。

**方案 A**（推荐）：如果 `build_ws_datasets.py` 支持 `--uniform-weights` 参数：
```bash
python -u scripts/phaseA1/build_ws_datasets.py \
    --uniform-weights \
    --output-dir outputs/phaseA2/ws_uniform \
    > logs/build_ws_uniform.log 2>&1
```

**方案 B**：如果不支持该参数，需要临时修改代码中的 LF 权重为全 1.0，运行后恢复。具体步骤：
1. 备份 `src/weak_supervision/labeling_functions/` 下的权重配置
2. 将所有 LF 权重改为 1.0
3. 运行 `build_ws_datasets.py --output-dir outputs/phaseA2/ws_uniform`
4. 恢复原始权重

**确认数据生成后再执行训练**：

```bash
# A1: Density uniform (训练样本 ~36K，取决于 uniform 聚合结果)
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_uniform/density \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aagg_uniform \
    > logs/train_mws_density_aagg_uniform.log 2>&1 &

# A2: Size uniform
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_uniform/size \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aagg_uniform \
    > logs/train_mws_size_aagg_uniform.log 2>&1 &

# A3: Location uniform
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_uniform/location \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag aagg_uniform \
    > logs/train_mws_location_aagg_uniform.log 2>&1 &
```

**Weighted = 主结果 M1-M3，直接复用。**

### 2.7 消融 A-section (S1-S6)

#### 前置：按 Section 过滤 WS 数据

需要创建 section 过滤脚本（或使用现有的 `filter_ws_by_section.py`），将 WS 数据按 FINDINGS / IMPRESSION 段落过滤。

```bash
# 生成 findings_only 数据
python -u scripts/phaseA2/filter_ws_by_section.py \
    --section findings \
    --output-dir outputs/phaseA2/ws_findings \
    > logs/filter_ws_findings.log 2>&1

# 生成 impression_only 数据
python -u scripts/phaseA2/filter_ws_by_section.py \
    --section impression \
    --output-dir outputs/phaseA2/ws_impression \
    > logs/filter_ws_impression.log 2>&1
```

**注意**：如果 `filter_ws_by_section.py` 尚未实现，需要我先编写该脚本。请确认后告知。

**确认数据生成后再执行训练**：

```bash
# === Findings Only ===
# S1: Density findings
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_findings/density \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_findings \
    > logs/train_mws_density_asec_findings.log 2>&1 &

# S2: Size findings
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_findings/size \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_findings \
    > logs/train_mws_size_asec_findings.log 2>&1 &

# S3: Location findings
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_findings/location \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_findings \
    > logs/train_mws_location_asec_findings.log 2>&1 &

# === Impression Only ===
# S4: Density impression
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_impression/density \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_impression \
    > logs/train_mws_density_asec_impression.log 2>&1 &

# S5: Size impression
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_impression/size \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_impression \
    > logs/train_mws_size_asec_impression.log 2>&1 &

# S6: Location impression
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 \
    --ws-data-dir outputs/phaseA2/ws_impression/location \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag asec_impression \
    > logs/train_mws_location_asec_impression.log 2>&1 &
```

**findings+impression = 主结果 M1-M3，直接复用。**

### 2.8 参数讨论 P1: max_seq_length (P1a-P1l)

```bash
# === max_length=64 ===
# P1a: Density len64
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --max-length 64 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len64 \
    > logs/train_mws_density_p1_len64.log 2>&1 &

# P1b: Size len64
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --max-length 64 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len64 \
    > logs/train_mws_size_p1_len64.log 2>&1 &

# P1c: Location len64
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --max-length 64 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len64 \
    > logs/train_mws_location_p1_len64.log 2>&1 &

# === max_length=96 ===
# P1d: Density len96
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --max-length 96 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len96 \
    > logs/train_mws_density_p1_len96.log 2>&1 &

# P1e: Size len96
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --max-length 96 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len96 \
    > logs/train_mws_size_p1_len96.log 2>&1 &

# P1f: Location len96
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --max-length 96 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len96 \
    > logs/train_mws_location_p1_len96.log 2>&1 &

# === max_length=160 ===
# P1g: Density len160
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --max-length 160 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len160 \
    > logs/train_mws_density_p1_len160.log 2>&1 &

# P1h: Size len160
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --max-length 160 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len160 \
    > logs/train_mws_size_p1_len160.log 2>&1 &

# P1i: Location len160
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --max-length 160 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len160 \
    > logs/train_mws_location_p1_len160.log 2>&1 &

# === max_length=192 ===
# P1j: Density len192
nohup python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --max-length 192 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len192 \
    > logs/train_mws_density_p1_len192.log 2>&1 &

# P1k: Size len192
nohup python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --max-length 192 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len192 \
    > logs/train_mws_size_p1_len192.log 2>&1 &

# P1l: Location len192
nohup python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --max-length 192 \
    --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag p1_len192 \
    > logs/train_mws_location_p1_len192.log 2>&1 &
```

**128 = 主结果 M1-M3，直接复用。**

### 2.9 参数讨论 P2 / P3

- **P2**：与 A-section (S1-S6) + A-window (W1-W3) + 主结果 (M1-M3) 共享，无需额外训练。
- **P3**：与 A-qg (Q1-Q12) + 主结果 (M1-M3) 完全共享，无需额外训练。


---

## 3. 推荐资源配置

### 3.1 硬件环境

| 资源 | 规格 |
|------|------|
| GPU | NVIDIA RTX 3090 24GB |
| CPU | 64 cores |
| 内存 | 需确认（建议 >= 64GB） |

### 3.2 按实验规模分级配置

#### 大规模（训练样本 > 100K：Size G1/G2/G4, Location G1/G2/G4）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| train-batch-size | 32 | Phase 5 实际使用值，3090 可承受 |
| gradient-accumulation-steps | 2 | 有效 batch=64，提升收敛稳定性 |
| eval-batch-size | 64 | 评估不需要梯度，可以更大 |
| fp16 | 自动启用 | 代码中 `fp16=torch.cuda.is_available()` |
| dataloader-num-workers | 8 | 64 核 CPU 可以支撑更多 workers |
| max-length | 128 | mention_text 默认长度 |
| epochs | 10 | 默认值，early stopping patience=3 |
| 预估显存 | ~10-12GB | BiomedBERT-base + batch32 + fp16 + seq128 |
| 预估时间 | 60-80min | 基于 Phase 5 实测 ~40min/5epochs 推算 |

#### 中规模（训练样本 30K-100K：Density G1/G2/G4, Location G3）

| 参数 | 推荐值 |
|------|--------|
| train-batch-size | 32 |
| gradient-accumulation-steps | 2 |
| dataloader-num-workers | 8 |
| 预估显存 | ~10-12GB |
| 预估时间 | 20-40min |

#### 小规模（训练样本 < 30K：Density G3/G5, Size G3/G5, Location G5）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| train-batch-size | 32 | 样本少但 batch 不需要降 |
| gradient-accumulation-steps | 1 | 样本少时不需要大有效 batch |
| dataloader-num-workers | 4 | 数据量小，不需要太多 workers |
| 预估显存 | ~8-10GB |
| 预估时间 | 5-15min |

#### full_text 输入（A-window W1-W3）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| train-batch-size | 16 | full_text 更长，降低 batch 防 OOM |
| gradient-accumulation-steps | 4 | 保持有效 batch=64 |
| max-length | 128 | 保持与主结果一致（控制变量） |
| 预估显存 | ~10-12GB | max_length=128 时与 mention_text 相同 |

### 3.3 显存估算公式

BiomedBERT-base (110M params) + fp16:
- 模型参数：~220MB (fp16)
- 优化器状态：~880MB (AdamW fp32)
- 激活值：~batch_size * seq_len * hidden_dim * num_layers * 2 bytes
- 估算：batch=32, seq=128 → ~8GB 总显存
- 估算：batch=32, seq=192 → ~10GB 总显存
- 估算：batch=16, seq=512 → ~12GB 总显存

**3090 24GB 安全线**：单任务 batch=32 + seq=128 占用 ~10GB，可同时跑 2 个 mention_text 任务。

### 3.4 并行执行建议

| 组合 | 总显存 | 可行性 |
|------|--------|--------|
| Density + Size (mention_text) | ~20GB | ✅ 可并行 |
| Density + Location (mention_text) | ~20GB | ✅ 可并行 |
| Size + Location (mention_text) | ~22GB | ⚠️ 勉强，建议监控 |
| 任意 2 个 full_text 任务 | ~22GB | ⚠️ 勉强 |
| 3 个任务同时 | ~30GB | ❌ 超出 24GB |

---

## 4. 执行顺序建议

### Wave 1（优先，核心结论）

**目标**：获得主结果 + Gold 对比 + Gate 敏感度全景。

```
顺序 1: M1 (density G2) + M2 (size G2) 并行     → ~70min
顺序 2: M3 (location G2)                         → ~55min
顺序 3: G1 (gold eval)                           → ~2min
顺序 4: Q1+Q2 (density+size G1) 并行             → ~70min
顺序 5: Q3 (location G1)                         → ~57min
顺序 6: Q4+Q5 (density+size G3) 并行             → ~15min
顺序 7: Q6 (location G3)                         → ~12min
顺序 8: Q7+Q8 (density+size G4) 并行             → ~70min
顺序 9: Q9 (location G4)                         → ~55min
顺序 10: Q10+Q11 (density+size G5) 并行          → ~15min
顺序 11: Q12 (location G5)                       → ~10min
```

**Wave 1 总时间**：~7-8 小时（含并行优化）

### Wave 2（次优先，输入敏感度）

**目标**：A-window + P1 序列长度敏感度。

```
顺序 12: W1+P1a (density full_text + density len64) 并行  → ~35min
顺序 13: W2 (size full_text)                               → ~90min
顺序 14: W3+P1b (location full_text + size len64) 并行     → ~75min
顺序 15: P1c (location len64)                              → ~45min
顺序 16: P1d+P1e (density+size len96) 并行                 → ~60min
顺序 17: P1f (location len96)                              → ~50min
顺序 18: P1g+P1h (density+size len160) 并行                → ~75min
顺序 19: P1i (location len160)                             → ~60min
顺序 20: P1j+P1k (density+size len192) 并行                → ~80min
顺序 21: P1l (location len192)                             → ~65min
```

**Wave 2 总时间**：~9-10 小时

### Wave 3（最后，需数据预处理）

**目标**：A-agg + A-section。

```
前置 A: 生成 uniform WS 数据                              → ~30min
前置 B: 生成 section-filtered WS 数据                     → ~30min
顺序 22: A1+S1 (density uniform + density findings) 并行  → ~25min
顺序 23: A2+S2 (size uniform + size findings) 并行        → ~70min
顺序 24: A3+S3 (location uniform + location findings) 并行 → ~55min
顺序 25: S4+S5 (density+size impression) 并行             → ~60min
顺序 26: S6 (location impression)                         → ~35min
```

**Wave 3 总时间**：~5-6 小时（含数据预处理）

### 总时间估算

| Wave | 时间 | 累计 |
|------|------|------|
| Wave 1 | ~7-8h | ~8h |
| Wave 2 | ~9-10h | ~18h |
| Wave 3 | ~5-6h | ~24h |
| **总计** | **~21-24h** | 约 1-1.5 天连续运行 |

---

## 5. 断点续跑与失败恢复

### 5.1 HuggingFace Trainer 的 Checkpoint 机制

训练脚本使用 `save_strategy="epoch"`，每个 epoch 结束时保存 checkpoint 到模型目录：
```
outputs/phaseA2/models/{task}_mws_cfe_{tag}/
├── checkpoint-{step}/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
└── ...
```

`--resume-from-checkpoint` 已添加到 CLI。使用方式：

```bash
python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --tag main_g2 \
    --resume-from-checkpoint outputs/phaseA2/models/density_mws_cfe_main_g2/checkpoint-1000
```

### 5.2 可复用产物

| 产物 | 路径 | 复用条件 |
|------|------|---------|
| Phase 5 Vanilla 结果 | `outputs/phase5/results/` | 直接复用，不重训 |
| Phase A1 WS 数据 | `outputs/phaseA1/{task}/` | 所有实验共享（除 A-agg/A-section） |
| Split 核验结果 | `outputs/phaseA2/split_verification.json` | 已通过，不需重跑 |
| 已完成的模型 | `outputs/phaseA2/models/{task}_mws_cfe_{tag}/` | 含 model.safetensors 即为完整 |
| 已完成的结果 | `outputs/phaseA2/results/mws_cfe_{task}_results_{tag}.json` | 存在即可复用 |

### 5.3 失败后处理方案

| 失败类型 | 处理方式 |
|---------|---------|
| OOM (显存不足) | 降低 `--train-batch-size` 到 16，相应增加 `--gradient-accumulation-steps` 到 4 |
| 训练中断（kill/断电） | 使用 `--resume-from-checkpoint` 从最近 checkpoint 恢复 |
| 训练发散（loss 爆炸） | 降低 `--learning-rate` 到 1e-5，或检查数据是否有异常 |
| 数据文件缺失 | 检查 `outputs/phaseA1/{task}/` 目录，必要时重跑 `build_ws_datasets.py` |
| 模型下载失败 | 确认 `HF_ENDPOINT=https://hf-mirror.com` 已设置，或手动下载模型到本地 |
| Early stopping 过早 | 增加 `--patience` 到 5，或检查 val 数据是否太小 |

### 5.4 哪些实验失败后不需要整组重跑

- **A-qg 各 Gate 之间完全独立**：G1 失败不影响 G3/G4/G5，单独重跑即可
- **P1 各 max_length 之间完全独立**：len64 失败不影响 len96/160/192
- **3 个任务之间完全独立**：density 失败不影响 size/location
- **Gold 评测可反复运行**：不修改模型，只读取模型做推理
- **A-agg/A-section 的数据预处理与训练独立**：数据生成成功后，训练失败只需重跑训练

---

## 6. 资源利用率优化建议

### 6.1 提高 GPU 利用率

**当前问题**：Phase 5 训练日志显示初始迭代速度 ~1-3 it/s，稳定后 ~12-14 it/s，说明数据加载初始化开销大。

**优化措施**：

1. **增大 batch_size 到 32**（Phase 5 实际用的就是 32）
   - 默认值 16 浪费了约 50% 的 GPU 计算能力
   - 3090 24GB 在 fp16 + seq128 下完全可以承受 batch=32

2. **启用 gradient_accumulation_steps=2**
   - 有效 batch=64，更稳定的梯度估计
   - 不增加显存，只增加计算时间（可忽略）

3. **启用 tf32 精度**（代码已包含）
   - `torch.set_float32_matmul_precision("high")` 已在脚本中设置
   - 3090 Ampere 架构支持 tf32，矩阵乘法速度提升 ~2x

4. **监控 GPU 利用率**
   ```bash
   # 实时监控（每 2 秒刷新）
   watch -n 2 nvidia-smi

   # 或使用 gpustat（更简洁）
   pip install gpustat
   watch -n 2 gpustat
   ```

### 6.2 提高 DataLoader 吞吐

**当前配置**：`num_workers=4`, `pin_memory=True`, `persistent_workers=True`（Phase A2 已优化）

**进一步优化**：

1. **增加 num_workers 到 8**
   ```bash
   --dataloader-num-workers 8
   ```
   64 核 CPU 完全可以支撑 8 个 worker。每个 worker 负责预取和 tokenize 一个 batch。

2. **确认 persistent_workers 已启用**
   - 代码中已有：`dataloader_persistent_workers=True if args.dataloader_num_workers > 0 else False`
   - 避免每个 epoch 重新创建 worker 进程的开销

3. **确认 pin_memory 已启用**
   - 代码中已有：`dataloader_pin_memory=True`
   - 加速 CPU→GPU 数据传输

### 6.3 减少 Tokenizer / I/O 瓶颈

1. **LazyMentionDataset 已实现按需 tokenize**
   - 当前实现在 `__getitem__` 中 tokenize，避免一次性加载全部数据到内存
   - 对于 200K 样本的数据集，这是正确的策略

2. **JSONL 文件 I/O 优化**
   - 数据文件已按行存储（JSONL），加载速度快
   - 如果 I/O 成为瓶颈（SSD 读取慢），可考虑预先 tokenize 并缓存为 Arrow 格式

3. **HuggingFace 模型缓存**
   - 首次运行会下载 BiomedBERT 模型（~440MB）
   - 后续运行从本地缓存加载，无网络开销
   - 缓存路径：`~/.cache/huggingface/hub/`

### 6.4 串行 vs 批量脚本化

**推荐：按 Wave 编写批量执行脚本**

```bash
#!/bin/bash
# scripts/phaseA2/run_wave1.sh
set -e
cd /home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup
export HF_ENDPOINT=https://hf-mirror.com

echo "=== Wave 1: 主结果 ==="

# M1+M2 并行
python -u scripts/phaseA2/train_mws_density.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag main_g2 \
    > logs/train_mws_density_main_g2.log 2>&1 &
PID_D=$!

python -u scripts/phaseA2/train_mws_size.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag main_g2 \
    > logs/train_mws_size_main_g2.log 2>&1 &
PID_S=$!

echo "等待 Density ($PID_D) 和 Size ($PID_S) 完成..."
wait $PID_D $PID_S

# M3 单独
python -u scripts/phaseA2/train_mws_location.py \
    --gate g2 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag main_g2 \
    > logs/train_mws_location_main_g2.log 2>&1
echo "主结果完成"

# G1: Gold 评测
python -u scripts/phaseA2/run_gold_eval_mws.py \
    --gate main_g2 \
    --model-base-dir outputs/phaseA2/models \
    > logs/gold_eval_mws_main_g2.log 2>&1
echo "Gold 评测完成"

echo "=== Wave 1: A-qg G1 ==="
python -u scripts/phaseA2/train_mws_density.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g1 \
    > logs/train_mws_density_aqg_g1.log 2>&1 &
PID_D=$!

python -u scripts/phaseA2/train_mws_size.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g1 \
    > logs/train_mws_size_aqg_g1.log 2>&1 &
PID_S=$!

wait $PID_D $PID_S

python -u scripts/phaseA2/train_mws_location.py \
    --gate g1 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g1 \
    > logs/train_mws_location_aqg_g1.log 2>&1

echo "=== Wave 1: A-qg G3 ==="
# G3 数据量小，3 个任务可以更激进地并行
python -u scripts/phaseA2/train_mws_density.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g3 \
    > logs/train_mws_density_aqg_g3.log 2>&1 &
PID_D=$!

python -u scripts/phaseA2/train_mws_size.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g3 \
    > logs/train_mws_size_aqg_g3.log 2>&1 &
PID_S=$!

wait $PID_D $PID_S

python -u scripts/phaseA2/train_mws_location.py \
    --gate g3 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g3 \
    > logs/train_mws_location_aqg_g3.log 2>&1

echo "=== Wave 1: A-qg G4 ==="
python -u scripts/phaseA2/train_mws_density.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g4 \
    > logs/train_mws_density_aqg_g4.log 2>&1 &
PID_D=$!

python -u scripts/phaseA2/train_mws_size.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g4 \
    > logs/train_mws_size_aqg_g4.log 2>&1 &
PID_S=$!

wait $PID_D $PID_S

python -u scripts/phaseA2/train_mws_location.py \
    --gate g4 --train-batch-size 32 --gradient-accumulation-steps 2 \
    --dataloader-num-workers 8 --tag aqg_g4 \
    > logs/train_mws_location_aqg_g4.log 2>&1

echo "=== Wave 1: A-qg G5 ==="
python -u scripts/phaseA2/train_mws_density.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g5 \
    > logs/train_mws_density_aqg_g5.log 2>&1 &
PID_D=$!

python -u scripts/phaseA2/train_mws_size.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g5 \
    > logs/train_mws_size_aqg_g5.log 2>&1 &
PID_S=$!

wait $PID_D $PID_S

python -u scripts/phaseA2/train_mws_location.py \
    --gate g5 --train-batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 --tag aqg_g5 \
    > logs/train_mws_location_aqg_g5.log 2>&1

echo "=== Wave 1 全部完成 ==="
```

**使用方式**：
```bash
chmod +x scripts/phaseA2/run_wave1.sh
nohup bash scripts/phaseA2/run_wave1.sh > logs/wave1_orchestrator.log 2>&1 &
tail -f logs/wave1_orchestrator.log
```

### 6.5 实时监控 Checklist

```bash
# 1. 查看所有训练进程
ps aux | grep train_mws | grep -v grep

# 2. 查看 GPU 使用
nvidia-smi

# 3. 查看某个实验的实时日志
tail -f logs/train_mws_density_main_g2.log

# 4. 查看某个实验是否完成（搜索 [Done]）
grep "\[Done\]" logs/train_mws_density_main_g2.log

# 5. 批量检查所有实验状态
for f in logs/train_mws_*.log; do
    echo -n "$f: "
    if grep -q "\[Done\]" "$f" 2>/dev/null; then
        echo "✅ 完成"
    elif [ -f "$f" ]; then
        tail -1 "$f"
    else
        echo "❌ 未开始"
    fi
done

# 6. 查看已完成的结果文件
ls -la outputs/phaseA2/results/
```

---

## 附录 A：Location 标签对齐说明（决策3）

MWS-CFE 的 location 分类器按 **8 类**训练：
`["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"]`

推理/评测阶段对 `no_location` 单独处理：
- 当 location 侧无有效证据时（WS 标签为 ABSTAIN），走 `no_location` fallback
- 最终输出与 Phase 5 兼容的 **9 类**结果

代码实现位于 `train_mws_cfe_common.py` 的 `evaluate_on_phase5_test()` 函数：
- Phase 5 test set 中 `location_label == "no_location"` 的样本直接预测为 `no_location`
- 其余样本由 8 类模型正常预测

**报告中需明确写清**：`no_location` 不是由多源 location 分类器直接学习得到，而是由 location 缺失/无证据 fallback 组成。

## 附录 B：待确认/待实现事项

> 以下 4 项已全部补实现（2026-04-12），可直接使用。

1. ✅ **`--resume-from-checkpoint` 参数**：已添加到 `train_mws_cfe_common.py`。用法：`--resume-from-checkpoint outputs/phaseA2/models/.../checkpoint-500`
2. ✅ **`build_ws_datasets.py --uniform-weights`**：已添加。同时 `rebuild_ws_uniform.py` 已修复 API 调用 bug（原代码传 dict 给 weighted_majority_vote，应传 list[LFOutput]）。
3. ✅ **`filter_ws_by_section.py`**：已实现，支持 `--section findings|impression|findings_impression`，带计时和过滤率统计。
4. ✅ **Gold 评测 `--tag` 参数**：`run_gold_eval_mws.py` 新增 `--tag` 参数，与 `--gate` 解耦。模型目录 = `{task}_mws_cfe_{tag}`。主结果用 `--tag main_g2`。
