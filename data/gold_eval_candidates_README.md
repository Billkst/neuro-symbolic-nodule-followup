# Gold Evaluation Candidates v1

## 概述

从 Phase 5 test split 中分层抽样 80 条 mention-level 样本，用于人工核对 `density_category`、`size_mm`、`location_lobe` 三个 silver label 字段。

## 输入文件

| 项目 | 值 |
|---|---|
| 输入文件 | `outputs/phase5/datasets/density_test.jsonl` |
| 输入样本总数 | 42,057 条 mention-level 样本 |
| 输入 subject 数 | 4,973 |
| 数据构建脚本 | `scripts/phase5/build_datasets.py` |
| split 方式 | subject-level 70/15/15 (train/val/test)，seed=42 |
| split 无泄漏 | 已验证 subject_id 与 train/val 零重叠 |

## 抽样脚本

`scripts/export_gold_eval_candidates.py`

## 随机种子

`42`（与 Phase 5 数据集构建一致）

## 抽样逻辑

总计 80 条，分 4 层抽样，优先级 D > C > A > B（高优先级层先抽，已抽样本不重复进入低优先级层）：

| 层 | 名称 | 数量 | 筛选条件 |
|---|---|---|---|
| D | 稀有类 | 10 | `density_label == "part_solid"` (6条，优先 explicit) 或 `location_label == "lingula"` (4条，优先非 unclear density) |
| C | 边界/非标准 | 15 | `mention_text` 匹配 `multiple\|several\|numerous\|bilateral\|scattered` 且 `density_label != "unclear"` |
| A | 明确阳性 | 30 | `label_quality == "explicit"` 且 `density != "unclear"` 且 `has_size` 且 `location ∈ 真实叶位`；按 density 子分层 (ground_glass:14, solid:10, calcified:4, part_solid:2) |
| B | 缺失信息 | 25 | B1: unclear density 但有 size 或 location (8条)；B2: 无 size 且非 unclear density (8条)；B3: 无 location 且非 unclear density 且有 size (9条) |

## 字段映射

| 输出字段 | 源字段 | 说明 |
|---|---|---|
| `sample_id` | `sample_id` | 格式 `{note_id}__{mention_idx}` |
| `subject_id` | `subject_id` | 患者 ID |
| `note_id` | `note_id` | 放射报告 ID |
| `mention_text` | `mention_text` | 单条结节 mention 原文 |
| `text_window` | 由 `full_text` + `mention_text` 构造 | 在 whitespace-normalized `full_text` 中定位 `mention_text`，取前后各 200 字符窗口 |
| `silver_density_category` | `density_label` | solid / part_solid / ground_glass / calcified / unclear |
| `silver_has_size` | `has_size` | 布尔值 |
| `silver_size_mm` | `size_label` | 浮点数 (mm)，无则为空 |
| `silver_location_lobe` | `location_label` | RUL/RML/RLL/LUL/LLL/bilateral/lingula/unclear/no_location；源数据中 `null` 映射为 `no_location` |
| `split` | 硬编码 `"test"` | 所有样本均来自 test split |

## test split 保证

1. 输入文件 `density_test.jsonl` 本身即为 `build_datasets.py` 按 subject-level split 生成的 test 集
2. 导出后额外验证：80 条样本的 75 个 unique subject_id 与 train/val 的 subject_id 集合零重叠

## 复现

```bash
conda activate follow-up
python -u scripts/export_gold_eval_candidates.py
```

输出：`data/gold_eval_candidates_v1.csv`

## 实际分布

```
density:  calcified=11, ground_glass=31, part_solid=11, solid=19, unclear=8
location: LLL=7, LUL=11, RLL=7, RML=7, RUL=14, bilateral=5, lingula=4, no_location=22, unclear=3
has_size: True=60, False=20
```
