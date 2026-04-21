# 模块2 Plan B cue-only 方法学审计

> 日期：2026-04-21  
> 审计对象：`Cue-only rules` 在 Plan B 主表中所有任务均为 `100.00 +/- 0.00` 的原因  
> 审计范围：`train_planb_baselines.py`、`build_planb_density_two_stage.py`、Phase5 数据构造、Phase A1 weak supervision 标签来源、最终 Plan B 结果 JSON  
> 约束：本报告只做方法学审计和写作建议，不补实验，不修改实验结果。

## 1. 审计结论

`Cue-only rules` 不应作为普通 baseline 继续放在正文主表中与可学习模型并列比较。它更适合作为 deterministic label-construction reference，即“确定性标签构造参照”单独呈现，或移入附录。

原因是：当前 Plan B 的评测标签，尤其 Phase5 test 标签，本身就是由 `mention_text` 上的规则抽取函数构造出来的；而 `Cue-only rules` 预测时也直接在同一个 `mention_text` 上调用同一组规则抽取函数。因此，`100.00 +/- 0.00` 不是模型泛化能力的证据，而是标签构造规则与预测规则完全同源导致的闭环一致性。

这不是训练集泄漏意义上的 label leakage，因为 cue-only 不训练、预测时也不直接读取 label 字段。但它构成 evaluation target leakage / label-construction proxy：评测目标由规则生成，评测方法复用同一规则，自然得到 100%。因此，如果将 cue-only 与 TF-IDF、PubMedBERT、MWS-CFE 放在同一正文主表中，会误导读者把它理解为一个独立强 baseline。

## 2. 代码证据

### 2.1 cue-only 预测逻辑

`scripts/phaseA2/train_planb_baselines.py` 明确说明 `regex_cue` 是 cue-only rules、没有可学习参数。其预测函数 `regex_predict` 对所有任务都只读取 `mention_text`，然后调用 `src.extractors.nodule_extractor` 中的规则函数：

1. density_stage1：`extract_density(mention_text)`，如果结果属于显式密度集合，则预测 `explicit_density`，否则预测 `unclear_or_no_evidence`。
2. density_stage2：`extract_density(mention_text)`，如果属于显式密度集合，则直接预测该密度亚型。
3. size：`extract_size(mention_text)`，非空则预测 `has_size=1`。
4. location：`extract_location(mention_text)`，非空且在合法标签集合内则预测对应位置，否则预测 `no_location`。

对应代码位置：

1. `task_labels` 从评测行读取真值字段：`density_stage1_label`、`density_stage2_label`、`has_size`、`location_label`，见 `scripts/phaseA2/train_planb_baselines.py:109-121`。
2. `regex_predict` 使用 `mention_text` 和 `extract_density` / `extract_size` / `extract_location` 生成预测，见 `scripts/phaseA2/train_planb_baselines.py:161-179`。
3. `evaluate_task` 使用上述真值字段和 cue-only 预测直接计算指标，见 `scripts/phaseA2/train_planb_baselines.py:207-229`。

### 2.2 规则抽取函数本身

`src/extractors/nodule_extractor.py` 中的三个函数是 cue-only 的实际特征来源：

1. `extract_size` 使用 mm/cm、二维/三维尺寸、范围等正则抽取尺寸，见 `src/extractors/nodule_extractor.py:50-96`。
2. `extract_density` 使用 part-solid、ground-glass、calcified、solid 等正则抽取密度类别，见 `src/extractors/nodule_extractor.py:99-127`。
3. `extract_location` 使用 RUL/RML/RLL/LUL/LLL/lingula/bilateral 等正则抽取位置，见 `src/extractors/nodule_extractor.py:130-150`。

### 2.3 Phase5 数据标签构造与 cue-only 同源

`scripts/phase5/build_datasets.py` 在构建 Phase5 mention-level 数据时，对每个 `mention_text` 直接调用同一组 extractor：

1. `density_label, density_text = extract_density(mention_text)`
2. `size_mm, size_text = extract_size(mention_text)`
3. `location_label, location_text = extract_location(mention_text)`
4. 随后写入 `density_label`、`has_size = size_mm is not None`、`location_label`

对应代码位置为 `scripts/phase5/build_datasets.py:289-314`。

这意味着 size 和 location 的 Phase5 test 真值字段，正是 cue-only 在预测阶段复用的同一规则函数输出。对 density，Phase5 的 `density_label` 同样来自 `extract_density(mention_text)`，随后 Plan B 再将它映射为 Stage 1 / Stage 2 标签。

### 2.4 Plan B two-stage density 标签构造

`scripts/phaseA2/build_planb_density_two_stage.py` 没有重新人工标注 density，而是把已有 density label 重构为两阶段任务：

1. `stage1_label_from_density(label)` 将 `solid`、`part_solid`、`ground_glass`、`calcified` 映射为 `explicit_density`，其他映射为 `unclear_or_no_evidence`，见 `scripts/phaseA2/build_planb_density_two_stage.py:118-119`。
2. 训练集从 weak supervision 的 `ws_label` 或指定 LF label 得到 `source_density_label`，再构造 `density_stage1_label` 和 `density_stage2_label`，见 `scripts/phaseA2/build_planb_density_two_stage.py:112-166`。
3. 验证/测试集从 Phase5 的 `density_label` 读取标签，构造 `density_stage1_label`；若 density 是显式类别，再写入 `density_stage2_label`，见 `scripts/phaseA2/build_planb_density_two_stage.py:169-184`。

由于 Phase5 的 `density_label` 来自 `extract_density(mention_text)`，而 cue-only 的 Stage 1 / Stage 2 预测也来自 `extract_density(mention_text)`，所以两者在评测集上天然一致。

### 2.5 Phase A1 weak supervision 标签来源

Phase A1 的 weak supervision 训练标签来自 labeling functions 的聚合：

1. `scripts/phaseA1/build_ws_datasets.py` 对每个 record 运行任务对应的 LFs，聚合为 `ws_label`，见 `scripts/phaseA1/build_ws_datasets.py:64-118`。
2. 训练记录中，density 写入 `density_label = ws_label`，size 写入 `has_size = ws_label == "true"`，location 写入 `location_label = ws_label`，见 `scripts/phaseA1/build_ws_datasets.py:142-160`。
3. density LFs、size LFs、location LFs 大多也是基于 `mention_text` 的正则或 cue 规则，例如 `LF-D1`、`LF-S1`、`LF-L1` 分别对应显式 density、size、location cue。

因此，训练标签也与 cue 特征高度同源；不过主表 100% 的直接原因主要在评测标签同样由 extractor 构造，而不是训练标签。

## 3. 已生成结果对齐情况

读取 `outputs/phaseA2_planB/results/regex_cue_*_results_planb_full_seed13.json` 后，主表中 cue-only 的 100% 与混淆矩阵完全一致：

| Task | Test samples | Result | Evidence |
|---|---:|---|---|
| density_stage1 | 42057 | accuracy / F1 / AUPRC / AUROC 均为 1.0 | confusion matrix 为 `[[5576, 0], [0, 36481]]` |
| density_stage2 | 5576 | accuracy / Macro-F1 均为 1.0 | confusion matrix 对四个密度类别全为对角线 |
| size | 42057 | accuracy / precision / recall / F1 均为 1.0 | `phase5_test_results` 全部为 1.0 |
| location | 42057 | accuracy / Macro-F1 均为 1.0 | confusion matrix 对 9 个位置标签全为对角线 |

对当前 workspace 中存在的 Phase5 test JSONL 进行逐行复算也确认：

| Task | File | Rows | Mismatches between label and extractor prediction |
|---|---|---:|---:|
| size | `outputs/phase5/datasets/size_test.jsonl` | 42057 | 0 |
| location | `outputs/phase5/datasets/location_test.jsonl` | 42057 | 0 |

density two-stage 原始 JSONL 当前不在 workspace 的相对路径下，但结果 JSON 已给出全对角混淆矩阵；代码路径也说明其评测标签由 Phase5 `density_label` 映射而来，而 Phase5 `density_label` 由 `extract_density(mention_text)` 构造。

## 4. 对四种解释的判断

### 4.1 是否是标签泄漏？

不是传统意义上的训练/测试泄漏。cue-only 没有训练阶段，也没有在预测时直接读取 `density_stage1_label`、`density_stage2_label`、`has_size` 或 `location_label`。

但它是 evaluation target leakage / label-construction leakage：评测标签由规则函数构造，预测也调用同一规则函数。因此它不应作为独立泛化能力 baseline 解释。

### 4.2 是否是 label-construction proxy？

是。这是最准确的解释。

cue-only 的预测函数与标签构造函数共享同一输入字段 `mention_text` 和同一 extractor 逻辑。主表中 100% 说明 cue-only 完美复现了当前 silver/test label construction，而不是说明它在真实临床 gold label 上一定完美。

### 4.3 是否说明任务被规则完全决定？

在当前 silver label / Phase5 constructed label 口径下，是的。当前评测任务被显式 cue 规则完全决定。

但这句话不能外推到真实人工 gold label。真实人工标注可能会考虑上下文、省略表达、跨句关联、否定、历史对比等因素；cue-only 在这种独立 gold label 上不一定仍为 100%。

### 4.4 是否是合理但极强的确定性 baseline？

它是合理的确定性参照，但不是适合与 learned baselines 并列的普通 baseline。

它可以回答“当前标签构造规则本身能达到什么上限”，但不能回答“模型是否学会了超越规则的临床语义泛化”。因此它适合作为 deterministic reference 或 label-construction sanity check，而不是正文主表里的普通 baseline 行。

## 5. 写作与落表建议

### 5.1 正文主表是否保留 cue-only？

建议不保留在正文主表中。

理由：

1. cue-only 与评测标签同源，100% 会压制所有 learned models，使正文主表的主要比较焦点变成“规则复现规则标签”。
2. 它不是独立模型能力比较；与 TF-IDF、Vanilla PubMedBERT、MWS-CFE 并列表达会造成方法学误读。
3. 如果正文主表保留该行，读者或审稿人很可能质疑标签泄漏或 circular evaluation。

### 5.2 推荐呈现方式

推荐方式：

1. 正文主表：移除 `Cue-only rules`，只展示 learned models，即 TF-IDF + LR、TF-IDF + SVM、TF-IDF + MLP、Vanilla PubMedBERT、MWS-CFE。
2. 正文文字：单独说明 cue-only deterministic reference 在当前 label-construction 口径下达到 100%，这表明评测标签高度规则化，因此不作为 learned-model baseline 比较。
3. 附录表：保留 cue-only 完整结果，并命名为 `Label-construction reference` 或 `Deterministic cue reference`。
4. 附录方法学说明：明确写出 cue-only 复用 `extract_density` / `extract_size` / `extract_location`，而 Phase5 labels 也由这些 extractor 构造。

可直接写入论文的方法学说明：

> We report the cue-only system as a deterministic label-construction reference rather than a learned baseline. Because the constructed evaluation labels are generated from the same mention-level extraction rules, the cue-only system recovers these labels perfectly. We therefore exclude it from the main learned-model comparison and provide it in the appendix as a sanity check for the silver-label construction.

中文说明：

> cue-only 系统应作为“标签构造参照”而非普通 baseline。当前评测标签由同一组 mention-level 规则抽取函数构造，因此 cue-only 能够完全复现这些标签。正文主表应聚焦 learned models，cue-only 结果建议移入附录，并用于说明当前 silver-label 评测口径的规则化程度。

## 6. 对当前 final tables 的影响

当前 `main_table_final.csv` 中仍包含 `Cue-only rules`。从方法学审计角度看，这张表若用于正文主表，应删除或拆出 cue-only 行。

建议后续论文落表时形成两个表：

1. 正文主表：learned-model comparison，不含 cue-only。
2. 附录或正文旁注：deterministic cue reference，展示 cue-only 100%，并明确说明其与 label construction 同源。

当前实验结果本身不需要修改；需要修改的是呈现口径和表注。

## 7. 最终判断

当前正文主表不应该保留 cue-only 作为普通方法行。

最合适的处理是：把 cue-only 从正文主表移出，作为 deterministic reference 单独呈现或放入附录。正文主表应只比较 learned models，并在正文中说明 cue-only 达到 100% 的原因是它复现了同源规则构造的标签，而不是证明规则系统在独立人工 gold label 上具有完美泛化能力。
