# 模块2最终表格审阅与论文落表报告

> 日期：2026-04-19  
> 阶段：模块2 A2.5 strict 聚合完成后的最终表格审阅、论文落表与结果解释整理  
> 数据来源：`outputs/phaseA2/results` 与 `outputs/phaseA2/tables`  
> 聚合入口：`python -u scripts/phaseA2/aggregate_a2_5_results.py --strict`  
> 约束：本报告不启动训练，不调用旧 `scripts/phaseA2/build_main_table.py`，只审阅当前 A2.5 strict 聚合产物。

---

## 0. 总体判断

模块2已经进入“论文落表与结果解释”阶段。当前 `outputs/phaseA2/tables` 已包含正式主表、效率表、A2 quality-gate、A3 aggregation、P1 max_seq_length 和 P3 section/input strategy 的 CSV/JSON 聚合结果；`a2_5_manifest_report.json` 中 36 个预期 manifest 条目均完整，未发现缺 seed 条目。

但当前表格还不能全部原样放进论文。主要原因不是实验缺失，而是论文展示口径需要进一步整理：主表包含 `N` 工程列，效率表的 `Peak GPU Memory` 对 MWS-CFE 为 `—`，消融表包含 `Tag`、`Experiment`、`N` 等工程字段，且主表结果显示 `MWS-CFE (Ours)` 并没有在正式三任务指标上全面超过 baseline，需要在论文叙事中诚实处理。

---

## 1. 当前聚合产物清单

| 文件 | 行数 | 内容 | 当前状态 |
|---|---:|---|---|
| `outputs/phaseA2/tables/a2_5_main_table.csv` | 4 行结果 | 正式主表，4 个方法，3 个任务，5-seed mean ± std | 可作为正文主表来源，但需删除 `N` 列并优化表头 |
| `outputs/phaseA2/tables/a2_5_efficiency_table.csv` | 12 行结果 | 方法 × 任务的训练样本、训练时间、评估时间、best epoch、GPU 显存 | 可作为效率表来源，但 MWS-CFE 显存缺失，不能无脚注直接进正文 |
| `outputs/phaseA2/tables/a2_quality_gate_summary.csv` | 10 行结果 | G1/G2/G3/G4/G5 的 density 与 location 消融 | 适合附录；正文可摘取关键结论 |
| `outputs/phaseA2/tables/a3_aggregation_summary.csv` | 4 行结果 | weighted vote/main 与 uniform vote 对比 | 适合附录或与 A2 合并成核心消融小表 |
| `outputs/phaseA2/tables/p1_max_length_summary.csv` | 5 行结果 | density-only max_seq_length 64/96/128/160/192 | 适合附录；正文只写趋势 |
| `outputs/phaseA2/tables/p3_section_input_summary.csv` | 5 行结果 | density-only input strategy 对比 | 适合附录；可在正文讨论长文本噪声与摘要段落价值 |
| `outputs/phaseA2/tables/a2_5_manifest_report.json` | 36 个 manifest 条目 | 完整性、unmatched 文件、metadata warnings | 不进论文表格，只作为复现实验审计依据 |

---

## 2. 聚合完整性审计

### 2.1 strict 完整性

`a2_5_manifest_report.json` 显示：

| 项目 | 结果 |
|---|---:|
| `record_count` | 150 |
| 预期 manifest 条目 | 36 |
| 完整条目 | 36 |
| 不完整条目 | 0 |
| `unmatched_files` | 3 |
| `ignored_method_files` | 0 |
| `metadata_warnings` | 33 |

结论：A2.5 strict 聚合层面已经闭环，`incomplete_entries=0` 的判断成立。当前不需要因为表格完整性继续补跑模块2训练。

### 2.2 unmatched 文件解释

`unmatched_files` 包含：

1. `mws_cfe_density_results_pf_density_ftext_b64_s42.json`
2. `mws_cfe_density_results_pf_density_ftext_b96_s42.json`
3. `mws_cfe_density_results_pf_density_mtext_b160_s42.json`

这些文件不符合当前 A2.5 标准命名正则 `mws_cfe_{task}_results_{tag}_seed{seed}.json`，因此被聚合脚本识别为 unmatched。它们不是 A2.5 manifest 期望矩阵中的正式条目，不影响 strict 完整性。论文不需要展示这些文件名，但答辩或复现说明中可写为“历史 preflight / probe 文件未纳入正式聚合口径”。

### 2.3 metadata warnings 解释

33 条 `metadata_warnings` 均为：

```text
model_name does not record the local safetensors path
```

这说明部分 MWS-CFE JSON 中 `model_name` 仍记录为 Hugging Face 名称 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`，没有记录本地 safetensors 路径。该 warning 是元数据记录问题，不是指标缺失或训练失败问题。论文中不建议展开，但实验复现部分应说明 GPU 训练使用了本地 safetensors / offline 运行策略，历史 JSON 的 `model_name` 字段存在记录不完整。

---

## 3. 表格逐项审阅

### 3.1 正式主表

来源：`outputs/phaseA2/tables/a2_5_main_table.csv`

当前表头：

```text
Method,Density Acc,Density Macro-F1,Density N,Has_size Acc,Has_size F1,Has_size N,Location Acc,Location Macro-F1,Location N
```

审阅结论：

1. 方法名统一，当前 4 行为 `TF-IDF + LR`、`TF-IDF + SVM`、`Vanilla PubMedBERT`、`MWS-CFE (Ours)`。
2. 任务名基本统一，但 `Has_size` 不适合直接进论文，建议表头写为 `Has-size`，首次正文解释其对应代码任务 `has_size`。
3. `Density N`、`Has_size N`、`Location N` 是工程审计字段，不应进入正文主表；可在表注中统一写“all results are averaged over 5 seeds”。
4. 指标名适合正文使用，但建议统一为 `Acc.`、`Macro-F1`、`F1`，避免表头过宽。
5. 当前主表缺少 `Params (M)` 和严格 `Inference Time (ms/sample)`；这与 A2.5 当前计划一致，因为这些字段尚未被严格 benchmark。不要临时用 `Eval Time / seed` 冒充 per-sample latency。

当前关键数值：

| Method | Density Macro-F1 | Has-size F1 | Location Macro-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 17.64 ± 0.00 | 87.83 ± 0.00 | 95.95 ± 0.00 |
| TF-IDF + SVM | 18.52 ± 0.00 | 87.53 ± 0.00 | 96.73 ± 0.00 |
| Vanilla PubMedBERT | 43.76 ± 6.92 | 82.11 ± 1.28 | 97.13 ± 0.83 |
| MWS-CFE (Ours) | 23.17 ± 4.47 | 81.75 ± 1.26 | 96.63 ± 0.33 |

硬问题：`MWS-CFE (Ours)` 在三项主指标上都不是最优。Density Macro-F1 低于 Vanilla PubMedBERT，Has-size F1 低于 TF-IDF + LR/SVM，Location Macro-F1 低于 Vanilla PubMedBERT 和 TF-IDF + SVM。论文不能写“Ours 全面优于 baseline”。更稳妥的写法是：

1. 正文主表如实报告公平口径下的性能。
2. 把 MWS-CFE 的价值转向“多源弱监督框架可运行、5-seed 稳定、消融揭示质量门控与输入策略影响”，而不是声称性能全面领先。
3. 对 density 任务，说明 Vanilla PubMedBERT 在当前 silver / Phase5 test 口径上更强，MWS-CFE 后续应通过质量权重、类别不平衡处理或标签噪声建模继续改进。

### 3.2 效率表

来源：`outputs/phaseA2/tables/a2_5_efficiency_table.csv`

当前表头：

```text
Method,Task,Tag,N,Train Samples,Train Time / seed (s),Eval Time / seed (s),Best Epoch,Peak GPU Memory (GB)
```

审阅结论：

1. `Tag` 和 `N` 不应进入正文效率表；`Tag=main_g2` 可放表注。
2. `Eval Time / seed (s)` 是整次 evaluation 时间，不是严格 per-sample inference latency。若论文使用，应命名为 `Eval Time / seed (s)` 或 `Evaluation Time / seed (s)`，不能改写成 `Inference Time (ms/sample)`。
3. MWS-CFE 的 `Peak GPU Memory (GB)` 全部为 `—`，是效率表最大的硬问题。若正文放效率表，必须脚注说明 MWS-CFE 历史 JSON 未记录显存；否则建议把效率表放附录或只保留训练时间/评估时间。
4. Vanilla PubMedBERT 的 GPU memory 有值，MWS-CFE 无值，会引发导师追问“为什么 ours 没显存”。需要提前解释这是日志/JSON 元数据缺失，不代表没有使用 GPU。
5. `Best Epoch` 对传统 ML 为 `—` 是合理的，但应脚注说明。

可汇报的正向点：MWS-CFE 的训练时间在三任务上均略低于 Vanilla PubMedBERT：

| Task | Vanilla PubMedBERT Train Time / seed | MWS-CFE Train Time / seed |
|---|---:|---:|
| Density | 343.1 ± 79.2 s | 271.7 ± 105.4 s |
| Has-size | 1865.0 ± 366.1 s | 1745.8 ± 518.7 s |
| Location | 843.7 ± 227.6 s | 763.4 ± 104.8 s |

但该点只能写成“训练耗时相近或略低”，不能扩展成严格效率领先，因为缺少参数量、吞吐和 per-sample latency。

### 3.3 A2 quality-gate 表

来源：`outputs/phaseA2/tables/a2_quality_gate_summary.csv`

当前表头：

```text
Experiment,Variant,Task,Tag,N,Accuracy,Main Metric,Main Metric Value
```

审阅结论：

1. 适合附录，不建议完整放正文。
2. 若正文需要核心消融，可只摘出 density 和 location 的最佳/主配置对比。
3. `Experiment`、`Tag`、`N` 是工程字段，正文表应删除或转为表注。
4. `Main Metric` 每行重复为 `Macro-F1`，正文表可直接改列名为 `Macro-F1`。

关键结论：

1. Density 上 G3 最强：Accuracy `19.28 ± 4.26`，Macro-F1 `39.84 ± 3.60`，明显高于 G2/main 的 `23.17 ± 4.47`。
2. Location 上 G1 最强：Accuracy `99.75 ± 0.09`，Macro-F1 `99.31 ± 0.22`。
3. G5 对 location 明显失败：Macro-F1 `67.49 ± 2.11`，说明过强质量门控可能牺牲覆盖或改变样本分布。

导师可能质疑：为什么正式主表仍用 G2，而消融里 G3/G1 更好。建议解释为：G2 是预先锁定的统一公平主配置，quality-gate 表用于说明 gate 选择对不同任务敏感；不能事后按 test 最优 gate 重写主表，否则有选择偏差。

### 3.4 A3 aggregation 表

来源：`outputs/phaseA2/tables/a3_aggregation_summary.csv`

当前表头同 A2 消融表。

关键结论：

1. Weighted vote / main 与 Uniform vote 差异很小。
2. Density Macro-F1：weighted `23.17 ± 4.47`，uniform `23.24 ± 8.34`。
3. Location Macro-F1：weighted `96.63 ± 0.33`，uniform `96.53 ± 0.39`。

解释方向：当前多源弱监督的 aggregation 权重没有带来稳定、显著的收益。论文中可写为“aggregation strategy 对最终结果影响有限，说明当前性能瓶颈更可能来自 silver label 噪声、类别不平衡或模型训练策略，而非简单投票权重”。

表格位置：建议附录；正文只保留一句总结。

### 3.5 P1 max_seq_length 表

来源：`outputs/phaseA2/tables/p1_max_length_summary.csv`

关键结论：

1. 该表是 density-only，不代表 has_size/location。
2. Macro-F1 随 max length 从 64 到 192 整体上升，但方差较大。
3. `192` 的 Macro-F1 最高：`24.77 ± 6.10`。
4. 主配置 `128 / main` 为 `23.17 ± 4.47`，接近 160 和 192。

解释方向：更长输入能提供更多上下文，但收益不稳定；128 作为主配置是计算成本与性能之间的折中。不能写“越长越好”，因为 Accuracy 在各长度接近，且 Macro-F1 方差较大。

表格位置：附录参数讨论表。

### 3.6 P3 section/input strategy 表

来源：`outputs/phaseA2/tables/p3_section_input_summary.csv`

关键结论：

1. 该表是 density-only。
2. `impression` 的 Macro-F1 最高：`31.27 ± 7.21`。
3. `full_text` 的 Accuracy 最高：`22.14 ± 4.70`，但 Macro-F1 只有 `23.22 ± 4.62`，与主配置接近。
4. `findings` 的 Macro-F1 最低：`19.03 ± 2.23`。

解释方向：impression 对 density 分类更有判别信息，但稳定性较差；full_text 提高 Accuracy 可能来自多数类或更宽泛上下文，不必然改善类别均衡表现。论文应优先解读 Macro-F1，而不是只看 Accuracy。

表格位置：附录参数讨论表；正文可在方法或实验分析中作为“输入段落选择影响性能”的证据。

---

## 4. 论文落表映射

### 4.1 正文主表

使用文件：

```text
outputs/phaseA2/tables/a2_5_main_table.csv
```

建议最终表头：

| 当前列 | 论文列名 | 处理 |
|---|---|---|
| `Method` | `Method` | 保留 |
| `Density Acc` | `Density Acc.` | 保留 |
| `Density Macro-F1` | `Density Macro-F1` | 保留 |
| `Density N` | 删除 | 表注写 5 seeds |
| `Has_size Acc` | `Has-size Acc.` | 改名 |
| `Has_size F1` | `Has-size F1` | 改名 |
| `Has_size N` | 删除 | 表注写 5 seeds |
| `Location Acc` | `Location Acc.` | 保留 |
| `Location Macro-F1` | `Location Macro-F1` | 保留 |
| `Location N` | 删除 | 表注写 5 seeds |

建议表注：

```text
All trainable methods are evaluated under the same multi-source weak-supervision protocol, subject-level split, G2 gate, and five random seeds. Results are reported as mean ± standard deviation (%). The size task is reported as has-size detection.
```

### 4.2 效率表

使用文件：

```text
outputs/phaseA2/tables/a2_5_efficiency_table.csv
```

建议最终表头：

| 当前列 | 论文列名 | 处理 |
|---|---|---|
| `Method` | `Method` | 保留 |
| `Task` | `Task` | 保留，`Has_size` 改为 `Has-size` |
| `Tag` | 删除 | 表注说明均为 `main_g2` |
| `N` | 删除 | 表注说明 5 seeds |
| `Train Samples` | `Train Samples` | 可保留 |
| `Train Time / seed (s)` | `Train Time / seed (s)` | 可保留 |
| `Eval Time / seed (s)` | `Eval Time / seed (s)` | 可保留，不能改成 inference latency |
| `Best Epoch` | `Best Epoch` | 可保留 |
| `Peak GPU Memory (GB)` | `Peak GPU Memory (GB)` | 可保留但需脚注 MWS-CFE 缺失 |

建议位置：如果正文篇幅有限，效率表放附录；若导师要求工程成本对比，可放正文表2，但必须加脚注。

### 4.3 消融实验表

使用文件：

```text
outputs/phaseA2/tables/a2_quality_gate_summary.csv
outputs/phaseA2/tables/a3_aggregation_summary.csv
```

建议做法：

1. 正文不建议完整放 14 行消融。
2. 正文可做一个“核心消融摘要表”，只保留 `Variant`、`Task`、`Accuracy`、`Macro-F1`。
3. 完整 A2 和 A3 表放附录。

建议最终表头：

| 当前列 | 论文列名 | 处理 |
|---|---|---|
| `Experiment` | `Experiment` | 附录保留；正文摘要可删除 |
| `Variant` | `Variant` | 保留 |
| `Task` | `Task` | 保留 |
| `Tag` | 删除 | 附录可保留；正文删除 |
| `N` | 删除 | 表注写 5 seeds |
| `Accuracy` | `Acc.` | 保留 |
| `Main Metric` | 删除 | 因为均为 Macro-F1 |
| `Main Metric Value` | `Macro-F1` | 改名 |

### 4.4 参数讨论表

使用文件：

```text
outputs/phaseA2/tables/p1_max_length_summary.csv
outputs/phaseA2/tables/p3_section_input_summary.csv
```

建议位置：附录。正文只提最关键趋势：

1. P1：较长上下文提升 density Macro-F1 的趋势存在，但不稳定。
2. P3：impression 对 density 更有判别价值，full_text 提高 Accuracy 但不明显提升 Macro-F1。

### 4.5 建议放附录而不是正文的结果

1. 完整 A2 quality-gate 表。
2. 完整 A3 aggregation 表。
3. 完整 P1 max_seq_length 表。
4. 完整 P3 section/input strategy 表。
5. manifest 的 unmatched 文件和 metadata warnings 只进入复现说明，不作为论文表。
6. 历史 preflight 文件不进入论文。

---

## 5. 表格解释提纲

### 5.1 正式主表解释提纲

这张表要证明什么：在统一 multi-source weak supervision、统一 split、统一 G2 gate、统一 5-seed 口径下，比较传统线性 baseline、Vanilla PubMedBERT 与 MWS-CFE 在三项结构化抽取任务上的表现。

关键比较关系：

1. TF-IDF 系列在 has-size 上强，说明尺寸有无判断具有明显词面模式。
2. Vanilla PubMedBERT 在 density 和 location 主指标上最强，说明当前深度语义编码对密度类别更有效。
3. MWS-CFE 在当前正式口径下没有全面超过 baseline，应作为需要诚实呈现的实验发现。

Ours 解读方式：

1. 不写“全面最优”。
2. 可以写“MWS-CFE 在统一多源弱监督框架下完成了三任务 5-seed 闭环，并通过后续消融分析揭示质量门控、聚合方式和输入策略对性能的影响”。
3. 若论文需要强调创新，应把创新放在 weak-supervision pipeline 和可审计训练框架，而不是单表性能领先。

可能被质疑的点：

1. Density Acc 都很低，而 Macro-F1 差异大。需说明类别分布和 silver 口径可能导致 Accuracy 与 Macro-F1 不一致，Macro-F1 更能反映类别均衡表现。
2. MWS-CFE 不如 Vanilla PubMedBERT。需说明当前 MWS-CFE 版本未证明性能优势，应作为限制和后续优化方向。
3. TF-IDF 的 std 为 0。需说明传统 ML 在固定数据与确定性设置下跨 seed 结果稳定或随机性弱，不代表评估错误。

脚注建议：

1. `has_size` 在论文中统一写作 `Has-size` 或 `has-size detection`。
2. 所有结果为 5 seeds mean ± std，单位为百分比。
3. 主表只使用 `phase5_test_results`，不混用 WS validation/test 指标。

### 5.2 效率表解释提纲

这张表要证明什么：展示各方法在三任务上的训练成本和评估成本，而不是严格推理延迟。

关键比较关系：

1. TF-IDF 方法训练快且不占 GPU。
2. 神经模型训练时间显著更高。
3. MWS-CFE 与 Vanilla PubMedBERT 训练时间同量级，部分任务略低。

Ours 解读方式：

1. 可写“MWS-CFE 未显著增加训练成本”。
2. 不可写“MWS-CFE 推理最快”，因为没有 per-sample inference benchmark。

可能被质疑的点：

1. MWS-CFE GPU memory 缺失。
2. Eval time 不是 inference latency。
3. Train samples 对不同任务差异大，因为任务正例/样本定义不同。

脚注建议：

1. `Eval Time / seed` 为整套 evaluation 时间。
2. `—` 表示对应元数据未记录，不表示数值为 0。
3. 传统 ML 的 GPU memory 记为 0，因为 CPU 路径运行。

### 5.3 A2 quality-gate 表解释提纲

这张表要证明什么：质量门控会显著影响 MWS-CFE 在不同任务上的表现，且最优 gate 具有任务依赖性。

关键比较关系：

1. Density：G3 明显优于 G2/main。
2. Location：G1 最强，G5 明显退化。
3. 过强 gate 未必更好，尤其对 location 可能损害覆盖。

Ours 解读方式：

1. 质量门控是 MWS-CFE 的关键可调部件。
2. 当前主配置 G2 是公平预设，不是 test 后挑选最优。

可能被质疑的点：

1. 为什么不把 G3 作为 density 主结果。回答：主表需统一预注册口径；G3 属于消融发现。
2. 为什么不同任务最优 gate 不同。回答：density 与 location 的标签噪声、覆盖率和类别分布不同。

### 5.4 A3 aggregation 表解释提纲

这张表要证明什么：当前 weighted vote 与 uniform vote 的差异很小，aggregation 权重不是当前性能瓶颈。

关键比较关系：weighted 和 uniform 在 density/location 上都接近。

Ours 解读方式：当前多源弱监督框架的收益不能简单归因于 vote weighting；后续应更关注质量门控、输入策略和噪声鲁棒训练。

可能被质疑的点：为什么设计 weighted vote 但收益不明显。回答：在当前 LF 权重和标签分布下，投票策略变化不足以显著改变训练信号。

### 5.5 P1 max_seq_length 表解释提纲

这张表要证明什么：density 任务对上下文长度敏感，但收益不稳定。

关键比较关系：

1. 64 明显弱于 128/160/192。
2. 192 Macro-F1 最高，但 std 也较大。
3. 128/main 是成本与性能的折中。

可能被质疑的点：为什么不使用 192 作为主配置。回答：P1 是参数讨论；主配置需统一稳定，且 192 的收益存在较大方差。

### 5.6 P3 section/input strategy 表解释提纲

这张表要证明什么：输入段落选择会影响 density 任务，impression 对密度判断更集中。

关键比较关系：

1. impression Macro-F1 最优。
2. findings 单独使用较弱。
3. full_text Accuracy 最高但 Macro-F1 不突出，提示长文本可能强化多数类或噪声。

可能被质疑的点：为什么主表仍用 mention_text。回答：mention_text 是预设统一主口径；P3 说明不同输入策略可作为后续优化方向。

---

## 6. 表格层面硬问题与修正建议

| 问题 | 是否存在 | 影响 | 修正建议 |
|---|---|---|---|
| 表头命名不统一 | 是 | `Has_size`、`size`、`Has-size` 混用风险 | 论文统一写 `Has-size`，代码/文件说明中标注对应 `has_size` |
| 方法名不统一 | 轻微 | `Vanilla PubMedBERT` 与实际 `BiomedBERT` model_name 可能被追问 | 方法节说明 PubMedBERT/BiomedBERT checkpoint 来源；表内保持当前方法名 |
| 指标列缺失 | 是 | 主表无 Params、严格 inference latency；效率表无 per-sample latency | 不临时伪造；正文主表先用性能列，效率表注明 evaluation-time proxy |
| 工程列直接进论文 | 是 | `N`、`Tag`、`Experiment` 不适合正文 | 正文删除，附录可保留或脚注化 |
| 某张表不适合直接进论文 | 是 | 效率表 MWS-CFE GPU memory 为 `—`；消融表工程字段多 | 效率表加脚注或放附录；消融表重排后再进正文 |
| 聚合结果存在但不适合作最终展示 | 是 | unmatched preflight 文件、manifest warnings 不应论文展示 | 仅放复现审计说明，不进正文结果 |
| Ours 结果不优 | 是 | 论文主结论不能写性能领先 | 改为诚实结果分析，突出闭环、公平聚合、消融洞察和限制 |

---

## 7. 推荐正文与附录布局

### 7.1 正文布局

**表1：模块2正式主表**

来源：`outputs/phaseA2/tables/a2_5_main_table.csv`

展示 4 个方法在 density、has-size、location 三任务上的 Acc / F1 指标。删除三列 `N`。这张表必须放正文，因为它是模块2最终公平比较的核心结果。

**表2：模块2核心消融摘要表**

来源：

```text
outputs/phaseA2/tables/a2_quality_gate_summary.csv
outputs/phaseA2/tables/a3_aggregation_summary.csv
```

建议正文只放一个压缩表：展示 G2/main、density 最优 G3、location 最优 G1、uniform vote 对照。目的不是穷尽所有变体，而是支持“质量门控影响显著，aggregation 权重影响有限”的结论。

**表3：模块2效率概览表，可选**

来源：`outputs/phaseA2/tables/a2_5_efficiency_table.csv`

是否进正文取决于导师是否要求工程成本。若进正文，建议只保留 `Method`、`Task`、`Train Time / seed`、`Eval Time / seed`、`Best Epoch`，并把 GPU memory 放附录或脚注，因为 MWS-CFE 缺失显存。

### 7.2 附录布局

**表A1：完整效率表**

来源：`outputs/phaseA2/tables/a2_5_efficiency_table.csv`

保留全部 12 行，脚注说明 MWS-CFE GPU memory metadata 未记录。

**表A2：完整 A2 quality-gate 表**

来源：`outputs/phaseA2/tables/a2_quality_gate_summary.csv`

保留 G1/G2/G3/G4/G5 在 density/location 上的完整结果。

**表A3：完整 A3 aggregation 表**

来源：`outputs/phaseA2/tables/a3_aggregation_summary.csv`

展示 weighted vote 与 uniform vote。

**表A4：P1 max_seq_length 参数讨论表**

来源：`outputs/phaseA2/tables/p1_max_length_summary.csv`

标注 density-only。

**表A5：P3 section/input strategy 参数讨论表**

来源：`outputs/phaseA2/tables/p3_section_input_summary.csv`

标注 density-only。

**表A6：A2.5 聚合审计摘要，可选**

来源：`outputs/phaseA2/tables/a2_5_manifest_report.json`

不列所有 warning，只列 manifest 完整性：150 records、36/36 complete、0 incomplete、3 unmatched preflight files、33 metadata warnings。

---

## 8. 最终建议

1. 模块2不需要继续优先补跑训练；当前应进入论文落表、结果解释和正文/附录整理阶段。
2. 正文主表可以使用 `a2_5_main_table.csv`，但必须删除 `N` 列并把 `Has_size` 改成论文友好的 `Has-size`。
3. 效率表不要声称严格 inference latency；当前只有 train/eval time，MWS-CFE GPU memory 缺失需脚注。
4. 消融实验不建议全部堆进正文；正文只保留核心摘要，完整表放附录。
5. 主表结果对 Ours 不利，论文必须诚实解释，不能写成性能全面领先。
6. A2 quality-gate 是最有论文讨论价值的消融，尤其 density 的 G3 和 location 的 G1/G5 对比。
7. A3 aggregation 结果接近，适合支撑“投票权重不是当前瓶颈”的分析。
8. P1/P3 是参数讨论，不应提升为核心主结论。
9. manifest warnings 只影响复现元数据说明，不影响 strict 聚合完整性。
10. 下一步应开始整理 LaTeX 表格和结果段落，而不是继续补跑实验或切换模块3编码。
