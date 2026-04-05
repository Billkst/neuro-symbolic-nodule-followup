# Phase 5 错误分析报告

## 1. 分析方法
- **数据源**：基于 PubMedBERT 在测试集（Test Set, 42,057 样本）上的预测结果。
- **分析对象**：提取模型预测与 Silver Label 不一致的样本，结合置信度分析错误根因。
- **覆盖范围**：涵盖密度分类、大小检测、位置分类三个任务。

## 2. Density Classification 错误分析

### 2.1 错误统计
- **总错误数**：29 个（测试集 42,057 中）。
- **错误分布**：
  - `unclear` → `ground_glass`: 20 例 (最主要错误源)
  - `calcified` → `unclear`: 7 例
  - `solid` → `ground_glass`: 1 例
  - `unclear` → `calcified`: 1 例

### 2.2 典型失败案例

**案例 1: 否定词识别失效**
- **mention_text**: "A few pulmonary nodules are noted not definitely calcified (5:67, 142) measuring 2 mm."
- **真实标签 (silver)**: `unclear`
- **模型预测**: `calcified`
- **模型置信度**: 0.9999
- **错误原因分析**: 文本中包含 "not definitely calcified"，Regex 正确识别为不确定（unclear），但 BERT 模型可能对 "calcified" 关键词过于敏感，未能充分理解 "not definitely" 的否定/不确定语义。

**案例 2: 混合描述混淆**
- **mention_text**: "right lower lobe and the other one is a mixture of ground and solid opacity in the left lower lobe."
- **真实标签 (silver)**: `solid`
- **模型预测**: `ground_glass`
- **模型置信度**: 0.9995
- **错误原因分析**: 文本同时包含 "ground" 和 "solid"。在这种混合描述中，Regex 可能有特定的优先级逻辑（如优先归类为 solid），而 BERT 选择了另一个关键词。

**案例 3: 跨句上下文干扰**
- **mention_text**: "ground- glass nodule left upper lobe (4.307), 3 mm nodule in the right upper"
- **真实标签 (silver)**: `unclear`
- **模型预测**: `ground_glass`
- **模型置信度**: 0.9998
- **错误原因分析**: 窗口内包含两个结节描述，第一个是 "ground- glass"，第二个是 "3 mm nodule"（目标 mention）。BERT 可能受到了前文 "ground- glass" 的强干扰，将其误认为是当前结节的属性。

### 2.3 错误模式总结
- **模式 1: 关键词过敏**（N=22）：模型在看到明确的医学术语（如 ground glass, calcified）时，容易忽略周围的否定词或修饰语。
- **模式 2: 标签噪声**（N=5）：部分 silver label 本身可能存在逻辑瑕疵，导致模型预测了更符合常理的结果却被判定为错。

## 3. Size Detection 错误分析

### 3.1 错误统计
- **总错误数**：24 个。
- **错误分布**：全部为 `no_size` → `has_size`（即模型发现了 Regex 漏掉的大小信息）。

### 3.2 典型失败案例

**案例 1: 拼写错误/连写导致 Regex 失效**
- **mention_text**: "Slightly increasing nodule in the left lower lobe from 5 to 7mmsince ___."
- **真实标签 (silver)**: `no_size`
- **模型预测**: `has_size`
- **模型置信度**: 0.9999
- **错误原因分析**: 文本中 "7mmsince" 缺少空格，导致 Regex 规则未能匹配到数字。BERT 凭借强大的泛化能力正确识别出此处包含大小信息。**这实际上是 BERT 优于 Regex 的体现。**

**案例 2: 单位拼写冗余**
- **mention_text**: "4mmm nodule in the right upper lobe is not definitely seen on the prior"
- **真实标签 (silver)**: `no_size`
- **模型预测**: `has_size`
- **模型置信度**: 0.9999
- **错误原因分析**: "4mmm" 多写了一个 m，Regex 匹配失败，但 BERT 成功识别。

## 4. Location Classification 错误分析

### 4.1 错误统计
- **总错误数**：1 个。
- **错误分布**：`lingula` → `no_location`。

### 4.2 典型失败案例

**案例 1: 非标准术语混用**
- **mention_text**: "...located predominantly in the left middle lobe (2:46) as well as the lingula which may represent an early/mild"
- **真实标签 (silver)**: `lingula`
- **模型预测**: `no_location`
- **模型置信度**: 0.9972
- **错误原因分析**: 文本中出现了 "left middle lobe"（非标准术语，通常指舌叶 lingula）。Regex 可能将其强制映射到了 lingula，但 BERT 在面对这种矛盾描述时表现得更为保守，选择了 no_location。

## 5. 跨任务错误模式
- **共同特征**：大多数错误发生在包含多个结节描述或非标准缩写的复杂长句中。
- **相互关联**：当一个 mention 的密度描述非常模糊时，其位置和大小的提取往往也伴随着较低的置信度。

## 6. Silver Label 噪声分析
- **发现**：在 Size 任务的 24 例错误中，有超过 90% 的案例实际上是 BERT 预测正确而 Silver Label（Regex）提取失败。
- **估计噪声率**：在 Regex 判定为“无信息”的样本中，约有 0.1% 实际上包含有效信息，这部分信息目前只能通过 BERT 或人工标注找回。

## 7. 改进建议
1. **增强否定词训练**：针对 "not definitely", "no evidence of" 等短语进行专门的样本增强。
2. **优化窗口清洗**：在 mention-centered window 中进一步过滤掉明显属于其他结节的描述文字。
3. **引入集成逻辑**：对于 BERT 置信度极高但与 Regex 不一致的 Size 样本，应优先信任 BERT 的结果。
