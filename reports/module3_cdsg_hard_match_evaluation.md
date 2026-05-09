# M3-3A CDSG Hard-Match Deterministic Evaluation

## 定位

M3-3A 是 deterministic agent evaluation，不是 learned-model 主实验。本阶段评估的是 conservative CDSG hard-match agent 在当前 v2 case bundle 上能否给出结构化、可追踪、带指南锚点的确定性输出，以及它在关键事实缺失时是否按设计 abstain。strong-silver label 由同一个 CDSG / rule engine 生成，因此不能被当作公平 gold label 来报告 Accuracy、F1 或 learned-model 性能。

## 主要结果

- Full set: 253 cases。
- Actionable: 76/253 (30.0%)。
- Abstention: 177/253 (70.0%)。
- Schema valid: 253/253 (100.0%)。
- Guideline anchor non-empty: 253/253 (100.0%)。
- Reasoning path non-empty: 253/253 (100.0%)。
- Decision path non-empty: 253/253 (100.0%)。

## Abstention 分解

- missing_density: 104 cases。
- missing_size: 66 cases。
- no_structured_nodule: 7 cases。

这些 abstention 是 conservative CDSG 的预期行为：明确的 size / density / location 等结构化事实走符号硬匹配；关键事实缺失时，不允许自由生成或默认补全。

## 为什么不采用 aggregation

- same-case union 可将 actionable 提高 8，但 high-risk candidates = 4，存在跨 mention / 跨结节错配风险。
- dominant size-first + density-nearest 可将 actionable 提高 8，但 high-risk candidates = 4，同样不适合进入正式 conservative v2。
- confidence-gated same-note union 只带来 actionable_delta = 3，且仍属于启发式弱合并；本轮仅作为 appendix diagnostic，不覆盖正式 v2。

## 为什么不能默认补事实

- missing density 不能默认 solid。solid / ground-glass / part-solid 会直接改变 Lung-RADS 路由、category 和 follow-up interval；默认 solid 会把未知事实伪装成确定事实，破坏 CDSG 的 guideline anchor 与 reasoning path 可信度。
- size_mm 缺失不能用 has-size 概率补。has-size 只表示文本中可能存在尺寸表达，不是可用于阈值判断的毫米数值；Lung-RADS hard-rule 需要明确的 size_mm 才能进入阈值边。

## 当前优势

- 输出全部保留标准 schema、guideline anchor、reasoning path 和 decision path。
- 推荐来自 CDSG 终态节点或 abstention 节点，不来自自由文本生成。
- 对关键事实缺失采取 conservative abstention，避免 under-follow-up 或 hallucinated recommendation。

## 当前限制

- Actionable coverage 仅 76/253，主要受 density 和 size 结构化事实缺失限制。
- Conflict rows = 425，覆盖 150 cases；其中 high-risk density conflicts = 25，需要人工审查或更安全的 mention-to-nodule linking。
- 后续若要提升覆盖率，应优先设计人工 audit set、soft matching 或受约束的事实恢复，而不是直接进入 learned-model 主实验。

## 输出表格

- CSV: `outputs/phaseA3/final_tables_m3a`
- LaTeX: `outputs/phaseA3/final_tables_latex_m3a`

## 下一步判断

可以进入 M3-3B：人工 audit set 设计或 soft matching 设计。当前不建议进入 learned-model 主实验，因为可行动样本只有 76 条且 strong-silver label 来自 CDSG 自身，不能支撑公平性能表。
