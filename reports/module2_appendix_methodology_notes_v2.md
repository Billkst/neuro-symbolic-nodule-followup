# 模块2附录与方法学说明 v2

> 日期：2026-04-21  
> 目的：将 cue-only reference、P2 hybrid 诊断和方法学审计结论从正文公平主表中拆出，避免 circular evaluation 被误解为 learned-model 性能提升。

## 1. Cue-only reference

Cue-only rules 在 Plan B 的 density Stage 1、density Stage 2、Has-size 和 Location 上均为 `100.00 +/- 0.00`。该结果不作为正文公平主表 baseline，而只作为 deterministic label-construction reference。

原因是 Phase5 / Plan B 的 constructed labels 与 `extract_density`、`extract_size`、`extract_location` 等规则抽取函数高度同源。Cue-only 预测阶段复用同一类 mention-level cue，因此 100% 更准确地说明它复现了标签构造逻辑，而不是说明它在独立人工 gold labels 上具有完美泛化能力。

## 2. P2 hybrid diagnostic

Has-size rule-first hybrid 的 100% 只作为方法学诊断，不进入正文公平主表。其主要风险与 cue-only 相同：Has-size 标签由尺寸规则构造，而 rule-first hybrid 直接利用尺寸规则优先预测，存在 benchmark circularity / label-construction proxy 风险。

Location hybrid 在方案 A 下停止推进，不再补跑，也不作为公平性能结论。正文主表继续使用 learned-model fair comparison 下的 Location 结果。

## 3. 对正文主表的影响

v2 正文主表排除 cue-only 和 P2 hybrid，只比较 learned models。这样做会保留 Has-size 与 Location 的剩余差距，但能避免把规则同源评测误写成模型能力提升。

该处理方式比旧主表更稳健：density 任务的提升来自 P0/P1 的 learned-model 决策与配置修正，而不是来自直接复用标签构造规则。

## 4. 推荐论文表述

可以写：Cue-only 和 rule-first hybrid results are reported only as deterministic label-construction references, because the constructed evaluation labels are generated from the same family of mention-level extraction rules. They are therefore excluded from the main learned-model comparison.

中文表述：cue-only 与 P2 hybrid 只作为标签构造参照或方法学诊断保留，不纳入正文公平主表；正文只报告 learned models，以避免 rule-same-label circular evaluation。

## 5. 最终判断

当前正文主表不应保留 cue-only，也不应纳入 P2 hybrid。P2 停止后，模块2公平结论应限定为 P0+P1 对 density two-stage 的修复和提升，并诚实保留 Has-size / Location 的 learned-model 剩余差距。
