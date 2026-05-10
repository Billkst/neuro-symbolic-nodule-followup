# Module 3 M3-3D Codex-Assisted Audit Closure

## 定位

本报告是 M3-3D 的阶段性收口，固化 Codex-assisted pilot audit（Codex 辅助试点审计）的工程与方法学结论。这里的 prelabels（预标注）只能称为 Codex-assisted pre-annotation、model-assisted evidence audit（模型辅助证据审计）或 failure-mode audit（失败模式审计）。

这些结果不是 clinical gold（临床金标准）、不是 expert annotation（专家标注）、不是 manual gold benchmark（人工金标准基准）。没有医学专家参与，`non_clinical_reviewer_notes` 仅表示 Codex self-check; not human reviewed。

## Pilot 20 完成状态

- Codex-assisted pilot 20 已完成：20/20 completed。
- Codex confidence 分布：high=1，medium=12，low=7。
- needs_clinical_expert_review：10/20。
- obvious_codex_error：0。
- evidence grounding support rate：20/20。

结论：Codex evidence grounding 表现可用，但医学裁决信心不足。

## 必答问题

1. Codex-assisted pilot 20 是否完成？

已完成。20 条 case 均已填写 Codex suggestion 和 evidence-grounding self-check 字段，评估脚本也已生成 summary、risk flags 和 report。

2. Codex 是否可以替代医学专家？

不可以。Codex 能辅助定位证据和暴露失败模式，但不能替代医学专家进行 clinical adjudication（临床裁决）。10/20 case 需要 clinical expert review，因此不能把 Codex prelabels 扩展为 gold benchmark。

3. 为什么不扩展到完整 133？

不扩展。虽然 evidence grounding 为 20/20，但 low confidence 为 7/20，clinical expert review 为 10/20，且 high-risk density conflict 是高不确定性来源。继续扩展只会扩大非专家 prelabels 的规模，不能解决 gold validity（标签有效性）问题。

4. 哪些 audit group 最容易出现 clinical uncertainty？

- missing_density_priority_sample: expert review 4/4，uncertainty 4/4，low confidence 2/4。
- high_risk_density_conflict: expert review 5/5，uncertainty 4/5，low confidence 4/5。
- actionable_recommendation: expert review 6/8，uncertainty 5/8，low confidence 5/8。
- missing_size_priority_sample: expert review 0/4，uncertainty 3/4，low confidence 0/4。
- no_structured_nodule: expert review 0/4，uncertainty 0/4，low confidence 0/4。

最突出的是 `high_risk_density_conflict` 和 `missing_density_priority_sample`。前者 5/5 需要 expert review；后者 4/4 需要 expert review 且 4/4 有 Codex uncertainty。

5. 为什么这些 prelabels 不能作为 clinical gold？

原因是本轮没有医学专家参与，Codex 只做 evidence-grounded pre-annotation。missing density、dominant nodule selection、solid/part-solid/ground-glass 冲突、非肺部 nodule 混入等问题会改变 Lung-RADS 路径，必须由专家按固定协议裁决。

6. 当前模块3是否可以进入 learned-model 实验？

不可以。当前模块3不进入 learned-model 主实验，也不生成 learned-model performance table。Codex prelabels 不能作为训练或评估 gold。

7. 当前模块3是否可以进入 soft matching 正式接入？

不可以进入正式接入。soft matching（软匹配）可以作为后续工程设计讨论，但正式接入需要专家审核目标、可接受误差阈值和审计门禁；否则 soft matching 可能掩盖 fact adapter 的事实错误。

8. 当前模块3最稳妥的论文定位是什么？

最稳妥定位是 model-assisted failure-mode audit，而不是 expert annotation。论文中可报告阶段性工程成果：

1. CDSG hard-match deterministic agent；
2. module2-to-CDSG fact adapter；
3. conservative abstention mechanism；
4. evidence-grounded failure-mode audit；
5. Codex-assisted pilot audit demonstrating need for expert review。

9. 后续如果要做临床性能评估，需要什么条件？

- 医学专家参与并签署明确标注协议。
- 对 high-risk density conflict、missing density、missing size、dominant nodule selection 进行专家裁决。
- 固定 clinical gold benchmark，并明确 train/validation/test 分离。
- 建立双人或多轮 adjudication 机制，记录不一致与裁决。
- 在 gold benchmark 完成后再定义 learned-model performance experiment 和 soft matching acceptance criteria。

## 阶段性结论

当前 M3-3C/M3-3D 的结论是：Codex evidence grounding 表现可用，但医学裁决信心不足。10/20 需要 clinical expert review，因此不能把 Codex prelabels 扩展为 gold benchmark。当前模块3不进入 learned-model 主实验，也不进入 soft matching 正式流程。

模块3当前阶段性成果应表述为 deterministic CDSG pipeline plus evidence-grounded failure-mode audit，展示了保守 abstention 与专家复核需求，而不是展示 clinical performance。
