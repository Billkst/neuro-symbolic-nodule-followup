# M3-3C Codex-Assisted Pilot 20 Audit Report

## 定位

本报告是 Codex-assisted pre-annotation / model-assisted evidence audit 的 pilot 20 统计结果。它不是 clinical gold benchmark、不是 expert label、不是 manual gold，也不用于 learned-model performance table。

没有医学专家参与；非医学复核列仅表示 Codex evidence-grounding self-check。结果只能用于 failure-mode audit。

## 输入状态

- 输入文件：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.csv`
- 读取行数：20
- Codex 预标注完成行数：20
- 非医学证据复核完成行数：20
- 缺失必要列：无

## 主要统计

- needs_clinical_expert_review：10 / 20 (0.500000)
- obvious_codex_error：0 / 20 (0.000000)
- evidence grounding support：20 / 20 (1.000000)
- rows with Codex uncertainty：12 / 20 (0.600000)
- 是否建议扩展到 133 cases：no
- 不建议扩展原因：仍有多例需要 clinical expert review；low confidence case 占比偏高
- 当前是否可进入 learned-model performance experiment：no

## Codex Confidence 分布

- high: 1
- medium: 12
- low: 7

## Clinical Expert Review Queue

- CASE-10001401-001
- CASE-10002155-001
- CASE-10052992-001
- CASE-10048825-001
- CASE-10049041-001
- CASE-10053082-001
- CASE-10042810-001
- CASE-10049330-001
- CASE-10064049-001
- CASE-10090755-001

## Obvious Codex Error

- obvious Codex error 数量：0

## Group-wise uncertainty

- actionable_recommendation: uncertainty 5/8 (0.625000), expert review 6/8
- high_risk_density_conflict: uncertainty 4/5 (0.800000), expert review 5/5
- missing_density_priority_sample: uncertainty 4/4 (1.000000), expert review 4/4
- missing_size_priority_sample: uncertainty 3/4 (0.750000), expert review 0/4
- no_structured_nodule: uncertainty 0/4 (0.000000), expert review 0/4

## 风险标记

- 风险标记行数：10

## 使用限制

Codex suggestion 和非医学复核结果只用于 failure-mode audit。若要形成 clinical gold benchmark，需要医学专家复核并重新定义 gold 标注协议。当前仍不能进入 learned-model performance experiment。
