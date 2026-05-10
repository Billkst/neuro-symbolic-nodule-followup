# M3-3C Codex-Assisted Pilot 20 Audit Report

## 定位

本报告是 Codex-assisted evidence audit 的 pilot 20 统计结果。它不是 clinical gold benchmark，也不用于 learned-model performance table。

## 输入状态

- 输入文件：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv`
- 读取行数：20
- Codex 预标注完成行数：0
- 非医学证据复核完成行数：0
- 缺失必要列：无

## Dry-run 结论

当前输入尚未包含完整 Codex 预标注或非医学证据复核。脚本已正常生成空统计框架；填完 `module3_codex_assisted_pilot_20_filled.csv` 后重新运行即可得到正式 pilot 评估。

## Group-wise uncertainty

- actionable_recommendation: uncertainty 0/8 (0.000000), expert review 0/8
- high_risk_density_conflict: uncertainty 0/5 (0.000000), expert review 0/5
- missing_density_priority_sample: uncertainty 0/4 (0.000000), expert review 0/4
- missing_size_priority_sample: uncertainty 0/4 (0.000000), expert review 0/4
- no_structured_nodule: uncertainty 0/4 (0.000000), expert review 0/4

## 风险标记

- 风险标记行数：0

## 使用限制

Codex suggestion 和非医学复核结果只用于 failure-mode audit。若要形成 clinical gold benchmark，需要医学专家复核并重新定义 gold 标注协议。
