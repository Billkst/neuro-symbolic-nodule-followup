# M3-3C Manual Audit Label Evaluation

## 定位

本报告评估人工填写后的 M3 manual audit labels。该评分仅用于诊断 conservative CDSG agent 的人工审查结果，不是 learned-model 主实验，也不报告 Accuracy/F1。

## 输入状态

- 输入文件：`outputs/phaseA3/audit_sets/module3_manual_audit_label_template.csv`
- 读取行数：133
- 已标注行数：0
- 缺失标注列：无

## Dry-run 结论

当前输入尚未包含可评分的完整人工标签，脚本已正常生成空评分框架。完成人工标注后，用同一脚本重新运行即可得到正式 score summary、error breakdown 和 group-wise scores。

## 输出文件

- `outputs/phaseA3/tables/module3_manual_audit_score_summary.csv`
- `outputs/phaseA3/tables/module3_manual_audit_error_breakdown.csv`
- `outputs/phaseA3/tables/module3_manual_audit_group_scores.csv`

## 后续使用

若 pilot 20 显示 abstention 大多合理且 conflict 风险可控，可进入 soft matching / safer aggregation 设计；若 density 或 size fact 错误率较高，应先修正 fact recovery 或扩大人工审查集。不建议在缺少人工审查结论前进入 learned-model 主实验。
