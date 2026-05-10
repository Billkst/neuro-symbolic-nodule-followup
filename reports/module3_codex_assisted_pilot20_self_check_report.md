# M3-3C Codex-Assisted Pilot 20 Self-Check Report

## 定位

- 本文件记录 Codex-assisted pre-annotation 和 model-assisted evidence audit 的自检结果。
- 这些输出不是 clinical gold、不是 expert label、不是 manual gold benchmark。
- 没有医学专家参与；`non_clinical_reviewer_notes` 均为 Codex self-check; not human reviewed。
- 结果只能用于 failure-mode audit，不能用于 learned-model performance experiment。

## 完成情况

- 输入模板：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv`
- filled CSV：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.csv`
- filled JSONL：`outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.jsonl`
- pilot 行数：20
- Codex suggestion 完成行数：20
- evidence self-check 完成行数：20

## Confidence 分布

- high: 1
- medium: 12
- low: 7

## Evidence Grounding

- evidence grounding support rate：20/20 (1.000000)
- obvious Codex error：0

## 需要 Clinical Expert Review 的 Case

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

## 是否建议扩展到完整 133 Cases

- 建议：no
- 原因：many cases remain low-confidence or require clinical expert review, especially high-risk density conflicts

## Learned-Model Experiment 状态

- 当前仍不能进入 learned-model performance experiment。
- 原因：pilot 20 仍包含 low-confidence case、high-risk density conflict 和需要 clinical expert review 的 case；这些 prelabels 不能作为 gold benchmark。
