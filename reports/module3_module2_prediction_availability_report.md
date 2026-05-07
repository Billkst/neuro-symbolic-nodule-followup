# Module 2 Per-Sample Prediction Availability Audit

本报告只审计本地文件和可重导出路径，未训练、未加载 GPU、未运行模型推理。

## 结论

- `density_stage1`: 不可本地重导出：final_model_dir_absent_locally。
- `density_stage2`: 不可本地重导出：final_model_dir_absent_locally。
- `size`: 可用：已有 Wave5 has-size 逐样本概率；size_mm 仍只能来自已有 fact/mention 字段。
- `location`: 不可本地重导出：final_model_dir_absent_locally。

## 明细

### density_stage1

- final result: `outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_density_final_g3_len128_seed42.json` exists=True
- final model reference: `outputs/phaseA2_planB/models/density_stage1_mws_cfe_density_final_g3_len128_raw_seed42`
- local model exists: False
- aggregate result only: True
- existing prediction rows: 2018
- existing prediction source types: `constructed_fact_not_final_model:2018`
- final model prediction rows: 0
- size probability rows: 0
- dataset rows: 288894
- alignable fields: `note_id|subject_id|sample_id|mention_text`
- blocker: `final_model_dir_absent_locally`

### density_stage2

- final result: `outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed42.json` exists=True
- final model reference: `outputs/phaseA2_planB/models/density_stage2_mws_cfe_density_final_g3_len128_seed42`
- local model exists: False
- aggregate result only: True
- existing prediction rows: 2018
- existing prediction source types: `constructed_fact_not_final_model:2018`
- final model prediction rows: 0
- size probability rows: 0
- dataset rows: 288894
- alignable fields: `note_id|subject_id|sample_id|mention_text`
- blocker: `final_model_dir_absent_locally`

### size

- final result: `outputs/phaseA2_planB/results/mws_cfe_size_results_size_wave5_lexical_alone_seed42.json` exists=True
- final model reference: `/data/hcf/ljx/neuro-symbolic-nodule-followup/outputs/phaseA2_planB/models/size_wave5_lexical_expert_size_wave5_lexical_alone_seed42.joblib`
- local model exists: True
- aggregate result only: True
- existing prediction rows: 2018
- existing prediction source types: `constructed_fact_only_no_probability:1495|final_model_probability_file:523`
- final model prediction rows: 523
- size probability rows: 85280
- dataset rows: 288894
- alignable fields: `note_id|subject_id|sample_id|mention_text`
- blocker: `none_for_has_size_probability; size_mm remains constructed_fact_only`

### location

- final result: `outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed42.json` exists=True
- final model reference: `outputs/phaseA2_planB/models/location_mws_cfe_augmented_location_aug_g2_seed42`
- local model exists: False
- aggregate result only: True
- existing prediction rows: 2018
- existing prediction source types: `constructed_fact_not_final_model:2018`
- final model prediction rows: 0
- size probability rows: 0
- dataset rows: 288894
- alignable fields: `note_id|subject_id|sample_id|mention_text`
- blocker: `final_model_dir_absent_locally`

## HCF 导出命令

如需生成真正的 density/location final model 逐样本预测，应在 HCF 环境运行以下命令；本地未执行：

```bash
conda run -n follow-up python scripts/phaseA3/export_module2_predictions_for_module3.py --allow-model-inference --tasks density_stage1 --output-dir outputs/phaseA3/module2_predictions
conda run -n follow-up python scripts/phaseA3/export_module2_predictions_for_module3.py --allow-model-inference --tasks density_stage2 --output-dir outputs/phaseA3/module2_predictions
conda run -n follow-up python scripts/phaseA3/export_module2_predictions_for_module3.py --allow-model-inference --tasks location --output-dir outputs/phaseA3/module2_predictions
```
