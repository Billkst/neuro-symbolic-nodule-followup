# Schema 映射真实样例文档

> 生成日期：2026-04-03
>
> 说明：所有 `note_id`、`subject_id`、原始文本片段都来自真实数据；`extractor_version`、`generation_metadata` 等元数据字段是为了演示 schema 结构而补充的文档示例值。

本文面向开发者，展示原始 radiology report（放射报告）、discharge note（出院小结）与 guideline-based recommendation（基于指南的随访建议）如何稳定映射到四个 JSON schema。所有 JSON 都给出完整对象，不使用 `...` 省略。

## Part A: Radiology Report -> radiology_fact_schema 映射（5 examples）

### Example A1: 单发 4 mm 实性结节（腹盆 CT 偶然发现）

1. 来源信息
- `note_id`: `10046097-RR-34`
- `subject_id`: `10046097`
- `exam_name`: `CT ABD AND PELVIS WITH CONTRAST`

2. 原始文本片段
> LOWER CHEST: There is a 4 mm solid nodule in the right middle lobe (2:2).
>
> 3. 4 mm right middle lobe pulmonary nodule without prior study for comparison.
> For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended in a low-risk patient, and an optional CT in 12 months is recommended in a high-risk patient.

3. 映射后的 JSON
```json
{
  "note_id": "10046097-RR-34",
  "subject_id": 10046097,
  "exam_name": "CT ABD AND PELVIS WITH CONTRAST",
  "modality": "CT",
  "body_site": "chest_abdomen_pelvis",
  "report_text": "EXAMINATION: CT ABD AND PELVIS WITH CONTRAST\nINDICATION: NO_PO contrast; History: ___ with 1 day of n/v, BRBPR, diffuse abd tendernessNO_PO contrast// Colitis, other intraabdominal pathology present?\nTECHNIQUE: Single phase contrast: MDCT axial images were acquired through the abdomen and pelvis following intravenous contrast administration. Oral contrast was not administered. Coronal and sagittal reformations were performed and reviewed on PACS.\nCOMPARISON: None.\nFINDINGS: LOWER CHEST: There is a 4 mm solid nodule in the right middle lobe (2:2). Visualized lung fields are otherwise within normal limits. There is no evidence of pleural or pericardial effusion.\nIMPRESSION: 3. 4 mm right middle lobe pulmonary nodule without prior study for comparison. For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended in a low-risk patient, and an optional CT in 12 months is recommended in a high-risk patient.\nRECOMMENDATION(S): For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended in a low-risk patient, and an optional CT in 12 months is recommended in a high-risk patient.",
  "sections": {"indication": "NO_PO contrast; History: ___ with 1 day of n/v, BRBPR, diffuse abd tendernessNO_PO contrast// Colitis, other intraabdominal pathology present?", "technique": "Single phase contrast: MDCT axial images were acquired through the abdomen and pelvis following intravenous contrast administration. Oral contrast was not administered. Coronal and sagittal reformations were performed and reviewed on PACS.", "comparison": "None.", "findings": "LOWER CHEST: There is a 4 mm solid nodule in the right middle lobe (2:2). Visualized lung fields are otherwise within normal limits. There is no evidence of pleural or pericardial effusion.", "impression": "3. 4 mm right middle lobe pulmonary nodule without prior study for comparison. For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended in a low-risk patient, and an optional CT in 12 months is recommended in a high-risk patient."},
  "nodule_count": 1,
  "nodules": [{"nodule_id_in_report": 1, "size_mm": 4, "size_text": "4 mm", "density_category": "solid", "density_text": "solid", "location_lobe": "RML", "location_text": "right middle lobe", "count_type": "single", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended in a low-risk patient, and an optional CT in 12 months is recommended in a high-risk patient.", "evidence_span": "There is a 4 mm solid nodule in the right middle lobe (2:2).", "confidence": "high", "missing_flags": ["change_status", "change_text", "lung_rads_category"]}],
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
}
```

4. 缺失字段
- `change_status` / `change_text`: 报告明确写明 `without prior study for comparison`，因此不能标为 `stable` 或 `new`。
- `lung_rads_category`: 这是 incidental nodule 场景，不是筛查型 `Lung-RADS` 报告。

5. 规范化决策
- `right middle lobe` 规范化为 `RML`；`solid` 规范化为 `density_category = solid`。
- `body_site` 记为 `chest_abdomen_pelvis`，对应 schema 枚举中的腹盆联合扫描类型；虽然结节位于肺部，但检查范围覆盖胸腹盆，因此按检查覆盖区域而非结节位置赋值。
- 报告中的随访句子被完整复制到 `recommendation_cue`，方便下游监督 `text -> recommendation` 对齐。

### Example A2: 多发 part-solid 结节与高危形态学

1. 来源信息
- `note_id`: `10001401-RR-8`
- `subject_id`: `10001401`
- `exam_name`: `CT CHEST W/CONTRAST`

2. 原始文本片段
> There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect). Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186). Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate.
>
> Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum.

3. 映射后的 JSON
```json
{
  "note_id": "10001401-RR-8",
  "subject_id": 10001401,
  "exam_name": "CT CHEST W/CONTRAST",
  "modality": "CT",
  "body_site": "chest",
  "report_text": "EXAMINATION: CT CHEST W/CONTRAST\nINDICATION: ___ year old woman with bladder ca, schedule for radical cystectomy // please evaluate for any abnormalities, mets\nTECHNIQUE: Non contrasted CT chest\nCOMPARISON: No prior chest CT available for comparison.\nFINDINGS: Biapical pleural-parenchymal scarring. There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect). Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186). Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate. Mild centrilobular emphysematous changes. A few punctate calcified granulomas. Incidental lung cysts.\nIMPRESSION: Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum. Small indeterminate round 3 mm pulmonary nodule seen in the left lower lobe.\nRECOMMENDATION(S): The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.",
  "sections": {
    "indication": "___ year old woman with bladder ca, schedule for radical cystectomy // please evaluate for any abnormalities, mets",
    "technique": "Non contrasted CT chest",
    "comparison": "No prior chest CT available for comparison.",
    "findings": "Biapical pleural-parenchymal scarring. There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect). Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186). Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate. Mild centrilobular emphysematous changes. A few punctate calcified granulomas. Incidental lung cysts.",
    "impression": "Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum. Small indeterminate round 3 mm pulmonary nodule seen in the left lower lobe."
  },
  "nodule_count": 3,
  "nodules": [
    {"nodule_id_in_report": 1, "size_mm": 10.5, "size_text": "10.5 mm", "density_category": "part_solid", "density_text": "part solid", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": true, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect).", "confidence": "high", "missing_flags": ["change_status", "change_text", "lung_rads_category"]},
    {"nodule_id_in_report": 2, "size_mm": 7, "size_text": "7 mm", "density_category": "part_solid", "density_text": "part solid", "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186).", "confidence": "high", "missing_flags": ["change_status", "change_text", "lung_rads_category"]},
    {"nodule_id_in_report": 3, "size_mm": 3, "size_text": "3 mm", "density_category": "unclear", "density_text": null, "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate.", "confidence": "medium", "missing_flags": ["density_text", "change_status", "change_text", "lung_rads_category"]}
  ],
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
}
```

4. 缺失字段
- 三个结节对象的 `change_status` / `change_text` 都为 `null`，因为 `No prior chest CT available for comparison.`
- `lung_rads_category` 为 `null`，因为原始检查不是筛查型 LDCT，报告也未显式给出 `Lung-RADS`。
- 第 3 个结节的 `density_text` 为 `null`，因为原文只写 `indeterminate`，未给出明确密度类型。

5. 规范化决策
- 第一枚结节优先采用 FINDINGS 中的 `10.5 mm`，而不是 IMPRESSION 中四舍五入后的 `11 mm`。
- `Few small pulmonary nodules measuring 3 mm ... (6, 181) and (6, 118)` 被聚合成一个 `count_type = multiple` 的 grouped finding，而不是拆成两个 3 mm 对象。
- 只有第一枚结节明确出现 `spiculated`，因此 `spiculation = true` 只赋给第一枚。

### Example A3: LDCT 筛查场景下的稳定微小结节

1. 来源信息
- `note_id`: `10002221-RR-133`
- `subject_id`: `10002221`
- `exam_name`: `CT LOW DOSE LUNG SCREENING`

2. 原始文本片段
> Stable 2 mm calcified nodule in the right upper lobe (5, 103).
> Stable 1 mm solid right middle lobe pulmonary nodule (5, 145.
> Stable 1 mm right middle lobe pulmonary nodule (5, 147).
> Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).
>
> Stable tiny pulmonary nodules ranging in size from 1-2 mm.
>
> Lung-RADS category: 2
>
> RECOMMENDATION(S): Continue low-dose lung cancer screening CT in 12 months.

3. 映射后的 JSON
```json
{
  "note_id": "10002221-RR-133",
  "subject_id": 10002221,
  "exam_name": "CT LOW DOSE LUNG SCREENING",
  "modality": "LDCT",
  "body_site": "chest",
  "report_text": "EXAMINATION: CT LOW DOSE LUNG SCREENING\nINDICATION: ___ yr old, former smoker (1), 40 pk yrs, asymptomatic// Ct Lung Cancer Screening Baseline Ct Lung Cancer Screening Baseline\nTECHNIQUE: Volumetric CT acquisitions over the entire thorax in inspiration, no administration of intravenous contrast material, multiplanar reconstructions.\nCOMPARISON: Baseline screening CT. Comparison to prior CT chest done on ___.\nFINDINGS: Other nodules: Stable 2 mm calcified nodule in the right upper lobe (5, 103). Stable 1 mm solid right middle lobe pulmonary nodule (5, 145. Stable 1 mm right middle lobe pulmonary nodule (5, 147). Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).\nIMPRESSION: Stable tiny pulmonary nodules ranging in size from 1-2 mm. Mild upper lobe predominant emphysema. Lung-RADS category: 2\nRECOMMENDATION(S): Continue low-dose lung cancer screening CT in 12 months.",
  "sections": {"indication": "___ yr old, former smoker (1), 40 pk yrs, asymptomatic// Ct Lung Cancer Screening Baseline Ct Lung Cancer Screening Baseline", "technique": "Volumetric CT acquisitions over the entire thorax in inspiration, no administration of intravenous contrast material, multiplanar reconstructions.", "comparison": "Baseline screening CT. Comparison to prior CT chest done on ___.", "findings": "Other nodules: Stable 2 mm calcified nodule in the right upper lobe (5, 103). Stable 1 mm solid right middle lobe pulmonary nodule (5, 145. Stable 1 mm right middle lobe pulmonary nodule (5, 147). Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).", "impression": "Stable tiny pulmonary nodules ranging in size from 1-2 mm. Mild upper lobe predominant emphysema. Lung-RADS category: 2"},
  "nodule_count": 4,
  "nodules": [
    {"nodule_id_in_report": 1, "size_mm": 2, "size_text": "2 mm", "density_category": "calcified", "density_text": "calcified", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": true, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 2 mm calcified nodule in the right upper lobe (5, 103).", "confidence": "high", "missing_flags": []},
    {"nodule_id_in_report": 2, "size_mm": 1, "size_text": "1 mm", "density_category": "solid", "density_text": "solid", "location_lobe": "RML", "location_text": "right middle lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm solid right middle lobe pulmonary nodule (5, 145.", "confidence": "high", "missing_flags": []},
    {"nodule_id_in_report": 3, "size_mm": 1, "size_text": "1 mm", "density_category": "unclear", "density_text": null, "location_lobe": "RML", "location_text": "right middle lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm right middle lobe pulmonary nodule (5, 147).", "confidence": "medium", "missing_flags": ["density_text"]},
    {"nodule_id_in_report": 4, "size_mm": 1, "size_text": "1 mm", "density_category": "solid", "density_text": "solid", "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).", "confidence": "high", "missing_flags": []}
  ],
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
}
```

4. 缺失字段
- 第 3 枚结节 `density_text = null`，因为原文未说明其是 `solid`、`calcified` 还是 `ground-glass`。
- 其余结节对象没有 `null` 字段；筛查报告给出了足够完整的 size / stability / category / recommendation 信息。

5. 规范化决策
- 这是最标准的 screening case：`modality = LDCT`，并把 `Lung-RADS category: 2` 复制到每个 nodule 对象的 `lung_rads_category`。
- 第一枚结节同时编码 `density_category = calcified` 与 `calcification = true`，便于分别支持分类任务和布尔属性任务。
- INDICATION 中的 `former smoker (1), 40 pk yrs` 没有直接进入 radiology schema，但为后续 case bundle 的弱吸烟标签提供链接依据。

### Example A4: 1-2 mm 模糊大小范围与钙化不确定性

1. 来源信息
- `note_id`: `10001338-RR-42`
- `subject_id`: `10001338`
- `exam_name`: `CTA CHEST`

2. 原始文本片段
> Tiny calcified granuloma is present at the right base (2:46). A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).
>
> There are four, less than 4 mm pulmonary nodules, at least one of which is calcified with additional finding of calcified lymph node in the right paratracheal region. These may relate to sequelae of prior granulomatous disease. In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.

3. 映射后的 JSON
```json
{
  "note_id": "10001338-RR-42",
  "subject_id": 10001338,
  "exam_name": "CTA CHEST",
  "modality": "CTA",
  "body_site": "chest",
  "report_text": "EXAMINATION: CTA CHEST\nINDICATION: ___ female with faintness and ischemic lesion in cecum. Question PE.\nTECHNIQUE: Unenhanced axial images were obtained of the chest using a low-dose protocol. Subsequently, images were obtained after the uneventful administration of 100 mL Omnipaque intravenous contrast. Coronal and sagittal and oblique reformatted images were constructed.\nCOMPARISON: ___.\nFINDINGS: Tiny calcified granuloma is present at the right base (2:46). A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).\nIMPRESSION: 2. There are four, less than 4 mm pulmonary nodules, at least one of which is calcified with additional finding of calcified lymph node in the right paratracheal region. These may relate to sequelae of prior granulomatous disease. In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.",
  "sections": {"indication": "___ female with faintness and ischemic lesion in cecum. Question PE.", "technique": "Unenhanced axial images were obtained of the chest using a low-dose protocol. Subsequently, images were obtained after the uneventful administration of 100 mL Omnipaque intravenous contrast. Coronal and sagittal and oblique reformatted images were constructed.", "comparison": "___.", "findings": "Tiny calcified granuloma is present at the right base (2:46). A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "impression": "2. There are four, less than 4 mm pulmonary nodules, at least one of which is calcified with additional finding of calcified lymph node in the right paratracheal region. These may relate to sequelae of prior granulomatous disease. In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary."},
  "nodule_count": 3,
  "nodules": [
    {"nodule_id_in_report": 1, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]},
    {"nodule_id_in_report": 2, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "lingula", "location_text": "lingula", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]},
    {"nodule_id_in_report": 3, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "RLL", "location_text": "right lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]}
  ],
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
}
```

4. 缺失字段
- 三个结节对象的 `change_status` / `change_text` 都为 `null`，因为没有 prior-growth 证据。
- `calcification` 设为 `null` 而不是 `false`，因为原文是 `not definitely calcified`，属于明确不确定。
- `lung_rads_category` 为 `null`，因为这不是筛查型 `Lung-RADS` 报告。

5. 规范化决策
- `1-2 mm` 以区间中点近似成 `size_mm = 1.5`，同时保留原文 `size_text = "1-2 mm"`。
- 本例只展开 3 个 `additional` nodules；句首那个 `tiny calcified granuloma` 没有纳入对象列表，因此 `nodule_count = 3` 是示例级抽取结果，而不是全文总病灶数。
- `right upper lobe` / `lingula` / `right lower lobe` 分别映射为 `RUL` / `lingula` / `RLL`。

### Example A5: perifissural 结节 + 条件式随访建议

1. 来源信息
- `note_id`: `10001176-RR-13`
- `subject_id`: `10001176`
- `exam_name`: `CT ABDOMEN AND PELVIS`

2. 原始文本片段
> Perifissural nodule is seen on the uppermost slice on the right major fissure measuring 3 mm. A right lower lobe nodule measures 3mm.
>
> 3-mm nodule seen along the right major fissure and right lower lobe. According to ___ guidelines, in the absence of risk factors, no further followup is needed. If patient has risk factors such as smoking, followup chest CT at 12 months is recommended to document stability.

3. 映射后的 JSON
```json
{
  "note_id": "10001176-RR-13",
  "subject_id": 10001176,
  "exam_name": "CT ABDOMEN AND PELVIS",
  "modality": "CT",
  "body_site": "other",
  "report_text": "EXAMINATION: CT ABDOMEN AND PELVIS\nINDICATION: ___ female with fevers, nausea, vomiting, abdominal pain status post appendectomy and cholecystectomy. Question colitis.\nCOMPARISON: None.\nTECHNIQUE: Helical CT images were acquired of the abdomen and pelvis following the administration of intravenous contrast and reformatted into coronal and sagittal planes.\nFINDINGS: LUNG BASES: Perifissural nodule is seen on the uppermost slice on the right major fissure measuring 3 mm. A right lower lobe nodule measures 3mm. There is bibasilar atelectasis, but no pleural effusion. Note is made of coronary arterial calcification, the heart is normal in size.\nIMPRESSION: 2. 3-mm nodule seen along the right major fissure and right lower lobe. According to ___ guidelines, in the absence of risk factors, no further followup is needed. If patient has risk factors such as smoking, followup chest CT at 12 months is recommended to document stability.",
  "sections": {"indication": "___ female with fevers, nausea, vomiting, abdominal pain status post appendectomy and cholecystectomy. Question colitis.", "technique": "Helical CT images were acquired of the abdomen and pelvis following the administration of intravenous contrast and reformatted into coronal and sagittal planes.", "comparison": "None.", "findings": "LUNG BASES: Perifissural nodule is seen on the uppermost slice on the right major fissure measuring 3 mm. A right lower lobe nodule measures 3mm. There is bibasilar atelectasis, but no pleural effusion. Note is made of coronary arterial calcification, the heart is normal in size.", "impression": "2. 3-mm nodule seen along the right major fissure and right lower lobe. According to ___ guidelines, in the absence of risk factors, no further followup is needed. If patient has risk factors such as smoking, followup chest CT at 12 months is recommended to document stability."},
  "nodule_count": 2,
  "nodules": [
    {"nodule_id_in_report": 1, "size_mm": 3, "size_text": "3 mm", "density_category": "unclear", "density_text": null, "location_lobe": "unclear", "location_text": "right major fissure", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": true, "lung_rads_category": null, "recommendation_cue": "According to ___ guidelines, in the absence of risk factors, no further followup is needed. If patient has risk factors such as smoking, followup chest CT at 12 months is recommended to document stability.", "evidence_span": "Perifissural nodule is seen on the uppermost slice on the right major fissure measuring 3 mm.", "confidence": "medium", "missing_flags": ["density_text", "change_status", "change_text", "lung_rads_category"]},
    {"nodule_id_in_report": 2, "size_mm": 3, "size_text": "3mm", "density_category": "unclear", "density_text": null, "location_lobe": "RLL", "location_text": "right lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "According to ___ guidelines, in the absence of risk factors, no further followup is needed. If patient has risk factors such as smoking, followup chest CT at 12 months is recommended to document stability.", "evidence_span": "A right lower lobe nodule measures 3mm.", "confidence": "high", "missing_flags": ["density_text", "change_status", "change_text", "lung_rads_category"]}
  ],
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
}
```

4. 缺失字段
- 两枚结节都没有 prior comparison，因此 `change_status` / `change_text` 为 `null`。
- 两枚结节都缺少明确的 density 描述，因此 `density_category = unclear` 且 `density_text = null`。
- `lung_rads_category` 为 `null`，因为这是 incidental abdominal CT 发现，不适用筛查分级。

5. 规范化决策
- 第一枚结节保留 `perifissural = true`，同时把位置原文 `right major fissure` 存入 `location_text`；`location_lobe` 则保守映射为 `unclear`。
- 第二枚结节 `A right lower lobe nodule measures 3mm` 可稳定映射为 `RLL`。
- 条件式建议被完整放进 `recommendation_cue`，保留 `absence of risk factors` 与 `risk factors such as smoking` 这两条分支。

## Part B: Discharge Note -> smoking_eligibility_schema 映射（3 examples）

### Example B1: former smoker + 明确 pack-year

1. 来源信息
- `note_id`: `10014967-DS-11`
- `subject_id`: `10014967`

2. 原始文本片段
> Ms. ___ is a ___ ___ female with history of poorly controlled HTN, GERD and former 46 pack year history of smoking who presents with ongoing symptoms of dyspnea and cough

3. 映射后的 JSON
```json
{
  "subject_id": 10014967,
  "note_id": "10014967-DS-11",
  "source_section": "HPI",
  "smoking_status_raw": "Ms. ___ is a ___ ___ female with history of poorly controlled HTN, GERD and former 46 pack year history of smoking who presents with ongoing symptoms of dyspnea and cough",
  "smoking_status_norm": "former_smoker",
  "pack_year_value": 46,
  "pack_year_text": "46 pack year history of smoking",
  "ppd_value": null,
  "ppd_text": null,
  "years_smoked_value": null,
  "years_smoked_text": null,
  "quit_years_value": null,
  "quit_years_text": null,
  "evidence_span": "Ms. ___ is a ___ ___ female with history of poorly controlled HTN, GERD and former 46 pack year history of smoking who presents with ongoing symptoms of dyspnea and cough",
  "ever_smoker_flag": true,
  "eligible_for_high_risk_screening": "unknown",
  "eligibility_criteria_applied": "USPSTF_2021",
  "eligibility_reason": "Former smoker with explicit 46 pack-years, but age and years since quitting are unavailable in the note snippet.",
  "evidence_quality": "high",
  "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"},
  "missing_flags": ["ppd_value", "ppd_text", "years_smoked_value", "years_smoked_text", "quit_years_value", "quit_years_text"],
  "data_quality_notes": "Pack-year is explicit, but de-identified age and quit timing prevent a final screening-eligibility call."
}
```

4. 缺失字段
- `ppd_value` / `ppd_text`: 原文没有每日吸烟量。
- `years_smoked_value` / `years_smoked_text`: 原文没有给出吸烟持续年数。
- `quit_years_value` / `quit_years_text`: 只说 `former`，没有具体戒烟时长。

5. 规范化决策
- `former 46 pack year history of smoking` 直接归一到 `smoking_status_norm = former_smoker` 与 `pack_year_value = 46`。
- 虽然 pack-years 已满足高危阈值，但年龄缺失，因此 `eligible_for_high_risk_screening = unknown`。

### Example B2: PPD + duration -> 计算 pack-year

1. 来源信息
- `note_id`: `10010399-DS-8`
- `subject_id`: `10010399`

2. 原始文本片段
> Smokes 1.5 ppd for 30+ years, trying to quit

3. 映射后的 JSON
```json
{"subject_id": 10010399, "note_id": "10010399-DS-8", "source_section": "Social History", "smoking_status_raw": "Smokes 1.5 ppd for 30+ years, trying to quit", "smoking_status_norm": "current_smoker", "pack_year_value": 45, "pack_year_text": null, "ppd_value": 1.5, "ppd_text": "1.5 ppd", "years_smoked_value": 30, "years_smoked_text": "30+ years", "quit_years_value": null, "quit_years_text": null, "evidence_span": "Smokes 1.5 ppd for 30+ years, trying to quit", "ever_smoker_flag": true, "eligible_for_high_risk_screening": "unknown", "eligibility_criteria_applied": "USPSTF_2021", "eligibility_reason": "Current smoker with explicit PPD and duration; pack-year threshold is met after computation, but age is unavailable.", "evidence_quality": "high", "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}, "missing_flags": ["pack_year_text", "quit_years_value", "quit_years_text"], "data_quality_notes": "pack_year_value is computed as 1.5 × 30 = 45 using the lower bound of '30+ years'; the note does not state a pack-year value directly."}
```

4. 缺失字段
- `pack_year_text`: 原文没有显式出现 `45 pack-years` 这样的短语，因此只能保留数值、不保留 raw text。
- `quit_years_value` / `quit_years_text`: `trying to quit` 不等于已经戒烟。

5. 规范化决策
- `smokes` 明确落到 `current_smoker`。
- `pack_year_value = 45` 是计算值，不是直接抽取值；这里按下界 `1.5 × 30` 处理，并在 `data_quality_notes` 中显式标注。
- `30+ years` 被规范化为 `years_smoked_value = 30`，同时保留 `years_smoked_text = "30+ years"` 以表达截断不确定性。

### Example B3: 口语化戒烟时间表达

1. 来源信息
- `note_id`: `10000032-DS-21`
- `subject_id`: `10000032`

2. 原始文本片段
> She quit smoking a couple of years ago.

3. 映射后的 JSON
```json
{"subject_id": 10000032, "note_id": "10000032-DS-21", "source_section": "Family History", "smoking_status_raw": "She quit smoking a couple of years ago.", "smoking_status_norm": "former_smoker", "pack_year_value": null, "pack_year_text": null, "ppd_value": null, "ppd_text": null, "years_smoked_value": null, "years_smoked_text": null, "quit_years_value": 2, "quit_years_text": "a couple of years ago", "evidence_span": "She quit smoking a couple of years ago.", "ever_smoker_flag": true, "eligible_for_high_risk_screening": "unknown", "eligibility_criteria_applied": "USPSTF_2021", "eligibility_reason": "Former smoker status is clear and quit timing is approximately within 15 years, but total tobacco exposure and age are missing.", "evidence_quality": "medium", "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}, "missing_flags": ["pack_year_value", "pack_year_text", "ppd_value", "ppd_text", "years_smoked_value", "years_smoked_text"], "data_quality_notes": "quit_years_value=2 is an approximation derived from the colloquial phrase 'a couple of years ago'."}
```

4. 缺失字段
- `pack_year_value` / `pack_year_text`: 没有累计暴露信息。
- `ppd_value` / `ppd_text`: 没有每日吸烟量信息。
- `years_smoked_value` / `years_smoked_text`: 没有吸烟总年限。

5. 规范化决策
- `quit smoking` 明确映射为 `former_smoker`，`ever_smoker_flag = true`。
- `a couple of years ago` 近似规范化为 `quit_years_value = 2`，但必须在 `data_quality_notes` 中保留“不确定近似”说明。

## Part C: Recommendation Schema 样例（5 examples, rule-constructed）

### Example C1: 基于 A1 的 Fleischner 无需随访样例

1. 案例依据
- 依据病例：`A1`
- 规则来源：`Fleischner_2017`

2. 映射后的 JSON
```json
{"case_id": "REC-10046097-A1", "recommendation_level": "no_followup_needed", "recommendation_action": "对于 low-risk 患者，该 4 mm 单发 solid nodule 无需进一步 CT 随访。", "followup_interval": "none", "followup_modality": "none", "lung_rads_category": null, "guideline_source": "Fleischner_2017", "guideline_anchor": "Single solid nodule < 6 mm in low-risk adult: no routine follow-up.", "reasoning_path": ["solid_nodule", "size_4mm < 6mm_threshold", "single_nodule", "low_risk_patient", "→ no_followup_needed"], "triggered_rules": ["fleischner_single_solid_lt6_low_risk"], "input_facts_used": {"nodule_size_mm": 4, "nodule_density": "solid", "nodule_count": 1, "change_status": null, "patient_risk_level": "low_risk", "smoking_eligible": null}, "missing_information": [], "uncertainty_note": "风险分层来自报告原文中的 low-risk / high-risk 条件句，而不是来自独立吸烟结构化记录。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
```

3. 缺失字段
- `lung_rads_category = null`，因为本例不属于 lung cancer screening 报告，也不是 `Lung-RADS` 规则入口。

4. 规范化决策
- `followup_interval` 与 `followup_modality` 分别编码为 `none`，避免用 `null` 混淆“未知”与“不适用”。
- `patient_risk_level = low_risk` 来自原始 recommendation cue 的条件句，而不是单独吸烟模型输出。

### Example C2: 基于 A2 的 diagnostic workup 样例

1. 案例依据
- 依据病例：`A2`
- 规则来源：`Lung-RADS_v2022`

2. 映射后的 JSON
```json
{"case_id": "REC-10001401-A2", "recommendation_level": "diagnostic_workup", "recommendation_action": "优先进入 diagnostic_workup 路径，尽快行 PET_CT 进一步表征；若暂不实施，可在 3_months 内复查。", "followup_interval": "immediate", "followup_modality": "PET_CT", "lung_rads_category": "4A", "guideline_source": "Lung-RADS_v2022", "guideline_anchor": "Part-solid nodule with solid component >= 8 mm and suspicious morphology requires diagnostic evaluation.", "reasoning_path": ["part_solid_nodule", "solid_component >= 8mm", "spiculated_margins", "→ Lung-RADS_4A", "→ diagnostic_workup"], "triggered_rules": ["lung_rads_part_solid_ge_8mm", "suspicious_spiculation_escalation"], "input_facts_used": {"nodule_size_mm": 10.5, "nodule_density": "part_solid", "nodule_count": 3, "change_status": null, "patient_risk_level": null, "smoking_eligible": null}, "missing_information": [], "uncertainty_note": "FINDINGS 写 10.5 mm，而 IMPRESSION 四舍五入为 11 mm；本例优先采用更细粒度的 10.5 mm 测量值。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
```

3. 缺失字段
- `input_facts_used.patient_risk_level = null`，因为本例主要由结节形态学与尺寸驱动，而不是患者风险分层驱动。

4. 规范化决策
- `followup_interval = immediate` 表示应优先进入诊断性评估，而不是简单例行复查。
- `followup_modality = PET_CT` 与原始报告建议保持一致；`3 month follow-up CT` 被保留在 `recommendation_action` 与 `uncertainty_note` 中。

### Example C3: 基于 A3 的 annual screening 样例

1. 案例依据
- 依据病例：`A3`
- 规则来源：`Lung-RADS_v2022`

2. 映射后的 JSON
```json
{"case_id": "REC-10002221-A3", "recommendation_level": "routine_screening", "recommendation_action": "维持 annual LDCT 筛查路径，12_months 后重复低剂量胸部 CT。", "followup_interval": "12_months", "followup_modality": "LDCT", "lung_rads_category": "2", "guideline_source": "Lung-RADS_v2022", "guideline_anchor": "Lung-RADS Category 2: benign appearance or behavior; continue annual screening with LDCT.", "reasoning_path": ["stable_tiny_nodules", "lung_rads_category_2", "screening_context", "→ routine_screening"], "triggered_rules": ["lung_rads_category_2_annual_ldct"], "input_facts_used": {"nodule_size_mm": 2, "nodule_density": "mixed_tiny_nodules", "nodule_count": 4, "change_status": "stable", "patient_risk_level": "screening_population", "smoking_eligible": "unknown"}, "missing_information": [], "uncertainty_note": "该建议主要由报告内显式给出的 Lung-RADS 2 驱动，不依赖单独的吸烟资格推断。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
```

3. 缺失字段
- 无；该 recommendation 对象没有 `null` 字段。

4. 规范化决策
- `lung_rads_category = 2` 直接继承自报告原文，而不是后验推断。
- `smoking_eligible = unknown` 不影响输出，因为 annual screening recommendation 已由 screening context 和 category 明确给出。

### Example C4: 基于 A5 的高风险 12 个月复查样例

1. 案例依据
- 依据病例：`A5`
- 规则来源：`Fleischner_2017`

2. 映射后的 JSON
```json
{"case_id": "REC-10001176-A5", "recommendation_level": "short_interval_followup", "recommendation_action": "若患者属于 high-risk 人群，建议 12_months 后复查胸部 CT 以确认稳定。", "followup_interval": "12_months", "followup_modality": "LDCT", "lung_rads_category": null, "guideline_source": "Fleischner_2017", "guideline_anchor": "Small incidental nodules in high-risk patients may undergo optional 12-month CT follow-up.", "reasoning_path": ["perifissural_nodule", "size_3mm < 6mm_threshold", "multiple_small_nodules", "high_risk_patient", "→ optional_12_month_followup"], "triggered_rules": ["fleischner_small_nodule_high_risk_optional_12m"], "input_facts_used": {"nodule_size_mm": 3, "nodule_density": "unclear", "nodule_count": 2, "change_status": null, "patient_risk_level": "high_risk", "smoking_eligible": null}, "missing_information": [], "uncertainty_note": "recommendation_schema 不含通用 noncontrast chest CT 枚举，因此此处用 LDCT 表示 12 个月胸部 CT 随访。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
```

3. 缺失字段
- `lung_rads_category = null`，因为这不是筛查型 `Lung-RADS` 处置。

4. 规范化决策
- 尽管第一枚结节是 `perifissural`，这里仍保留 `short_interval_followup` 示例，是为了展示条件式高风险分支如何被规则化。
- 由于 schema 没有通用 chest CT 枚举，此处把 12 个月胸部 CT 随访编码为 `LDCT`，并在 `uncertainty_note` 中显式说明这个 schema gap。

### Example C5: insufficient_data 占位输出

1. 案例依据
- 依据场景：缺少关键输入事实
- 输出类型：`rule_based` fallback

2. 映射后的 JSON
```json
{"case_id": "REC-INSUFFICIENT-DATA-001", "recommendation_level": "insufficient_data", "recommendation_action": "缺少足够事实，当前仅能输出 insufficient_data 占位结果。", "followup_interval": null, "followup_modality": null, "lung_rads_category": null, "guideline_source": "none", "guideline_anchor": null, "reasoning_path": ["required_fact_missing", "nodule_size_absent_or_unreliable", "prior_imaging_absent", "patient_risk_factors_absent", "→ insufficient_data"], "triggered_rules": ["fallback_insufficient_data"], "input_facts_used": {"nodule_size_mm": null, "nodule_density": null, "nodule_count": null, "change_status": null, "patient_risk_level": null, "smoking_eligible": null}, "missing_information": ["nodule_size", "prior_imaging", "patient_risk_factors"], "uncertainty_note": "当关键结构化输入缺失时，规则引擎回退到该占位输出，而不是强行给出具体管理建议。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
```

3. 缺失字段
- `followup_interval` / `followup_modality` / `lung_rads_category` / `guideline_anchor` 都为 `null`，因为无法稳定落到具体指南分支。
- `input_facts_used` 中所有字段均为 `null`，表示上游结构化抽取没有提供足够输入。

4. 规范化决策
- `guideline_source = none` 与 `recommendation_level = insufficient_data` 配套出现，用来区分“没有规则入口”而不是“规则推理后建议不随访”。
- `missing_information` 显式列出 `nodule_size`、`prior_imaging`、`patient_risk_factors`，方便下游错误分析。

## Part D: 完整 Case Bundle 样例（3 examples）

### Example D1: fully labeled (silver) case bundle

1. 来源信息
- `case_id`: `CASE-10001401-001`
- `subject_id`: `10001401`
- 放射来源：`10001401-RR-8`
- 出院小结来源：`10001401-DS-17`

2. 映射后的 JSON
```json
{
  "case_id": "CASE-10001401-001",
  "subject_id": 10001401,
  "demographics": {"age": null, "sex": null, "race": null, "insurance": null, "source": null, "missing_flags": ["age", "sex", "race", "insurance", "source"]},
  "radiology_facts": [
    {
      "note_id": "10001401-RR-8",
      "subject_id": 10001401,
      "exam_name": "CT CHEST W/CONTRAST",
      "modality": "CT",
      "body_site": "chest",
      "report_text": "EXAMINATION: CT CHEST W/CONTRAST\nINDICATION: ___ year old woman with bladder ca, schedule for radical cystectomy // please evaluate for any abnormalities, mets\nTECHNIQUE: Non contrasted CT chest\nCOMPARISON: No prior chest CT available for comparison.\nFINDINGS: Biapical pleural-parenchymal scarring. There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect). Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186). Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate. Mild centrilobular emphysematous changes. A few punctate calcified granulomas. Incidental lung cysts.\nIMPRESSION: Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum. Small indeterminate round 3 mm pulmonary nodule seen in the left lower lobe.\nRECOMMENDATION(S): The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.",
      "sections": {
        "indication": "___ year old woman with bladder ca, schedule for radical cystectomy // please evaluate for any abnormalities, mets",
        "technique": "Non contrasted CT chest",
        "comparison": "No prior chest CT available for comparison.",
        "findings": "Biapical pleural-parenchymal scarring. There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect). Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186). Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate. Mild centrilobular emphysematous changes. A few punctate calcified granulomas. Incidental lung cysts.",
        "impression": "Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum. Small indeterminate round 3 mm pulmonary nodule seen in the left lower lobe."
      },
      "nodule_count": 3,
      "nodules": [
        {"nodule_id_in_report": 1, "size_mm": 10.5, "size_text": "10.5 mm", "density_category": "part_solid", "density_text": "part solid", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": true, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe (6, 146) with the nodule and solid component measuring 10.5 mm (the sub solid component is seen in its inferior aspect).", "confidence": "high", "missing_flags": ["change_status", "change_text", "lung_rads_category"]},
        {"nodule_id_in_report": 2, "size_mm": 7, "size_text": "7 mm", "density_category": "part_solid", "density_text": "part solid", "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe (6, 186).", "confidence": "high", "missing_flags": ["change_status", "change_text", "lung_rads_category"]},
        {"nodule_id_in_report": 3, "size_mm": 3, "size_text": "3 mm", "density_category": "unclear", "density_text": null, "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.", "evidence_span": "Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe (6, 181) and (6, 118) which are indeterminate.", "confidence": "medium", "missing_flags": ["density_text", "change_status", "change_text", "lung_rads_category"]}
      ],
      "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
    }
  ],
  "smoking_eligibility": {"subject_id": 10001401, "note_id": "10001401-DS-17", "source_section": null, "smoking_status_raw": null, "smoking_status_norm": null, "pack_year_value": null, "pack_year_text": null, "ppd_value": null, "ppd_text": null, "years_smoked_value": null, "years_smoked_text": null, "quit_years_value": null, "quit_years_text": null, "evidence_span": null, "ever_smoker_flag": null, "eligible_for_high_risk_screening": "unknown", "eligibility_criteria_applied": "none", "eligibility_reason": "已链接真实 discharge note 10001401-DS-17，但未检出 smoking/tobacco 相关证据。", "evidence_quality": "none", "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}, "missing_flags": ["source_section", "smoking_status_raw", "smoking_status_norm", "pack_year_value", "pack_year_text", "ppd_value", "ppd_text", "years_smoked_value", "years_smoked_text", "quit_years_value", "quit_years_text", "evidence_span", "ever_smoker_flag"], "data_quality_notes": "真实出院小结 10001401-DS-17 已检索，但没有可用于吸烟结构化抽取的原文证据。"},
  "recommendation_target": {
    "ground_truth_action": "The larger spiculated part solid nodule in the right upper lobe measures 11 mm in diameter. Thus, PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.",
    "ground_truth_source": "extracted_from_report",
    "ground_truth_interval": "immediate_or_3_months",
    "recommendation_output": {"case_id": "REC-10001401-A2", "recommendation_level": "diagnostic_workup", "recommendation_action": "优先进入 diagnostic_workup 路径，尽快行 PET_CT 进一步表征；若暂不实施，可在 3_months 内复查。", "followup_interval": "immediate", "followup_modality": "PET_CT", "lung_rads_category": "4A", "guideline_source": "Lung-RADS_v2022", "guideline_anchor": "Part-solid nodule with solid component >= 8 mm and suspicious morphology requires diagnostic evaluation.", "reasoning_path": ["part_solid_nodule", "solid_component >= 8mm", "spiculated_margins", "→ Lung-RADS_4A", "→ diagnostic_workup"], "triggered_rules": ["lung_rads_part_solid_ge_8mm", "suspicious_spiculation_escalation"], "input_facts_used": {"nodule_size_mm": 10.5, "nodule_density": "part_solid", "nodule_count": 3, "change_status": null, "patient_risk_level": null, "smoking_eligible": null}, "missing_information": [], "uncertainty_note": "FINDINGS 写 10.5 mm，而 IMPRESSION 四舍五入为 11 mm；本例优先采用更细粒度的 10.5 mm 测量值。", "output_type": "rule_based", "generation_metadata": {"engine_version": "schema_examples_v1", "generation_timestamp": "2026-04-03T00:00:00Z", "rules_version": "guideline_rules_v1"}}
  },
  "provenance": {"radiology_note_ids": ["10001401-RR-8"], "discharge_note_id": "10001401-DS-17", "data_version": "mimic-iv-note-2.2", "extraction_date": "2026-04-03", "pipeline_version": "schema_examples_v1"},
  "split": "test",
  "label_quality": "silver",
  "case_notes": "银标 bundle：放射事实来自 10001401-RR-8，推荐输出按 Lung-RADS/FINDINGS 规则构造；同 subject 的 discharge note 已链接但未发现吸烟证据。"
}
```

3. 缺失字段
- `demographics` 中 `age` / `sex` / `race` / `insurance` / `source` 全为 `null`，因为本示例只依赖 note-level 真实文本，不额外回表人口学文件。
- 内嵌 `smoking_eligibility` 对象的大多数字段为 `null`，因为真实 discharge note 已检索但未发现 smoking evidence。

4. 规范化决策
- 尽管 linked discharge note 没有吸烟证据，`smoking_eligibility` 仍保留为非空对象，从而满足 `case_bundle_schema` 对 `silver` 样本的约束。
- `recommendation_target.recommendation_output` 直接复用 `C2` 的 rule-based recommendation，用于展示 silver-level 结构化标签。

### Example D2: partially labeled (weak) case bundle

1. 来源信息
- `case_id`: `CASE-10002221-001`
- `subject_id`: `10002221`
- 放射来源：`10002221-RR-133`
- 出院小结来源：`10002221-DS-8`

2. 映射后的 JSON
```json
{
  "case_id": "CASE-10002221-001",
  "subject_id": 10002221,
  "demographics": {"age": null, "sex": null, "race": null, "insurance": null, "source": null, "missing_flags": ["age", "sex", "race", "insurance", "source"]},
  "radiology_facts": [
    {
      "note_id": "10002221-RR-133",
      "subject_id": 10002221,
      "exam_name": "CT LOW DOSE LUNG SCREENING",
      "modality": "LDCT",
      "body_site": "chest",
      "report_text": "EXAMINATION: CT LOW DOSE LUNG SCREENING\nINDICATION: ___ yr old, former smoker (1), 40 pk yrs, asymptomatic// Ct Lung Cancer Screening Baseline Ct Lung Cancer Screening Baseline\nTECHNIQUE: Volumetric CT acquisitions over the entire thorax in inspiration, no administration of intravenous contrast material, multiplanar reconstructions.\nCOMPARISON: Baseline screening CT. Comparison to prior CT chest done on ___.\nFINDINGS: Other nodules: Stable 2 mm calcified nodule in the right upper lobe (5, 103). Stable 1 mm solid right middle lobe pulmonary nodule (5, 145. Stable 1 mm right middle lobe pulmonary nodule (5, 147). Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).\nIMPRESSION: Stable tiny pulmonary nodules ranging in size from 1-2 mm. Mild upper lobe predominant emphysema. Lung-RADS category: 2\nRECOMMENDATION(S): Continue low-dose lung cancer screening CT in 12 months.",
      "sections": {"indication": "___ yr old, former smoker (1), 40 pk yrs, asymptomatic// Ct Lung Cancer Screening Baseline Ct Lung Cancer Screening Baseline", "technique": "Volumetric CT acquisitions over the entire thorax in inspiration, no administration of intravenous contrast material, multiplanar reconstructions.", "comparison": "Baseline screening CT. Comparison to prior CT chest done on ___.", "findings": "Other nodules: Stable 2 mm calcified nodule in the right upper lobe (5, 103). Stable 1 mm solid right middle lobe pulmonary nodule (5, 145. Stable 1 mm right middle lobe pulmonary nodule (5, 147). Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).", "impression": "Stable tiny pulmonary nodules ranging in size from 1-2 mm. Mild upper lobe predominant emphysema. Lung-RADS category: 2"},
      "nodule_count": 4,
      "nodules": [
        {"nodule_id_in_report": 1, "size_mm": 2, "size_text": "2 mm", "density_category": "calcified", "density_text": "calcified", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": true, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 2 mm calcified nodule in the right upper lobe (5, 103).", "confidence": "high", "missing_flags": []},
        {"nodule_id_in_report": 2, "size_mm": 1, "size_text": "1 mm", "density_category": "solid", "density_text": "solid", "location_lobe": "RML", "location_text": "right middle lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm solid right middle lobe pulmonary nodule (5, 145.", "confidence": "high", "missing_flags": []},
        {"nodule_id_in_report": 3, "size_mm": 1, "size_text": "1 mm", "density_category": "unclear", "density_text": null, "location_lobe": "RML", "location_text": "right middle lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm right middle lobe pulmonary nodule (5, 147).", "confidence": "medium", "missing_flags": ["density_text"]},
        {"nodule_id_in_report": 4, "size_mm": 1, "size_text": "1 mm", "density_category": "solid", "density_text": "solid", "location_lobe": "LLL", "location_text": "left lower lobe", "count_type": "multiple", "change_status": "stable", "change_text": "Stable", "calcification": false, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": "2", "recommendation_cue": "Continue low-dose lung cancer screening CT in 12 months.", "evidence_span": "Stable 1 mm solid left lower lobe pulmonary nodule (5, 175).", "confidence": "high", "missing_flags": []}
      ],
      "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
    }
  ],
  "smoking_eligibility": {
    "subject_id": 10002221,
    "note_id": "10002221-DS-8",
    "source_section": "Transitional Issues + linked radiology indication",
    "smoking_status_raw": "# Cigarette smoking: The patient quit smoking on admission to the ER ___, please provide encouragement and resources regarding smoking cessation.",
    "smoking_status_norm": "former_smoker",
    "pack_year_value": 40,
    "pack_year_text": "40 pk yrs",
    "ppd_value": null,
    "ppd_text": null,
    "years_smoked_value": null,
    "years_smoked_text": null,
    "quit_years_value": 0,
    "quit_years_text": "quit smoking on admission to the ER ___",
    "evidence_span": "former smoker (1), 40 pk yrs; # Cigarette smoking: The patient quit smoking on admission to the ER ___, please provide encouragement and resources regarding smoking cessation.",
    "ever_smoker_flag": true,
    "eligible_for_high_risk_screening": "unknown",
    "eligibility_criteria_applied": "USPSTF_2021",
    "eligibility_reason": "linked radiology indication 给出 40 pack-years，discharge note 仅支持近期戒烟；由于年龄缺失，因此资格仍为 unknown。",
    "evidence_quality": "low",
    "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"},
    "missing_flags": ["ppd_value", "ppd_text", "years_smoked_value", "years_smoked_text"],
    "data_quality_notes": "弱标签构造：pack-year 字段由同一 subject 的放射报告 indication 回填，出院小结仅提供 recent quit 证据。"
  },
  "recommendation_target": {"ground_truth_action": "Continue low-dose lung cancer screening CT in 12 months.", "ground_truth_source": "extracted_from_report", "ground_truth_interval": "12_months", "recommendation_output": null},
  "provenance": {"radiology_note_ids": ["10002221-RR-133"], "discharge_note_id": "10002221-DS-8", "data_version": "mimic-iv-note-2.2", "extraction_date": "2026-04-03", "pipeline_version": "schema_examples_v1"},
  "split": "test",
  "label_quality": "weak",
  "case_notes": "弱标 bundle：ground truth action 直接取自筛查报告；吸烟资格对象保留真实 DS note_id，但 pack-year 字段弱回填自 linked radiology indication。"
}
```

3. 缺失字段
- `recommendation_target.recommendation_output = null`，因为 `weak` 样本只保留 ground-truth action，不强行生成结构化 recommendation label。
- 内嵌 `smoking_eligibility` 里的 `ppd` 与 `years_smoked` 仍然缺失；这里只能得到 weak pack-year signal。

4. 规范化决策
- 本例严格遵循用户要求：ground truth recommendation 取报告原文 `Continue low-dose lung cancer screening CT in 12 months.`。
- `pack_year_value = 40` 来自 radiology indication 的 weak linkage，而 `quit_years_value = 0` 则来自真实 discharge note 中 `quit smoking on admission` 的语义规范化。

### Example D3: unlabeled inference-only case bundle

1. 来源信息
- `case_id`: `CASE-10001338-001`
- `subject_id`: `10001338`
- 放射来源：`10001338-RR-42`

2. 映射后的 JSON
```json
{
  "case_id": "CASE-10001338-001",
  "subject_id": 10001338,
  "demographics": {"age": null, "sex": null, "race": null, "insurance": null, "source": null, "missing_flags": ["age", "sex", "race", "insurance", "source"]},
  "radiology_facts": [
    {
      "note_id": "10001338-RR-42",
      "subject_id": 10001338,
      "exam_name": "CTA CHEST",
      "modality": "CTA",
      "body_site": "chest",
      "report_text": "EXAMINATION: CTA CHEST\nINDICATION: ___ female with faintness and ischemic lesion in cecum. Question PE.\nTECHNIQUE: Unenhanced axial images were obtained of the chest using a low-dose protocol. Subsequently, images were obtained after the uneventful administration of 100 mL Omnipaque intravenous contrast. Coronal and sagittal and oblique reformatted images were constructed.\nCOMPARISON: ___.\nFINDINGS: Tiny calcified granuloma is present at the right base (2:46). A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).\nIMPRESSION: 2. There are four, less than 4 mm pulmonary nodules, at least one of which is calcified with additional finding of calcified lymph node in the right paratracheal region. These may relate to sequelae of prior granulomatous disease. In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.",
      "sections": {"indication": "___ female with faintness and ischemic lesion in cecum. Question PE.", "technique": "Unenhanced axial images were obtained of the chest using a low-dose protocol. Subsequently, images were obtained after the uneventful administration of 100 mL Omnipaque intravenous contrast. Coronal and sagittal and oblique reformatted images were constructed.", "comparison": "___.", "findings": "Tiny calcified granuloma is present at the right base (2:46). A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "impression": "2. There are four, less than 4 mm pulmonary nodules, at least one of which is calcified with additional finding of calcified lymph node in the right paratracheal region. These may relate to sequelae of prior granulomatous disease. In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary."},
      "nodule_count": 3,
      "nodules": [
        {"nodule_id_in_report": 1, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "RUL", "location_text": "right upper lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]},
        {"nodule_id_in_report": 2, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "lingula", "location_text": "lingula", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]},
        {"nodule_id_in_report": 3, "size_mm": 1.5, "size_text": "1-2 mm", "density_category": "unclear", "density_text": "not definitely calcified", "location_lobe": "RLL", "location_text": "right lower lobe", "count_type": "multiple", "change_status": null, "change_text": null, "calcification": null, "spiculation": false, "lobulation": false, "cavitation": false, "perifissural": false, "lung_rads_category": null, "recommendation_cue": "In a low-risk patient no followup is needed. In a high-risk patient, a followup CT at 12 months may be obtained and if unchanged, no further followup would be necessary.", "evidence_span": "A few additional 1-2 mm pulmonary nodules are not definitely calcified in the right upper lobe (2:21), lingula (2:27), right lower lobe (2:34).", "confidence": "medium", "missing_flags": ["change_status", "change_text", "calcification", "lung_rads_category"]}
      ],
      "extraction_metadata": {"extractor_version": "schema_examples_v1", "extraction_timestamp": "2026-04-03T00:00:00Z", "model_name": "manual_schema_mapping"}
    }
  ],
  "smoking_eligibility": null,
  "recommendation_target": {"ground_truth_action": null, "ground_truth_source": "none", "ground_truth_interval": null, "recommendation_output": null},
  "provenance": {"radiology_note_ids": ["10001338-RR-42"], "discharge_note_id": null, "data_version": "mimic-iv-note-2.2", "extraction_date": "2026-04-03", "pipeline_version": "schema_examples_v1"},
  "split": "unlabeled",
  "label_quality": "unlabeled",
  "case_notes": "仅保留放射事实，未链接可用吸烟记录，也没有报告级 ground truth recommendation。"
}
```

3. 缺失字段
- `smoking_eligibility = null`，因为本例未链接到可用的 discharge note。
- `ground_truth_action` / `ground_truth_interval` / `recommendation_output` 全为 `null`，符合 `unlabeled` 样本定义。

4. 规范化决策
- `split = unlabeled` 与 `label_quality = unlabeled` 必须同时出现，才能满足 `case_bundle_schema` 的条件约束。
- 该 bundle 适合 inference-only 或 semi-supervised 场景：保留 radiology facts，但不假设任何 gold/silver recommendation。

## 使用提示

- 本文中的 `report_text` 采用“真实 section 原文拼接版”，目的是在控制篇幅的同时保留可核对的原始证据。
- 如果你的 extraction pipeline 直接消费整篇 note text，可以保持相同字段结构，只把 `report_text` 替换成完整原始报告全文。
