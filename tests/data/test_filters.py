import pandas as pd

from src.data.filters import filter_chest_ct, filter_nodule_reports


def _base_radiology_df():
    return pd.DataFrame(
        [
            {"note_id": 1, "subject_id": 100, "text": "4 mm solid nodule in RUL."},
            {"note_id": 2, "subject_id": 101, "text": "No nodules identified."},
            {"note_id": 3, "subject_id": 102, "text": "No pulmonary nodule."},
        ]
    )


def test_filter_chest_ct_matches_ct_chest_with_contrast():
    radiology_df = _base_radiology_df()
    detail_df = pd.DataFrame(
        [
            {"note_id": 1, "field_name": "exam_name", "field_value": "CT CHEST W/CONTRAST"},
            {"note_id": 2, "field_name": "exam_name", "field_value": "CT HEAD"},
        ]
    )
    result = filter_chest_ct(radiology_df, detail_df)
    assert list(result["note_id"]) == [1]
    assert result.iloc[0]["exam_name"] == "CT CHEST W/CONTRAST"


def test_filter_chest_ct_matches_low_dose_lung_screening():
    radiology_df = _base_radiology_df()
    detail_df = pd.DataFrame(
        [
            {"note_id": 2, "field_name": "exam_name", "field_value": "CT LOW DOSE LUNG SCREENING"},
        ]
    )
    result = filter_chest_ct(radiology_df, detail_df)
    assert list(result["note_id"]) == [2]


def test_filter_chest_ct_excludes_ct_head():
    radiology_df = _base_radiology_df()
    detail_df = pd.DataFrame(
        [
            {"note_id": 3, "field_name": "exam_name", "field_value": "CT HEAD"},
        ]
    )
    result = filter_chest_ct(radiology_df, detail_df)
    assert result.empty


def test_filter_nodule_reports_includes_positive_nodule():
    df = pd.DataFrame([{"text": "There is a 4 mm solid nodule in the right upper lobe."}])
    result = filter_nodule_reports(df)
    assert len(result) == 1


def test_filter_nodule_reports_excludes_negated_only_mentions():
    df = pd.DataFrame([{"text": "No nodules identified in either lung."}])
    result = filter_nodule_reports(df)
    assert result.empty
