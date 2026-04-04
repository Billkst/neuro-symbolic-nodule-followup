import json
from pathlib import Path

import pytest

from src.extractors.smoking_extractor import extract_smoking_eligibility


def _note(social_history: str) -> str:
    return f"Admission Date: 2020-01-01\nSocial History: {social_history}\nFamily History: none"


def test_deidentified_social_history_unknown_absent():
    result = extract_smoking_eligibility(10000001, "10000001-DS-1", _note("___"))
    assert result["smoking_status_norm"] == "unknown"
    assert result["evidence_quality"] == "none"


def test_current_smoker_with_pack_years():
    result = extract_smoking_eligibility(
        10000002,
        "10000002-DS-2",
        _note("Current smoker, 40 pack-year history"),
    )
    assert result["smoking_status_norm"] == "current_smoker"
    assert result["pack_year_value"] == 40.0


def test_former_smoker_quit_years():
    result = extract_smoking_eligibility(
        10000003,
        "10000003-DS-3",
        _note("Former smoker, quit 5 years ago"),
    )
    assert result["smoking_status_norm"] == "former_smoker"
    assert result["quit_years_value"] == 5.0


def test_never_smoker_denies_tobacco():
    result = extract_smoking_eligibility(
        10000004,
        "10000004-DS-4",
        _note("Denies tobacco use"),
    )
    assert result["smoking_status_norm"] == "never_smoker"
    assert result["eligible_for_high_risk_screening"] == "not_eligible"


def test_ppd_with_pack_year_calculation():
    result = extract_smoking_eligibility(
        10000005,
        "10000005-DS-5",
        _note("Smokes 1 ppd x 20 years"),
    )
    assert result["ppd_value"] == 1.0
    assert result["years_smoked_value"] == 20.0
    assert result["pack_year_value"] == 20.0


def test_ppd_disambiguation_postpartum():
    result = extract_smoking_eligibility(
        10000006,
        "10000006-DS-6",
        _note("PPD 2, postpartum day 2"),
    )
    assert result["ppd_value"] is None


def test_ppd_disambiguation_tb_test():
    result = extract_smoking_eligibility(
        10000007,
        "10000007-DS-7",
        _note("PPD placed, read at 0mm"),
    )
    assert result["ppd_value"] is None


def test_ambiguous_tobacco_weak_evidence():
    result = extract_smoking_eligibility(
        10000008,
        "10000008-DS-8",
        _note("Social history significant for tobacco"),
    )
    assert result["smoking_status_norm"] in {"current_smoker", "former_smoker"}
    assert result["evidence_quality"] == "low"


def test_no_social_history_section_returns_unknown_absent():
    text = "Admission Date: 2020-01-01\nHistory of Present Illness: cough\nFamily History: none"
    result = extract_smoking_eligibility(10000009, "10000009-DS-9", text)
    assert result["source_section"] is None
    assert result["smoking_status_norm"] == "unknown"
    assert result["evidence_quality"] == "none"


def test_fulltext_fallback_when_social_history_deidentified():
    text = (
        "Admission Date: 2020-01-01\n"
        "Social History: ___\n"
        "Family History: none\n"
        "History of Present Illness: Patient is a former smoker with 40 pack-year history."
    )
    result = extract_smoking_eligibility(10000012, "10000012-DS-12", text)
    assert result["source_section"] == "full_text_fallback"
    assert result["smoking_status_norm"] == "former_smoker"
    assert result["evidence_quality"] == "low"
    assert "full-text fallback" in (result["data_quality_notes"] or "")


def test_fulltext_fallback_when_social_history_missing():
    text = (
        "Admission Date: 2020-01-01\n"
        "History of Present Illness: Patient quit smoking a couple of years ago.\n"
        "Family History: none"
    )
    result = extract_smoking_eligibility(10000013, "10000013-DS-13", text)
    assert result["source_section"] == "full_text_fallback"
    assert result["smoking_status_norm"] in {"former_smoker", "current_smoker"}
    assert result["quit_years_value"] == 2.0


def test_fulltext_fallback_ppd_disambiguation_preserved():
    text = (
        "Admission Date: 2020-01-01\n"
        "Social History: ___\n"
        "Hospital Course: PPD 2, postpartum day 2. Patient recovering well."
    )
    result = extract_smoking_eligibility(10000014, "10000014-DS-14", text)
    assert result["ppd_value"] is None


def test_fulltext_fallback_no_smoking_cues_stays_unknown():
    text = (
        "Admission Date: 2020-01-01\n"
        "Social History: ___\n"
        "History of Present Illness: Patient presents with chest pain."
    )
    result = extract_smoking_eligibility(10000015, "10000015-DS-15", text)
    assert result["smoking_status_norm"] == "unknown"
    assert result["evidence_quality"] == "none"


def test_social_history_preferred_over_fallback():
    text = (
        "Admission Date: 2020-01-01\n"
        "Social History: Never smoked\n"
        "History of Present Illness: Patient is a former smoker."
    )
    result = extract_smoking_eligibility(10000016, "10000016-DS-16", text)
    assert result["source_section"] == "Social History"
    assert result["smoking_status_norm"] == "never_smoker"


def test_complex_former_smoker_full_quantitative():
    result = extract_smoking_eligibility(
        10000010,
        "10000010-DS-10",
        _note("Former 2ppd smoker x 30 years, quit 10 years ago"),
    )
    assert result["smoking_status_norm"] == "former_smoker"
    assert result["ppd_value"] == 2.0
    assert result["years_smoked_value"] == 30.0
    assert result["pack_year_value"] == 60.0
    assert result["quit_years_value"] == 10.0


def test_output_passes_schema_validation():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path("/home/UserData/ljx/Project_3/neuro-symbolic-nodule-followup/schemas/smoking_eligibility_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    result = extract_smoking_eligibility(
        10000011,
        "10000011-DS-11",
        _note("Former smoker, 40 pack-year history, quit 5 years ago"),
    )
    jsonschema.validate(instance=result, schema=schema)
