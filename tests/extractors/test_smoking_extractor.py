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
