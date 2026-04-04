from collections import Counter

from src.eval.smoking_metrics import (
    compute_smoking_coverage_summary,
    evaluate_smoking,
    evaluate_smoking_single,
)


def _make_result(
    note_id: str,
    *,
    status: str = "unknown",
    source_section: str | None = "Social History",
    pack_year_value: float | None = None,
    ppd_value: float | None = None,
    years_smoked_value: float | None = None,
    quit_years_value: float | None = None,
    ppd_text: str | None = None,
    ever_smoker_flag: bool | None = None,
    eligible_for_high_risk_screening: str | bool | None = "unknown",
    evidence_quality: str = "none",
    data_quality_notes: str | None = None,
) -> dict:
    return {
        "subject_id": 10000001,
        "note_id": note_id,
        "source_section": source_section,
        "smoking_status_raw": None,
        "smoking_status_norm": status,
        "pack_year_value": pack_year_value,
        "pack_year_text": None,
        "ppd_value": ppd_value,
        "ppd_text": ppd_text,
        "years_smoked_value": years_smoked_value,
        "years_smoked_text": None,
        "quit_years_value": quit_years_value,
        "quit_years_text": None,
        "evidence_span": None,
        "ever_smoker_flag": ever_smoker_flag,
        "eligible_for_high_risk_screening": eligible_for_high_risk_screening,
        "eligibility_criteria_applied": "USPSTF_2021",
        "eligibility_reason": None,
        "evidence_quality": evidence_quality,
        "extraction_metadata": {
            "extractor_version": "v1",
            "extraction_timestamp": "2026-01-01T00:00:00Z",
            "model_name": "unit-test",
        },
        "missing_flags": [],
        "data_quality_notes": data_quality_notes,
    }


def test_evaluate_smoking_empty_input():
    metrics = evaluate_smoking([])
    assert metrics["total_notes"] == 0
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["non_unknown_rate"] == 0.0
    assert metrics["quit_years_parse_rate"] == 0.0
    assert metrics["evidence_quality_distribution"] == Counter()
    assert metrics["status_distribution"] == Counter()


def test_evaluate_smoking_all_unknown_results():
    results = [
        _make_result("100-DS-1", status="unknown", evidence_quality="none"),
        _make_result("100-DS-2", status="unknown", evidence_quality="none"),
    ]
    metrics = evaluate_smoking(results)

    assert metrics["total_notes"] == 2
    assert metrics["unknown_rate"] == 1.0
    assert metrics["non_unknown_rate"] == 0.0
    assert metrics["ever_smoker_rate"] == 0.0
    assert metrics["eligible_rate"] == 0.0
    assert metrics["pack_year_parse_rate"] == 0.0
    assert metrics["quit_years_parse_rate"] == 0.0


def test_evaluate_smoking_mixed_status_and_quantitative_rates():
    results = [
        _make_result(
            "100-DS-3",
            status="current_smoker",
            pack_year_value=30.0,
            ppd_value=1.0,
            years_smoked_value=30.0,
            ever_smoker_flag=True,
            eligible_for_high_risk_screening="eligible",
            evidence_quality="high",
        ),
        _make_result(
            "100-DS-4",
            status="former_smoker",
            pack_year_value=40.0,
            ppd_value=2.0,
            years_smoked_value=20.0,
            quit_years_value=5.0,
            ever_smoker_flag=True,
            eligible_for_high_risk_screening="eligible",
            evidence_quality="high",
        ),
        _make_result(
            "100-DS-5",
            status="never_smoker",
            ever_smoker_flag=False,
            eligible_for_high_risk_screening="not_eligible",
            evidence_quality="medium",
        ),
    ]
    metrics = evaluate_smoking(results)

    assert metrics["non_unknown_rate"] == 1.0
    assert metrics["unknown_rate"] == 0.0
    assert metrics["ever_smoker_rate"] == 2.0 / 3.0
    assert metrics["eligible_rate"] == 2.0 / 3.0
    assert metrics["pack_year_parse_rate"] == 2.0 / 3.0
    assert metrics["ppd_parse_rate"] == 2.0 / 3.0
    assert metrics["years_smoked_parse_rate"] == 2.0 / 3.0
    assert metrics["quit_years_parse_rate"] == 1.0
    assert metrics["status_distribution"] == Counter(
        {"current": 1, "former": 1, "never": 1}
    )


def test_evaluate_smoking_fallback_trigger_detection():
    results = [
        _make_result("100-DS-6", status="current_smoker", source_section="Social History"),
        _make_result("100-DS-7", status="former_smoker", source_section="full_text_fallback"),
        _make_result("100-DS-8", status="unknown", source_section=None),
    ]
    metrics = evaluate_smoking(results)

    assert metrics["fallback_trigger_rate"] == 2.0 / 3.0
    assert metrics["social_history_only_rate"] == 1.0 / 3.0


def test_evaluate_smoking_explicit_subset_matching():
    results = [
        _make_result(
            "100-DS-9",
            status="current_smoker",
            pack_year_value=40.0,
            ppd_value=1.0,
        ),
        _make_result(
            "100-DS-10",
            status="former_smoker",
            pack_year_value=10.0,
            ppd_value=0.5,
        ),
    ]
    manifest = {
        "explicit_status_labels": {
            "100-DS-9": "current",
            "100-DS-10": "never",
        },
        "explicit_quantitative_labels": {
            "100-DS-9": {"pack_year": 39.4, "ppd": 1.05},
            "100-DS-10": {"pack_year": 12.0, "ppd": 0.7},
        },
    }
    metrics = evaluate_smoking(results, manifest=manifest)

    assert metrics["explicit_status_accuracy"] == 0.5
    assert metrics["explicit_pack_year_accuracy"] == 0.5
    assert metrics["explicit_ppd_accuracy"] == 0.5


def test_single_and_coverage_include_ppd_ambiguity_and_deidentified_rate():
    result = _make_result(
        "100-DS-11",
        status="former_smoker",
        source_section="full_text_fallback",
        ppd_value=None,
        ppd_text="PPD 2",
        quit_years_value=None,
        data_quality_notes="Social History is de-identified.",
    )
    single = evaluate_smoking_single(result)
    summary = compute_smoking_coverage_summary([result])
    metrics = evaluate_smoking([result])

    assert single["ppd_ambiguity_protected"] is True
    assert single["fallback_triggered"] is True
    assert summary["fallback_trigger_rate"] == 1.0
    assert summary["quit_years_parse_rate"] == 0.0
    assert metrics["ppd_ambiguity_protected_count"] == 1
    assert metrics["deidentified_rate"] == 1.0
