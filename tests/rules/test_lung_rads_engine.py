from src.pipeline.schema_validator import validate_instance
from src.rules.lung_rads_engine import generate_recommendation


def make_nodule(
    size_mm: float | None,
    density_category: str | None,
    change_status: str | None = None,
    change_text: str | None = None,
    evidence_span: str = "Pulmonary nodule.",
    recommendation_cue: str | None = "Follow-up recommended.",
) -> dict:
    return {
        "nodule_id_in_report": 1,
        "size_mm": size_mm,
        "size_text": None if size_mm is None else f"{size_mm} mm",
        "density_category": density_category,
        "density_text": density_category,
        "location_lobe": "RUL",
        "location_text": "right upper lobe",
        "count_type": "single",
        "change_status": change_status,
        "change_text": change_text,
        "calcification": False,
        "spiculation": False,
        "lobulation": False,
        "cavitation": False,
        "perifissural": False,
        "lung_rads_category": None,
        "recommendation_cue": recommendation_cue,
        "evidence_span": evidence_span,
        "confidence": "high",
        "missing_flags": [],
    }


def make_radiology_fact(subject_id: int, note_id: str, nodules: list[dict]) -> dict:
    return {
        "note_id": note_id,
        "subject_id": subject_id,
        "exam_name": "CT CHEST",
        "modality": "CT",
        "body_site": "chest",
        "report_text": "Synthetic report text.",
        "sections": {
            "indication": "screening",
            "technique": "CT chest",
            "comparison": None,
            "findings": "Synthetic finding",
            "impression": "Synthetic impression",
        },
        "nodule_count": len(nodules),
        "nodules": nodules,
        "extraction_metadata": {
            "extractor_version": "test_extractor_v1",
            "extraction_timestamp": "2026-04-03T00:00:00Z",
            "model_name": "pytest_fixture",
        },
    }


def make_case_bundle(nodules: list[dict], smoking_eligibility: dict | None = None) -> dict:
    subject_id = 10001401
    return {
        "case_id": "CASE-10001401-001",
        "subject_id": subject_id,
        "radiology_facts": [make_radiology_fact(subject_id, f"{subject_id}-RR-1", nodules)],
        "smoking_eligibility": smoking_eligibility,
    }


def make_smoking_eligibility(status: str = "eligible") -> dict:
    return {
        "eligible_for_high_risk_screening": status,
        "note_id": "10001401-DS-1",
    }


def assert_recommendation_valid(recommendation: dict):
    errors = validate_instance(recommendation, "recommendation_schema.json")
    assert errors == []


def test_solid_nodule_lt_6mm_category_2_and_12_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(5.0, "solid")]))
    assert recommendation["lung_rads_category"] == "2"
    assert recommendation["followup_interval"] == "12_months"
    assert_recommendation_valid(recommendation)


def test_solid_nodule_6_to_7mm_category_3_and_6_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(6.0, "solid")]))
    assert recommendation["lung_rads_category"] == "3"
    assert recommendation["followup_interval"] == "6_months"
    assert_recommendation_valid(recommendation)


def test_solid_nodule_8_to_14mm_category_4a_and_3_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(10.0, "solid")]))
    assert recommendation["lung_rads_category"] == "4A"
    assert recommendation["followup_interval"] == "3_months"
    assert_recommendation_valid(recommendation)


def test_solid_nodule_ge_15mm_category_4b_and_immediate():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(16.0, "solid")]))
    assert recommendation["lung_rads_category"] == "4B"
    assert recommendation["followup_interval"] == "immediate"
    assert_recommendation_valid(recommendation)


def test_ground_glass_lt_30mm_category_2_and_12_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(12.0, "ground_glass")]))
    assert recommendation["lung_rads_category"] == "2"
    assert recommendation["followup_interval"] == "12_months"
    assert_recommendation_valid(recommendation)


def test_ground_glass_ge_30mm_category_3_and_6_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(30.0, "ground_glass")]))
    assert recommendation["lung_rads_category"] == "3"
    assert recommendation["followup_interval"] == "6_months"
    assert_recommendation_valid(recommendation)


def test_part_solid_lt_6mm_category_2_and_12_months():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(5.0, "part_solid")]))
    assert recommendation["lung_rads_category"] == "2"
    assert recommendation["followup_interval"] == "12_months"
    assert_recommendation_valid(recommendation)


def test_part_solid_ge_6mm_with_large_solid_component_category_4a():
    recommendation = generate_recommendation(
        make_case_bundle(
            [
                make_nodule(
                    12.0,
                    "part_solid",
                    evidence_span="Part-solid nodule with solid component measuring 7 mm.",
                )
            ]
        )
    )
    assert recommendation["lung_rads_category"] == "4A"
    assert recommendation["followup_interval"] == "3_months"
    assert_recommendation_valid(recommendation)


def test_new_solid_nodule_5mm_category_3():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(5.0, "solid", change_status="new")]))
    assert recommendation["lung_rads_category"] == "3"
    assert recommendation["recommendation_level"] in {"short_interval_followup", "diagnostic_workup"}
    assert_recommendation_valid(recommendation)


def test_stable_nodule_for_two_years_downgrades_to_category_2():
    recommendation = generate_recommendation(
        make_case_bundle([make_nodule(8.0, "solid", change_status="stable", change_text="stable for 2 years")])
    )
    assert recommendation["lung_rads_category"] == "2"
    assert recommendation["followup_interval"] == "12_months"
    assert_recommendation_valid(recommendation)


def test_growing_nodule_upgrades_category():
    recommendation = generate_recommendation(
        make_case_bundle([make_nodule(6.0, "solid", change_status="increased", change_text="interval increase")])
    )
    assert recommendation["lung_rads_category"] == "4A"
    assert "growth_modifier_upgrade" in recommendation["triggered_rules"]
    assert_recommendation_valid(recommendation)


def test_missing_size_returns_insufficient_data():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(None, "solid")]))
    assert recommendation["recommendation_level"] == "insufficient_data"
    assert recommendation["lung_rads_category"] is None
    assert_recommendation_valid(recommendation)


def test_unclear_density_defaults_to_solid_pathway():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(7.0, "unclear")]))
    assert recommendation["lung_rads_category"] == "3"
    assert "density_category" in recommendation["missing_information"]
    assert_recommendation_valid(recommendation)


def test_missing_smoking_info_is_recorded_but_not_blocking():
    recommendation = generate_recommendation(make_case_bundle([make_nodule(5.0, "solid")], smoking_eligibility=None))
    assert recommendation["recommendation_level"] == "routine_screening"
    assert "smoking_eligibility" in recommendation["missing_information"]
    assert_recommendation_valid(recommendation)


def test_multiple_nodules_use_most_concerning_one():
    recommendation = generate_recommendation(
        make_case_bundle(
            [
                make_nodule(4.0, "solid"),
                make_nodule(
                    12.0,
                    "part_solid",
                    evidence_span="Part-solid nodule with solid component measuring 7 mm.",
                ),
            ],
            smoking_eligibility=make_smoking_eligibility(),
        )
    )
    assert recommendation["lung_rads_category"] == "4A"
    assert recommendation["input_facts_used"]["nodule_size_mm"] == 12.0
    assert recommendation["input_facts_used"]["nodule_count"] == 2
    assert_recommendation_valid(recommendation)
