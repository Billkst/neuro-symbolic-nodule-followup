from src.assemblers.case_bundle_assembler import assemble_case_bundles
from src.pipeline.schema_validator import validate_instance


def make_radiology_fact(
    subject_id: int,
    note_id: str,
    size_mm: float | None = 8.0,
    density_category: str | None = "solid",
    confidence: str = "high",
    recommendation_cue: str | None = "Repeat LDCT in 3 months.",
    change_status: str | None = None,
    change_text: str | None = None,
) -> dict:
    missing_flags = []
    if size_mm is None:
        missing_flags.append("size_mm")
    if density_category in {None, "unclear"}:
        missing_flags.append("density_category")
    if recommendation_cue is None:
        missing_flags.append("recommendation_cue")

    return {
        "note_id": note_id,
        "subject_id": subject_id,
        "exam_name": "CT CHEST W/CONTRAST",
        "modality": "CT",
        "body_site": "chest",
        "report_text": "Synthetic radiology report.",
        "sections": {
            "indication": "Follow-up",
            "technique": "CT chest",
            "comparison": None,
            "findings": "Pulmonary nodule described.",
            "impression": recommendation_cue,
        },
        "nodule_count": 1,
        "nodules": [
            {
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
                "evidence_span": "8 mm pulmonary nodule.",
                "confidence": confidence,
                "missing_flags": missing_flags,
            }
        ],
        "extraction_metadata": {
            "extractor_version": "test_extractor_v1",
            "extraction_timestamp": "2026-04-03T00:00:00Z",
            "model_name": "pytest_fixture",
        },
    }


def make_smoking_result(subject_id: int, evidence_quality: str = "high") -> dict:
    return {
        "subject_id": subject_id,
        "note_id": f"{subject_id}-DS-1",
        "source_section": "Social History",
        "smoking_status_raw": "former smoker, 35 pack years",
        "smoking_status_norm": "former_smoker",
        "pack_year_value": 35,
        "pack_year_text": "35 pack years",
        "ppd_value": None,
        "ppd_text": None,
        "years_smoked_value": None,
        "years_smoked_text": None,
        "quit_years_value": 5,
        "quit_years_text": "quit 5 years ago",
        "evidence_span": "former smoker, 35 pack years",
        "ever_smoker_flag": True,
        "eligible_for_high_risk_screening": "eligible",
        "eligibility_criteria_applied": "USPSTF_2021",
        "eligibility_reason": "Meets screening criteria.",
        "evidence_quality": evidence_quality,
        "extraction_metadata": {
            "extractor_version": "test_smoking_v1",
            "extraction_timestamp": "2026-04-03T00:00:00Z",
            "model_name": "pytest_fixture",
        },
        "missing_flags": ["ppd_value", "ppd_text", "years_smoked_value", "years_smoked_text"],
        "data_quality_notes": None,
    }


def make_demographics() -> dict:
    return {
        "age": 67,
        "sex": "F",
        "race": "WHITE",
        "insurance": "Medicare",
        "source": "synthetic_demo_table",
    }


def assert_case_bundle_valid(bundle: dict):
    errors = validate_instance(bundle, "case_bundle_schema.json")
    assert errors == []


def test_full_bundle_with_radiology_smoking_and_demographics():
    subject_id = 10001401
    bundles = assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1")],
        smoking_results={subject_id: make_smoking_result(subject_id)},
        demographics={subject_id: make_demographics()},
    )

    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["subject_id"] == subject_id
    assert bundle["demographics"]["age"] == 67
    assert bundle["smoking_eligibility"]["note_id"] == f"{subject_id}-DS-1"
    assert_case_bundle_valid(bundle)


def test_bundle_with_no_smoking_results():
    subject_id = 10002221
    bundle = assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1")],
        smoking_results=None,
        demographics={subject_id: make_demographics()},
    )[0]

    assert bundle["smoking_eligibility"] is None
    assert bundle["label_quality"] == "weak"
    assert bundle["recommendation_target"]["recommendation_output"] is None
    assert_case_bundle_valid(bundle)


def test_bundle_with_no_demographics_uses_nulls_and_missing_flags():
    subject_id = 10001338
    bundle = assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1")],
        smoking_results={subject_id: make_smoking_result(subject_id, evidence_quality="low")},
        demographics=None,
    )[0]

    assert bundle["demographics"] == {
        "age": None,
        "sex": None,
        "race": None,
        "insurance": None,
        "source": None,
        "missing_flags": ["age", "sex", "race", "insurance", "source"],
    }
    assert_case_bundle_valid(bundle)


def test_bundle_with_multiple_radiology_facts_for_same_subject():
    subject_id = 10009999
    fact_one = make_radiology_fact(subject_id, f"{subject_id}-RR-1", size_mm=4.0)
    fact_two = make_radiology_fact(subject_id, f"{subject_id}-RR-2", size_mm=12.0)
    bundle = assemble_case_bundles(
        radiology_facts=[fact_one, fact_two],
        smoking_results={subject_id: make_smoking_result(subject_id, evidence_quality="low")},
        demographics={subject_id: make_demographics()},
    )[0]

    assert len(bundle["radiology_facts"]) == 2
    assert bundle["provenance"]["radiology_note_ids"] == [f"{subject_id}-RR-1", f"{subject_id}-RR-2"]
    assert_case_bundle_valid(bundle)


def test_label_quality_silver_with_recommendation_output():
    subject_id = 10007777
    bundle = assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1", size_mm=10.0)],
        smoking_results={subject_id: make_smoking_result(subject_id, evidence_quality="high")},
        demographics={subject_id: make_demographics()},
    )[0]

    assert bundle["label_quality"] == "silver"
    assert bundle["recommendation_target"]["recommendation_output"] is not None
    assert_case_bundle_valid(bundle)


def test_label_quality_weak_with_partial_data():
    subject_id = 10006666
    bundle = assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1", confidence="medium")],
        smoking_results={subject_id: make_smoking_result(subject_id, evidence_quality="low")},
        demographics={subject_id: make_demographics()},
    )[0]

    assert bundle["label_quality"] == "weak"
    assert bundle["recommendation_target"]["recommendation_output"] is None
    assert_case_bundle_valid(bundle)


def test_label_quality_unlabeled_for_minimal_extraction():
    subject_id = 10005555
    bundle = assemble_case_bundles(
        radiology_facts=[
            make_radiology_fact(
                subject_id,
                f"{subject_id}-RR-1",
                size_mm=None,
                density_category="unclear",
                confidence="low",
                recommendation_cue=None,
            )
        ],
        smoking_results=None,
        demographics=None,
    )[0]

    assert bundle["label_quality"] == "unlabeled"
    assert bundle["smoking_eligibility"] is None
    assert bundle["recommendation_target"]["ground_truth_action"] is None
    assert bundle["recommendation_target"]["ground_truth_source"] == "none"
    assert bundle["split"] == "unlabeled"
    assert_case_bundle_valid(bundle)


def test_case_id_format_matches_schema_pattern():
    bundles = assemble_case_bundles(
        radiology_facts=[
            make_radiology_fact(10001111, "10001111-RR-1"),
            make_radiology_fact(10002222, "10002222-RR-1"),
        ],
        smoking_results=None,
        demographics=None,
    )

    assert [bundle["case_id"] for bundle in bundles] == [
        "CASE-10001111-001",
        "CASE-10002222-001",
    ]
