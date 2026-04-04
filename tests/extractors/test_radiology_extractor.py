from src.extractors.radiology_extractor import extract_radiology_facts


def test_extract_example_a1_single_4mm_solid_rml():
    report_text = (
        "FINDINGS: LOWER CHEST: There is a 4 mm solid nodule in the right middle lobe (2:2). "
        "IMPRESSION: 4 mm right middle lobe pulmonary nodule without prior study for comparison. "
        "For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended."
    )
    sections = {
        "findings": "LOWER CHEST: There is a 4 mm solid nodule in the right middle lobe (2:2).",
        "impression": "4 mm right middle lobe pulmonary nodule without prior study for comparison. For incidentally detected single solid pulmonary nodule smaller than 6 mm, no CT follow-up is recommended.",
    }

    facts = extract_radiology_facts(
        note_id="10046097-RR-34",
        subject_id=10046097,
        exam_name="CT ABD AND PELVIS WITH CONTRAST",
        report_text=report_text,
        sections=sections,
    )

    assert facts["modality"] == "CT"
    assert facts["body_site"] == "chest_abdomen_pelvis"
    assert facts["nodule_count"] == 1
    nodule = facts["nodules"][0]
    assert nodule["size_mm"] == 4.0
    assert nodule["density_category"] == "solid"
    assert nodule["location_lobe"] == "RML"
    assert nodule["count_type"] == "single"
    assert isinstance(facts["extraction_metadata"]["extraction_timestamp"], str)


def test_extract_example_a2_multiple_part_solid_with_spiculation():
    findings = (
        "There are 2 spiculated irregular part solid nodules with associated bronchiolectasis "
        "the largest in the right upper lobe with the nodule and solid component measuring 10.5 mm. "
        "Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe. "
        "Few small pulmonary nodules measuring 3 mm in diameter seen in the left lower lobe which are indeterminate."
    )
    impression = (
        "Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. "
        "Small indeterminate round 3 mm pulmonary nodule seen in the left lower lobe."
    )
    report_text = f"FINDINGS: {findings} IMPRESSION: {impression}"
    sections = {"findings": findings, "impression": impression}

    facts = extract_radiology_facts(
        note_id="10001401-RR-8",
        subject_id=10001401,
        exam_name="CT CHEST W/CONTRAST",
        report_text=report_text,
        sections=sections,
    )

    assert facts["nodule_count"] == 3
    assert all(n["count_type"] == "multiple" for n in facts["nodules"])
    assert facts["nodules"][0]["density_category"] == "part_solid"
    assert facts["nodules"][0]["spiculation"] is True
    assert facts["nodules"][2]["density_category"] in {"unclear", "solid", "part_solid", "calcified", "ground_glass"}


def test_extract_example_a3_ldct_lung_rads_2():
    findings = (
        "Stable 2 mm calcified nodule in the right upper lobe. "
        "Stable 1 mm solid right middle lobe pulmonary nodule. "
        "Stable 1 mm right middle lobe pulmonary nodule. "
        "Stable 1 mm solid left lower lobe pulmonary nodule."
    )
    impression = "Stable tiny pulmonary nodules ranging in size from 1-2 mm. Lung-RADS category: 2"
    report_text = (
        f"FINDINGS: {findings} "
        f"IMPRESSION: {impression} "
        "RECOMMENDATION(S): Continue low-dose lung cancer screening CT in 12 months."
    )
    sections = {"findings": findings, "impression": impression}

    facts = extract_radiology_facts(
        note_id="10002221-RR-133",
        subject_id=10002221,
        exam_name="CT LOW DOSE LUNG SCREENING",
        report_text=report_text,
        sections=sections,
    )

    assert facts["modality"] == "LDCT"
    assert facts["body_site"] == "chest"
    assert facts["nodule_count"] == 4
    assert all(n["lung_rads_category"] == "2" for n in facts["nodules"])
    assert all(n["change_status"] == "stable" for n in facts["nodules"])
    assert facts["nodules"][0]["recommendation_cue"] is not None


def test_extract_report_with_no_nodules():
    findings = "No focal consolidation, pleural effusion, or pneumothorax."
    impression = "No acute cardiopulmonary abnormality."
    report_text = f"FINDINGS: {findings} IMPRESSION: {impression}"
    sections = {"findings": findings, "impression": impression}

    facts = extract_radiology_facts(
        note_id="12345678-RR-1",
        subject_id=12345678,
        exam_name="CT CHEST",
        report_text=report_text,
        sections=sections,
    )

    assert facts["nodule_count"] == 0
    assert facts["nodules"] == []
    assert facts["extraction_metadata"]["model_name"] == "regex_baseline"


def test_extract_cm_measurement_converted_to_mm():
    findings = "There is a 1.2 cm part-solid nodule in the left upper lobe."
    impression = "Recommend follow-up CT in 3 months."
    report_text = f"FINDINGS: {findings} IMPRESSION: {impression}"
    sections = {"findings": findings, "impression": impression}

    facts = extract_radiology_facts(
        note_id="87654321-RR-22",
        subject_id=87654321,
        exam_name="CT CHEST",
        report_text=report_text,
        sections=sections,
    )

    assert facts["nodule_count"] == 1
    assert facts["nodules"][0]["size_mm"] == 12.0
    assert facts["nodules"][0]["density_category"] == "part_solid"
    assert facts["nodules"][0]["location_lobe"] == "LUL"
    assert facts["nodules"][0]["confidence"] in {"high", "medium", "low"}
