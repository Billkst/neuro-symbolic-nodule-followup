from src.extractors.modality_classifier import classify_body_site, classify_modality


def test_ct_chest_with_contrast():
    exam_name = "CT CHEST W/CONTRAST"
    assert classify_modality(exam_name) == "CT"
    assert classify_body_site(exam_name) == "chest"


def test_ldct_lung_screening():
    exam_name = "CT LOW DOSE LUNG SCREENING"
    assert classify_modality(exam_name) == "LDCT"
    assert classify_body_site(exam_name) == "chest"


def test_cta_chest():
    exam_name = "CTA CHEST"
    assert classify_modality(exam_name) == "CTA"
    assert classify_body_site(exam_name) == "chest"


def test_ct_abd_and_pelvis_with_contrast():
    exam_name = "CT ABD AND PELVIS WITH CONTRAST"
    assert classify_modality(exam_name) == "CT"
    assert classify_body_site(exam_name) == "chest_abdomen_pelvis"


def test_chest_xray_pa_lat():
    exam_name = "CHEST (PA AND LAT)"
    assert classify_modality(exam_name) == "X-ray"
    assert classify_body_site(exam_name) == "chest"


def test_unknown_exam():
    exam_name = "WHOLE BODY NUCLEAR MEDICINE"
    assert classify_modality(exam_name) == "other"
    assert classify_body_site(exam_name) == "other"
