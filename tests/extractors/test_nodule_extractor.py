from src.extractors.nodule_extractor import (
    extract_change_status,
    extract_density,
    extract_location,
    extract_lung_rads,
    extract_morphology,
    extract_recommendation_cue,
    extract_size,
    segment_nodule_mentions,
)


def test_extract_size_mm():
    size, text = extract_size("4 mm nodule")
    assert size == 4.0
    assert text == "4 mm"


def test_extract_size_cm_to_mm():
    size, text = extract_size("1.2 cm nodule")
    assert size == 12.0
    assert text == "1.2 cm"


def test_extract_size_2d_mm_takes_max():
    size, text = extract_size("nodule measures 4 x 3 mm")
    assert size == 4.0
    assert text == "4 x 3 mm"


def test_extract_size_2d_cm_takes_max_and_convert():
    size, text = extract_size("lesion is 1.2 x 0.8 cm")
    assert size == 12.0
    assert text == "1.2 x 0.8 cm"


def test_extract_size_with_hyphen_mm():
    size, text = extract_size("a 4-mm nodule")
    assert size == 4.0
    assert text == "4-mm"


def test_extract_size_range_mm():
    size, text = extract_size("tiny nodules ranging 1-2 mm")
    assert size == 1.5
    assert text == "1-2 mm"


def test_extract_size_not_found():
    size, text = extract_size("pulmonary nodule without explicit size")
    assert size is None
    assert text is None


def test_extract_density_solid():
    category, text = extract_density("solid nodule")
    assert category == "solid"
    assert text is not None
    assert text.lower() == "solid"


def test_extract_density_part_solid():
    category, text = extract_density("part-solid nodule")
    assert category == "part_solid"
    assert text is not None
    assert text.lower() == "part-solid"


def test_extract_density_ground_glass():
    category, text = extract_density("ground-glass opacity")
    assert category == "ground_glass"
    assert text is not None
    assert text.lower() == "ground-glass"


def test_extract_density_calcified():
    category, text = extract_density("calcified granuloma")
    assert category == "calcified"
    assert text is not None
    assert text.lower() == "calcified"


def test_extract_density_unclear():
    category, text = extract_density("indeterminate pulmonary nodule")
    assert category == "unclear"
    assert text is None


def test_extract_location_rul():
    location, text = extract_location("nodule in right upper lobe")
    assert location == "RUL"
    assert text is not None
    assert text.lower() == "right upper lobe"


def test_extract_location_lll():
    location, text = extract_location("left lower lobe pulmonary nodule")
    assert location == "LLL"
    assert text is not None
    assert text.lower() == "left lower lobe"


def test_extract_location_lingula():
    location, text = extract_location("nodule in the lingula")
    assert location == "lingula"
    assert text is not None
    assert text.lower() == "lingula"


def test_extract_location_unclear_side_only():
    location, text = extract_location("small nodule in the right lung")
    assert location == "unclear"
    assert text is not None
    assert text.lower() == "right lung"


def test_extract_change_stable():
    status, text = extract_change_status("stable nodule")
    assert status == "stable"
    assert text is not None
    assert text.lower() == "stable"


def test_extract_change_new():
    status, text = extract_change_status("new 4 mm nodule")
    assert status == "new"
    assert text is not None
    assert text.lower() == "new"


def test_extract_change_increased():
    status, text = extract_change_status("nodule increased in size")
    assert status == "increased"
    assert text is not None
    assert text.lower() == "increased"


def test_extract_morphology_spiculation():
    morphology = extract_morphology("spiculated mass")
    assert morphology["spiculation"] is True
    assert morphology["calcification"] is False


def test_extract_morphology_calcification():
    morphology = extract_morphology("calcified nodule")
    assert morphology["calcification"] is True


def test_extract_recommendation_cue():
    text = "There is a 4 mm nodule. Follow-up CT in 12 months is recommended."
    cue = extract_recommendation_cue(text)
    assert cue is not None
    assert "follow-up" in cue.lower() or "recommended" in cue.lower()


def test_extract_lung_rads():
    category = extract_lung_rads("Lung-RADS category: 2")
    assert category == "2"


def test_segment_nodule_mentions_basic():
    text = (
        "There is a 4 mm solid nodule in the right middle lobe. "
        "No pleural effusion. "
        "A new 3 mm nodule is seen in the left lower lobe."
    )
    mentions = segment_nodule_mentions(text)
    assert len(mentions) == 2
    assert "4 mm" in mentions[0]
    assert "3 mm" in mentions[1]
