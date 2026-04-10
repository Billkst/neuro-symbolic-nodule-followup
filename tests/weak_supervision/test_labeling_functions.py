import pytest

from src.weak_supervision.base import ABSTAIN, MentionRecord
from src.weak_supervision.labeling_functions.density_lfs import (
    DENSITY_LFS,
    lf_density_keyword_exact,
    lf_density_keyword_fuzzy,
    lf_density_negation_aware,
    lf_density_multi_density,
    lf_density_impression_cue,
)
from src.weak_supervision.labeling_functions.size_lfs import (
    SIZE_LFS,
    lf_size_regex_standard,
    lf_size_regex_tolerant,
    lf_size_numeric_context,
    lf_size_subcentimeter_cue,
    lf_size_no_size_negative,
)
from src.weak_supervision.labeling_functions.location_lfs import (
    LOCATION_LFS,
    lf_location_lobe_exact,
    lf_location_multi_lobe,
    lf_location_bilateral_keyword,
    lf_location_laterality_inference,
    lf_location_context_window,
)


def _make_record(**kwargs) -> MentionRecord:
    base = {
        "sample_id": "test__1",
        "note_id": "test",
        "subject_id": 1,
        "exam_name": "CT CHEST",
        "section": "findings",
        "mention_text": "",
        "full_text": "",
        "density_label": "unclear",
        "size_label": None,
        "size_text": None,
        "has_size": False,
        "location_label": None,
        "has_location": False,
        "label_quality": "weak",
    }
    base.update(kwargs)
    return base


class TestDensityLFs:
    def test_d1_solid(self):
        r = _make_record(mention_text="A solid nodule in the lung.")
        out = lf_density_keyword_exact(r)
        assert out.label == "solid"
        assert out.lf_name == "LF-D1"

    def test_d1_part_solid_priority(self):
        r = _make_record(mention_text="A part-solid nodule with solid component.")
        out = lf_density_keyword_exact(r)
        assert out.label == "part_solid"

    def test_d1_ground_glass(self):
        r = _make_record(mention_text="GGO in the right upper lobe.")
        out = lf_density_keyword_exact(r)
        assert out.label == "ground_glass"

    def test_d1_calcified(self):
        r = _make_record(mention_text="Calcified granuloma noted.")
        out = lf_density_keyword_exact(r)
        assert out.label == "calcified"

    def test_d1_abstain_no_density(self):
        r = _make_record(mention_text="A nodule in the right upper lobe.")
        out = lf_density_keyword_exact(r)
        assert out.is_abstain

    def test_d2_subsolid(self):
        r = _make_record(mention_text="A subsolid opacity was noted.")
        out = lf_density_keyword_fuzzy(r)
        assert out.label == "part_solid"

    def test_d2_hazy(self):
        r = _make_record(mention_text="Hazy opacity in the left lung.")
        out = lf_density_keyword_fuzzy(r)
        assert out.label == "ground_glass"

    def test_d2_abstain_standard(self):
        r = _make_record(mention_text="A solid nodule.")
        out = lf_density_keyword_fuzzy(r)
        assert out.is_abstain

    def test_d3_negation_fires(self):
        r = _make_record(mention_text="No evidence of calcification.")
        out = lf_density_negation_aware(r)
        assert out.label == "unclear"
        assert out.lf_name == "LF-D3"

    def test_d3_no_negation_abstains(self):
        r = _make_record(mention_text="A solid nodule.")
        out = lf_density_negation_aware(r)
        assert out.is_abstain

    def test_d4_multi_density(self):
        r = _make_record(mention_text="Mixed solid and ground-glass nodule.")
        out = lf_density_multi_density(r)
        assert out.label == "unclear"

    def test_d4_single_density_abstains(self):
        r = _make_record(mention_text="A solid nodule.")
        out = lf_density_multi_density(r)
        assert out.is_abstain

    def test_d5_impression_cue(self):
        r = _make_record(
            section="findings",
            mention_text="A nodule in the right upper lobe.",
            full_text="FINDINGS:\nA nodule in the right upper lobe.\n\nIMPRESSION:\nGround-glass nodule in the right upper lobe.",
        )
        out = lf_density_impression_cue(r)
        assert out.label == "ground_glass"

    def test_d5_impression_section_abstains(self):
        r = _make_record(
            section="impression",
            mention_text="Ground-glass nodule.",
            full_text="IMPRESSION:\nGround-glass nodule.",
        )
        out = lf_density_impression_cue(r)
        assert out.is_abstain

    def test_d5_no_impression_abstains(self):
        r = _make_record(
            section="findings",
            mention_text="A nodule.",
            full_text="FINDINGS:\nA nodule.",
        )
        out = lf_density_impression_cue(r)
        assert out.is_abstain

    def test_density_lfs_count(self):
        assert len(DENSITY_LFS) == 5


class TestSizeLFs:
    def test_s1_standard_mm(self):
        r = _make_record(mention_text="Nodule measuring 6 mm in diameter.")
        out = lf_size_regex_standard(r)
        assert out.label == "true"
        assert "6 mm" in out.evidence_span

    def test_s1_range(self):
        r = _make_record(mention_text="Nodule 10-15 mm.")
        out = lf_size_regex_standard(r)
        assert out.label == "true"

    def test_s1_2d(self):
        r = _make_record(mention_text="Mass 10 x 15 mm.")
        out = lf_size_regex_standard(r)
        assert out.label == "true"

    def test_s1_cm(self):
        r = _make_record(mention_text="Mass 2.5 cm.")
        out = lf_size_regex_standard(r)
        assert out.label == "true"

    def test_s1_abstain(self):
        r = _make_record(mention_text="A nodule in the right upper lobe.")
        out = lf_size_regex_standard(r)
        assert out.is_abstain

    def test_s2_missing_space(self):
        r = _make_record(mention_text="Nodule 7mm in the lung.")
        out = lf_size_regex_tolerant(r)
        assert out.label == "true"

    def test_s3_numeric_context(self):
        r = _make_record(mention_text="Nodule measuring approximately 6 in diameter.")
        out = lf_size_numeric_context(r)
        assert out.label == "true"

    def test_s4_subcentimeter(self):
        r = _make_record(mention_text="Subcentimeter nodule in the lung.")
        out = lf_size_subcentimeter_cue(r)
        assert out.label == "true"

    def test_s4_tiny(self):
        r = _make_record(mention_text="Tiny nodule noted.")
        out = lf_size_subcentimeter_cue(r)
        assert out.label == "true"

    def test_s5_no_size(self):
        r = _make_record(mention_text="A nodule in the right upper lobe.")
        out = lf_size_no_size_negative(r)
        assert out.label == "false"

    def test_s5_has_digits_abstains(self):
        r = _make_record(mention_text="Nodule seen on image 8.")
        out = lf_size_no_size_negative(r)
        assert out.is_abstain

    def test_size_lfs_count(self):
        assert len(SIZE_LFS) == 5


class TestLocationLFs:
    def test_l1_rul(self):
        r = _make_record(mention_text="Nodule in the right upper lobe.")
        out = lf_location_lobe_exact(r)
        assert out.label == "RUL"

    def test_l1_bilateral(self):
        r = _make_record(mention_text="Bilateral pulmonary nodules.")
        out = lf_location_lobe_exact(r)
        assert out.label == "bilateral"

    def test_l1_lingula(self):
        r = _make_record(mention_text="Nodule in the lingula.")
        out = lf_location_lobe_exact(r)
        assert out.label == "lingula"

    def test_l1_abstain(self):
        r = _make_record(mention_text="A nodule was noted.")
        out = lf_location_lobe_exact(r)
        assert out.is_abstain

    def test_l2_multi_lobe_bilateral(self):
        r = _make_record(mention_text="Nodules in the right upper lobe and left lower lobe.")
        out = lf_location_multi_lobe(r)
        assert out.label == "bilateral"

    def test_l2_single_lobe_abstains(self):
        r = _make_record(mention_text="Nodule in the right upper lobe.")
        out = lf_location_multi_lobe(r)
        assert out.is_abstain

    def test_l3_both_lungs(self):
        r = _make_record(mention_text="Scattered nodules in both lungs.")
        out = lf_location_bilateral_keyword(r)
        assert out.label == "bilateral"

    def test_l3_diffuse(self):
        r = _make_record(mention_text="Diffuse pulmonary nodules.")
        out = lf_location_bilateral_keyword(r)
        assert out.label == "bilateral"

    def test_l4_right_lung(self):
        r = _make_record(mention_text="Opacity in the right lung.")
        out = lf_location_laterality_inference(r)
        assert out.label == "unclear"

    def test_l4_specific_lobe_abstains(self):
        r = _make_record(mention_text="Nodule in the right upper lobe.")
        out = lf_location_laterality_inference(r)
        assert out.is_abstain

    def test_l5_context_window(self):
        r = _make_record(
            mention_text="A small nodule was noted.",
            full_text="Right upper lobe findings: A small nodule was noted. Follow-up recommended.",
        )
        out = lf_location_context_window(r)
        assert out.label == "RUL"

    def test_l5_mention_has_lobe_abstains(self):
        r = _make_record(
            mention_text="Nodule in the right upper lobe.",
            full_text="FINDINGS:\nNodule in the right upper lobe.",
        )
        out = lf_location_context_window(r)
        assert out.is_abstain

    def test_location_lfs_count(self):
        assert len(LOCATION_LFS) == 5


class TestEdgeCases:
    def test_empty_mention(self):
        r = _make_record(mention_text="")
        for task, lfs in [("density", DENSITY_LFS), ("size", SIZE_LFS), ("location", LOCATION_LFS)]:
            for lf in lfs:
                out = lf(r)
                assert isinstance(out.label, str), f"{lf.__name__} failed on empty mention"

    def test_none_mention(self):
        r = _make_record(mention_text=None)
        for task, lfs in [("density", DENSITY_LFS), ("size", SIZE_LFS), ("location", LOCATION_LFS)]:
            for lf in lfs:
                out = lf(r)
                assert isinstance(out.label, str), f"{lf.__name__} failed on None mention"

    def test_negation_with_density(self):
        r = _make_record(mention_text="Not calcified nodule in the right upper lobe.")
        d3 = lf_density_negation_aware(r)
        assert d3.label == "unclear"

    def test_multi_density_part_solid_not_double_counted(self):
        r = _make_record(mention_text="A part-solid nodule was identified.")
        d4 = lf_density_multi_density(r)
        assert d4.is_abstain
