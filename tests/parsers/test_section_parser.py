from src.parsers.section_parser import parse_sections


def test_standard_report_with_all_sections():
    text = """INDICATION: Lung nodule follow-up
TECHNIQUE: CT chest without contrast
COMPARISON: Prior CT 2022
FINDINGS: 4 mm right upper lobe nodule.
IMPRESSION: Stable small pulmonary nodule.
"""
    parsed = parse_sections(text)
    assert parsed["indication"] == "Lung nodule follow-up"
    assert parsed["technique"] == "CT chest without contrast"
    assert parsed["comparison"] == "Prior CT 2022"
    assert parsed["findings"] == "4 mm right upper lobe nodule."
    assert parsed["impression"] == "Stable small pulmonary nodule."


def test_report_missing_comparison():
    text = """INDICATION: Follow-up
TECHNIQUE: CT chest
FINDINGS: Nodule unchanged.
IMPRESSION: No interval change.
"""
    parsed = parse_sections(text)
    assert parsed["comparison"] is None
    assert parsed["findings"] == "Nodule unchanged."


def test_report_missing_indication():
    text = """TECHNIQUE: CT chest
COMPARISON: None
FINDINGS: Tiny nodule.
IMPRESSION: Benign-appearing nodule.
"""
    parsed = parse_sections(text)
    assert parsed["indication"] is None
    assert parsed["impression"] == "Benign-appearing nodule."


def test_recommendation_used_as_impression_when_missing():
    text = """FINDINGS: Small right lower lobe nodule.
RECOMMENDATION(S): Follow-up low-dose CT in 12 months.
"""
    parsed = parse_sections(text)
    assert parsed["findings"] == "Small right lower lobe nodule."
    assert parsed["impression"] == "Follow-up low-dose CT in 12 months."


def test_no_recognizable_sections():
    text = "No formal headers. Tiny pulmonary nodule described in free text."
    parsed = parse_sections(text)
    assert parsed == {
        "indication": None,
        "technique": None,
        "comparison": None,
        "findings": None,
        "impression": None,
    }


def test_colon_style_headers_with_spaces():
    text = """FINDINGS :   5 mm nodule in left upper lobe.
IMPRESSION : Likely benign.
"""
    parsed = parse_sections(text)
    assert parsed["findings"] == "5 mm nodule in left upper lobe."
    assert parsed["impression"] == "Likely benign."


def test_newline_style_headers():
    text = """FINDINGS
There is a stable 4 mm nodule.
IMPRESSION
No suspicious change.
"""
    parsed = parse_sections(text)
    assert parsed["findings"] == "There is a stable 4 mm nodule."
    assert parsed["impression"] == "No suspicious change."


def test_extra_whitespace_trimmed():
    text = """  INDICATION:    screening   
  TECHNIQUE:   low dose CT chest   
  FINDINGS:   mild emphysema, tiny nodule   
  IMPRESSION:   Lung-RADS 2   
"""
    parsed = parse_sections(text)
    assert parsed["indication"] == "screening"
    assert parsed["technique"] == "low dose CT chest"
    assert parsed["findings"] == "mild emphysema, tiny nodule"
    assert parsed["impression"] == "Lung-RADS 2"
