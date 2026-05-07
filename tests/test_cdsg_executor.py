from src.pipeline.schema_validator import validate_instance
from src.rules.cdsg_executor import generate_cdsg_recommendation


def make_nodule(
    size_mm: float | None,
    density_category: str | None,
    change_status: str | None = None,
    evidence_span: str = "Pulmonary nodule.",
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
        "change_text": None,
        "calcification": False,
        "spiculation": False,
        "lobulation": False,
        "cavitation": False,
        "perifissural": False,
        "lung_rads_category": None,
        "recommendation_cue": None,
        "evidence_span": evidence_span,
        "confidence": "high",
        "missing_flags": [],
    }


def make_case_bundle(nodules: list[dict]) -> dict:
    subject_id = 10001401
    return {
        "case_id": "CASE-10001401-001",
        "subject_id": subject_id,
        "radiology_facts": [
            {
                "note_id": f"{subject_id}-RR-1",
                "subject_id": subject_id,
                "exam_name": "CT CHEST",
                "modality": "CT",
                "body_site": "chest",
                "report_text": "Synthetic report text.",
                "sections": {},
                "nodule_count": len(nodules),
                "nodules": nodules,
            }
        ],
        "smoking_eligibility": {"eligible_for_high_risk_screening": "eligible"},
    }


def assert_module3_recommendation_valid(recommendation: dict) -> None:
    errors = validate_instance(recommendation, "module3_recommendation_schema.json")
    assert errors == []


def test_solid_nodule_threshold_path():
    recommendation = generate_cdsg_recommendation(make_case_bundle([make_nodule(10.0, "solid")]))
    assert recommendation["lung_rads_category"] == "4A"
    assert recommendation["followup_interval"] == "3_months"
    assert "SOLID_EXISTING_SIZE" in recommendation["visited_nodes"]
    assert "E_SOLID_8_14" in recommendation["matched_edges"]
    assert recommendation["abstention_reason"] is None
    assert_module3_recommendation_valid(recommendation)


def test_subsolid_and_ground_glass_paths():
    part_solid = generate_cdsg_recommendation(
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
    assert part_solid["lung_rads_category"] == "4A"
    assert "PART_SOLID_EXISTING_SIZE" in part_solid["visited_nodes"]
    assert "E_PART_SOLID_GE6_SOLID_COMPONENT_GE6" in part_solid["matched_edges"]
    assert_module3_recommendation_valid(part_solid)

    ground_glass = generate_cdsg_recommendation(make_case_bundle([make_nodule(30.0, "ground_glass")]))
    assert ground_glass["lung_rads_category"] == "3"
    assert "GROUND_GLASS_EXISTING_SIZE" in ground_glass["visited_nodes"]
    assert "E_GGO_GE30" in ground_glass["matched_edges"]
    assert_module3_recommendation_valid(ground_glass)


def test_missing_size_abstention():
    recommendation = generate_cdsg_recommendation(make_case_bundle([make_nodule(None, "solid")]))
    assert recommendation["recommendation_level"] == "insufficient_data"
    assert recommendation["lung_rads_category"] is None
    assert recommendation["abstention_reason"] == "missing_nodule_size"
    assert "nodule_size" in recommendation["missing_info"]
    assert "A_MISSING_SIZE" in recommendation["visited_nodes"]
    assert_module3_recommendation_valid(recommendation)


def test_missing_density_abstention():
    recommendation = generate_cdsg_recommendation(make_case_bundle([make_nodule(7.0, None)]))
    assert recommendation["recommendation_level"] == "insufficient_data"
    assert recommendation["lung_rads_category"] is None
    assert recommendation["abstention_reason"] == "missing_nodule_density"
    assert "density_category" in recommendation["missing_info"]
    assert "A_MISSING_DENSITY" in recommendation["visited_nodes"]
    assert_module3_recommendation_valid(recommendation)


def test_terminal_recommendation_schema_valid():
    recommendation = generate_cdsg_recommendation(make_case_bundle([make_nodule(5.0, "solid")]))
    assert recommendation["recommendation"]["level"] == "routine_screening"
    assert recommendation["recommendation_level"] == "routine_screening"
    assert recommendation["decision_path"]
    assert recommendation["visited_nodes"]
    assert recommendation["generation_metadata"]["terminal_node_id"] == "T_SOLID_LT6_CAT2"
    assert_module3_recommendation_valid(recommendation)


def test_guideline_anchor_non_empty():
    recommendation = generate_cdsg_recommendation(make_case_bundle([make_nodule(6.0, "solid")]))
    assert recommendation["guideline_anchor"]
    assert any(anchor["graph_element_type"] == "node" for anchor in recommendation["guideline_anchor"])
    assert any(anchor["graph_element_type"] == "edge" for anchor in recommendation["guideline_anchor"])
    assert_module3_recommendation_valid(recommendation)
