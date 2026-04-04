from src.eval.recommendation_metrics import (
    compute_recommendation_quality_summary,
    evaluate_recommendation_single,
    evaluate_recommendations,
)


def make_recommendation(
    case_id: str,
    recommendation_level: str = "insufficient_data",
    recommendation_action: str = "Insufficient structured facts.",
    followup_interval: str | None = None,
    followup_modality: str | None = None,
    lung_rads_category: str | None = None,
    guideline_anchor: str | None = "Rule anchor",
    reasoning_path: list[str] | None = None,
    triggered_rules: list[str] | None = None,
    missing_information: list[str] | None = None,
    density: str | None = "solid",
) -> dict:
    return {
        "case_id": case_id,
        "recommendation_level": recommendation_level,
        "recommendation_action": recommendation_action,
        "followup_interval": followup_interval,
        "followup_modality": followup_modality,
        "lung_rads_category": lung_rads_category,
        "guideline_source": "Lung-RADS_v2022",
        "guideline_anchor": guideline_anchor,
        "reasoning_path": reasoning_path if reasoning_path is not None else ["step_1"],
        "triggered_rules": triggered_rules if triggered_rules is not None else ["rule_1"],
        "input_facts_used": {
            "nodule_size_mm": 8.0,
            "nodule_density": density,
            "nodule_count": 1,
            "change_status": "stable",
            "patient_risk_level": "high_risk",
            "smoking_eligible": "eligible",
        },
        "missing_information": missing_information if missing_information is not None else [],
        "uncertainty_note": None,
        "output_type": "rule_based",
        "generation_metadata": {
            "engine_version": "test_engine_v1",
            "generation_timestamp": "2026-04-04T00:00:00Z",
            "rules_version": "test_rules_v1",
        },
    }


def test_empty_input_returns_zero_rates_and_empty_distributions():
    metrics = evaluate_recommendations([])

    assert metrics["total_recommendations"] == 0
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["actionable_rate"] == 0.0
    assert metrics["monitoring_rate"] == 0.0
    assert metrics["insufficient_data_rate"] == 0.0
    assert metrics["missing_field_distribution"] == {}
    assert metrics["recommendation_by_density"] == {}


def test_single_actionable_recommendation_metrics():
    rec = make_recommendation(
        case_id="case-actionable",
        recommendation_level="actionable",
        recommendation_action="LDCT follow-up in 6 months",
        followup_interval="6_months",
        followup_modality="LDCT",
        lung_rads_category="3",
    )
    metrics = evaluate_recommendations([rec])

    assert metrics["total_recommendations"] == 1
    assert metrics["actionable_rate"] == 1.0
    assert metrics["monitoring_rate"] == 0.0
    assert metrics["insufficient_data_rate"] == 0.0
    assert metrics["guideline_anchor_presence_rate"] == 1.0
    assert metrics["reasoning_path_nonempty_rate"] == 1.0
    assert metrics["triggered_rules_nonempty_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 0.0


def test_single_insufficient_data_recommendation_metrics():
    rec = make_recommendation(
        case_id="case-insufficient",
        recommendation_level="insufficient_data",
        recommendation_action="Insufficient structured facts.",
        followup_interval=None,
        followup_modality=None,
        lung_rads_category=None,
        guideline_anchor=None,
        missing_information=["nodule_size_mm", "nodule_density"],
        density=None,
    )
    metrics = evaluate_recommendations([rec])

    assert metrics["insufficient_data_rate"] == 1.0
    assert metrics["avg_missing_fields"] == 2.0
    assert metrics["followup_interval_distribution"][None] == 1
    assert metrics["followup_modality_distribution"][None] == 1


def test_multiple_recommendations_distribution_and_density_grouping():
    recs = [
        make_recommendation(
            case_id="case-1",
            recommendation_level="actionable",
            missing_information=["smoking_eligible"],
            density="solid",
        ),
        make_recommendation(
            case_id="case-2",
            recommendation_level="monitoring",
            missing_information=[],
            density="ground_glass",
        ),
        make_recommendation(
            case_id="case-3",
            recommendation_level="insufficient_data",
            missing_information=["nodule_size_mm", "change_status"],
            density="solid",
            guideline_anchor=None,
            reasoning_path=[],
            triggered_rules=[],
        ),
    ]
    metrics = evaluate_recommendations(recs)

    assert metrics["actionable_rate"] == 1 / 3
    assert metrics["monitoring_rate"] == 1 / 3
    assert metrics["insufficient_data_rate"] == 1 / 3
    assert metrics["recommendation_level_distribution"] == {
        "actionable": 1,
        "monitoring": 1,
        "insufficient_data": 1,
    }
    assert metrics["missing_field_distribution"] == {
        "smoking_eligible": 1,
        "nodule_size_mm": 1,
        "change_status": 1,
    }
    assert metrics["recommendation_by_density"] == {
        "solid": {"actionable": 1, "insufficient_data": 1},
        "ground_glass": {"monitoring": 1},
    }


def test_explicit_and_rule_subset_agreement_rates():
    recs = [
        make_recommendation(
            case_id="case-a",
            recommendation_level="monitoring",
            recommendation_action="Recommend LDCT followup in 6 months.",
            lung_rads_category="3",
        ),
        make_recommendation(
            case_id="case-b",
            recommendation_level="monitoring",
            recommendation_action="Immediate biopsy is recommended.",
            lung_rads_category="4A",
        ),
    ]
    manifest = {
        "explicit_cue_labels": {
            "case-a": "LDCT follow-up in 6 months",
            "case-b": "PET-CT in 3 months",
        },
        "rule_derived_labels": {
            "case-a": "3",
            "case-b": "4B",
        },
    }

    metrics = evaluate_recommendations(recs, manifest=manifest)

    assert metrics["explicit_cue_agreement_rate"] == 0.5
    assert metrics["rule_agreement_rate"] == 0.5


def test_single_and_quality_summary_function_outputs():
    rec = make_recommendation(
        case_id="case-single",
        recommendation_level="insufficient_data",
        guideline_anchor=None,
        reasoning_path=["only_step"],
        triggered_rules=[],
    )

    single = evaluate_recommendation_single(rec)
    assert single["is_insufficient_data"] is True
    assert single["has_guideline_anchor"] is False
    assert single["has_triggered_rules"] is False

    summary = compute_recommendation_quality_summary([rec])
    assert summary["guideline_anchor_presence_rate"] == 0.0
    assert summary["reasoning_path_nonempty_rate"] == 1.0
    assert summary["triggered_rules_nonempty_rate"] == 0.0
