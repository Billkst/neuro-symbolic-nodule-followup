from src.assemblers.case_bundle_assembler import assemble_case_bundles
from src.eval.bundle_metrics import (
    compute_bundle_completeness_summary,
    evaluate_bundle_single,
    evaluate_bundles,
)
from tests.assemblers.test_case_bundle_assembler import (
    make_demographics,
    make_radiology_fact,
    make_smoking_result,
)


def build_silver_bundle(subject_id: int) -> dict:
    return assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1", size_mm=10.0)],
        smoking_results={subject_id: make_smoking_result(subject_id, evidence_quality="high")},
        demographics={subject_id: make_demographics()},
    )[0]


def build_weak_bundle_no_smoking(subject_id: int) -> dict:
    return assemble_case_bundles(
        radiology_facts=[make_radiology_fact(subject_id, f"{subject_id}-RR-1", confidence="medium")],
        smoking_results=None,
        demographics={subject_id: make_demographics()},
    )[0]


def build_unlabeled_bundle(subject_id: int) -> dict:
    return assemble_case_bundles(
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


def test_evaluate_bundles_empty_input_returns_zero_metrics():
    metrics = evaluate_bundles([])
    summary = compute_bundle_completeness_summary([])

    assert metrics["total_bundles"] == 0
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["bundle_with_radiology_rate"] == 0.0
    assert metrics["label_quality_distribution"] == {}
    assert metrics["ground_truth_source_distribution"] == {}
    assert metrics["avg_nodules_per_bundle"] == 0.0
    assert summary["provenance_complete_rate"] == 0.0


def test_evaluate_bundles_complete_silver_bundle():
    bundle = build_silver_bundle(10009001)
    single = evaluate_bundle_single(bundle)
    metrics = evaluate_bundles([bundle])

    assert bundle["label_quality"] == "silver"
    assert single["schema_valid"] is True
    assert single["has_smoking"] is True
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["bundle_with_recommendation_rate"] == 1.0
    assert metrics["label_quality_distribution"] == {"silver": 1}
    assert metrics["ground_truth_source_distribution"] == {"extracted_from_report": 1}


def test_evaluate_bundles_weak_bundle_no_smoking():
    bundle = build_weak_bundle_no_smoking(10009002)
    single = evaluate_bundle_single(bundle)
    metrics = evaluate_bundles([bundle])

    assert bundle["label_quality"] == "weak"
    assert single["has_smoking"] is False
    assert single["has_recommendation"] is False
    assert metrics["bundle_with_smoking_rate"] == 0.0
    assert metrics["bundle_with_recommendation_rate"] == 0.0
    assert metrics["label_quality_distribution"] == {"weak": 1}


def test_evaluate_bundles_unlabeled_bundle_no_radiology_signal():
    bundle = build_unlabeled_bundle(10009003)
    metrics = evaluate_bundles([bundle])

    assert bundle["label_quality"] == "unlabeled"
    assert metrics["ground_truth_action_present_rate"] == 0.0
    assert metrics["bundle_with_smoking_rate"] == 0.0
    assert metrics["bundle_with_radiology_rate"] == 1.0
    assert metrics["label_quality_distribution"] == {"unlabeled": 1}
    assert metrics["ground_truth_source_distribution"] == {"none": 1}


def test_evaluate_bundles_mixed_quality_distribution_and_completeness_summary():
    bundles = [
        build_silver_bundle(10009011),
        build_weak_bundle_no_smoking(10009012),
        build_unlabeled_bundle(10009013),
    ]

    metrics = evaluate_bundles(bundles)
    summary = compute_bundle_completeness_summary(bundles)

    assert metrics["total_bundles"] == 3
    assert metrics["label_quality_distribution"] == {"silver": 1, "weak": 1, "unlabeled": 1}
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["bundle_with_smoking_rate"] == 1 / 3
    assert metrics["bundle_with_recommendation_rate"] == 1 / 3
    assert metrics["has_discharge_note_rate"] == 1 / 3
    assert metrics["ground_truth_source_distribution"] == {"extracted_from_report": 2, "none": 1}
    assert summary["bundle_with_demographics_rate"] == 2 / 3
    assert metrics["avg_radiology_facts_per_bundle"] == 1.0
