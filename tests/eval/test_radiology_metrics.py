from src.eval.radiology_metrics import (
    compute_field_extraction_summary,
    evaluate_radiology,
    evaluate_radiology_single,
)


def _build_nodule(
    *,
    size_mm: float | None = 4.0,
    density_category: str | None = "solid",
    location_lobe: str | None = "RUL",
    change_status: str | None = "stable",
    recommendation_cue: str | None = "follow-up ct in 12 months",
    confidence: str = "high",
    missing_flags: list[str] | None = None,
) -> dict:
    return {
        "nodule_id_in_report": 1,
        "size_mm": size_mm,
        "size_text": "4 mm" if size_mm is not None else None,
        "density_category": density_category,
        "density_text": "solid" if density_category not in (None, "unclear") else None,
        "location_lobe": location_lobe,
        "location_text": "right upper lobe" if location_lobe not in (None, "unclear") else None,
        "count_type": "single",
        "change_status": change_status,
        "change_text": "stable" if change_status not in (None, "unclear") else None,
        "calcification": False,
        "spiculation": False,
        "lobulation": False,
        "cavitation": False,
        "perifissural": False,
        "lung_rads_category": "2",
        "recommendation_cue": recommendation_cue,
        "evidence_span": "4 mm solid nodule in right upper lobe",
        "confidence": confidence,
        "missing_flags": missing_flags or [],
    }


def _build_fact(note_id: str, nodules: list[dict], nodule_count: int | None = None) -> dict:
    return {
        "note_id": note_id,
        "subject_id": 10000001,
        "exam_name": "CT CHEST",
        "modality": "CT",
        "body_site": "chest",
        "report_text": "FINDINGS: pulmonary nodule",
        "sections": {
            "indication": "screening",
            "technique": "ct chest without contrast",
            "comparison": "none",
            "findings": "4 mm solid nodule",
            "impression": "follow-up",
        },
        "nodule_count": len(nodules) if nodule_count is None else nodule_count,
        "nodules": nodules,
        "extraction_metadata": {
            "extractor_version": "regex_baseline_0.1",
            "extraction_timestamp": "2026-01-01T00:00:00Z",
            "model_name": "regex_baseline",
        },
    }


def test_evaluate_radiology_empty_input():
    metrics = evaluate_radiology([])

    assert metrics["total_notes"] == 0
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["note_level_success_rate"] == 0.0
    assert metrics["nodule_detect_rate"] == 0.0
    assert metrics["avg_nodules_per_note"] == 0.0
    assert metrics["total_nodules"] == 0
    assert metrics["size_mm_extract_rate"] == 0.0
    assert metrics["explicit_size_exact_rate"] == 0.0
    assert len(metrics["confidence_distribution"]) == 0


def test_single_note_with_complete_nodule():
    fact = _build_fact("10000001-RR-1", [_build_nodule()])
    metrics = evaluate_radiology([fact])
    single = evaluate_radiology_single(fact)

    assert metrics["total_notes"] == 1
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["note_level_success_rate"] == 1.0
    assert metrics["nodule_detect_rate"] == 1.0
    assert metrics["avg_nodules_per_note"] == 1.0
    assert metrics["total_nodules"] == 1
    assert metrics["size_mm_extract_rate"] == 1.0
    assert metrics["density_category_extract_rate"] == 1.0
    assert metrics["location_lobe_extract_rate"] == 1.0
    assert metrics["change_status_extract_rate"] == 1.0
    assert metrics["recommendation_cue_extract_rate"] == 1.0
    assert metrics["confidence_distribution"]["high"] == 1
    assert single["schema_valid"] == 1.0
    assert single["total_nodules"] == 1


def test_single_note_with_missing_fields():
    nodule = _build_nodule(
        size_mm=None,
        density_category="unclear",
        location_lobe=None,
        change_status="unclear",
        recommendation_cue=None,
        confidence="low",
        missing_flags=["size_mm", "density_category", "location_lobe", "change_status", "recommendation_cue"],
    )
    fact = _build_fact("10000001-RR-2", [nodule])
    metrics = evaluate_radiology([fact])
    summary = compute_field_extraction_summary([fact])

    assert metrics["size_mm_extract_rate"] == 0.0
    assert metrics["density_category_extract_rate"] == 0.0
    assert metrics["location_lobe_extract_rate"] == 0.0
    assert metrics["change_status_extract_rate"] == 0.0
    assert metrics["recommendation_cue_extract_rate"] == 0.0
    assert metrics["confidence_distribution"]["low"] == 1
    assert metrics["missing_field_distribution"]["size_mm"] == 1
    assert summary["size_mm_extract_rate"] == 0.0


def test_multiple_notes_mixed_counts_and_distributions():
    fact_a = _build_fact("10000001-RR-3", [_build_nodule(size_mm=5.0, location_lobe="RML", confidence="high")])
    fact_b = _build_fact("10000001-RR-4", [], nodule_count=0)
    fact_c = _build_fact(
        "10000001-RR-5",
        [
            _build_nodule(
                size_mm=None,
                density_category="unclear",
                location_lobe="LLL",
                change_status="new",
                recommendation_cue=None,
                confidence="medium",
                missing_flags=["size_mm", "recommendation_cue"],
            )
        ],
    )

    metrics = evaluate_radiology([fact_a, fact_b, fact_c])

    assert metrics["total_notes"] == 3
    assert metrics["total_nodules"] == 2
    assert metrics["note_level_success_rate"] == 2 / 3
    assert metrics["nodule_detect_rate"] == 2 / 3
    assert metrics["avg_nodules_per_note"] == 2 / 3
    assert metrics["size_mm_extract_rate"] == 0.5
    assert metrics["density_distribution"]["solid"] == 1
    assert metrics["density_distribution"]["unclear"] == 1
    assert metrics["location_distribution"]["RML"] == 1
    assert metrics["location_distribution"]["LLL"] == 1
    assert metrics["change_status_distribution"]["stable"] == 1
    assert metrics["change_status_distribution"]["new"] == 1


def test_explicit_subset_matching_rates():
    fact_a = _build_fact("10000001-RR-6", [_build_nodule(size_mm=4.2, density_category="solid", location_lobe="RUL", change_status="stable")])
    fact_b = _build_fact("10000001-RR-7", [_build_nodule(size_mm=8.0, density_category="ground_glass", location_lobe="LUL", change_status="new")])

    manifest = {
        "explicit_labels": {
            "10000001-RR-6": {
                "size_mm": 4.6,
                "density_category": "solid",
                "location_lobe": "RUL",
                "change_status": "stable",
            },
            "10000001-RR-7": {
                "size_mm": 7.0,
                "density_category": "solid",
                "location_lobe": "LLL",
                "change_status": "new",
            },
        }
    }

    metrics = evaluate_radiology([fact_a, fact_b], manifest=manifest)

    assert metrics["explicit_size_exact_rate"] == 0.5
    assert metrics["explicit_density_exact_rate"] == 0.5
    assert metrics["explicit_location_exact_rate"] == 0.5
    assert metrics["explicit_change_exact_rate"] == 1.0
