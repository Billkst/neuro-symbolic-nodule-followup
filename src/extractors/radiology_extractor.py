from datetime import datetime, timezone

from src.extractors.modality_classifier import classify_body_site, classify_modality
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


def _normalize_sections(sections: dict | None) -> dict:
    sections = sections or {}
    return {
        "indication": sections.get("indication"),
        "technique": sections.get("technique"),
        "comparison": sections.get("comparison"),
        "findings": sections.get("findings"),
        "impression": sections.get("impression"),
    }


def _confidence(size_mm: float | None, density_category: str | None, location_lobe: str | None) -> str:
    score = 0
    if size_mm is not None:
        score += 1
    if density_category not in (None, "unclear"):
        score += 1
    if location_lobe not in (None, "unclear"):
        score += 1

    if score == 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"


def extract_radiology_facts(
    note_id: str,
    subject_id: int,
    exam_name: str,
    report_text: str,
    sections: dict,
) -> dict:
    norm_sections = _normalize_sections(sections)
    findings_text = norm_sections.get("findings") or ""
    impression_text = norm_sections.get("impression") or ""

    modality = classify_modality(exam_name)
    body_site = classify_body_site(exam_name)

    nodule_mentions = segment_nodule_mentions(findings_text)
    if not nodule_mentions:
        nodule_mentions = segment_nodule_mentions(impression_text)

    recommendation_cue = extract_recommendation_cue(f"{impression_text} {report_text}")
    lung_rads = extract_lung_rads(f"{impression_text} {report_text}")

    nodule_count = len(nodule_mentions)
    count_type = "single" if nodule_count == 1 else ("multiple" if nodule_count > 1 else "unclear")

    nodules = []
    for idx, mention in enumerate(nodule_mentions, start=1):
        size_mm, size_text = extract_size(mention)
        density_category, density_text = extract_density(mention)
        location_lobe, location_text = extract_location(mention)
        change_status, change_text = extract_change_status(mention)
        morphology = extract_morphology(mention)

        confidence = _confidence(size_mm, density_category, location_lobe)

        nodule_obj = {
            "nodule_id_in_report": idx,
            "size_mm": size_mm,
            "size_text": size_text,
            "density_category": density_category,
            "density_text": density_text,
            "location_lobe": location_lobe,
            "location_text": location_text,
            "count_type": count_type,
            "change_status": change_status,
            "change_text": change_text,
            "calcification": morphology["calcification"],
            "spiculation": morphology["spiculation"],
            "lobulation": morphology["lobulation"],
            "cavitation": morphology["cavitation"],
            "perifissural": morphology["perifissural"],
            "lung_rads_category": lung_rads,
            "recommendation_cue": recommendation_cue,
            "evidence_span": mention,
            "confidence": confidence,
            "missing_flags": [],
        }

        missing_flags = []
        for key, value in nodule_obj.items():
            if key == "missing_flags":
                continue
            if value is None:
                missing_flags.append(key)
        nodule_obj["missing_flags"] = missing_flags
        nodules.append(nodule_obj)

    result = {
        "note_id": note_id,
        "subject_id": subject_id,
        "exam_name": exam_name,
        "modality": modality,
        "body_site": body_site,
        "report_text": report_text,
        "sections": norm_sections,
        "nodule_count": nodule_count,
        "nodules": nodules,
        "extraction_metadata": {
            "extractor_version": "regex_baseline_0.1",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_name": "regex_baseline",
        },
    }
    return result
