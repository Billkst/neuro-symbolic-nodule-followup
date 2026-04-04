from collections import defaultdict
from datetime import date

from src.rules.lung_rads_engine import generate_recommendation


def _normalize_sex(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"m", "male"}:
        return "M"
    if text in {"f", "female"}:
        return "F"
    if text in {"unknown", "unk", "other"}:
        return "unknown"
    return value


def _build_demographics_block(demographics_row: dict | None) -> dict:
    demographics_row = demographics_row or {}
    block = {
        "age": demographics_row.get("age"),
        "sex": _normalize_sex(demographics_row.get("sex")),
        "race": demographics_row.get("race"),
        "insurance": demographics_row.get("insurance"),
        "source": demographics_row.get("source"),
        "missing_flags": [],
    }

    missing_flags = []
    for key in ("age", "sex", "race", "insurance", "source"):
        if block[key] is None:
            missing_flags.append(key)
    block["missing_flags"] = missing_flags
    return block


def _flatten_nodules(radiology_facts: list[dict]) -> list[dict]:
    nodules = []
    for fact in radiology_facts:
        nodules.extend(fact.get("nodules", []))
    return nodules


def _first_recommendation_cue(radiology_facts: list[dict]) -> str | None:
    for fact in radiology_facts:
        for nodule in fact.get("nodules", []):
            cue = nodule.get("recommendation_cue")
            if cue:
                return cue
    return None


def _has_high_confidence_nodule(radiology_facts: list[dict]) -> bool:
    for nodule in _flatten_nodules(radiology_facts):
        if nodule.get("confidence") == "high" and nodule.get("size_mm") is not None:
            return True
    return False


def _has_medium_confidence_nodule(radiology_facts: list[dict]) -> bool:
    for nodule in _flatten_nodules(radiology_facts):
        if nodule.get("confidence") == "medium":
            return True
    return False


def _is_minimal_radiology_signal(radiology_facts: list[dict]) -> bool:
    nodules = _flatten_nodules(radiology_facts)
    if not nodules:
        return True

    informative_nodules = 0
    for nodule in nodules:
        if (
            nodule.get("size_mm") is not None
            or nodule.get("density_category") not in {None, "unclear"}
            or nodule.get("recommendation_cue")
            or nodule.get("confidence") in {"medium", "high"}
        ):
            informative_nodules += 1

    return informative_nodules == 0


def _unique_note_ids(radiology_facts: list[dict]) -> list[str]:
    note_ids = []
    seen = set()
    for fact in radiology_facts:
        note_id = fact.get("note_id")
        if note_id and note_id not in seen:
            seen.add(note_id)
            note_ids.append(note_id)
    return note_ids


def classify_label_quality(radiology_facts: list[dict], smoking_eligibility: dict | None) -> str:
    if not radiology_facts or _is_minimal_radiology_signal(radiology_facts):
        return "unlabeled"

    smoking_quality = None
    if smoking_eligibility is not None:
        smoking_quality = str(smoking_eligibility.get("evidence_quality") or "").lower()

    has_recommendation_cue = _first_recommendation_cue(radiology_facts) is not None
    if (
        _has_high_confidence_nodule(radiology_facts)
        and has_recommendation_cue
        and smoking_quality in {"high", "medium", "strong", "moderate"}
    ):
        return "silver"

    if radiology_facts and (smoking_eligibility is None or smoking_quality in {None, "", "low", "none"}):
        return "weak"

    if _has_medium_confidence_nodule(radiology_facts):
        return "weak"

    return "weak"


def assemble_case_bundles(
    radiology_facts: list[dict],
    smoking_results: dict[int, dict] | None,
    demographics: dict[int, dict] | None,
    pipeline_version: str = "0.1.0",
    data_version: str = "mimic-iv-note-2.2",
) -> list[dict]:
    if not radiology_facts:
        return []

    grouped_facts = defaultdict(list)
    for fact in radiology_facts:
        grouped_facts[fact["subject_id"]].append(fact)

    bundles = []
    smoking_results = smoking_results or {}
    demographics = demographics or {}
    extraction_date = date.today().isoformat()

    for subject_id in sorted(grouped_facts):
        subject_facts = grouped_facts[subject_id]
        smoking_eligibility = smoking_results.get(subject_id)
        label_quality = classify_label_quality(subject_facts, smoking_eligibility)
        ground_truth_action = _first_recommendation_cue(subject_facts)
        ground_truth_source = "extracted_from_report" if ground_truth_action else "none"

        if label_quality == "unlabeled":
            smoking_eligibility = None
            ground_truth_action = None
            ground_truth_source = "none"

        case_bundle = {
            "case_id": f"CASE-{subject_id}-001",
            "subject_id": subject_id,
            "demographics": _build_demographics_block(demographics.get(subject_id)),
            "radiology_facts": subject_facts,
            "smoking_eligibility": smoking_eligibility,
            "recommendation_target": {
                "ground_truth_action": ground_truth_action,
                "ground_truth_source": ground_truth_source,
                "ground_truth_interval": None,
                "recommendation_output": None,
            },
            "provenance": {
                "radiology_note_ids": _unique_note_ids(subject_facts),
                "discharge_note_id": smoking_eligibility.get("note_id") if smoking_eligibility else None,
                "data_version": data_version,
                "extraction_date": extraction_date,
                "pipeline_version": pipeline_version,
            },
            "split": "unlabeled",
            "label_quality": label_quality,
            "case_notes": None,
        }

        if label_quality == "silver":
            recommendation_output = generate_recommendation(case_bundle)
            if recommendation_output["recommendation_level"] == "insufficient_data":
                case_bundle["label_quality"] = "weak"
            else:
                case_bundle["recommendation_target"]["recommendation_output"] = recommendation_output

        bundles.append(case_bundle)

    return bundles
