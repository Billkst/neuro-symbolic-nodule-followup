from collections import Counter
from pathlib import Path
from typing import Callable, cast

from src.pipeline import schema_validator


def _validate_schema(data: dict, schema_path: str | Path) -> tuple[bool, list[str]]:
    validator_fn = cast(
        Callable[[dict, str | Path], tuple[bool, list[str]]] | None,
        getattr(schema_validator, "validate_against_schema", None),
    )
    if callable(validator_fn):
        return validator_fn(data, schema_path)
    errors = schema_validator.validate_instance(data, str(schema_path))
    return len(errors) == 0, errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "radiology_fact_schema.json"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _note_nodule_count(fact: dict) -> int:
    raw_count = fact.get("nodule_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    nodules = fact.get("nodules")
    if isinstance(nodules, list):
        return len(nodules)
    return 0


def _note_nodules(fact: dict) -> list[dict]:
    nodules = fact.get("nodules")
    if isinstance(nodules, list):
        return [n for n in nodules if isinstance(n, dict)]
    return []


def _size_match(extracted_size: object, expected_size: object) -> bool:
    if expected_size is None:
        return extracted_size is None
    if not isinstance(expected_size, (int, float)):
        return False
    if not isinstance(extracted_size, (int, float)):
        return False
    return abs(float(extracted_size) - float(expected_size)) <= 0.5


def _exact_match(extracted_value: object, expected_value: object) -> bool:
    return extracted_value == expected_value


def compute_field_extraction_summary(facts: list[dict]) -> dict:
    all_nodules: list[dict] = []
    for fact in facts:
        all_nodules.extend(_note_nodules(fact))

    total_nodules = len(all_nodules)
    size_mm_hits = 0
    density_hits = 0
    location_hits = 0
    change_hits = 0
    recommendation_hits = 0

    for nodule in all_nodules:
        if nodule.get("size_mm") is not None:
            size_mm_hits += 1
        if nodule.get("density_category") not in (None, "unclear"):
            density_hits += 1
        if nodule.get("location_lobe") not in (None, "unclear"):
            location_hits += 1
        if nodule.get("change_status") not in (None, "unclear"):
            change_hits += 1
        if nodule.get("recommendation_cue") is not None:
            recommendation_hits += 1

    return {
        "size_mm_extract_rate": _safe_rate(size_mm_hits, total_nodules),
        "density_category_extract_rate": _safe_rate(density_hits, total_nodules),
        "location_lobe_extract_rate": _safe_rate(location_hits, total_nodules),
        "change_status_extract_rate": _safe_rate(change_hits, total_nodules),
        "recommendation_cue_extract_rate": _safe_rate(recommendation_hits, total_nodules),
    }


def evaluate_radiology_single(fact: dict) -> dict:
    note_id = fact.get("note_id")
    nodules = _note_nodules(fact)
    nodule_count = _note_nodule_count(fact)
    schema_valid, _ = _validate_schema(fact, SCHEMA_PATH)

    field_summary = compute_field_extraction_summary([fact])
    confidence_distribution = Counter(n.get("confidence") for n in nodules)
    missing_field_distribution = Counter()
    density_distribution = Counter(n.get("density_category") for n in nodules)
    location_distribution = Counter(n.get("location_lobe") for n in nodules)
    change_status_distribution = Counter(n.get("change_status") for n in nodules)

    for nodule in nodules:
        missing_flags = nodule.get("missing_flags")
        if isinstance(missing_flags, list):
            missing_field_distribution.update(missing_flags)

    metrics = {
        "note_id": note_id,
        "schema_valid": float(1.0 if schema_valid else 0.0),
        "note_level_success": float(1.0 if len(nodules) > 0 else 0.0),
        "nodule_detected": float(1.0 if nodule_count > 0 else 0.0),
        "nodule_count": nodule_count,
        "total_nodules": nodule_count,
        "avg_nodules_per_note": float(nodule_count),
        "confidence_distribution": confidence_distribution,
        "missing_field_distribution": missing_field_distribution,
        "density_distribution": density_distribution,
        "location_distribution": location_distribution,
        "change_status_distribution": change_status_distribution,
    }
    metrics.update(field_summary)
    return metrics


def evaluate_radiology(facts: list[dict], manifest: dict | None = None) -> dict:
    total_notes = len(facts)
    schema_valid_count = 0
    note_level_success_count = 0
    nodule_detect_count = 0
    total_nodules = 0
    all_nodules: list[dict] = []

    for fact in facts:
        is_valid, _ = _validate_schema(fact, SCHEMA_PATH)
        if is_valid:
            schema_valid_count += 1

        nodules = _note_nodules(fact)
        nodule_count = _note_nodule_count(fact)

        if len(nodules) > 0:
            note_level_success_count += 1
        if nodule_count > 0:
            nodule_detect_count += 1

        total_nodules += nodule_count
        all_nodules.extend(nodules)

    field_summary = compute_field_extraction_summary(facts)

    confidence_distribution = Counter(n.get("confidence") for n in all_nodules)
    missing_field_distribution = Counter()
    density_distribution = Counter(n.get("density_category") for n in all_nodules)
    location_distribution = Counter(n.get("location_lobe") for n in all_nodules)
    change_status_distribution = Counter(n.get("change_status") for n in all_nodules)

    for nodule in all_nodules:
        missing_flags = nodule.get("missing_flags")
        if isinstance(missing_flags, list):
            missing_field_distribution.update(missing_flags)

    explicit_size_total = 0
    explicit_size_match = 0
    explicit_density_total = 0
    explicit_density_match = 0
    explicit_location_total = 0
    explicit_location_match = 0
    explicit_change_total = 0
    explicit_change_match = 0

    explicit_labels = {}
    if isinstance(manifest, dict):
        labels = manifest.get("explicit_labels")
        if isinstance(labels, dict):
            explicit_labels = labels

    facts_by_note_id = {}
    for fact in facts:
        note_id = fact.get("note_id")
        if isinstance(note_id, str):
            facts_by_note_id[note_id] = fact

    for note_id, expected in explicit_labels.items():
        if not isinstance(expected, dict):
            continue
        fact = facts_by_note_id.get(note_id)
        nodules = _note_nodules(fact) if isinstance(fact, dict) else []
        extracted = nodules[0] if nodules else {}

        if "size_mm" in expected:
            explicit_size_total += 1
            if _size_match(extracted.get("size_mm"), expected.get("size_mm")):
                explicit_size_match += 1
        if "density_category" in expected:
            explicit_density_total += 1
            if _exact_match(extracted.get("density_category"), expected.get("density_category")):
                explicit_density_match += 1
        if "location_lobe" in expected:
            explicit_location_total += 1
            if _exact_match(extracted.get("location_lobe"), expected.get("location_lobe")):
                explicit_location_match += 1
        if "change_status" in expected:
            explicit_change_total += 1
            if _exact_match(extracted.get("change_status"), expected.get("change_status")):
                explicit_change_match += 1

    results = {
        "total_notes": total_notes,
        "schema_valid_rate": _safe_rate(schema_valid_count, total_notes),
        "note_level_success_rate": _safe_rate(note_level_success_count, total_notes),
        "nodule_detect_rate": _safe_rate(nodule_detect_count, total_notes),
        "avg_nodules_per_note": _safe_rate(total_nodules, total_notes),
        "total_nodules": total_nodules,
        "confidence_distribution": confidence_distribution,
        "missing_field_distribution": missing_field_distribution,
        "density_distribution": density_distribution,
        "location_distribution": location_distribution,
        "change_status_distribution": change_status_distribution,
        "explicit_size_exact_rate": _safe_rate(explicit_size_match, explicit_size_total),
        "explicit_density_exact_rate": _safe_rate(explicit_density_match, explicit_density_total),
        "explicit_location_exact_rate": _safe_rate(explicit_location_match, explicit_location_total),
        "explicit_change_exact_rate": _safe_rate(explicit_change_match, explicit_change_total),
    }
    results.update(field_summary)
    return results
