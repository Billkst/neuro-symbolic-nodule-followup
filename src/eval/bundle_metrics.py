from collections import Counter

from src.pipeline import schema_validator


_VALIDATE_AGAINST_SCHEMA = getattr(schema_validator, "validate_against_schema", None)


def validate_against_schema(instance: dict, schema_path: str):
    if _VALIDATE_AGAINST_SCHEMA is not None:
        return _VALIDATE_AGAINST_SCHEMA(instance, schema_path)
    schema_name = schema_path.split("/")[-1]
    return len(schema_validator.validate_instance(instance, schema_name)) == 0


SCHEMA_PATH = "schemas/case_bundle_schema.json"


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return []


def _is_schema_valid(bundle: dict) -> bool:
    result = validate_against_schema(bundle, SCHEMA_PATH)
    if isinstance(result, bool):
        return result
    if isinstance(result, (list, tuple, set)):
        return len(result) == 0
    if isinstance(result, dict):
        if "is_valid" in result:
            return bool(result["is_valid"])
        if "errors" in result:
            return len(result["errors"]) == 0
    return False


def evaluate_bundle_single(bundle: dict) -> dict:
    demographics = bundle.get("demographics") or {}
    radiology_facts = _as_list(bundle.get("radiology_facts"))
    recommendation_target = bundle.get("recommendation_target") or {}
    provenance = bundle.get("provenance") or {}

    nodules_count = 0
    for fact in radiology_facts:
        nodules_count += len(_as_list(fact.get("nodules")))

    radiology_note_ids = _as_list(provenance.get("radiology_note_ids"))
    missing_flags = _as_list(demographics.get("missing_flags"))

    return {
        "schema_valid": _is_schema_valid(bundle),
        "has_radiology": len(radiology_facts) > 0,
        "has_smoking": bundle.get("smoking_eligibility") is not None,
        "has_recommendation": recommendation_target.get("recommendation_output") is not None,
        "has_demographics": any(demographics.get(key) is not None for key in ("age", "sex", "race")),
        "label_quality": bundle.get("label_quality"),
        "provenance_complete": (
            len(radiology_note_ids) > 0
            and provenance.get("data_version") is not None
            and provenance.get("pipeline_version") is not None
        ),
        "has_discharge_note": provenance.get("discharge_note_id") is not None,
        "ground_truth_action_present": recommendation_target.get("ground_truth_action") is not None,
        "ground_truth_source": recommendation_target.get("ground_truth_source"),
        "demographics_age_present": demographics.get("age") is not None,
        "demographics_sex_present": demographics.get("sex") is not None,
        "demographics_missing_fields_count": len(missing_flags),
        "radiology_facts_count": len(radiology_facts),
        "nodules_count": nodules_count,
    }


def compute_bundle_completeness_summary(bundles: list[dict]) -> dict:
    total = len(bundles)
    if total == 0:
        return {
            "total_bundles": 0,
            "schema_valid_rate": 0.0,
            "bundle_with_radiology_rate": 0.0,
            "bundle_with_smoking_rate": 0.0,
            "bundle_with_recommendation_rate": 0.0,
            "bundle_with_demographics_rate": 0.0,
            "provenance_complete_rate": 0.0,
            "has_discharge_note_rate": 0.0,
            "ground_truth_action_present_rate": 0.0,
            "demographics_age_present_rate": 0.0,
            "demographics_sex_present_rate": 0.0,
        }

    per_bundle = [evaluate_bundle_single(bundle) for bundle in bundles]
    return {
        "total_bundles": total,
        "schema_valid_rate": _rate(sum(1 for x in per_bundle if x["schema_valid"]), total),
        "bundle_with_radiology_rate": _rate(sum(1 for x in per_bundle if x["has_radiology"]), total),
        "bundle_with_smoking_rate": _rate(sum(1 for x in per_bundle if x["has_smoking"]), total),
        "bundle_with_recommendation_rate": _rate(sum(1 for x in per_bundle if x["has_recommendation"]), total),
        "bundle_with_demographics_rate": _rate(sum(1 for x in per_bundle if x["has_demographics"]), total),
        "provenance_complete_rate": _rate(sum(1 for x in per_bundle if x["provenance_complete"]), total),
        "has_discharge_note_rate": _rate(sum(1 for x in per_bundle if x["has_discharge_note"]), total),
        "ground_truth_action_present_rate": _rate(sum(1 for x in per_bundle if x["ground_truth_action_present"]), total),
        "demographics_age_present_rate": _rate(sum(1 for x in per_bundle if x["demographics_age_present"]), total),
        "demographics_sex_present_rate": _rate(sum(1 for x in per_bundle if x["demographics_sex_present"]), total),
    }


def evaluate_bundles(bundles: list[dict]) -> dict:
    summary = compute_bundle_completeness_summary(bundles)
    total = summary["total_bundles"]
    if total == 0:
        return {
            **summary,
            "label_quality_distribution": {},
            "ground_truth_source_distribution": {},
            "avg_demographics_missing_fields": 0.0,
            "avg_radiology_facts_per_bundle": 0.0,
            "avg_nodules_per_bundle": 0.0,
        }

    per_bundle = [evaluate_bundle_single(bundle) for bundle in bundles]
    label_quality_counter = Counter()
    ground_truth_source_counter = Counter()
    for item in per_bundle:
        if item["label_quality"] is not None:
            label_quality_counter[item["label_quality"]] += 1
        if item["ground_truth_source"] is not None:
            ground_truth_source_counter[item["ground_truth_source"]] += 1

    return {
        **summary,
        "label_quality_distribution": dict(label_quality_counter),
        "ground_truth_source_distribution": dict(ground_truth_source_counter),
        "avg_demographics_missing_fields": float(
            sum(x["demographics_missing_fields_count"] for x in per_bundle) / total
        ),
        "avg_radiology_facts_per_bundle": float(sum(x["radiology_facts_count"] for x in per_bundle) / total),
        "avg_nodules_per_bundle": float(sum(x["nodules_count"] for x in per_bundle) / total),
    }
