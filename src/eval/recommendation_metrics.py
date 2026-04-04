import re
from collections import Counter

from src.pipeline import schema_validator


def validate_against_schema(instance: dict, schema_name: str) -> bool:
    validator = getattr(schema_validator, "validate_against_schema", None)
    if callable(validator):
        return bool(validator(instance, schema_name))
    return bool(schema_validator.is_valid(instance, schema_name))


_KEY_TERM_HINTS = {
    "follow",
    "followup",
    "follow-up",
    "ct",
    "ldct",
    "pet",
    "petct",
    "biopsy",
    "month",
    "months",
    "year",
    "years",
    "immediate",
    "annual",
}


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _as_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    lowered = text.lower().replace("pet-ct", "petct").replace("follow-up", "followup")
    return set(re.findall(r"[a-z0-9]+", lowered))


def _extract_expected_key_terms(cue_text: str | None) -> set[str]:
    tokens = _as_tokens(cue_text)
    if not tokens:
        return set()

    selected = set()
    for token in tokens:
        if token in _KEY_TERM_HINTS:
            selected.add(token)
            continue
        if "month" in token or "year" in token or token.endswith("ct"):
            selected.add(token)

    if selected:
        return selected

    return {token for token in tokens if len(token) >= 3}


def evaluate_recommendation_single(rec: dict) -> dict:
    recommendation_level = rec.get("recommendation_level")
    guideline_anchor = rec.get("guideline_anchor")
    reasoning_path = rec.get("reasoning_path") or []
    triggered_rules = rec.get("triggered_rules") or []
    missing_information = rec.get("missing_information") or []
    input_facts = rec.get("input_facts_used") or {}

    _ACTIONABLE_LEVELS = {"actionable", "short_interval_followup", "tissue_sampling", "diagnostic_workup"}
    _MONITORING_LEVELS = {"monitoring", "routine_screening"}
    _INSUFFICIENT_LEVELS = {"insufficient_data"}

    return {
        "schema_valid": bool(validate_against_schema(rec, "recommendation_schema.json")),
        "is_actionable": recommendation_level in _ACTIONABLE_LEVELS,
        "is_monitoring": recommendation_level in _MONITORING_LEVELS,
        "is_insufficient_data": recommendation_level in _INSUFFICIENT_LEVELS,
        "has_guideline_anchor": guideline_anchor is not None and str(guideline_anchor).strip() != "",
        "has_reasoning_path": isinstance(reasoning_path, list) and len(reasoning_path) > 0,
        "has_triggered_rules": isinstance(triggered_rules, list) and len(triggered_rules) > 0,
        "missing_fields_count": len(missing_information) if isinstance(missing_information, list) else 0,
        "missing_fields": missing_information if isinstance(missing_information, list) else [],
        "lung_rads_category": rec.get("lung_rads_category"),
        "recommendation_level": recommendation_level,
        "followup_interval": rec.get("followup_interval"),
        "followup_modality": rec.get("followup_modality"),
        "density_type": input_facts.get("nodule_density"),
    }


def compute_recommendation_quality_summary(recommendations: list[dict]) -> dict:
    total = len(recommendations)
    if total == 0:
        return {
            "schema_valid_rate": 0.0,
            "guideline_anchor_presence_rate": 0.0,
            "reasoning_path_nonempty_rate": 0.0,
            "triggered_rules_nonempty_rate": 0.0,
        }

    single_metrics = [evaluate_recommendation_single(rec) for rec in recommendations]
    schema_valid_count = sum(1 for item in single_metrics if item["schema_valid"])
    anchor_count = sum(1 for item in single_metrics if item["has_guideline_anchor"])
    reasoning_count = sum(1 for item in single_metrics if item["has_reasoning_path"])
    rules_count = sum(1 for item in single_metrics if item["has_triggered_rules"])

    return {
        "schema_valid_rate": _safe_rate(schema_valid_count, total),
        "guideline_anchor_presence_rate": _safe_rate(anchor_count, total),
        "reasoning_path_nonempty_rate": _safe_rate(reasoning_count, total),
        "triggered_rules_nonempty_rate": _safe_rate(rules_count, total),
    }


def evaluate_recommendations(recommendations: list[dict], manifest: dict | None = None) -> dict:
    manifest = manifest or {}
    total = len(recommendations)
    single_metrics = [evaluate_recommendation_single(rec) for rec in recommendations]

    quality_summary = compute_recommendation_quality_summary(recommendations)

    actionable_count = sum(1 for item in single_metrics if item["is_actionable"])
    monitoring_count = sum(1 for item in single_metrics if item["is_monitoring"])
    insufficient_count = sum(1 for item in single_metrics if item["is_insufficient_data"])

    missing_field_distribution = Counter()
    for item in single_metrics:
        missing_field_distribution.update(item["missing_fields"])

    lung_rads_distribution = Counter(item["lung_rads_category"] for item in single_metrics)
    recommendation_level_distribution = Counter(item["recommendation_level"] for item in single_metrics)
    followup_interval_distribution = Counter(item["followup_interval"] for item in single_metrics)
    followup_modality_distribution = Counter(item["followup_modality"] for item in single_metrics)

    by_density: dict[str | None, Counter] = {}
    for item in single_metrics:
        density = item["density_type"]
        if density not in by_density:
            by_density[density] = Counter()
        by_density[density].update([item["recommendation_level"]])

    explicit_labels = manifest.get("explicit_cue_labels") if isinstance(manifest, dict) else None
    explicit_matches = 0
    explicit_total = 0
    if isinstance(explicit_labels, dict):
        by_case_id = {rec.get("case_id"): rec for rec in recommendations}
        for case_id, expected_cue in explicit_labels.items():
            rec = by_case_id.get(case_id)
            if rec is None:
                continue
            explicit_total += 1
            action_tokens = _as_tokens(rec.get("recommendation_action"))
            expected_terms = _extract_expected_key_terms(expected_cue)
            if expected_terms and expected_terms.issubset(action_tokens):
                explicit_matches += 1

    rule_labels = manifest.get("rule_derived_labels") if isinstance(manifest, dict) else None
    rule_matches = 0
    rule_total = 0
    if isinstance(rule_labels, dict):
        by_case_id = {rec.get("case_id"): rec for rec in recommendations}
        for case_id, expected_category in rule_labels.items():
            rec = by_case_id.get(case_id)
            if rec is None:
                continue
            rule_total += 1
            if rec.get("lung_rads_category") == expected_category:
                rule_matches += 1

    avg_missing_fields = 0.0
    if total > 0:
        avg_missing_fields = float(sum(item["missing_fields_count"] for item in single_metrics) / total)

    return {
        "total_recommendations": total,
        "schema_valid_rate": quality_summary["schema_valid_rate"],
        "actionable_rate": _safe_rate(actionable_count, total),
        "monitoring_rate": _safe_rate(monitoring_count, total),
        "insufficient_data_rate": _safe_rate(insufficient_count, total),
        "guideline_anchor_presence_rate": quality_summary["guideline_anchor_presence_rate"],
        "reasoning_path_nonempty_rate": quality_summary["reasoning_path_nonempty_rate"],
        "triggered_rules_nonempty_rate": quality_summary["triggered_rules_nonempty_rate"],
        "avg_missing_fields": avg_missing_fields,
        "missing_field_distribution": dict(missing_field_distribution),
        "lung_rads_distribution": dict(lung_rads_distribution),
        "recommendation_level_distribution": dict(recommendation_level_distribution),
        "followup_interval_distribution": dict(followup_interval_distribution),
        "followup_modality_distribution": dict(followup_modality_distribution),
        "explicit_cue_agreement_rate": _safe_rate(explicit_matches, explicit_total),
        "rule_agreement_rate": _safe_rate(rule_matches, rule_total),
        "recommendation_by_density": {density: dict(counter) for density, counter in by_density.items()},
    }
