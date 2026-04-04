from collections import Counter

from src.pipeline import schema_validator

_validate_against_schema = getattr(schema_validator, "validate_against_schema", None)

def validate_against_schema(instance: dict, schema_name: str):
    if callable(_validate_against_schema):
        return _validate_against_schema(instance, schema_name)
    errors = schema_validator.validate_instance(instance, schema_name)
    return len(errors) == 0, errors


_SCHEMA_PATH = "schemas/smoking_eligibility_schema.json"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _normalize_status(status: str | None) -> str:
    if status in {"current_smoker", "current"}:
        return "current"
    if status in {"former_smoker", "former"}:
        return "former"
    if status in {"never_smoker", "never"}:
        return "never"
    return "unknown"


def _is_schema_valid(result: dict) -> bool:
    for schema_arg in (_SCHEMA_PATH, "smoking_eligibility_schema.json"):
        try:
            response = validate_against_schema(result, schema_arg)
        except FileNotFoundError:
            continue
        except TypeError:
            continue
        except Exception:
            return False

        if isinstance(response, bool):
            return response
        if isinstance(response, tuple) and response:
            return bool(response[0])
        if isinstance(response, dict):
            if "is_valid" in response:
                return bool(response["is_valid"])
            if "valid" in response:
                return bool(response["valid"])
        if isinstance(response, list):
            return len(response) == 0
        return bool(response)
    return False


def _is_eligible(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value == "eligible"
    return False


def _contains_deidentified(note: str | None) -> bool:
    if not note:
        return False
    lowered = note.lower()
    return "de-identified" in lowered or "deidentified" in lowered


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def evaluate_smoking_single(result: dict) -> dict:
    status = _normalize_status(result.get("smoking_status_norm"))
    is_non_unknown = status != "unknown"
    is_former = status == "former"
    source_section = result.get("source_section")

    return {
        "schema_valid": _is_schema_valid(result),
        "status": status,
        "is_unknown": not is_non_unknown,
        "is_non_unknown": is_non_unknown,
        "ever_smoker_true": bool(result.get("ever_smoker_flag") is True),
        "eligible_true": _is_eligible(result.get("eligible_for_high_risk_screening")),
        "fallback_triggered": source_section != "Social History",
        "social_history_only": source_section == "Social History",
        "pack_year_parsed": result.get("pack_year_value") is not None,
        "ppd_parsed": result.get("ppd_value") is not None,
        "years_smoked_parsed": result.get("years_smoked_value") is not None,
        "quit_years_parsed": bool(is_former and result.get("quit_years_value") is not None),
        "ppd_ambiguity_protected": bool(
            result.get("ppd_value") is None and result.get("ppd_text") is not None
        ),
        "deidentified_note": _contains_deidentified(result.get("data_quality_notes")),
    }


def compute_smoking_coverage_summary(results: list[dict]) -> dict:
    total = len(results)
    per_note = [evaluate_smoking_single(item) for item in results]
    non_unknown = [item for item in per_note if item["is_non_unknown"]]
    former = [item for item in per_note if item["status"] == "former"]

    return {
        "total_notes": total,
        "non_unknown_rate": _safe_rate(len(non_unknown), total),
        "unknown_rate": _safe_rate(sum(1 for item in per_note if item["is_unknown"]), total),
        "fallback_trigger_rate": _safe_rate(
            sum(1 for item in per_note if item["fallback_triggered"]), total
        ),
        "social_history_only_rate": _safe_rate(
            sum(1 for item in per_note if item["social_history_only"]), total
        ),
        "pack_year_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["pack_year_parsed"]), len(non_unknown)
        ),
        "ppd_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["ppd_parsed"]), len(non_unknown)
        ),
        "years_smoked_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["years_smoked_parsed"]), len(non_unknown)
        ),
        "quit_years_parse_rate": _safe_rate(
            sum(1 for item in former if item["quit_years_parsed"]), len(former)
        ),
    }


def evaluate_smoking(results: list[dict], manifest: dict | None = None) -> dict:
    total = len(results)
    per_note = [evaluate_smoking_single(item) for item in results]

    schema_valid_count = sum(1 for item in per_note if item["schema_valid"])
    non_unknown = [item for item in per_note if item["is_non_unknown"]]
    former = [item for item in per_note if item["status"] == "former"]

    status_distribution = Counter(item["status"] for item in per_note)
    evidence_quality_distribution = Counter(
        (result.get("evidence_quality") or "none") for result in results
    )

    explicit_status_accuracy = 0.0
    explicit_pack_year_accuracy = 0.0
    explicit_ppd_accuracy = 0.0

    if manifest:
        note_to_result = {result.get("note_id"): result for result in results}

        explicit_status_labels = manifest.get("explicit_status_labels") or {}
        status_total = 0
        status_correct = 0
        for note_id, expected in explicit_status_labels.items():
            if note_id not in note_to_result:
                continue
            status_total += 1
            predicted = _normalize_status(note_to_result[note_id].get("smoking_status_norm"))
            if predicted == _normalize_status(expected):
                status_correct += 1
        explicit_status_accuracy = _safe_rate(status_correct, status_total)

        explicit_quantitative_labels = manifest.get("explicit_quantitative_labels") or {}
        pack_total = 0
        pack_correct = 0
        ppd_total = 0
        ppd_correct = 0
        for note_id, expected_values in explicit_quantitative_labels.items():
            if note_id not in note_to_result or not isinstance(expected_values, dict):
                continue

            expected_pack = expected_values.get("pack_year_value")
            if expected_pack is None:
                expected_pack = expected_values.get("pack_year")
            expected_pack_float = _to_float(expected_pack)
            if expected_pack_float is not None:
                pack_total += 1
                predicted_pack = _to_float(note_to_result[note_id].get("pack_year_value"))
                if predicted_pack is not None and abs(predicted_pack - expected_pack_float) <= 1.0:
                    pack_correct += 1

            expected_ppd = expected_values.get("ppd_value")
            if expected_ppd is None:
                expected_ppd = expected_values.get("ppd")
            expected_ppd_float = _to_float(expected_ppd)
            if expected_ppd_float is not None:
                ppd_total += 1
                predicted_ppd = _to_float(note_to_result[note_id].get("ppd_value"))
                if predicted_ppd is not None and abs(predicted_ppd - expected_ppd_float) <= 0.1:
                    ppd_correct += 1

        explicit_pack_year_accuracy = _safe_rate(pack_correct, pack_total)
        explicit_ppd_accuracy = _safe_rate(ppd_correct, ppd_total)

    deidentified_count = sum(
        1 for result in results if _contains_deidentified(result.get("data_quality_notes"))
    )

    output = {
        "total_notes": total,
        "schema_valid_rate": _safe_rate(schema_valid_count, total),
        "non_unknown_rate": _safe_rate(len(non_unknown), total),
        "ever_smoker_rate": _safe_rate(
            sum(1 for item in non_unknown if item["ever_smoker_true"]), len(non_unknown)
        ),
        "eligible_rate": _safe_rate(
            sum(1 for item in non_unknown if item["eligible_true"]), len(non_unknown)
        ),
        "evidence_quality_distribution": evidence_quality_distribution,
        "status_distribution": status_distribution,
        "fallback_trigger_rate": _safe_rate(
            sum(1 for item in per_note if item["fallback_triggered"]), total
        ),
        "social_history_only_rate": _safe_rate(
            sum(1 for item in per_note if item["social_history_only"]), total
        ),
        "pack_year_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["pack_year_parsed"]), len(non_unknown)
        ),
        "ppd_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["ppd_parsed"]), len(non_unknown)
        ),
        "years_smoked_parse_rate": _safe_rate(
            sum(1 for item in non_unknown if item["years_smoked_parsed"]), len(non_unknown)
        ),
        "quit_years_parse_rate": _safe_rate(
            sum(1 for item in former if item["quit_years_parsed"]), len(former)
        ),
        "ppd_ambiguity_protected_count": sum(
            1 for item in per_note if item["ppd_ambiguity_protected"]
        ),
        "explicit_status_accuracy": explicit_status_accuracy,
        "explicit_pack_year_accuracy": explicit_pack_year_accuracy,
        "explicit_ppd_accuracy": explicit_ppd_accuracy,
        "unknown_rate": _safe_rate(sum(1 for item in per_note if item["is_unknown"]), total),
        "deidentified_rate": _safe_rate(deidentified_count, total),
    }
    return output
