import re
from datetime import datetime, timezone


_CATEGORY_ORDER = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4A": 4,
    "4B": 5,
    "4X": 6,
    "S": 7,
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _flatten_nodule_candidates(case_bundle: dict) -> list[dict]:
    candidates = []
    for fact in case_bundle.get("radiology_facts", []):
        report_nodule_count = fact.get("nodule_count")
        for nodule in fact.get("nodules", []):
            candidates.append(
                {
                    "note_id": fact.get("note_id"),
                    "report_nodule_count": report_nodule_count,
                    "report_text": fact.get("report_text"),
                    "sections": fact.get("sections") or {},
                    "nodule": nodule,
                }
            )
    return candidates


def _extract_solid_component_mm(nodule: dict) -> float | None:
    texts = [
        nodule.get("density_text"),
        nodule.get("evidence_span"),
        nodule.get("recommendation_cue"),
    ]
    pattern = re.compile(
        r"solid\s+component(?:\s+(?:measuring|measures|measure(?:d)?|of))?\s*(\d+(?:\.\d+)?)\s*mm\b",
        re.IGNORECASE,
    )

    for text in texts:
        if not text:
            continue
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _normalize_density(nodule: dict, missing_information: list[str], uncertainty_notes: list[str]) -> str:
    density = nodule.get("density_category")
    if density in {"solid", "part_solid", "ground_glass"}:
        return density
    if density in {"calcified", "fat_containing"}:
        return density
    if density in {None, "unclear"}:
        missing_information.append("density_category")
        uncertainty_notes.append("density_category 缺失或不明确，按 solid 路径保守处理。")
        return "solid"
    missing_information.append("density_category")
    uncertainty_notes.append("density_category 超出最小规则集，按 solid 路径保守处理。")
    return "solid"


def _is_stable_for_two_years(nodule: dict) -> bool:
    change_text = (nodule.get("change_text") or "")
    evidence_span = (nodule.get("evidence_span") or "")
    combined_text = f"{change_text} {evidence_span}".lower()
    return bool(
        re.search(r"\b(2\s*years?|two\s+years?|24\s*months?|2-year|two-year)\b", combined_text)
    )


def _base_result_for_category(category: str, is_new: bool) -> dict:
    if category == "2":
        return {
            "recommendation_level": "routine_screening",
            "recommendation_action": "继续 annual LDCT 筛查，12 个月后复查。",
            "followup_interval": "12_months",
            "followup_modality": "LDCT",
        }

    if category == "3":
        return {
            "recommendation_level": "short_interval_followup",
            "recommendation_action": "建议 6 个月后复查 LDCT。",
            "followup_interval": "6_months",
            "followup_modality": "LDCT",
        }

    if category == "4A":
        if is_new:
            return {
                "recommendation_level": "diagnostic_workup",
                "recommendation_action": "建议尽快进入诊断性评估，可行 PET-CT，并在 3 个月内完成复评。",
                "followup_interval": "3_months",
                "followup_modality": "PET_CT",
            }
        return {
            "recommendation_level": "short_interval_followup",
            "recommendation_action": "建议 3 个月后复查 LDCT，必要时补充 PET-CT。",
            "followup_interval": "3_months",
            "followup_modality": "LDCT",
        }

    return {
        "recommendation_level": "tissue_sampling",
        "recommendation_action": "建议立即进一步诊断评估，优先考虑组织取样或 PET-CT。",
        "followup_interval": "immediate",
        "followup_modality": "biopsy",
    }


def _upgrade_category(category: str) -> str:
    if category == "2":
        return "3"
    if category == "3":
        return "4A"
    if category == "4A":
        return "4B"
    return category


def _evaluate_nodule(candidate: dict) -> dict:
    nodule = candidate["nodule"]
    size_mm = nodule.get("size_mm")
    missing_information = []
    uncertainty_notes = []
    reasoning_path = ["flatten_radiology_facts"]
    triggered_rules = []

    density = _normalize_density(nodule, missing_information, uncertainty_notes)
    change_status = nodule.get("change_status")
    is_new = change_status == "new"
    solid_component_mm = None

    if size_mm is None:
        return {
            "category": None,
            "guideline_anchor": None,
            "reasoning_path": reasoning_path + ["required_fact_missing", "nodule_size_missing"],
            "triggered_rules": ["fallback_missing_size"],
            "missing_information": ["nodule_size"],
            "uncertainty_notes": ["缺少 size_mm，无法进入 Lung-RADS 尺寸分支。"],
            "nodule": nodule,
            "density": density,
            "size_mm": None,
            "change_status": change_status,
            "solid_component_mm": None,
            "severity": -1,
        }

    if density in {"calcified", "fat_containing"}:
        category = "2"
        guideline_anchor = "Benign attenuation pattern treated as Lung-RADS category 2 in minimal engine."
        reasoning_path.extend([f"density={density}", "benign_density_pattern", "category_2"])
        triggered_rules.append("benign_density_category_2")
    elif density == "solid":
        reasoning_path.append("solid_pathway")
        if is_new:
            if size_mm < 4:
                category = "2"
                guideline_anchor = "New solid nodule < 4 mm -> category 2 with annual screening."
                triggered_rules.append("solid_new_lt4_category_2")
            elif size_mm < 8:
                category = "3"
                guideline_anchor = "New solid nodule 4-7 mm -> category 3 with 6-month follow-up."
                triggered_rules.append("solid_new_4_to_7_category_3")
            elif size_mm < 15:
                category = "4A"
                guideline_anchor = "New solid nodule 8-14 mm -> category 4A with diagnostic workup or short follow-up."
                triggered_rules.append("solid_new_8_to_14_category_4A")
            else:
                category = "4B"
                guideline_anchor = "New solid nodule >= 15 mm -> category 4B with immediate workup."
                triggered_rules.append("solid_new_ge15_category_4B")
        else:
            if size_mm < 6:
                category = "2"
                guideline_anchor = "Solid nodule < 6 mm -> category 2 with annual screening."
                triggered_rules.append("solid_lt6_category_2")
            elif size_mm < 8:
                category = "3"
                guideline_anchor = "Solid nodule 6-7 mm -> category 3 with 6-month follow-up."
                triggered_rules.append("solid_6_to_7_category_3")
            elif size_mm < 15:
                category = "4A"
                guideline_anchor = "Solid nodule 8-14 mm -> category 4A with 3-month follow-up or PET-CT."
                triggered_rules.append("solid_8_to_14_category_4A")
            else:
                category = "4B"
                guideline_anchor = "Solid nodule >= 15 mm -> category 4B with immediate diagnostic action."
                triggered_rules.append("solid_ge15_category_4B")
        reasoning_path.append(f"size_mm={size_mm}")
    elif density == "part_solid":
        reasoning_path.append("part_solid_pathway")
        solid_component_mm = _extract_solid_component_mm(nodule)
        if is_new:
            category = "3"
            guideline_anchor = "New part-solid nodule -> category 3 with 6-month follow-up."
            triggered_rules.append("part_solid_new_category_3")
        elif size_mm < 6:
            category = "2"
            guideline_anchor = "Part-solid nodule < 6 mm -> category 2 with annual screening."
            triggered_rules.append("part_solid_lt6_category_2")
        elif solid_component_mm is None:
            category = "4A"
            guideline_anchor = "Part-solid nodule >= 6 mm with missing solid component -> conservative category 4A fallback."
            triggered_rules.append("part_solid_missing_solid_component_conservative_4A")
            missing_information.append("solid_component_mm")
            uncertainty_notes.append("part-solid 结节缺少 solid component，按保守 4A 处理。")
        elif solid_component_mm < 6:
            category = "3"
            guideline_anchor = "Part-solid nodule >= 6 mm with solid component < 6 mm -> category 3."
            triggered_rules.append("part_solid_ge6_solid_component_lt6_category_3")
        else:
            category = "4A"
            guideline_anchor = "Part-solid nodule >= 6 mm with solid component >= 6 mm -> category 4A."
            triggered_rules.append("part_solid_ge6_solid_component_ge6_category_4A")
        reasoning_path.append(f"size_mm={size_mm}")
        if solid_component_mm is not None:
            reasoning_path.append(f"solid_component_mm={solid_component_mm}")
    else:
        reasoning_path.append("ground_glass_pathway")
        if is_new:
            if size_mm < 30:
                category = "2"
                guideline_anchor = "New ground-glass nodule < 30 mm -> category 2 with annual follow-up."
                triggered_rules.append("ground_glass_new_lt30_category_2")
            else:
                category = "3"
                guideline_anchor = "New ground-glass nodule >= 30 mm -> category 3 with 6-month follow-up."
                triggered_rules.append("ground_glass_new_ge30_category_3")
        elif size_mm < 30:
            category = "2"
            guideline_anchor = "Ground-glass nodule < 30 mm -> category 2 with annual screening."
            triggered_rules.append("ground_glass_lt30_category_2")
        else:
            category = "3"
            guideline_anchor = "Ground-glass nodule >= 30 mm -> category 3 with 6-month follow-up."
            triggered_rules.append("ground_glass_ge30_category_3")
        reasoning_path.append(f"size_mm={size_mm}")

    if change_status == "stable" and _is_stable_for_two_years(nodule) and category in {"3", "4A"}:
        category = "2"
        triggered_rules.append("stable_ge_2y_downgrade_to_2")
        reasoning_path.append("stable_for_2_years_downgrade")
        uncertainty_notes.append("稳定至少 2 年，按类别 2 处理。")
    elif change_status == "increased":
        upgraded = _upgrade_category(category)
        if upgraded != category:
            category = upgraded
            triggered_rules.append("growth_modifier_upgrade")
            reasoning_path.append("growth_modifier_upgrade")
            uncertainty_notes.append("存在增长证据，类别上调一级。")
    elif change_status == "decreased":
        triggered_rules.append("decrease_modifier_maintain")
        reasoning_path.append("decrease_modifier_maintain")
        uncertainty_notes.append("存在缩小证据，维持当前类别并提示改善。")

    severity = _CATEGORY_ORDER.get(category, -1)
    result = _base_result_for_category(category, is_new)
    result.update(
        {
            "category": category,
            "guideline_anchor": guideline_anchor,
            "reasoning_path": reasoning_path + [f"final_category={category}"],
            "triggered_rules": triggered_rules,
            "missing_information": missing_information,
            "uncertainty_notes": uncertainty_notes,
            "nodule": nodule,
            "density": density,
            "size_mm": size_mm,
            "change_status": change_status,
            "solid_component_mm": solid_component_mm,
            "severity": severity,
        }
    )
    return result


def _pick_dominant_evaluation(evaluations: list[dict]) -> dict | None:
    if not evaluations:
        return None

    return max(
        evaluations,
        key=lambda item: (
            item.get("severity", -1),
            item.get("size_mm") if item.get("size_mm") is not None else -1,
        ),
    )


def _deduplicate(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _insufficient_recommendation(
    case_id: str,
    smoking_eligibility: dict | None,
    engine_version: str,
    rules_version: str,
) -> dict:
    missing_information = ["nodule_size"]
    if smoking_eligibility is None:
        missing_information.append("smoking_eligibility")

    return {
        "case_id": case_id,
        "recommendation_level": "insufficient_data",
        "recommendation_action": "缺少足够结构化事实，当前只能返回 insufficient_data。",
        "followup_interval": None,
        "followup_modality": None,
        "lung_rads_category": None,
        "guideline_source": "Lung-RADS_v2022",
        "guideline_anchor": None,
        "reasoning_path": ["required_fact_missing", "nodule_size_absent", "insufficient_data_output"],
        "triggered_rules": ["fallback_insufficient_data"],
        "input_facts_used": {
            "nodule_size_mm": None,
            "nodule_density": None,
            "nodule_count": None,
            "change_status": None,
            "patient_risk_level": None,
            "smoking_eligible": smoking_eligibility.get("eligible_for_high_risk_screening") if smoking_eligibility else None,
        },
        "missing_information": missing_information,
        "uncertainty_note": "关键尺寸信息缺失，无法稳定进入 Lung-RADS 最小规则集。",
        "output_type": "rule_based",
        "generation_metadata": {
            "engine_version": engine_version,
            "generation_timestamp": _timestamp(),
            "rules_version": rules_version,
        },
    }


def generate_recommendation(
    case_bundle: dict,
    engine_version: str = "lung_rads_minimal_0.1",
    rules_version: str = "lung_rads_v2022_minimal_0.1",
) -> dict:
    case_id = case_bundle.get("case_id") or "unknown_case"
    smoking_eligibility = case_bundle.get("smoking_eligibility")
    candidates = _flatten_nodule_candidates(case_bundle)

    if not candidates:
        return _insufficient_recommendation(case_id, smoking_eligibility, engine_version, rules_version)

    evaluations = [_evaluate_nodule(candidate) for candidate in candidates]
    dominant = _pick_dominant_evaluation(evaluations)
    if dominant is None or dominant.get("size_mm") is None:
        return _insufficient_recommendation(case_id, smoking_eligibility, engine_version, rules_version)

    patient_risk_level = None
    smoking_eligible = None
    missing_information = list(dominant["missing_information"])
    uncertainty_notes = list(dominant["uncertainty_notes"])

    if smoking_eligibility is None:
        missing_information.append("smoking_eligibility")
        uncertainty_notes.append("缺少 smoking_eligibility，不阻断规则，但限制风险分层解释。")
    else:
        smoking_eligible = smoking_eligibility.get("eligible_for_high_risk_screening")
        if smoking_eligible == "eligible":
            patient_risk_level = "high_risk"
        elif smoking_eligible == "not_eligible":
            patient_risk_level = "not_high_risk"
        else:
            patient_risk_level = "unknown"

    all_nodules = [candidate["nodule"] for candidate in candidates]
    reasoning_path = list(dominant["reasoning_path"])
    reasoning_path.insert(1, f"selected_dominant_nodule_size={dominant['size_mm']}")
    reasoning_path.append(f"recommendation_level={dominant['recommendation_level']}")

    recommendation = {
        "case_id": case_id,
        "recommendation_level": dominant["recommendation_level"],
        "recommendation_action": dominant["recommendation_action"],
        "followup_interval": dominant["followup_interval"],
        "followup_modality": dominant["followup_modality"],
        "lung_rads_category": dominant["category"],
        "guideline_source": "Lung-RADS_v2022",
        "guideline_anchor": dominant["guideline_anchor"],
        "reasoning_path": reasoning_path,
        "triggered_rules": _deduplicate(dominant["triggered_rules"]),
        "input_facts_used": {
            "nodule_size_mm": dominant["size_mm"],
            "nodule_density": dominant["density"],
            "nodule_count": len(all_nodules),
            "change_status": dominant["change_status"],
            "patient_risk_level": patient_risk_level,
            "smoking_eligible": smoking_eligible,
        },
        "missing_information": _deduplicate(missing_information),
        "uncertainty_note": " ".join(_deduplicate(uncertainty_notes)) or None,
        "output_type": "rule_based",
        "generation_metadata": {
            "engine_version": engine_version,
            "generation_timestamp": _timestamp(),
            "rules_version": rules_version,
        },
    }
    return recommendation
