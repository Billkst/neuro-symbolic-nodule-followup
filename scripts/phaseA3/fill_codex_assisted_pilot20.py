#!/usr/bin/env python3
"""Fill M3-3C Codex-assisted pilot 20 audit outputs.

This script writes Codex-assisted pre-annotations and evidence-grounding
self-checks. The outputs are failure-mode audit artifacts only; they are not
clinical gold labels and do not involve medical expert review.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TEMPLATE_PATH = Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_template.csv")
FILLED_CSV_PATH = Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.csv")
FILLED_JSONL_PATH = Path("outputs/phaseA3/audit_sets/module3_codex_assisted_pilot_20_filled.jsonl")
SELF_CHECK_REPORT_PATH = Path("reports/module3_codex_assisted_pilot20_self_check_report.md")

NOTE = "Codex self-check; not human reviewed"


CODEX_COLUMNS = [
    "codex_gold_actionable_suggestion",
    "codex_lung_rads_category_suggestion",
    "codex_recommendation_level_suggestion",
    "codex_cdsg_recommendation_correct_suggestion",
    "codex_abstention_appropriate_suggestion",
    "codex_under_followup_risk_suggestion",
    "codex_over_followup_risk_suggestion",
    "codex_confidence",
    "codex_rationale",
    "codex_cited_evidence",
]

REVIEW_COLUMNS = [
    "evidence_has_nodule",
    "evidence_has_size",
    "evidence_has_density",
    "evidence_has_location",
    "size_text_present",
    "density_text_present",
    "codex_cited_evidence_exists",
    "codex_rationale_supported_by_evidence",
    "obvious_codex_error",
    "needs_clinical_expert_review",
    "non_clinical_reviewer_notes",
]


ANNOTATIONS: dict[str, dict[str, str]] = {
    "CASE-10001338-001": {
        "codex_gold_actionable_suggestion": "yes",
        "codex_lung_rads_category_suggestion": "2",
        "codex_recommendation_level_suggestion": "routine_screening",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "Evidence supports small pulmonary nodules below 4 mm with at least one calcified nodule, so routine category-2 style follow-up is supported as a Codex prelabel.",
        "codex_cited_evidence": "There are four, less than 4 mm pulmonary nodules, at least one of which is calcified; mm pulmonary nodules are not definitely calcified in the right upper lobe",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10001401-001": {
        "codex_gold_actionable_suggestion": "yes",
        "codex_lung_rads_category_suggestion": "4A",
        "codex_recommendation_level_suggestion": "short_interval_followup",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "low",
        "codex_confidence": "medium",
        "codex_rationale": "The 11 mm spiculated part-solid RUL nodule supports a suspicious short-interval prelabel, but high-risk density conflict and missing solid-component size require expert review.",
        "codex_cited_evidence": "Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter; conflict density_category high selected=part_solid alts=part_solid|solid",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10001919-001": {
        "codex_gold_actionable_suggestion": "yes",
        "codex_lung_rads_category_suggestion": "2",
        "codex_recommendation_level_suggestion": "routine_screening",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "low",
        "codex_confidence": "medium",
        "codex_rationale": "A small RLL nodule measuring 5 x 4 mm supports a routine low-risk prelabel, but the calcified density is not clearly tied to that same pulmonary nodule.",
        "codex_cited_evidence": "A right lower lobe nodule (3:207) now measures 5 x 4 mm, previously 7; partially peripherally calcified nodule in the right lobe measuring up to 2.5",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "uncertain",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10002155-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "uncertain",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "low",
        "codex_rationale": "The 11 x 11 mm enlarging mixed solid and ground-glass LUL opacity may not be a pure ground-glass nodule, so the exact category is uncertain and expert review is needed.",
        "codex_cited_evidence": "cluster of mixed solid and ground-glass opacity in the anterior segment of the left upper lobe (4A:101) is larger measuring 11 x 11 mm",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10052992-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "The case has new pulmonary nodules up to 1.8 cm but no density text, so insufficient-data abstention is appropriate while under-follow-up risk remains a concern.",
        "codex_cited_evidence": "Numerous pulmonary nodules are scattered lungs measuring up to 1.8 cm in the left upper lobe (6:69), all new since the prior chest CT; missing_info=density_category",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10048825-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "A 6 mm RLL pulmonary nodule is evidenced, but density is absent; density could change the path, so abstention and expert review are appropriate.",
        "codex_cited_evidence": "there is a 6 mm nodule in the right lower lobe (6, 211); missing_info=density_category",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10049041-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "uncertain",
        "codex_over_followup_risk_suggestion": "uncertain",
        "codex_confidence": "low",
        "codex_rationale": "The selected 3.3 cm pleural-based lesion and separate small nodules create uncertain nodule applicability and no usable density, so no category should be inferred.",
        "codex_cited_evidence": "There is a 2.8 x 3.3 cm intermediate density pleural based lesion at the right lung base; Additional small nodule measures 4 mm; missing_info=density_category",
        "evidence_has_nodule": "uncertain",
        "evidence_has_size": "yes",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10053082-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "uncertain",
        "codex_over_followup_risk_suggestion": "uncertain",
        "codex_confidence": "low",
        "codex_rationale": "The evidence centers on target lesions or a right infrahilar soft-tissue mass and also says no other lung masses or nodules, so Lung-RADS applicability is uncertain.",
        "codex_cited_evidence": "target lesion #1. The right infrahilar soft tissue mass measures 4.8 x 3.9 cm; There are no other lung masses or nodules; No lung lesion",
        "evidence_has_nodule": "uncertain",
        "evidence_has_size": "yes",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10036909-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "partially",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "high",
        "codex_rationale": "The only cited nodule evidence is adrenal rather than pulmonary, so no Lung-RADS category is applicable and abstention is appropriate.",
        "codex_cited_evidence": "right adrenal nodule is seen on a background of bilateral adrenal nodularity",
        "evidence_has_nodule": "no",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "no",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10006431-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "low",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "Pulmonary nodule evidence is present but only described as millimetric with no explicit size or density, so insufficient-data abstention is supported.",
        "codex_cited_evidence": "PARENCHYMA: Millimetric pulmonary nodule in the left lower lobe is; No new or growing pulmonary nodules",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10039272-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "low",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "Calcified pulmonary/granulomatous evidence is present but lacks usable size text, so a Lung-RADS category should not be inferred from size.",
        "codex_cited_evidence": "A larger, calcified granuloma is found in the right lower lobe; Geographic soft tissue lesion in the right upper lobe containing eccentric calcification",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "no",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "no",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10063856-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "low",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "A stable subpleural pulmonary nodule is mentioned, but no usable size or density text is present, so insufficient data is appropriate.",
        "codex_cited_evidence": "Subpleural nodule in the superior segment of the left lower lobe (03:21) is stable; no new lung nodules",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "yes",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10011365-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "The template has no structured nodule evidence for this case, so abstention is supported and no Lung-RADS category is applicable.",
        "codex_cited_evidence": "no_direct_evidence_found; abstention_reason=no_structured_nodule; missing_info=nodule|smoking_eligibility",
        "evidence_has_nodule": "no",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "no",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10041127-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "The template has no structured nodule evidence for this case, so abstention is supported and no Lung-RADS category is applicable.",
        "codex_cited_evidence": "no_direct_evidence_found; abstention_reason=no_structured_nodule; missing_info=nodule|smoking_eligibility",
        "evidence_has_nodule": "no",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "no",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10048899-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "The template has no structured nodule evidence for this case, so abstention is supported and no Lung-RADS category is applicable.",
        "codex_cited_evidence": "no_direct_evidence_found; abstention_reason=no_structured_nodule; missing_info=nodule|smoking_eligibility",
        "evidence_has_nodule": "no",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "no",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10054464-001": {
        "codex_gold_actionable_suggestion": "no",
        "codex_lung_rads_category_suggestion": "not_applicable",
        "codex_recommendation_level_suggestion": "insufficient_data",
        "codex_cdsg_recommendation_correct_suggestion": "yes",
        "codex_abstention_appropriate_suggestion": "yes",
        "codex_under_followup_risk_suggestion": "none",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "medium",
        "codex_rationale": "The template has no structured nodule evidence for this case, so abstention is supported and no Lung-RADS category is applicable.",
        "codex_cited_evidence": "no_direct_evidence_found; abstention_reason=no_structured_nodule; missing_info=nodule|smoking_eligibility",
        "evidence_has_nodule": "no",
        "evidence_has_size": "no",
        "evidence_has_density": "no",
        "evidence_has_location": "no",
        "size_text_present": "no",
        "density_text_present": "no",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "no",
    },
    "CASE-10042810-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "uncertain",
        "codex_cdsg_recommendation_correct_suggestion": "uncertain",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "high",
        "codex_over_followup_risk_suggestion": "none",
        "codex_confidence": "low",
        "codex_rationale": "A 2.5 x 1.4 cm enlarging LUL nodule with surrounding ground-glass is present, but high-risk density conflict means the category could be higher than the CDSG path.",
        "codex_cited_evidence": "left upper lobe, there is a nodule measuring 2.5 x 1.4 cm with surrounding ground-glass changes, increased in size; conflict density_category high selected=ground_glass alts=calcified|ground_glass",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10049330-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "uncertain",
        "codex_cdsg_recommendation_correct_suggestion": "no",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "low",
        "codex_confidence": "low",
        "codex_rationale": "The new 7 mm RLL nodule lacks direct calcified-density support, while other small solid/calcified nodules create a high-risk density conflict.",
        "codex_cited_evidence": "New right lower lobe 7 mm nodule (7, 200); Stable 3 mm solid right upper lobe nodule; conflict density_category high selected=calcified alts=calcified|solid",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "uncertain",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10064049-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "uncertain",
        "codex_cdsg_recommendation_correct_suggestion": "uncertain",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "medium",
        "codex_over_followup_risk_suggestion": "low",
        "codex_confidence": "low",
        "codex_rationale": "The evidence mixes a large RLL rounded opacity/nodule with ground-glass and calcified granuloma mentions, so density and dominant-lesion selection need expert review.",
        "codex_cited_evidence": "Rounded opacity in the right lower lobe has slightly decreased in size measuring 24 x 22 mm; Decreased size of right lower lobe nodule and surrounding ground-glass; conflict density_category high selected=ground_glass alts=calcified|ground_glass",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
    "CASE-10090755-001": {
        "codex_gold_actionable_suggestion": "uncertain",
        "codex_lung_rads_category_suggestion": "uncertain",
        "codex_recommendation_level_suggestion": "uncertain",
        "codex_cdsg_recommendation_correct_suggestion": "no",
        "codex_abstention_appropriate_suggestion": "not_applicable",
        "codex_under_followup_risk_suggestion": "uncertain",
        "codex_over_followup_risk_suggestion": "high",
        "codex_confidence": "low",
        "codex_rationale": "The 4.5 cm value appears to come from a posterior chest-wall fat-attenuation nodule rather than a pulmonary nodule, while pulmonary nodules are smaller and conflicted.",
        "codex_cited_evidence": "Fat attenuation nodule in the posterior chest wall, measuring 4.5 x 2.0 cm; Two solid 6 mm nodules stand out, in both lower lung bases; An 11 mm left upper lobe pulmonary nodule",
        "evidence_has_nodule": "yes",
        "evidence_has_size": "yes",
        "evidence_has_density": "yes",
        "evidence_has_location": "yes",
        "size_text_present": "yes",
        "density_text_present": "yes",
        "codex_cited_evidence_exists": "yes",
        "codex_rationale_supported_by_evidence": "yes",
        "obvious_codex_error": "no",
        "needs_clinical_expert_review": "yes",
    },
}


def _read_template() -> tuple[list[dict[str, str]], list[str]]:
    with TEMPLATE_PATH.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    FILLED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FILLED_CSV_PATH.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: list[dict[str, str]]) -> None:
    FILLED_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FILLED_JSONL_PATH.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _yes(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip().lower() in {"yes", "y", "true", "1"}


def _expert_review_cases(rows: list[dict[str, str]]) -> list[str]:
    return [row["case_id"] for row in rows if _yes(row, "needs_clinical_expert_review")]


def _grounding_stats(rows: list[dict[str, str]]) -> tuple[int, int, float]:
    denominator = sum(
        1
        for row in rows
        if row.get("codex_cited_evidence_exists", "").strip()
        and row.get("codex_rationale_supported_by_evidence", "").strip()
    )
    supported = sum(
        1
        for row in rows
        if _yes(row, "codex_cited_evidence_exists")
        and _yes(row, "codex_rationale_supported_by_evidence")
    )
    rate = supported / denominator if denominator else 0.0
    return supported, denominator, rate


def _recommend_expansion(rows: list[dict[str, str]]) -> tuple[str, str]:
    _, _, grounding_rate = _grounding_stats(rows)
    low_confidence = sum(1 for row in rows if row.get("codex_confidence") == "low")
    expert_review = sum(1 for row in rows if _yes(row, "needs_clinical_expert_review"))
    obvious_errors = sum(1 for row in rows if _yes(row, "obvious_codex_error"))
    if grounding_rate < 0.8:
        return "no", "evidence grounding support rate is below the pilot threshold"
    if obvious_errors > 2:
        return "no", "obvious Codex error count is above the pilot threshold"
    if low_confidence >= 6 or expert_review >= 8:
        return (
            "no",
            "many cases remain low-confidence or require clinical expert review, especially high-risk density conflicts",
        )
    return "yes", "pilot evidence grounding is complete enough for a broader failure-mode audit, not for performance evaluation"


def _write_self_check_report(rows: list[dict[str, str]]) -> None:
    SELF_CHECK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    confidence = Counter(row.get("codex_confidence", "empty") or "empty" for row in rows)
    expert_cases = _expert_review_cases(rows)
    obvious_errors = sum(1 for row in rows if _yes(row, "obvious_codex_error"))
    supported, denominator, rate = _grounding_stats(rows)
    expand, expand_reason = _recommend_expansion(rows)

    lines = [
        "# M3-3C Codex-Assisted Pilot 20 Self-Check Report",
        "",
        "## 定位",
        "",
        "- 本文件记录 Codex-assisted pre-annotation 和 model-assisted evidence audit 的自检结果。",
        "- 这些输出不是 clinical gold、不是 expert label、不是 manual gold benchmark。",
        "- 没有医学专家参与；`non_clinical_reviewer_notes` 均为 Codex self-check; not human reviewed。",
        "- 结果只能用于 failure-mode audit，不能用于 learned-model performance experiment。",
        "",
        "## 完成情况",
        "",
        f"- 输入模板：`{TEMPLATE_PATH}`",
        f"- filled CSV：`{FILLED_CSV_PATH}`",
        f"- filled JSONL：`{FILLED_JSONL_PATH}`",
        f"- pilot 行数：{len(rows)}",
        f"- Codex suggestion 完成行数：{sum(1 for row in rows if all(row.get(col, '').strip() for col in CODEX_COLUMNS))}",
        f"- evidence self-check 完成行数：{sum(1 for row in rows if all(row.get(col, '').strip() for col in REVIEW_COLUMNS))}",
        "",
        "## Confidence 分布",
        "",
    ]
    for label in ["high", "medium", "low"]:
        lines.append(f"- {label}: {confidence.get(label, 0)}")

    lines.extend(
        [
            "",
            "## Evidence Grounding",
            "",
            f"- evidence grounding support rate：{supported}/{denominator} ({rate:.6f})",
            f"- obvious Codex error：{obvious_errors}",
            "",
            "## 需要 Clinical Expert Review 的 Case",
            "",
        ]
    )
    if expert_cases:
        lines.extend(f"- {case_id}" for case_id in expert_cases)
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 是否建议扩展到完整 133 Cases",
            "",
            f"- 建议：{expand}",
            f"- 原因：{expand_reason}",
            "",
            "## Learned-Model Experiment 状态",
            "",
            "- 当前仍不能进入 learned-model performance experiment。",
            "- 原因：pilot 20 仍包含 low-confidence case、high-risk density conflict 和需要 clinical expert review 的 case；这些 prelabels 不能作为 gold benchmark。",
        ]
    )
    SELF_CHECK_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fill_rows() -> list[dict[str, str]]:
    rows, fieldnames = _read_template()
    case_ids = {row["case_id"] for row in rows}
    missing = sorted(set(ANNOTATIONS) - case_ids)
    extra = sorted(case_ids - set(ANNOTATIONS))
    if missing or extra:
        raise RuntimeError(f"annotation/template case mismatch: missing={missing}; extra={extra}")

    for row in rows:
        values = ANNOTATIONS[row["case_id"]]
        for col in CODEX_COLUMNS + REVIEW_COLUMNS:
            if col == "non_clinical_reviewer_notes":
                row[col] = NOTE
            else:
                row[col] = values[col]

    _write_csv(rows, fieldnames)
    _write_jsonl(rows)
    _write_self_check_report(rows)
    return rows


def main() -> int:
    rows = fill_rows()
    supported, denominator, rate = _grounding_stats(rows)
    result: dict[str, Any] = {
        "filled_csv": str(FILLED_CSV_PATH),
        "filled_jsonl": str(FILLED_JSONL_PATH),
        "self_check_report": str(SELF_CHECK_REPORT_PATH),
        "rows": len(rows),
        "confidence_distribution": dict(Counter(row["codex_confidence"] for row in rows)),
        "needs_clinical_expert_review": len(_expert_review_cases(rows)),
        "obvious_codex_error": sum(1 for row in rows if _yes(row, "obvious_codex_error")),
        "evidence_grounding_support": {"count": supported, "denominator": denominator, "rate": rate},
        "recommend_expand_to_133_cases": _recommend_expansion(rows)[0],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
