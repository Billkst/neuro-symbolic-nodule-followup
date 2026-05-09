#!/usr/bin/env python3
"""Audit Module 3 abstentions against Module 2 prediction evidence.

This script is diagnostic only. It does not modify case bundles or generate new
labels. The goal is to explain why CDSG still abstains after anchor-based
Module 2 predictions have been exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_DENSITIES = {"solid", "part_solid", "ground_glass", "calcified", "fat_containing"}
VALID_LOCATIONS = {"RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral"}
PULMONARY_CUE_RE = re.compile(
    r"\b(lung|lungs|pulmonary|nodule|nodules|lobe|lobes|lingula|pleural|ground[- ]glass|ggo)\b",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_density(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "partsolid": "part_solid",
        "groundglass": "ground_glass",
        "ggo": "ground_glass",
        "fat": "fat_containing",
    }
    return aliases.get(text, text)


def _is_valid_density(value: Any) -> bool:
    return _normalize_density(value) in VALID_DENSITIES


def _is_valid_location(value: Any) -> bool:
    return value in VALID_LOCATIONS


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _mention_key(row: dict[str, Any]) -> str | None:
    value = row.get("sample_id") or row.get("mention_id")
    return str(value) if value not in {None, ""} else None


def _load_prediction_mentions(prediction_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    files = {
        "density_stage1": prediction_dir / "module2_density_stage1_predictions.jsonl",
        "density_stage2": prediction_dir / "module2_density_stage2_predictions.jsonl",
        "size": prediction_dir / "module2_size_predictions.jsonl",
        "location": prediction_dir / "module2_location_predictions.jsonl",
    }
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for task, path in files.items():
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            case_id = row.get("case_id")
            key = _mention_key(row)
            if not case_id or not key:
                continue
            mention = by_case[str(case_id)].setdefault(
                key,
                {
                    "case_id": str(case_id),
                    "mention_id": row.get("mention_id") or key,
                    "sample_id": row.get("sample_id") or key,
                    "note_id": row.get("note_id") or row.get("report_id"),
                    "mention_text": row.get("mention_text"),
                    "tasks": {},
                },
            )
            mention["tasks"][task] = row
    return by_case


def _flatten_nodules(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_idx, fact in enumerate(bundle.get("radiology_facts", []) or []):
        note_id = fact.get("note_id")
        for nodule_idx, nodule in enumerate(fact.get("nodules", []) or [], start=1):
            source_ids = nodule.get("source_mention_ids")
            if not source_ids and nodule.get("mention_id"):
                source_ids = [nodule.get("mention_id")]
            rows.append(
                {
                    "case_id": bundle.get("case_id"),
                    "note_id": note_id,
                    "fact_index": fact_idx,
                    "nodule_index": nodule_idx,
                    "nodule": nodule,
                    "mention_id": nodule.get("mention_id"),
                    "source_mention_ids": [str(item) for item in (source_ids or [])],
                }
            )
    for nodule_idx, nodule in enumerate(bundle.get("nodules", []) or [], start=1):
        rows.append(
            {
                "case_id": bundle.get("case_id"),
                "note_id": None,
                "fact_index": None,
                "nodule_index": nodule_idx,
                "nodule": nodule,
                "mention_id": nodule.get("mention_id"),
                "source_mention_ids": [str(nodule.get("mention_id"))] if nodule.get("mention_id") else [],
            }
        )
    return rows


def _candidate_counts(nodules: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids_with_density: set[str] = set()
    candidate_ids_with_size: set[str] = set()
    candidate_ids_with_location: set[str] = set()
    candidate_ids_with_both_size_density: set[str] = set()
    source_ids: set[str] = set()
    source_ids_with_density: set[str] = set()
    source_ids_with_size: set[str] = set()
    source_ids_with_location: set[str] = set()

    for item in nodules:
        nodule = item["nodule"]
        candidate_id = str(nodule.get("module2_candidate_id") or nodule.get("nodule_id_in_report"))
        for source_id in item.get("source_mention_ids") or []:
            source_ids.add(source_id)

        has_density = _is_valid_density(nodule.get("density_category"))
        has_size = _as_float(nodule.get("size_mm")) is not None
        has_location = _is_valid_location(nodule.get("location_lobe"))

        if has_density:
            candidate_ids_with_density.add(candidate_id)
            source_ids_with_density.update(item.get("source_mention_ids") or [])
        if has_size:
            candidate_ids_with_size.add(candidate_id)
            source_ids_with_size.update(item.get("source_mention_ids") or [])
        if has_location:
            candidate_ids_with_location.add(candidate_id)
            source_ids_with_location.update(item.get("source_mention_ids") or [])
        if has_size and has_density:
            candidate_ids_with_both_size_density.add(candidate_id)

    return {
        "candidate_count": len(nodules),
        "candidate_with_density": len(candidate_ids_with_density),
        "candidate_with_size": len(candidate_ids_with_size),
        "candidate_with_location": len(candidate_ids_with_location),
        "candidate_with_both_size_density": len(candidate_ids_with_both_size_density),
        "candidate_with_size_without_density": len(candidate_ids_with_size - candidate_ids_with_both_size_density),
        "candidate_with_density_without_size": len(candidate_ids_with_density - candidate_ids_with_both_size_density),
        "candidate_source_mention_ids": source_ids,
        "candidate_source_ids_with_density": source_ids_with_density,
        "candidate_source_ids_with_size": source_ids_with_size,
        "candidate_source_ids_with_location": source_ids_with_location,
    }


def _prediction_counts(mentions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stage1_explicit: list[str] = []
    stage2_subtype: list[str] = []
    stage2_applicable: list[str] = []
    size_mm: list[str] = []
    location: list[str] = []
    pulmonary_cues: list[str] = []
    stage2_labels: list[str] = []

    for key, mention in mentions.items():
        tasks = mention.get("tasks") or {}
        stage1 = tasks.get("density_stage1") or {}
        stage2 = tasks.get("density_stage2") or {}
        size = tasks.get("size") or {}
        loc = tasks.get("location") or {}

        if stage1.get("predicted_label") == "explicit_density":
            stage1_explicit.append(key)
        if _is_valid_density(stage2.get("predicted_label")):
            stage2_subtype.append(key)
            stage2_labels.append(str(stage2.get("predicted_label")))
        if _bool_value(stage2.get("stage2_applicable")):
            stage2_applicable.append(key)
        if _as_float(size.get("size_mm")) is not None:
            size_mm.append(key)
        if _is_valid_location(loc.get("predicted_label")):
            location.append(key)
        if PULMONARY_CUE_RE.search(str(mention.get("mention_text") or "")):
            pulmonary_cues.append(key)

    return {
        "prediction_mentions": len(mentions),
        "stage1_explicit_density_mentions": len(stage1_explicit),
        "stage2_subtype_mentions": len(stage2_subtype),
        "stage2_applicable_true_mentions": len(stage2_applicable),
        "stage2_subtype_but_not_applicable_mentions": len(set(stage2_subtype) - set(stage2_applicable)),
        "size_mm_prediction_mentions": len(size_mm),
        "location_prediction_mentions": len(location),
        "pulmonary_cue_prediction_mentions": len(pulmonary_cues),
        "stage1_explicit_ids": stage1_explicit,
        "stage2_subtype_ids": stage2_subtype,
        "stage2_applicable_ids": stage2_applicable,
        "size_mm_ids": size_mm,
        "location_ids": location,
        "pulmonary_cue_ids": pulmonary_cues,
        "stage2_label_distribution": dict(Counter(stage2_labels)),
    }


def _selected_input_facts(strong_record: dict[str, Any]) -> dict[str, Any]:
    recommendation_object = strong_record.get("recommendation_object") or {}
    return recommendation_object.get("input_facts_used") or {}


def _first_values(values: list[str] | set[str], limit: int = 5) -> str:
    return "|".join(sorted(str(value) for value in values)[:limit])


def _density_root_cause(pred: dict[str, Any], cand: dict[str, Any], selected_density: Any) -> str:
    if pred["stage1_explicit_density_mentions"] == 0:
        if pred["stage2_subtype_mentions"] > 0:
            return "stage1_no_explicit_density_stage2_subtypes_not_applicable"
        return "no_explicit_density_prediction"
    if cand["candidate_with_density"] == 0:
        return "explicit_density_prediction_not_written_to_candidate"
    if cand["candidate_with_both_size_density"] == 0 and cand["candidate_with_size"] > 0:
        return "size_and_density_split_across_candidates"
    if selected_density in {None, "", "unclear"}:
        return "dominant_selection_or_graph_routing_selected_unclear_density"
    return "other_density_abstention"


def _size_root_cause(pred: dict[str, Any], cand: dict[str, Any], selected_size: Any) -> str:
    if pred["size_mm_prediction_mentions"] == 0:
        return "no_size_mm_prediction"
    if cand["candidate_with_size"] == 0:
        return "size_mm_prediction_not_written_to_candidate"
    if cand["candidate_with_both_size_density"] == 0 and cand["candidate_with_density"] > 0:
        return "size_and_density_split_across_candidates"
    if _as_float(selected_size) is None:
        return "dominant_selection_or_graph_routing_selected_missing_size"
    return "other_size_abstention"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Module 3 abstentions against Module 2 predictions.")
    parser.add_argument("--case-bundles", default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl")
    parser.add_argument("--strong-silver", default="outputs/phaseA3/datasets/module3_strong_silver_v2.jsonl")
    parser.add_argument("--prediction-dir", default="outputs/phaseA3/module2_predictions")
    parser.add_argument("--case-audit", default="outputs/phaseA3/tables/module3_abstention_case_audit.csv")
    parser.add_argument("--missing-density-audit", default="outputs/phaseA3/tables/module3_missing_density_audit.csv")
    parser.add_argument("--missing-size-audit", default="outputs/phaseA3/tables/module3_missing_size_audit.csv")
    parser.add_argument("--no-structured-nodule-audit", default="outputs/phaseA3/tables/module3_no_structured_nodule_audit.csv")
    args = parser.parse_args()

    bundles = {str(row.get("case_id")): row for row in _load_jsonl(Path(args.case_bundles))}
    strong_rows = _load_jsonl(Path(args.strong_silver))
    predictions_by_case = _load_prediction_mentions(Path(args.prediction_dir))

    case_rows: list[dict[str, Any]] = []
    missing_density_rows: list[dict[str, Any]] = []
    missing_size_rows: list[dict[str, Any]] = []
    no_nodule_rows: list[dict[str, Any]] = []
    root_counter: Counter[str] = Counter()
    abstention_counter: Counter[str] = Counter()

    for strong in strong_rows:
        case_id = str(strong.get("case_id"))
        bundle = bundles.get(case_id) or strong.get("case_bundle") or {}
        abstention_reason = strong.get("abstention_reason")
        abstention_counter[str(abstention_reason)] += 1
        nodules = _flatten_nodules(bundle)
        cand = _candidate_counts(nodules)
        pred = _prediction_counts(predictions_by_case.get(case_id, {}))
        input_facts = _selected_input_facts(strong)
        selected_density = input_facts.get("nodule_density")
        selected_size = input_facts.get("nodule_size_mm")
        selected_note_id = input_facts.get("note_id")

        density_stage2_without_adapter_density = (
            pred["stage2_subtype_mentions"] > 0 and cand["candidate_with_density"] == 0
        )
        stage2_filtered_by_applicability = (
            pred["stage2_subtype_mentions"] > 0
            and pred["stage2_applicable_true_mentions"] == 0
            and pred["stage1_explicit_density_mentions"] == 0
        )
        candidate_selection_missed_density = (
            abstention_reason == "missing_nodule_density"
            and cand["candidate_with_density"] > 0
            and selected_density in {None, "", "unclear"}
        )
        candidate_selection_missed_size = (
            abstention_reason == "missing_nodule_size"
            and cand["candidate_with_size"] > 0
            and _as_float(selected_size) is None
        )
        size_prediction_not_in_candidate = len(set(pred["size_mm_ids"]) - set(cand["candidate_source_ids_with_size"]))
        density_prediction_not_in_candidate = len(set(pred["stage1_explicit_ids"]) - set(cand["candidate_source_ids_with_density"]))

        density_root = ""
        size_root = ""
        no_nodule_root = ""
        if abstention_reason == "missing_nodule_density":
            density_root = _density_root_cause(pred, cand, selected_density)
            root_counter[density_root] += 1
        elif abstention_reason == "missing_nodule_size":
            size_root = _size_root_cause(pred, cand, selected_size)
            root_counter[size_root] += 1
        elif abstention_reason == "no_structured_nodule":
            no_nodule_root = (
                "pulmonary_cue_prediction_exists_but_no_structured_candidate"
                if pred["pulmonary_cue_prediction_mentions"] > 0
                else "no_phase4_aligned_pulmonary_prediction_cue"
            )
            root_counter[no_nodule_root] += 1

        base = {
            "case_id": case_id,
            "abstention_reason": abstention_reason,
            "missing_information": "|".join(str(item) for item in (strong.get("missing_information") or [])),
            "recommendation_level": strong.get("recommendation_level"),
            "selected_size_mm": selected_size,
            "selected_density": selected_density,
            "selected_note_id": selected_note_id,
            "candidate_count": cand["candidate_count"],
            "candidate_with_density": cand["candidate_with_density"],
            "candidate_with_size": cand["candidate_with_size"],
            "candidate_with_location": cand["candidate_with_location"],
            "candidate_with_both_size_density": cand["candidate_with_both_size_density"],
            "candidate_with_size_without_density": cand["candidate_with_size_without_density"],
            "candidate_with_density_without_size": cand["candidate_with_density_without_size"],
            "prediction_mentions": pred["prediction_mentions"],
            "stage1_explicit_density_mentions": pred["stage1_explicit_density_mentions"],
            "stage2_subtype_mentions": pred["stage2_subtype_mentions"],
            "stage2_applicable_true_mentions": pred["stage2_applicable_true_mentions"],
            "stage2_subtype_but_not_applicable_mentions": pred["stage2_subtype_but_not_applicable_mentions"],
            "size_mm_prediction_mentions": pred["size_mm_prediction_mentions"],
            "location_prediction_mentions": pred["location_prediction_mentions"],
            "pulmonary_cue_prediction_mentions": pred["pulmonary_cue_prediction_mentions"],
            "density_stage2_without_adapter_density": density_stage2_without_adapter_density,
            "stage2_filtered_by_applicability": stage2_filtered_by_applicability,
            "candidate_selection_missed_density": candidate_selection_missed_density,
            "candidate_selection_missed_size": candidate_selection_missed_size,
            "size_prediction_not_in_candidate_mentions": size_prediction_not_in_candidate,
            "density_prediction_not_in_candidate_mentions": density_prediction_not_in_candidate,
            "density_root_cause": density_root,
            "size_root_cause": size_root,
            "no_structured_nodule_root_cause": no_nodule_root,
            "example_stage1_explicit_ids": _first_values(pred["stage1_explicit_ids"]),
            "example_stage2_subtype_ids": _first_values(pred["stage2_subtype_ids"]),
            "example_size_mm_ids": _first_values(pred["size_mm_ids"]),
            "example_pulmonary_cue_ids": _first_values(pred["pulmonary_cue_ids"]),
            "stage2_label_distribution": json.dumps(pred["stage2_label_distribution"], ensure_ascii=False, sort_keys=True),
        }
        case_rows.append(base)
        if abstention_reason == "missing_nodule_density":
            missing_density_rows.append(base)
        elif abstention_reason == "missing_nodule_size":
            missing_size_rows.append(base)
        elif abstention_reason == "no_structured_nodule":
            no_nodule_rows.append(base)

    fieldnames = [
        "case_id",
        "abstention_reason",
        "missing_information",
        "recommendation_level",
        "selected_size_mm",
        "selected_density",
        "selected_note_id",
        "candidate_count",
        "candidate_with_density",
        "candidate_with_size",
        "candidate_with_location",
        "candidate_with_both_size_density",
        "candidate_with_size_without_density",
        "candidate_with_density_without_size",
        "prediction_mentions",
        "stage1_explicit_density_mentions",
        "stage2_subtype_mentions",
        "stage2_applicable_true_mentions",
        "stage2_subtype_but_not_applicable_mentions",
        "size_mm_prediction_mentions",
        "location_prediction_mentions",
        "pulmonary_cue_prediction_mentions",
        "density_stage2_without_adapter_density",
        "stage2_filtered_by_applicability",
        "candidate_selection_missed_density",
        "candidate_selection_missed_size",
        "size_prediction_not_in_candidate_mentions",
        "density_prediction_not_in_candidate_mentions",
        "density_root_cause",
        "size_root_cause",
        "no_structured_nodule_root_cause",
        "example_stage1_explicit_ids",
        "example_stage2_subtype_ids",
        "example_size_mm_ids",
        "example_pulmonary_cue_ids",
        "stage2_label_distribution",
    ]
    _write_csv(Path(args.case_audit), case_rows, fieldnames)
    _write_csv(Path(args.missing_density_audit), missing_density_rows, fieldnames)
    _write_csv(Path(args.missing_size_audit), missing_size_rows, fieldnames)
    _write_csv(Path(args.no_structured_nodule_audit), no_nodule_rows, fieldnames)

    print(
        json.dumps(
            {
                "cases": len(case_rows),
                "missing_density_cases": len(missing_density_rows),
                "missing_size_cases": len(missing_size_rows),
                "no_structured_nodule_cases": len(no_nodule_rows),
                "abstention_distribution": dict(abstention_counter),
                "root_cause_distribution": dict(root_counter),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
