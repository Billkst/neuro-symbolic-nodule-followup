#!/usr/bin/env python3
"""Adapt exported Module 2 predictions into Module 3 case bundles.

Version 2 consumes the unified prediction JSONL files produced by
export_module2_predictions_for_module3.py. It preserves provenance, records
conflicts, and appends candidate nodules without overwriting original Phase4
facts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


VALID_DENSITIES = {"solid", "part_solid", "ground_glass", "calcified", "fat_containing"}
VALID_LOCATIONS = {"RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral"}
NON_PULMONARY_TERMS = {
    "adrenal",
    "hepatic",
    "liver",
    "renal",
    "kidney",
    "spleen",
    "splenic",
    "thyroid",
    "breast",
    "pancreas",
    "pancreatic",
    "lymph node",
    "lymph nodes",
    "mediastinal node",
    "hilar node",
    "paratracheal",
    "osseous",
    "bone",
    "rib lesion",
}
PULMONARY_TERMS = {
    "lung",
    "lungs",
    "pulmonary",
    "nodule",
    "nodules",
    "granuloma",
    "lobe",
    "lobes",
    "lingula",
    "pleural",
    "ground-glass",
    "ground glass",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
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


def _confidence_label(value: float | None, source_type: str | None = None) -> str:
    if value is None:
        return "low" if source_type == "constructed_fact_not_final_model" else "unknown"
    if value >= 0.9:
        return "high"
    if value >= 0.7:
        return "medium"
    return "low"


def _is_reliably_pulmonary_mention(text: str) -> tuple[bool, str | None]:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False, "empty_mention_text"
    has_pulmonary = any(term in normalized for term in PULMONARY_TERMS)
    has_non_pulmonary = any(term in normalized for term in NON_PULMONARY_TERMS)
    if has_non_pulmonary and not re.search(r"\b(lung|pulmonary|lobe|lingula|pleural|ground[- ]glass)\b", normalized):
        return False, "non_pulmonary_mention_filtered"
    if not has_pulmonary:
        return False, "no_pulmonary_nodule_cue"
    return True, None


def _fact_source(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record.get("prediction_source_type"),
        "confidence": _confidence_label(_as_float(record.get("confidence")), record.get("prediction_source_type")),
        "confidence_value": _as_float(record.get("confidence")),
        "model_tag": record.get("model_tag"),
        "mention_id": record.get("mention_id") or record.get("sample_id"),
        "sample_id": record.get("sample_id"),
        "original_text": record.get("mention_text"),
        "note_id": record.get("note_id"),
        "source_split": record.get("source_split"),
        "source_path": record.get("source_path"),
        "gold_or_constructed_label": record.get("gold_or_constructed_label"),
        "failure_reason": record.get("failure_reason"),
    }


def _record_confidence(record: dict[str, Any] | None) -> float:
    if not record:
        return 0.0
    value = _as_float(record.get("confidence"))
    if value is None:
        value = _as_float(record.get("confidence_value"))
    if value is not None:
        return value
    if record.get("prediction_source_type") == "constructed_fact_not_final_model" or record.get("source") == "constructed_fact_not_final_model":
        return 0.5
    return 0.0


def _note_to_case(bundles: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bundle in bundles:
        case_id = str(bundle.get("case_id"))
        for fact in bundle.get("radiology_facts", []) or []:
            note_id = fact.get("note_id")
            if note_id:
                mapping[str(note_id)] = case_id
    return mapping


def _merge_prediction_records(
    prediction_paths: list[Path],
    note_to_case: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    mentions: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for path in prediction_paths:
        task_rows = _load_jsonl(path)
        counters[f"prediction_rows.{path.name}"] += len(task_rows)
        for record in task_rows:
            note_id = record.get("note_id") or record.get("report_id")
            sample_id = record.get("sample_id") or record.get("mention_id")
            task = record.get("task")
            if not note_id:
                unmatched.append(
                    {
                        "case_id": record.get("case_id"),
                        "note_id": None,
                        "sample_id": sample_id,
                        "task": task,
                        "reason": "missing_note_id",
                        "mention_text": record.get("mention_text"),
                        "source_prediction_file": str(path),
                    }
                )
                counters["unmatched.missing_note_id"] += 1
                continue
            if str(note_id) not in note_to_case:
                unmatched.append(
                    {
                        "case_id": record.get("case_id"),
                        "note_id": note_id,
                        "sample_id": sample_id,
                        "task": task,
                        "reason": "note_id_not_in_phase4_case_bundle",
                        "mention_text": record.get("mention_text"),
                        "source_prediction_file": str(path),
                    }
                )
                counters["unmatched.note_id_not_in_phase4_case_bundle"] += 1
                continue
            if not sample_id:
                unmatched.append(
                    {
                        "case_id": note_to_case[str(note_id)],
                        "note_id": note_id,
                        "sample_id": None,
                        "task": task,
                        "reason": "missing_sample_or_mention_id",
                        "mention_text": record.get("mention_text"),
                        "source_prediction_file": str(path),
                    }
                )
                counters["unmatched.missing_sample_or_mention_id"] += 1
                continue

            mention = mentions.setdefault(
                str(sample_id),
                {
                    "sample_id": str(sample_id),
                    "mention_id": record.get("mention_id") or sample_id,
                    "case_id": note_to_case[str(note_id)],
                    "note_id": str(note_id),
                    "subject_id": record.get("subject_id"),
                    "mention_text": record.get("mention_text"),
                    "tasks": {},
                },
            )
            mention["tasks"][str(task)] = record
            counters[f"aligned.{task}"] += 1

    counters["phase4_aligned_mentions"] = len(mentions)
    return mentions, unmatched, counters


def _facts_from_mention(mention: dict[str, Any]) -> dict[str, Any]:
    tasks = mention.get("tasks") or {}
    stage1 = tasks.get("density_stage1")
    stage2 = tasks.get("density_stage2")
    size = tasks.get("size")
    location = tasks.get("location")

    density = None
    if stage1 and stage1.get("predicted_label") == "explicit_density" and stage2:
        candidate_density = _normalize_density(stage2.get("predicted_label"))
        if candidate_density in VALID_DENSITIES:
            density = candidate_density

    has_size = bool(size and size.get("predicted_label") == "has_size")
    size_mm = _as_float(size.get("size_mm")) if size and has_size else None
    location_label = location.get("predicted_label") if location else None
    location_lobe = location_label if location_label in VALID_LOCATIONS else None

    sources = {}
    if density and stage2:
        sources["density_category"] = _fact_source(stage2)
    if size:
        sources["has_size"] = _fact_source(size)
        if size_mm is not None:
            sources["size_mm"] = _fact_source(size)
    if location_lobe and location:
        sources["location_lobe"] = _fact_source(location)

    confidence_values = [_record_confidence(record) for record in [stage1, stage2, size, location] if record]
    confidence_value = max(confidence_values) if confidence_values else None

    return {
        "size_mm": size_mm,
        "size_text": size.get("size_text") if size and size_mm is not None else None,
        "has_size": has_size if size else None,
        "density_category": density,
        "density_text": density,
        "location_lobe": location_lobe,
        "location_text": location_lobe,
        "confidence_value": confidence_value,
        "confidence": _confidence_label(confidence_value),
        "fact_sources": sources,
    }


def _candidate_from_facts(
    *,
    facts: dict[str, Any],
    mention: dict[str, Any],
    rank: int,
    candidate_kind: str,
    evidence_span: str | None = None,
    source_mentions: list[str] | None = None,
) -> dict[str, Any] | None:
    size_mm = facts.get("size_mm")
    density = _normalize_density(facts.get("density_category"))
    location = facts.get("location_lobe")
    has_any = size_mm is not None or density in VALID_DENSITIES or location in VALID_LOCATIONS
    if not has_any:
        return None

    missing_flags: list[str] = []
    if size_mm is None:
        missing_flags.extend(["size_mm", "size_text"])
    if density not in VALID_DENSITIES:
        missing_flags.extend(["density_category", "density_text"])
    if location not in VALID_LOCATIONS:
        missing_flags.extend(["location_lobe", "location_text"])

    candidate_id = f"module2_v2:{candidate_kind}:{mention.get('sample_id')}"
    return {
        "nodule_id_in_report": candidate_id,
        "module2_candidate_id": candidate_id,
        "mention_id": mention.get("mention_id") or mention.get("sample_id"),
        "source_mention_ids": source_mentions or [str(mention.get("mention_id") or mention.get("sample_id"))],
        "size_mm": size_mm,
        "size_text": facts.get("size_text"),
        "has_size": facts.get("has_size"),
        "density_category": density if density in VALID_DENSITIES else "unclear",
        "density_text": facts.get("density_text") if density in VALID_DENSITIES else None,
        "location_lobe": location if location in VALID_LOCATIONS else None,
        "location_text": facts.get("location_text") if location in VALID_LOCATIONS else None,
        "count_type": "candidate",
        "change_status": None,
        "change_text": None,
        "calcification": density == "calcified",
        "spiculation": False,
        "lobulation": False,
        "cavitation": False,
        "perifissural": False,
        "lung_rads_category": None,
        "recommendation_cue": None,
        "evidence_span": evidence_span or mention.get("mention_text"),
        "confidence": facts.get("confidence"),
        "missing_flags": missing_flags,
        "source": "module2_to_case_bundle_adapter_v2",
        "source_confidence": facts.get("confidence_value"),
        "fact_sources": facts.get("fact_sources") or {},
        "dominant_candidate_rank": rank,
        "dominant_selection_fields": {
            "size_mm": size_mm,
            "density_category": density,
            "location_lobe": location,
        },
        "adapter_notes": {
            "adapter_version": "module3_m2_adapter_v2",
            "candidate_kind": candidate_kind,
            "no_deterministic_rule_override": True,
            "no_free_text_value_generation": True,
        },
    }


def _candidate_score(candidate: dict[str, Any]) -> tuple[int, float, float, str]:
    has_density = 1 if candidate.get("density_category") not in {None, "unclear"} else 0
    size = _as_float(candidate.get("size_mm")) or -1.0
    confidence = _as_float(candidate.get("source_confidence")) or 0.0
    return (has_density, size, confidence, str(candidate.get("module2_candidate_id")))


def _select_value(
    *,
    field: str,
    candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]],
    case_id: str,
    note_id: str,
    conflicts: list[dict[str, Any]],
    preferred_mention_id: str | None,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
    usable = [(value, mention, source) for value, mention, source in candidates if value not in {None, "", "unclear"}]
    if not usable:
        return None, None, None

    values = {str(value) for value, _, _ in usable}
    if preferred_mention_id:
        preferred = [
            item
            for item in usable
            if str(item[1].get("mention_id") or item[1].get("sample_id")) == str(preferred_mention_id)
        ]
        if preferred:
            selected = sorted(preferred, key=lambda item: _record_confidence(item[2]), reverse=True)[0]
        else:
            selected = None
    else:
        selected = None

    if selected is None:
        if field == "size_mm":
            selected = sorted(usable, key=lambda item: (_as_float(item[0]) or -1.0, _record_confidence(item[2])), reverse=True)[0]
        else:
            selected = sorted(usable, key=lambda item: (_record_confidence(item[2]), str(item[0])), reverse=True)[0]

    if len(values) > 1:
        conflicts.append(
            {
                "case_id": case_id,
                "note_id": note_id,
                "field": field,
                "selected_value": selected[0],
                "alternative_values": "|".join(sorted(values)),
                "source_mention_ids": "|".join(
                    sorted(str(mention.get("mention_id") or mention.get("sample_id")) for _, mention, _ in usable)
                ),
                "resolution": "recorded_conflict_selected_dominant_candidate_value",
            }
        )
    return selected


def _aggregate_candidate_for_note(
    *,
    case_id: str,
    note_id: str,
    mentions: list[dict[str, Any]],
    rank: int,
    conflicts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    size_candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    density_candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    location_candidates: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    facts_by_sample: dict[str, dict[str, Any]] = {}
    pulmonary_mentions: list[dict[str, Any]] = []

    for mention in mentions:
        ok, _ = _is_reliably_pulmonary_mention(str(mention.get("mention_text") or ""))
        if not ok:
            continue
        pulmonary_mentions.append(mention)
        facts = _facts_from_mention(mention)
        facts_by_sample[str(mention.get("sample_id"))] = facts
        if facts.get("size_mm") is not None:
            source = (facts.get("fact_sources") or {}).get("size_mm") or {}
            size_candidates.append((facts["size_mm"], mention, source))
        density = facts.get("density_category")
        if density in VALID_DENSITIES:
            source = (facts.get("fact_sources") or {}).get("density_category") or {}
            density_candidates.append((density, mention, source))
        location = facts.get("location_lobe")
        if location in VALID_LOCATIONS:
            source = (facts.get("fact_sources") or {}).get("location_lobe") or {}
            location_candidates.append((location, mention, source))

    if not pulmonary_mentions:
        return None

    selected_size, size_mention, size_source = _select_value(
        field="size_mm",
        candidates=size_candidates,
        case_id=case_id,
        note_id=note_id,
        conflicts=conflicts,
        preferred_mention_id=None,
    )
    preferred = str(size_mention.get("mention_id") or size_mention.get("sample_id")) if size_mention else None
    selected_density, density_mention, density_source = _select_value(
        field="density_category",
        candidates=density_candidates,
        case_id=case_id,
        note_id=note_id,
        conflicts=conflicts,
        preferred_mention_id=preferred,
    )
    selected_location, location_mention, location_source = _select_value(
        field="location_lobe",
        candidates=location_candidates,
        case_id=case_id,
        note_id=note_id,
        conflicts=conflicts,
        preferred_mention_id=preferred,
    )

    source_mentions = [
        str(mention.get("mention_id") or mention.get("sample_id"))
        for mention in [size_mention, density_mention, location_mention]
        if mention
    ]
    if len(set(source_mentions)) <= 1:
        return None

    fact_sources: dict[str, Any] = {}
    if size_source:
        fact_sources["size_mm"] = size_source
        fact_sources["has_size"] = size_source
    if density_source:
        fact_sources["density_category"] = density_source
    if location_source:
        fact_sources["location_lobe"] = location_source
    confidence_values = [_record_confidence(source) for source in [size_source, density_source, location_source] if source]
    confidence = max(confidence_values) if confidence_values else None
    evidence_parts = [
        str(mention.get("mention_text") or "")
        for mention in [size_mention, density_mention, location_mention]
        if mention and mention.get("mention_text")
    ]
    facts = {
        "size_mm": selected_size,
        "size_text": size_source.get("gold_or_constructed_label") if size_source else None,
        "has_size": selected_size is not None,
        "density_category": selected_density,
        "density_text": selected_density,
        "location_lobe": selected_location,
        "location_text": selected_location,
        "confidence_value": confidence,
        "confidence": _confidence_label(confidence),
        "fact_sources": fact_sources,
    }
    aggregate_mention = {
        "sample_id": f"{note_id}:aggregate",
        "mention_id": f"{note_id}:aggregate",
        "note_id": note_id,
        "case_id": case_id,
        "mention_text": " | ".join(evidence_parts),
    }
    return _candidate_from_facts(
        facts=facts,
        mention=aggregate_mention,
        rank=rank,
        candidate_kind="note_aggregate",
        evidence_span=" | ".join(evidence_parts),
        source_mentions=sorted(set(source_mentions)),
    )


def _append_candidates(
    *,
    bundle: dict[str, Any],
    mentions_by_note: dict[str, list[dict[str, Any]]],
    conflicts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    adapted = deepcopy(bundle)
    case_id = str(adapted.get("case_id"))
    unmatched: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for fact in adapted.get("radiology_facts", []) or []:
        note_id = str(fact.get("note_id"))
        mentions = list(mentions_by_note.get(note_id, []))
        candidates: list[dict[str, Any]] = []
        for mention in mentions:
            ok, reason = _is_reliably_pulmonary_mention(str(mention.get("mention_text") or ""))
            if not ok:
                unmatched.append(
                    {
                        "case_id": case_id,
                        "note_id": note_id,
                        "sample_id": mention.get("sample_id"),
                        "task": "merged",
                        "reason": reason or "not_reliably_pulmonary",
                        "mention_text": mention.get("mention_text"),
                        "source_prediction_file": "merged_module2_predictions",
                    }
                )
                counters[f"unmatched.{reason}"] += 1
                continue
            facts = _facts_from_mention(mention)
            candidate = _candidate_from_facts(
                facts=facts,
                mention=mention,
                rank=0,
                candidate_kind="mention",
            )
            if candidate is None:
                unmatched.append(
                    {
                        "case_id": case_id,
                        "note_id": note_id,
                        "sample_id": mention.get("sample_id"),
                        "task": "merged",
                        "reason": "no_direct_module2_fact",
                        "mention_text": mention.get("mention_text"),
                        "source_prediction_file": "merged_module2_predictions",
                    }
                )
                counters["unmatched.no_direct_module2_fact"] += 1
                continue
            candidates.append(candidate)

        aggregate = _aggregate_candidate_for_note(
            case_id=case_id,
            note_id=note_id,
            mentions=mentions,
            rank=0,
            conflicts=conflicts,
        )
        if aggregate is not None:
            candidates.append(aggregate)
            counters["note_aggregate_candidates_appended"] += 1

        candidates.sort(key=_candidate_score, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["dominant_candidate_rank"] = rank
            fact.setdefault("nodules", []).append(candidate)
            counters["candidate_nodules_appended"] += 1
            if candidate.get("density_category") not in {None, "unclear"}:
                counters["candidate_with_density"] += 1
            if candidate.get("size_mm") is not None:
                counters["candidate_with_size"] += 1
            if candidate.get("location_lobe") is not None:
                counters["candidate_with_location"] += 1
        if candidates:
            fact["nodule_count"] = len(fact.get("nodules") or [])
            fact.setdefault("module2_adapter_metadata", {})
            fact["module2_adapter_metadata"].update(
                {
                    "adapter_version": "module3_m2_adapter_v2",
                    "module2_candidate_nodule_count": len(candidates),
                    "note_id": note_id,
                }
            )

    adapted["module3_adapter_metadata"] = {
        "adapter_version": "module3_m2_adapter_v2",
        "source": "module2_prediction_export_jsonl",
        "no_deterministic_rule_override": True,
        "no_free_text_value_generation": True,
        "candidate_nodules_appended": counters["candidate_nodules_appended"],
        "note_aggregate_candidates_appended": counters["note_aggregate_candidates_appended"],
    }
    return adapted, unmatched, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Module2 exported predictions into Phase4 case bundles.")
    parser.add_argument("--case-bundles", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument("--prediction-dir", default="outputs/phaseA3/module2_predictions")
    parser.add_argument("--output", default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_summary.csv")
    parser.add_argument("--unmatched", default="outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_unmatched.csv")
    parser.add_argument("--conflicts", default="outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_conflicts.csv")
    args = parser.parse_args()

    bundles = _load_jsonl(Path(args.case_bundles))
    note_case = _note_to_case(bundles)
    prediction_dir = Path(args.prediction_dir)
    prediction_paths = [
        prediction_dir / "module2_density_stage1_predictions.jsonl",
        prediction_dir / "module2_density_stage2_predictions.jsonl",
        prediction_dir / "module2_size_predictions.jsonl",
        prediction_dir / "module2_location_predictions.jsonl",
    ]

    mentions, unmatched_rows, load_counters = _merge_prediction_records(prediction_paths, note_case)
    mentions_by_note: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions.values():
        mentions_by_note[str(mention.get("note_id"))].append(mention)

    adapted_bundles: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    append_counters: Counter[str] = Counter()
    cases_with_candidates = 0

    for bundle in bundles:
        adapted, unmatched, counters = _append_candidates(
            bundle=bundle,
            mentions_by_note=mentions_by_note,
            conflicts=conflicts,
        )
        adapted_bundles.append(adapted)
        unmatched_rows.extend(unmatched)
        append_counters.update(counters)
        if adapted.get("module3_adapter_metadata", {}).get("candidate_nodules_appended", 0) > 0:
            cases_with_candidates += 1

    summary_rows: list[dict[str, Any]] = [
        {"metric": "input_cases", "value": len(bundles)},
        {"metric": "output_cases", "value": len(adapted_bundles)},
        {"metric": "phase4_radiology_notes", "value": len(note_case)},
        {"metric": "phase4_aligned_mentions", "value": load_counters["phase4_aligned_mentions"]},
        {"metric": "cases_with_module2_candidates", "value": cases_with_candidates},
        {"metric": "candidate_nodules_appended", "value": append_counters["candidate_nodules_appended"]},
        {"metric": "note_aggregate_candidates_appended", "value": append_counters["note_aggregate_candidates_appended"]},
        {"metric": "candidate_with_density", "value": append_counters["candidate_with_density"]},
        {"metric": "candidate_with_size", "value": append_counters["candidate_with_size"]},
        {"metric": "candidate_with_location", "value": append_counters["candidate_with_location"]},
        {"metric": "conflict_rows", "value": len(conflicts)},
        {"metric": "unmatched_rows", "value": len(unmatched_rows)},
        {"metric": "adapter_version", "value": "module3_m2_adapter_v2"},
        {"metric": "no_deterministic_rule_override", "value": True},
    ]
    for key, value in sorted(load_counters.items()):
        summary_rows.append({"metric": key, "value": value})
    for key, value in sorted(append_counters.items()):
        if key.startswith("unmatched."):
            summary_rows.append({"metric": key, "value": value})

    _write_jsonl(Path(args.output), adapted_bundles)
    _write_csv(Path(args.summary), summary_rows, ["metric", "value"])
    _write_csv(
        Path(args.unmatched),
        unmatched_rows,
        ["case_id", "note_id", "sample_id", "task", "reason", "mention_text", "source_prediction_file"],
    )
    _write_csv(
        Path(args.conflicts),
        conflicts,
        ["case_id", "note_id", "field", "selected_value", "alternative_values", "source_mention_ids", "resolution"],
    )

    print(
        json.dumps(
            {
                "input_cases": len(bundles),
                "output_cases": len(adapted_bundles),
                "phase4_aligned_mentions": load_counters["phase4_aligned_mentions"],
                "candidate_nodules_appended": append_counters["candidate_nodules_appended"],
                "note_aggregate_candidates_appended": append_counters["note_aggregate_candidates_appended"],
                "cases_with_module2_candidates": cases_with_candidates,
                "conflict_rows": len(conflicts),
                "unmatched_rows": len(unmatched_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
