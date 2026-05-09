#!/usr/bin/env python3
"""Simulate case/note-level fact aggregation for Module 3.

This is a diagnostic simulation only. It appends clearly marked aggregate
candidates to copied case bundles, runs the deterministic CDSG executor, and
does not overwrite the official v2 dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.cdsg_executor import CDSGExecutor, load_cdsg_graph


VALID_DENSITIES = {"solid", "part_solid", "ground_glass", "calcified", "fat_containing"}
VALID_LOCATIONS = {"RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _valid_density(value: Any) -> bool:
    return _normalize_density(value) in VALID_DENSITIES


def _valid_location(value: Any) -> bool:
    return value in VALID_LOCATIONS


def _confidence_to_float(value: Any) -> float | None:
    numeric = _as_float(value)
    if numeric is not None:
        return numeric
    if value == "high":
        return 0.9
    if value == "medium":
        return 0.75
    if value == "low":
        return 0.4
    return None


def _flatten_nodules(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_idx, fact in enumerate(bundle.get("radiology_facts", []) or []):
        for nodule in fact.get("nodules", []) or []:
            rows.append(
                {
                    "case_id": bundle.get("case_id"),
                    "note_id": fact.get("note_id"),
                    "fact_index": fact_idx,
                    "nodule": nodule,
                    "candidate_id": nodule.get("module2_candidate_id") or nodule.get("nodule_id_in_report"),
                    "mention_id": nodule.get("mention_id"),
                    "source_mention_ids": [str(item) for item in (nodule.get("source_mention_ids") or [])],
                }
            )
    return rows


def _field_confidence(item: dict[str, Any] | None, field: str) -> float:
    if not item:
        return 0.0
    nodule = item.get("nodule") or {}
    fact_sources = nodule.get("fact_sources") or {}
    source = fact_sources.get(field) or fact_sources.get("has_size") or {}
    for value in [source.get("confidence_value"), source.get("confidence"), nodule.get("source_confidence"), nodule.get("confidence")]:
        numeric = _confidence_to_float(value)
        if numeric is not None:
            return numeric
    return 0.0


def _source_descriptor(item: dict[str, Any] | None, field: str) -> dict[str, Any] | None:
    if not item:
        return None
    nodule = item.get("nodule") or {}
    source = (nodule.get("fact_sources") or {}).get(field) or {}
    return {
        "source": "module3_fact_aggregation_simulation",
        "source_field": field,
        "source_candidate_id": item.get("candidate_id"),
        "source_note_id": item.get("note_id"),
        "source_mention_ids": item.get("source_mention_ids") or ([item.get("mention_id")] if item.get("mention_id") else []),
        "source_candidate_source": nodule.get("source"),
        "source_confidence": source.get("confidence"),
        "confidence_value": source.get("confidence_value"),
        "original_text": nodule.get("evidence_span"),
        "no_deterministic_rule_override": True,
        "no_free_text_value_generation": True,
    }


def _mention_order(item: dict[str, Any] | None) -> int | None:
    if not item:
        return None
    text = str(item.get("mention_id") or item.get("candidate_id") or "")
    match = re.search(r"__(\d+)$", text)
    return int(match.group(1)) if match else None


def _facts_by_type(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {"size": [], "density": [], "location": [], "both_size_density": []}
    for item in items:
        nodule = item["nodule"]
        has_size = _as_float(nodule.get("size_mm")) is not None
        has_density = _valid_density(nodule.get("density_category"))
        has_location = _valid_location(nodule.get("location_lobe"))
        if has_size:
            out["size"].append(item)
        if has_density:
            out["density"].append(item)
        if has_location:
            out["location"].append(item)
        if has_size and has_density:
            out["both_size_density"].append(item)
    return out


def _largest_size(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return sorted(items, key=lambda item: (_as_float(item["nodule"].get("size_mm")) or -1.0, _field_confidence(item, "size_mm")), reverse=True)[0]


def _best_density(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    severity = {"fat_containing": 0, "calcified": 0, "ground_glass": 1, "solid": 2, "part_solid": 3}
    return sorted(
        items,
        key=lambda item: (
            _field_confidence(item, "density_category"),
            severity.get(str(_normalize_density(item["nodule"].get("density_category"))), -1),
        ),
        reverse=True,
    )[0]


def _best_location(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return sorted(items, key=lambda item: _field_confidence(item, "location_lobe"), reverse=True)[0]


def _nearest_density(size_item: dict[str, Any] | None, density_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not size_item or not density_items:
        return None
    size_order = _mention_order(size_item)
    size_note = size_item.get("note_id")

    def score(item: dict[str, Any]) -> tuple[int, int, float]:
        same_note = 1 if item.get("note_id") == size_note else 0
        item_order = _mention_order(item)
        if size_order is not None and item_order is not None:
            distance = -abs(size_order - item_order)
        else:
            distance = -9999
        return (same_note, distance, _field_confidence(item, "density_category"))

    return sorted(density_items, key=score, reverse=True)[0]


def _risk(size_item: dict[str, Any] | None, density_item: dict[str, Any] | None, location_item: dict[str, Any] | None) -> tuple[str, list[str]]:
    items = [item for item in [size_item, density_item, location_item] if item]
    notes = {str(item.get("note_id")) for item in items}
    mentions = {
        str(source_id)
        for item in items
        for source_id in (item.get("source_mention_ids") or ([item.get("mention_id")] if item.get("mention_id") else []))
    }
    reasons: list[str] = []
    if len(notes) > 1:
        reasons.append("cross_note_fact_union")
    if len(mentions) > 1:
        reasons.append("cross_mention_fact_union")
    if not reasons:
        return "low", []
    if "cross_note_fact_union" in reasons:
        return "high", reasons
    return "medium", reasons


def _build_aggregate_candidate(
    *,
    case_id: str,
    strategy: str,
    index: int,
    size_item: dict[str, Any] | None,
    density_item: dict[str, Any] | None,
    location_item: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not size_item or not density_item:
        return None
    size_mm = _as_float(size_item["nodule"].get("size_mm"))
    density = _normalize_density(density_item["nodule"].get("density_category"))
    if size_mm is None or density not in VALID_DENSITIES:
        return None

    location = location_item["nodule"].get("location_lobe") if location_item else None
    if location not in VALID_LOCATIONS:
        location = None
    risk_level, risk_reasons = _risk(size_item, density_item, location_item)
    source_items = [item for item in [size_item, density_item, location_item] if item]
    source_mentions = sorted(
        {
            str(source_id)
            for item in source_items
            for source_id in (item.get("source_mention_ids") or ([item.get("mention_id")] if item.get("mention_id") else []))
            if source_id
        }
    )
    evidence_parts = [
        str(item["nodule"].get("evidence_span") or "")
        for item in source_items
        if item["nodule"].get("evidence_span")
    ]
    confidence = max(
        _field_confidence(size_item, "size_mm"),
        _field_confidence(density_item, "density_category"),
        _field_confidence(location_item, "location_lobe") if location_item else 0.0,
    )
    return {
        "nodule_id_in_report": f"module3_sim:{strategy}:{case_id}:{index}",
        "module2_candidate_id": f"module3_sim:{strategy}:{case_id}:{index}",
        "mention_id": f"module3_sim:{strategy}:{case_id}:{index}",
        "source_mention_ids": source_mentions,
        "size_mm": size_mm,
        "size_text": size_item["nodule"].get("size_text") or f"{size_mm:g} mm",
        "has_size": True,
        "density_category": density,
        "density_text": density,
        "location_lobe": location,
        "location_text": location,
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
        "evidence_span": " | ".join(evidence_parts),
        "confidence": "high" if confidence >= 0.9 else "medium" if confidence >= 0.7 else "low",
        "source_confidence": confidence,
        "missing_flags": [] if location else ["location_lobe", "location_text"],
        "source": "module3_fact_aggregation_simulation",
        "fact_sources": {
            "size_mm": _source_descriptor(size_item, "size_mm"),
            "has_size": _source_descriptor(size_item, "has_size"),
            "density_category": _source_descriptor(density_item, "density_category"),
            "location_lobe": _source_descriptor(location_item, "location_lobe") if location_item else None,
        },
        "dominant_selection_fields": {
            "size_mm": size_mm,
            "density_category": density,
            "location_lobe": location,
        },
        "adapter_notes": {
            "adapter_version": "module3_fact_aggregation_simulation",
            "simulation_strategy": strategy,
            "no_deterministic_rule_override": True,
            "no_free_text_value_generation": True,
            "aggregation_risk_level": risk_level,
            "aggregation_risk_reasons": risk_reasons,
        },
    }


def _append_candidate(bundle: dict[str, Any], candidate: dict[str, Any], preferred_note_id: Any | None) -> None:
    facts = bundle.get("radiology_facts") or []
    if not facts:
        return
    target = None
    if preferred_note_id is not None:
        for fact in facts:
            if str(fact.get("note_id")) == str(preferred_note_id):
                target = fact
                break
    if target is None:
        target = facts[0]
    target.setdefault("nodules", []).append(candidate)
    target["nodule_count"] = len(target.get("nodules") or [])


def _simulate_strategy(
    bundle: dict[str, Any],
    *,
    strategy: str,
    confidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulated = deepcopy(bundle)
    case_id = str(simulated.get("case_id"))
    items = _flatten_nodules(simulated)
    by_type = _facts_by_type(items)
    appended: list[dict[str, Any]] = []

    if by_type["both_size_density"]:
        simulated["module3_aggregation_simulation"] = {
            "strategy": strategy,
            "candidates_appended": 0,
            "skipped_reason": "existing_size_density_candidate_present",
        }
        return simulated, appended

    candidate_specs: list[tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, Any | None]] = []

    if strategy == "same_case_union":
        size_item = _largest_size(by_type["size"])
        density_item = _best_density(by_type["density"])
        location_item = _best_location(by_type["location"])
        candidate_specs.append((size_item, density_item, location_item, size_item.get("note_id") if size_item else None))

    elif strategy == "same_note_union":
        note_ids = sorted({str(item.get("note_id")) for item in items if item.get("note_id")})
        for note_id in note_ids:
            note_items = [item for item in items if str(item.get("note_id")) == note_id]
            note_by_type = _facts_by_type(note_items)
            candidate_specs.append(
                (
                    _largest_size(note_by_type["size"]),
                    _best_density(note_by_type["density"]),
                    _best_location(note_by_type["location"]),
                    note_id,
                )
            )

    elif strategy == "dominant_size_density_nearest":
        size_item = _largest_size(by_type["size"])
        density_item = _nearest_density(size_item, by_type["density"])
        same_note_locations = [item for item in by_type["location"] if size_item and item.get("note_id") == size_item.get("note_id")]
        location_item = _best_location(same_note_locations) or _best_location(by_type["location"])
        candidate_specs.append((size_item, density_item, location_item, size_item.get("note_id") if size_item else None))

    elif strategy == "confidence_gated_same_note_union":
        note_ids = sorted({str(item.get("note_id")) for item in items if item.get("note_id")})
        for note_id in note_ids:
            note_items = [item for item in items if str(item.get("note_id")) == note_id]
            note_by_type = _facts_by_type(note_items)
            density_items = [
                item for item in note_by_type["density"] if _field_confidence(item, "density_category") >= confidence_threshold
            ]
            location_items = [
                item for item in note_by_type["location"] if _field_confidence(item, "location_lobe") >= confidence_threshold
            ]
            candidate_specs.append(
                (
                    _largest_size(note_by_type["size"]),
                    _best_density(density_items),
                    _best_location(location_items),
                    note_id,
                )
            )
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    for idx, (size_item, density_item, location_item, preferred_note_id) in enumerate(candidate_specs, start=1):
        candidate = _build_aggregate_candidate(
            case_id=case_id,
            strategy=strategy,
            index=idx,
            size_item=size_item,
            density_item=density_item,
            location_item=location_item,
        )
        if candidate is None:
            continue
        _append_candidate(simulated, candidate, preferred_note_id)
        appended.append(candidate)

    simulated["module3_aggregation_simulation"] = {
        "strategy": strategy,
        "candidates_appended": len(appended),
        "confidence_threshold": confidence_threshold if "confidence_gated" in strategy else None,
        "simulation_only": True,
        "does_not_replace_official_v2": True,
    }
    return simulated, appended


def _execute_all(executor: CDSGExecutor, bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [executor.execute(bundle) for bundle in bundles]


def _summarize_recommendations(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(recommendations)
    abstention_counter = Counter(str(row.get("abstention_reason")) for row in recommendations)
    actionable = sum(1 for row in recommendations if not row.get("abstention_reason"))
    return {
        "total_cases": total,
        "actionable": actionable,
        "abstention": total - actionable,
        "missing_density": abstention_counter["missing_nodule_density"],
        "missing_size": abstention_counter["missing_nodule_size"],
        "no_structured_nodule": abstention_counter["no_structured_nodule"],
        "abstention_distribution": dict(abstention_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Module 3 case-level fact aggregation.")
    parser.add_argument("--case-bundles", default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl")
    parser.add_argument("--graph", default="outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module3_fact_aggregation_simulation.csv")
    parser.add_argument(
        "--output",
        default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2_simulated_aggregation.jsonl",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()

    bundles = _load_jsonl(Path(args.case_bundles))
    executor = CDSGExecutor(load_cdsg_graph(args.graph))
    original_recommendations = _execute_all(executor, bundles)
    original_summary = _summarize_recommendations(original_recommendations)
    original_by_case = {str(bundle.get("case_id")): rec for bundle, rec in zip(bundles, original_recommendations, strict=False)}

    strategies = [
        "same_case_union",
        "same_note_union",
        "dominant_size_density_nearest",
        "confidence_gated_same_note_union",
    ]
    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        simulated_bundles: list[dict[str, Any]] = []
        appended_total = 0
        modified_cases = 0
        high_risk_candidates = 0
        medium_risk_candidates = 0
        recovered_cases = 0

        for bundle in bundles:
            simulated, appended = _simulate_strategy(
                bundle,
                strategy=strategy,
                confidence_threshold=args.confidence_threshold,
            )
            simulated_bundles.append(simulated)
            appended_total += len(appended)
            if appended:
                modified_cases += 1
            high_risk_candidates += sum(
                1 for item in appended if item.get("adapter_notes", {}).get("aggregation_risk_level") == "high"
            )
            medium_risk_candidates += sum(
                1 for item in appended if item.get("adapter_notes", {}).get("aggregation_risk_level") == "medium"
            )

        simulated_recommendations = _execute_all(executor, simulated_bundles)
        simulated_summary = _summarize_recommendations(simulated_recommendations)

        for bundle, simulated_bundle, recommendation in zip(bundles, simulated_bundles, simulated_recommendations, strict=False):
            case_id = str(bundle.get("case_id"))
            original_rec = original_by_case[case_id]
            recovered = bool(original_rec.get("abstention_reason") and not recommendation.get("abstention_reason"))
            if recovered:
                recovered_cases += 1
            output_rows.append(
                {
                    "simulation_strategy": strategy,
                    "case_id": case_id,
                    "case_bundle": simulated_bundle,
                    "aggregation_candidates_appended": simulated_bundle.get("module3_aggregation_simulation", {}).get("candidates_appended"),
                    "original_abstention_reason": original_rec.get("abstention_reason"),
                    "simulated_abstention_reason": recommendation.get("abstention_reason"),
                    "simulated_recommendation_level": recommendation.get("recommendation_level"),
                    "simulated_lung_rads_category": recommendation.get("lung_rads_category"),
                    "recovered_from_abstention": recovered,
                    "simulation_only": True,
                }
            )

        summary_rows.append(
            {
                "strategy": strategy,
                "original_actionable": original_summary["actionable"],
                "original_abstention": original_summary["abstention"],
                "original_missing_density": original_summary["missing_density"],
                "original_missing_size": original_summary["missing_size"],
                "original_no_structured_nodule": original_summary["no_structured_nodule"],
                "simulated_actionable": simulated_summary["actionable"],
                "simulated_abstention": simulated_summary["abstention"],
                "simulated_missing_density": simulated_summary["missing_density"],
                "simulated_missing_size": simulated_summary["missing_size"],
                "simulated_no_structured_nodule": simulated_summary["no_structured_nodule"],
                "actionable_delta": simulated_summary["actionable"] - original_summary["actionable"],
                "abstention_delta": simulated_summary["abstention"] - original_summary["abstention"],
                "missing_density_delta": simulated_summary["missing_density"] - original_summary["missing_density"],
                "missing_size_delta": simulated_summary["missing_size"] - original_summary["missing_size"],
                "recovered_cases": recovered_cases,
                "cases_modified": modified_cases,
                "aggregation_candidates_appended": appended_total,
                "high_risk_candidates": high_risk_candidates,
                "medium_risk_candidates": medium_risk_candidates,
                "confidence_threshold": args.confidence_threshold if "confidence_gated" in strategy else "",
                "risk_note": (
                    "high_risk"
                    if high_risk_candidates
                    else "medium_or_lower_risk"
                    if medium_risk_candidates
                    else "low_risk"
                    if appended_total
                    else "no_aggregate_candidates"
                ),
            }
        )

    _write_csv(
        Path(args.summary),
        summary_rows,
        [
            "strategy",
            "original_actionable",
            "original_abstention",
            "original_missing_density",
            "original_missing_size",
            "original_no_structured_nodule",
            "simulated_actionable",
            "simulated_abstention",
            "simulated_missing_density",
            "simulated_missing_size",
            "simulated_no_structured_nodule",
            "actionable_delta",
            "abstention_delta",
            "missing_density_delta",
            "missing_size_delta",
            "recovered_cases",
            "cases_modified",
            "aggregation_candidates_appended",
            "high_risk_candidates",
            "medium_risk_candidates",
            "confidence_threshold",
            "risk_note",
        ],
    )
    _write_jsonl(Path(args.output), output_rows)

    print(
        json.dumps(
            {
                "original": original_summary,
                "strategies": summary_rows,
                "output": args.output,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
