#!/usr/bin/env python3
"""Audit Module 3 candidate aggregation conflicts from adapter v2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _split_pipe(value: Any) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def _risk_for_conflict(field: str, alternatives: list[str], source_mentions: list[str]) -> tuple[str, str, bool]:
    alt_count = len(set(alternatives))
    mention_count = len(set(source_mentions))
    if field == "size_mm":
        if alt_count <= 1:
            return "low", "same_numeric_size_duplicate_mentions", False
        if alt_count <= 3:
            return "medium", "largest_size_case_or_note_level_resolution_possible", False
        return "medium", "many_size_values_largest_size_possible_but_review_recommended", True
    if field == "density_category":
        if alt_count <= 1:
            return "low", "same_density_duplicate_mentions", False
        if mention_count > 1:
            return "high", "density_subtype_conflict_changes_cdsg_path_manual_review", True
        return "medium", "density_conflict_single_mention_duplicate", True
    if field == "location_lobe":
        if alt_count <= 1:
            return "low", "same_location_duplicate_mentions", False
        if alt_count > 2:
            return "medium", "multiple_location_mentions_likely_multiple_nodules", True
        return "medium", "location_conflict_low_cdsg_impact_but_grounding_risk", False
    return "medium", "unknown_conflict_type", True


def _candidate_lookup(bundles: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        case_id = str(bundle.get("case_id"))
        for fact in bundle.get("radiology_facts", []) or []:
            note_id = str(fact.get("note_id"))
            for nodule in fact.get("nodules", []) or []:
                lookup[(case_id, note_id)].append(nodule)
    return lookup


def _evidence_for_mentions(candidates: list[dict[str, Any]], mention_ids: list[str]) -> str:
    snippets: list[str] = []
    wanted = set(mention_ids)
    for nodule in candidates:
        source_ids = set(str(item) for item in (nodule.get("source_mention_ids") or []))
        if nodule.get("mention_id"):
            source_ids.add(str(nodule.get("mention_id")))
        if source_ids & wanted and nodule.get("evidence_span"):
            snippets.append(str(nodule.get("evidence_span")).replace("\n", " ")[:160])
        if len(snippets) >= 3:
            break
    return " || ".join(snippets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Module 3 candidate conflicts.")
    parser.add_argument("--case-bundles", default="outputs/phaseA3/datasets/module3_ready_case_bundles_v2.jsonl")
    parser.add_argument("--conflicts", default="outputs/phaseA3/tables/module2_to_case_bundle_adapter_v2_conflicts.csv")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module3_candidate_conflict_summary.csv")
    parser.add_argument("--examples", default="outputs/phaseA3/tables/module3_candidate_conflict_examples.csv")
    args = parser.parse_args()

    bundles = _load_jsonl(Path(args.case_bundles))
    conflicts = _load_csv(Path(args.conflicts))
    candidates_by_case_note = _candidate_lookup(bundles)

    field_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    review_counter: Counter[str] = Counter()
    case_counter: Counter[str] = Counter()
    note_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in conflicts:
        field = str(row.get("field") or "unknown")
        alternatives = _split_pipe(row.get("alternative_values"))
        source_mentions = _split_pipe(row.get("source_mention_ids"))
        risk, resolution, needs_review = _risk_for_conflict(field, alternatives, source_mentions)
        field_counter[field] += 1
        risk_counter[risk] += 1
        resolution_counter[resolution] += 1
        review_counter["needs_manual_review" if needs_review else "aggregation_resolvable"] += 1
        case_counter[str(row.get("case_id"))] += 1
        note_counter[f"{row.get('case_id')}::{row.get('note_id')}"] += 1

        candidates = candidates_by_case_note.get((str(row.get("case_id")), str(row.get("note_id"))), [])
        examples.append(
            {
                "case_id": row.get("case_id"),
                "note_id": row.get("note_id"),
                "field": field,
                "selected_value": row.get("selected_value"),
                "alternative_values": row.get("alternative_values"),
                "alternative_value_count": len(set(alternatives)),
                "source_mention_count": len(set(source_mentions)),
                "source_mention_ids": row.get("source_mention_ids"),
                "risk_level": risk,
                "case_level_aggregation_resolution": resolution,
                "needs_manual_review": needs_review,
                "different_mentions_merged": len(set(source_mentions)) > 1,
                "example_evidence_spans": _evidence_for_mentions(candidates, source_mentions),
            }
        )

    summary_rows: list[dict[str, Any]] = [
        {"metric": "total_conflict_rows", "value": len(conflicts)},
        {"metric": "conflict_cases", "value": len(case_counter)},
        {"metric": "conflict_case_notes", "value": len(note_counter)},
        {"metric": "manual_review_rows", "value": review_counter["needs_manual_review"]},
        {"metric": "aggregation_resolvable_rows", "value": review_counter["aggregation_resolvable"]},
    ]
    for field, count in sorted(field_counter.items()):
        summary_rows.append({"metric": f"field.{field}", "value": count})
    for risk, count in sorted(risk_counter.items()):
        summary_rows.append({"metric": f"risk_level.{risk}", "value": count})
    for resolution, count in sorted(resolution_counter.items()):
        summary_rows.append({"metric": f"resolution.{resolution}", "value": count})

    examples.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(str(item["risk_level"]), 3),
            -int(item["alternative_value_count"]),
            str(item["case_id"]),
        )
    )
    _write_csv(Path(args.summary), summary_rows, ["metric", "value"])
    _write_csv(
        Path(args.examples),
        examples,
        [
            "case_id",
            "note_id",
            "field",
            "selected_value",
            "alternative_values",
            "alternative_value_count",
            "source_mention_count",
            "source_mention_ids",
            "risk_level",
            "case_level_aggregation_resolution",
            "needs_manual_review",
            "different_mentions_merged",
            "example_evidence_spans",
        ],
    )

    print(
        json.dumps(
            {
                "total_conflict_rows": len(conflicts),
                "field_distribution": dict(field_counter),
                "risk_distribution": dict(risk_counter),
                "manual_review_rows": review_counter["needs_manual_review"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
