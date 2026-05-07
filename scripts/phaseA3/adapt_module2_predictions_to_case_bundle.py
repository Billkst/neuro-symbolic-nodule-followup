#!/usr/bin/env python3
"""Adapt available Module 2 mention-level facts into Module 3 case bundles.

This adapter is intentionally conservative. It does not infer missing values
from text, does not override deterministic rules, and only appends candidate
nodules when a Phase5/Module2 mention row can be aligned to a Phase4 note.
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
}


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


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_unmatched(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "note_id",
        "sample_id",
        "reason",
        "mention_text",
        "source_dataset",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_result_tag(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return str(data.get("tag") or fallback)
    except json.JSONDecodeError:
        return fallback


def _read_chosen_threshold(path: Path, fallback: float = 0.5) -> float:
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        value = data.get("chosen_threshold")
        return float(value) if value is not None else fallback
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback


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


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _confidence_from_sources(label_quality: str | None, prob: float | None) -> str:
    if prob is not None:
        if prob >= 0.9:
            return "high"
        if prob >= 0.7:
            return "medium"
        return "low"
    if label_quality == "explicit":
        return "high"
    if label_quality in {"silver", "high"}:
        return "medium"
    return "low"


def _module2_confidence_value(label_quality: str | None, prob: float | None) -> float | None:
    if prob is not None:
        return float(prob)
    if label_quality == "explicit":
        return 1.0
    if label_quality == "silver":
        return 0.75
    if label_quality == "weak":
        return 0.5
    return None


def _is_reliably_pulmonary_mention(text: str) -> tuple[bool, str | None]:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False, "empty_mention_text"
    has_pulmonary_term = any(term in normalized for term in PULMONARY_TERMS)
    has_non_pulmonary_term = any(term in normalized for term in NON_PULMONARY_TERMS)
    if has_non_pulmonary_term and not re.search(r"\b(lung|pulmonary|lobe|lingula)\b", normalized):
        return False, "non_pulmonary_mention_filtered"
    if not has_pulmonary_term:
        return False, "no_pulmonary_nodule_cue"
    return True, None


def _load_size_probabilities(probability_dir: Path, tag: str) -> dict[str, dict[str, Any]]:
    probabilities: dict[str, dict[str, Any]] = {}
    if not probability_dir.exists():
        return probabilities
    for path in sorted(probability_dir.glob(f"{tag}_*_probs.jsonl")):
        split_name = path.name.replace(f"{tag}_", "").replace("_probs.jsonl", "")
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if not sample_id:
                    continue
                current = probabilities.get(sample_id)
                prob = _as_float(row.get("prob_has_size"))
                if current is None or (prob is not None and prob > (current.get("prob_has_size") or -1)):
                    probabilities[sample_id] = {
                        "prob_has_size": prob,
                        "probability_split": split_name,
                        "probability_path": str(path),
                    }
    return probabilities


def _phase5_dataset_paths(data_dir: Path) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []
    for task in ["density", "size", "location"]:
        for split in ["train", "val", "test"]:
            path = data_dir / f"{task}_{split}.jsonl"
            if path.exists():
                paths.append((task, split, path))
    return paths


def _collect_module2_mentions(
    *,
    phase5_data_dir: Path,
    note_to_case: dict[str, str],
    size_probabilities: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    mentions: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()

    for task, split, path in _phase5_dataset_paths(phase5_data_dir):
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                counters[f"{task}_{split}_rows_seen"] += 1
                row = json.loads(line)
                note_id = row.get("note_id")
                if note_id not in note_to_case:
                    counters[f"{task}_{split}_rows_note_not_in_phase4"] += 1
                    continue
                sample_id = row.get("sample_id")
                if not sample_id:
                    counters[f"{task}_{split}_rows_missing_sample_id"] += 1
                    continue

                mention = mentions.setdefault(
                    str(sample_id),
                    {
                        "sample_id": sample_id,
                        "case_id": note_to_case[note_id],
                        "note_id": note_id,
                        "subject_id": row.get("subject_id"),
                        "exam_name": row.get("exam_name"),
                        "section": row.get("section"),
                        "mention_text": row.get("mention_text"),
                        "full_text": row.get("full_text"),
                        "source_splits": set(),
                        "source_paths": set(),
                        "label_quality": row.get("label_quality"),
                    },
                )
                mention["source_splits"].add(split)
                mention["source_paths"].add(str(path))
                if row.get("label_quality") == "explicit":
                    mention["label_quality"] = "explicit"
                elif mention.get("label_quality") is None:
                    mention["label_quality"] = row.get("label_quality")

                if task == "density":
                    mention["density_label"] = row.get("density_label")
                elif task == "size":
                    mention["has_size"] = row.get("has_size")
                    mention["size_label"] = row.get("size_label")
                    mention["size_text"] = row.get("size_text")
                elif task == "location":
                    mention["location_label"] = row.get("location_label")
                    mention["has_location"] = row.get("has_location")

                if sample_id in size_probabilities:
                    mention["size_probability"] = size_probabilities[sample_id]

    for mention in mentions.values():
        mention["source_splits"] = sorted(mention["source_splits"])
        mention["source_paths"] = sorted(mention["source_paths"])
    counters["phase4_aligned_mentions"] = len(mentions)
    return mentions, counters


def _fact_source(
    *,
    source: str,
    model_tag: str,
    mention: dict[str, Any],
    confidence: str,
    confidence_value: float | None,
) -> dict[str, Any]:
    return {
        "source": source,
        "confidence": confidence,
        "confidence_value": confidence_value,
        "model_tag": model_tag,
        "mention_id": mention.get("sample_id"),
        "original_text": mention.get("mention_text"),
        "note_id": mention.get("note_id"),
        "source_splits": mention.get("source_splits") or [],
        "source_paths": mention.get("source_paths") or [],
        "label_quality": mention.get("label_quality"),
    }


def _candidate_from_mention(
    *,
    mention: dict[str, Any],
    rank: int,
    density_model_tag: str,
    size_model_tag: str,
    location_model_tag: str,
    size_threshold: float,
) -> tuple[dict[str, Any] | None, str | None]:
    ok, reason = _is_reliably_pulmonary_mention(str(mention.get("mention_text") or ""))
    if not ok:
        return None, reason

    label_quality = mention.get("label_quality")
    size_probability = mention.get("size_probability") or {}
    prob_has_size = _as_float(size_probability.get("prob_has_size"))
    size_allowed_by_prob = prob_has_size is None or prob_has_size >= size_threshold
    size_mm = None
    if mention.get("has_size") is True and size_allowed_by_prob:
        size_mm = _as_float(mention.get("size_label"))

    density = _normalize_density(mention.get("density_label"))
    if density not in VALID_DENSITIES:
        density = "unclear"

    location = mention.get("location_label")
    if location not in VALID_LOCATIONS:
        location = None

    missing_flags: list[str] = []
    if size_mm is None:
        missing_flags.extend(["size_mm", "size_text"])
    if density in {None, "unclear"}:
        missing_flags.extend(["density_category", "density_text"])
    if location is None:
        missing_flags.extend(["location_lobe", "location_text"])

    has_any_direct_fact = size_mm is not None or density not in {None, "unclear"} or location is not None
    if not has_any_direct_fact:
        return None, "no_direct_module2_fact"

    confidence = _confidence_from_sources(label_quality, prob_has_size)
    confidence_value = _module2_confidence_value(label_quality, prob_has_size)
    fact_sources = {
        "density_category": _fact_source(
            source="module2_phase5_density_dataset_label",
            model_tag=density_model_tag,
            mention=mention,
            confidence=confidence,
            confidence_value=confidence_value,
        ),
        "size_mm": _fact_source(
            source="module2_phase5_size_dataset_label_with_has_size_probability",
            model_tag=size_model_tag,
            mention=mention,
            confidence=confidence,
            confidence_value=confidence_value,
        ),
        "location_lobe": _fact_source(
            source="module2_phase5_location_dataset_label",
            model_tag=location_model_tag,
            mention=mention,
            confidence=confidence,
            confidence_value=confidence_value,
        ),
    }

    candidate = {
        "nodule_id_in_report": f"module2:{mention.get('sample_id')}",
        "module2_candidate_id": mention.get("sample_id"),
        "mention_id": mention.get("sample_id"),
        "size_mm": size_mm,
        "size_text": mention.get("size_text") if size_mm is not None else None,
        "density_category": density,
        "density_text": mention.get("density_label") if density != "unclear" else None,
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
        "evidence_span": mention.get("mention_text"),
        "confidence": confidence,
        "missing_flags": missing_flags,
        "source": "module2_to_case_bundle_adapter",
        "source_confidence": confidence_value,
        "fact_sources": fact_sources,
        "dominant_candidate_rank": rank,
        "dominant_selection_fields": {
            "size_mm": size_mm,
            "density_category": density,
            "location_lobe": location,
        },
        "adapter_notes": {
            "no_deterministic_rule_override": True,
            "size_probability_threshold": size_threshold,
            "prob_has_size": prob_has_size,
            "probability_split": size_probability.get("probability_split"),
        },
    }
    return candidate, None


def _candidate_sort_key(mention: dict[str, Any]) -> tuple[int, float, str]:
    density = _normalize_density(mention.get("density_label"))
    density_score = 1 if density in VALID_DENSITIES else 0
    size_score = _as_float(mention.get("size_label")) or -1.0
    return (density_score, size_score, str(mention.get("sample_id")))


def _append_module2_candidates(
    *,
    bundle: dict[str, Any],
    mentions_by_note: dict[str, list[dict[str, Any]]],
    density_model_tag: str,
    size_model_tag: str,
    location_model_tag: str,
    size_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    adapted = deepcopy(bundle)
    counters: Counter[str] = Counter()
    unmatched: list[dict[str, Any]] = []
    case_id = str(adapted.get("case_id"))

    for fact in adapted.get("radiology_facts", []) or []:
        note_id = fact.get("note_id")
        mentions = list(mentions_by_note.get(str(note_id), []))
        mentions.sort(key=_candidate_sort_key, reverse=True)
        appended = 0
        for rank, mention in enumerate(mentions, start=1):
            candidate, reason = _candidate_from_mention(
                mention=mention,
                rank=rank,
                density_model_tag=density_model_tag,
                size_model_tag=size_model_tag,
                location_model_tag=location_model_tag,
                size_threshold=size_threshold,
            )
            if candidate is None:
                unmatched.append(
                    {
                        "case_id": case_id,
                        "note_id": note_id,
                        "sample_id": mention.get("sample_id"),
                        "reason": reason or "unknown_alignment_failure",
                        "mention_text": mention.get("mention_text"),
                        "source_dataset": "|".join(mention.get("source_paths") or []),
                    }
                )
                counters[f"unmatched_reason.{reason}"] += 1
                continue
            fact.setdefault("nodules", []).append(candidate)
            appended += 1
            counters["candidate_nodules_appended"] += 1
            if candidate.get("density_category") not in {None, "unclear"}:
                counters["candidate_with_density"] += 1
            if candidate.get("size_mm") is not None:
                counters["candidate_with_size"] += 1
            if candidate.get("location_lobe") is not None:
                counters["candidate_with_location"] += 1
        if appended:
            fact["nodule_count"] = len(fact.get("nodules") or [])
            fact.setdefault("module2_adapter_metadata", {})
            fact["module2_adapter_metadata"].update(
                {
                    "adapter_version": "module3_m2_adapter_0.1",
                    "module2_candidate_nodule_count": appended,
                    "note_id": note_id,
                }
            )

    adapted["module3_adapter_metadata"] = {
        "adapter_version": "module3_m2_adapter_0.1",
        "source": "module2_phase5_mention_datasets",
        "no_deterministic_rule_override": True,
        "density_model_tag": density_model_tag,
        "size_model_tag": size_model_tag,
        "location_model_tag": location_model_tag,
        "candidate_nodules_appended": counters["candidate_nodules_appended"],
    }
    return adapted, unmatched, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Module 2 facts into Phase4 case bundles.")
    parser.add_argument("--case-bundles", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument(
        "--size-probability-dir",
        default="outputs/phaseA2_planB/size_wave5/probabilities",
    )
    parser.add_argument("--size-probability-tag", default="size_wave5_lexical_alone_seed42")
    parser.add_argument(
        "--density-result",
        default="outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed42.json",
    )
    parser.add_argument(
        "--size-result",
        default="outputs/phaseA2_planB/results/mws_cfe_size_results_size_wave5_lexical_alone_seed42.json",
    )
    parser.add_argument(
        "--location-result",
        default="outputs/phaseA2_planB/results/mws_cfe_location_results_planb_full_seed42.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/phaseA3/datasets/module3_ready_case_bundles.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="outputs/phaseA3/tables/module2_to_case_bundle_adapter_summary.csv",
    )
    parser.add_argument(
        "--unmatched",
        default="outputs/phaseA3/tables/module2_to_case_bundle_unmatched.csv",
    )
    args = parser.parse_args()

    bundles = _load_jsonl(Path(args.case_bundles))
    note_to_case: dict[str, str] = {}
    for bundle in bundles:
        case_id = str(bundle.get("case_id"))
        for fact in bundle.get("radiology_facts", []) or []:
            note_id = fact.get("note_id")
            if note_id:
                note_to_case[str(note_id)] = case_id

    size_probabilities = _load_size_probabilities(
        Path(args.size_probability_dir), args.size_probability_tag
    )
    mentions, load_counters = _collect_module2_mentions(
        phase5_data_dir=Path(args.phase5_data_dir),
        note_to_case=note_to_case,
        size_probabilities=size_probabilities,
    )

    mentions_by_note: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions.values():
        mentions_by_note[str(mention.get("note_id"))].append(mention)

    density_model_tag = _read_result_tag(Path(args.density_result), "mws_cfe_density_stage2_final")
    size_model_tag = f"{_read_result_tag(Path(args.size_result), args.size_probability_tag)}_probabilities"
    location_model_tag = _read_result_tag(Path(args.location_result), "mws_cfe_location_final")
    size_threshold = _read_chosen_threshold(Path(args.size_result), fallback=0.5)

    adapted_bundles: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    append_counters: Counter[str] = Counter()
    cases_with_candidates = 0

    for bundle in bundles:
        adapted, unmatched, counters = _append_module2_candidates(
            bundle=bundle,
            mentions_by_note=mentions_by_note,
            density_model_tag=density_model_tag,
            size_model_tag=size_model_tag,
            location_model_tag=location_model_tag,
            size_threshold=size_threshold,
        )
        adapted_bundles.append(adapted)
        unmatched_rows.extend(unmatched)
        append_counters.update(counters)
        if adapted.get("module3_adapter_metadata", {}).get("candidate_nodules_appended", 0) > 0:
            cases_with_candidates += 1

    summary_rows: list[dict[str, Any]] = [
        {"metric": "input_cases", "value": len(bundles)},
        {"metric": "output_cases", "value": len(adapted_bundles)},
        {"metric": "phase4_radiology_notes", "value": len(note_to_case)},
        {"metric": "size_probability_rows_loaded", "value": len(size_probabilities)},
        {"metric": "phase4_aligned_mentions", "value": load_counters["phase4_aligned_mentions"]},
        {"metric": "cases_with_module2_candidates", "value": cases_with_candidates},
        {"metric": "candidate_nodules_appended", "value": append_counters["candidate_nodules_appended"]},
        {"metric": "candidate_with_density", "value": append_counters["candidate_with_density"]},
        {"metric": "candidate_with_size", "value": append_counters["candidate_with_size"]},
        {"metric": "candidate_with_location", "value": append_counters["candidate_with_location"]},
        {"metric": "unmatched_rows", "value": len(unmatched_rows)},
        {"metric": "density_model_tag", "value": density_model_tag},
        {"metric": "size_model_tag", "value": size_model_tag},
        {"metric": "location_model_tag", "value": location_model_tag},
        {"metric": "size_probability_threshold", "value": f"{size_threshold:.6f}"},
        {
            "metric": "density_location_per_sample_prediction_note",
            "value": "aggregate_result_json_has_no_per_sample_predictions; adapter_uses_phase5_mention_fact_fields_with_source_tags",
        },
    ]
    for key, value in sorted(load_counters.items()):
        summary_rows.append({"metric": key, "value": value})
    for key, value in sorted(append_counters.items()):
        if key.startswith("unmatched_reason."):
            summary_rows.append({"metric": key, "value": value})

    _write_jsonl(Path(args.output), adapted_bundles)
    _write_summary(Path(args.summary), summary_rows)
    _write_unmatched(Path(args.unmatched), unmatched_rows)

    print(
        json.dumps(
            {
                "input_cases": len(bundles),
                "output_cases": len(adapted_bundles),
                "phase4_aligned_mentions": load_counters["phase4_aligned_mentions"],
                "candidate_nodules_appended": append_counters["candidate_nodules_appended"],
                "cases_with_module2_candidates": cases_with_candidates,
                "unmatched_rows": len(unmatched_rows),
                "output": args.output,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
