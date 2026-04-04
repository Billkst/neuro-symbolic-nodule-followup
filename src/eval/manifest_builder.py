import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.filters import filter_chest_ct, filter_nodule_reports
from src.data.loader import load_discharge, load_radiology, load_radiology_detail
from src.extractors.radiology_extractor import extract_radiology_facts
from src.extractors.smoking_extractor import (
    extract_smoking_eligibility,
    extract_smoking_status,
    find_social_history_section,
)
from src.parsers.section_parser import parse_sections


EXPLICIT_SIZE_RE = re.compile(
    r"(\d+\.?\d*)\s*(?:x\s*\d+\.?\d*\s*)?(?:mm|cm)\b", re.IGNORECASE
)
EXPLICIT_DENSITY_RE = re.compile(
    r"\b(solid|part[- ]?solid|ground[- ]?glass|ggo|ggn|calcified)\b", re.IGNORECASE
)
EXPLICIT_LOCATION_RE = re.compile(
    r"\b(right upper|right middle|right lower|left upper|left lower|lingula|RUL|RML|RLL|LUL|LLL)\b",
    re.IGNORECASE,
)
EXPLICIT_CHANGE_RE = re.compile(
    r"\b(new|stable|unchanged|increased|decreased|growing|enlarging|resolved|interval change)\b",
    re.IGNORECASE,
)
EXPLICIT_RECOMMENDATION_RE = re.compile(
    r"\b(recommend|follow[- ]?up|suggested|advised|should be|lung[- ]?rads|screening)\b",
    re.IGNORECASE,
)

SMOKING_STATUS_RE = re.compile(
    r"\b(smoker|former smoker|never smoker|non[- ]?smoker|current smoker|quit smoking|"
    r"tobacco use|pack[- ]?year|ppd|packs? per day|years? smoked)\b",
    re.IGNORECASE,
)
SMOKING_QUANTITATIVE_RE = re.compile(
    r"\b(\d+\.?\d*)\s*(?:pack[- ]?year|ppd|packs?\s*per\s*day|years?\s*(?:of\s+)?smok)",
    re.IGNORECASE,
)


def _manifest_hash(seed: int, subset_name: str, size: int) -> str:
    raw = f"{seed}-{subset_name}-{size}-phase4_v1"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _count_explicit_fields(text: str) -> dict:
    return {
        "has_size": bool(EXPLICIT_SIZE_RE.search(text)),
        "has_density": bool(EXPLICIT_DENSITY_RE.search(text)),
        "has_location": bool(EXPLICIT_LOCATION_RE.search(text)),
        "has_change": bool(EXPLICIT_CHANGE_RE.search(text)),
        "has_recommendation_cue": bool(EXPLICIT_RECOMMENDATION_RE.search(text)),
    }


def _count_true(d: dict) -> int:
    return sum(1 for v in d.values() if v)


def build_radiology_explicit_eval(
    nrows: int | None = None,
    target_size: int = 500,
    seed: int = 42,
    min_explicit_fields: int = 1,
    cache_dir: Path | None = None,
) -> dict:
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "radiology_candidates_cache.jsonl"
        if cache_path.exists():
            candidates = _load_jsonl(cache_path)
            return _select_radiology_subset(candidates, target_size, seed, min_explicit_fields)

    radiology_df = load_radiology(nrows=nrows)
    detail_df = load_radiology_detail(nrows=nrows)
    chest_ct_df = filter_chest_ct(radiology_df, detail_df)
    nodule_df = filter_nodule_reports(chest_ct_df)

    candidates = []
    for _, row in nodule_df.iterrows():
        text = str(row.get("text") or "")
        explicit_fields = _count_explicit_fields(text)
        explicit_count = _count_true(explicit_fields)

        candidates.append({
            "note_id": str(row.get("note_id", "")),
            "subject_id": int(row.get("subject_id", 0)),
            "exam_name": str(row.get("exam_name", "")),
            "text": text,
            "explicit_fields": explicit_fields,
            "explicit_field_count": explicit_count,
        })

    if cache_dir:
        cache_path = Path(cache_dir) / "radiology_candidates_cache.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_jsonl(candidates, cache_path)

    return _select_radiology_subset(candidates, target_size, seed, min_explicit_fields)


def _select_radiology_subset(
    candidates: list[dict],
    target_size: int,
    seed: int,
    min_explicit_fields: int,
) -> dict:
    qualified = [c for c in candidates if c["explicit_field_count"] >= min_explicit_fields]
    qualified.sort(key=lambda x: (-x["explicit_field_count"], x["note_id"]))

    import random
    rng = random.Random(seed)
    if len(qualified) > target_size:
        qualified = qualified[:target_size * 3]
        rng.shuffle(qualified)
        qualified = qualified[:target_size]
    qualified.sort(key=lambda x: x["note_id"])

    field_stats = Counter()
    for c in qualified:
        for field, present in c["explicit_fields"].items():
            if present:
                field_stats[field] += 1

    return {
        "subset_name": "radiology_explicit_eval",
        "manifest_version": "phase4_v1",
        "manifest_hash": _manifest_hash(seed, "radiology_explicit_eval", len(qualified)),
        "creation_date": date.today().isoformat(),
        "seed": seed,
        "total_candidates": len(candidates) if candidates else 0,
        "qualified_candidates": len([c for c in (candidates or []) if c.get("explicit_field_count", 0) >= min_explicit_fields]),
        "selected_count": len(qualified),
        "min_explicit_fields": min_explicit_fields,
        "field_coverage": dict(field_stats),
        "samples": qualified,
    }


def build_smoking_explicit_eval(
    nrows: int | None = None,
    target_size: int = 500,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> dict:
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "smoking_candidates_cache.jsonl"
        if cache_path.exists():
            candidates = _load_jsonl(cache_path)
            return _select_smoking_subset(candidates, target_size, seed)

    discharge_df = load_discharge(nrows=nrows)
    text_col = "text" if "text" in discharge_df.columns else "note_text"

    candidates = []
    for _, row in discharge_df.iterrows():
        text = str(row.get(text_col) or "")
        subject_id = int(row.get("subject_id", 0))
        note_id = str(row.get("note_id", f"{subject_id}-DS-0"))

        has_status_cue = bool(SMOKING_STATUS_RE.search(text))
        has_quantitative_cue = bool(SMOKING_QUANTITATIVE_RE.search(text))

        if not has_status_cue and not has_quantitative_cue:
            continue

        section_name, section_text = find_social_history_section(text)
        cue_source = "social_history" if section_name else "full_text_fallback"

        candidates.append({
            "note_id": note_id,
            "subject_id": subject_id,
            "text": text,
            "has_status_cue": has_status_cue,
            "has_quantitative_cue": has_quantitative_cue,
            "cue_source": cue_source,
        })

    if cache_dir:
        cache_path = Path(cache_dir) / "smoking_candidates_cache.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_jsonl(candidates, cache_path)

    return _select_smoking_subset(candidates, target_size, seed)


def _select_smoking_subset(
    candidates: list[dict],
    target_size: int,
    seed: int,
) -> dict:
    quantitative = [c for c in candidates if c["has_quantitative_cue"]]
    status_only = [c for c in candidates if c["has_status_cue"] and not c["has_quantitative_cue"]]

    import random
    rng = random.Random(seed)

    quant_target = min(len(quantitative), target_size // 3)
    status_target = min(len(status_only), target_size - quant_target)

    rng.shuffle(quantitative)
    rng.shuffle(status_only)

    selected = quantitative[:quant_target] + status_only[:status_target]
    selected.sort(key=lambda x: x["note_id"])

    cue_stats = Counter()
    for c in selected:
        if c["has_quantitative_cue"]:
            cue_stats["quantitative_cue"] += 1
        if c["has_status_cue"]:
            cue_stats["status_cue"] += 1
        cue_stats[c["cue_source"]] += 1

    return {
        "subset_name": "smoking_explicit_eval",
        "manifest_version": "phase4_v1",
        "manifest_hash": _manifest_hash(seed, "smoking_explicit_eval", len(selected)),
        "creation_date": date.today().isoformat(),
        "seed": seed,
        "total_candidates": len(candidates),
        "selected_count": len(selected),
        "cue_distribution": dict(cue_stats),
        "samples": selected,
    }


def build_recommendation_eval(
    radiology_facts: list[dict],
    case_bundles: list[dict] | None = None,
    target_size: int = 500,
    seed: int = 42,
) -> dict:
    cue_samples = []
    rule_samples = []
    insufficient_samples = []

    for fact in radiology_facts:
        has_cue = False
        has_size = False
        for nodule in fact.get("nodules", []):
            if nodule.get("recommendation_cue"):
                has_cue = True
            if nodule.get("size_mm") is not None:
                has_size = True

        if has_cue:
            cue_samples.append({
                "note_id": fact["note_id"],
                "subject_id": fact["subject_id"],
                "eval_type": "explicit_cue",
                "has_recommendation_cue": True,
                "has_size": has_size,
            })
        elif has_size:
            rule_samples.append({
                "note_id": fact["note_id"],
                "subject_id": fact["subject_id"],
                "eval_type": "rule_derived",
                "has_recommendation_cue": False,
                "has_size": True,
            })
        else:
            insufficient_samples.append({
                "note_id": fact["note_id"],
                "subject_id": fact["subject_id"],
                "eval_type": "insufficient_data",
                "has_recommendation_cue": False,
                "has_size": False,
            })

    import random
    rng = random.Random(seed)

    cue_target = min(len(cue_samples), target_size // 3)
    rule_target = min(len(rule_samples), target_size // 3)
    insuf_target = min(len(insufficient_samples), target_size - cue_target - rule_target)

    rng.shuffle(cue_samples)
    rng.shuffle(rule_samples)
    rng.shuffle(insufficient_samples)

    selected = (
        cue_samples[:cue_target]
        + rule_samples[:rule_target]
        + insufficient_samples[:insuf_target]
    )
    selected.sort(key=lambda x: x["note_id"])

    type_stats = Counter(s["eval_type"] for s in selected)

    return {
        "subset_name": "recommendation_eval",
        "manifest_version": "phase4_v1",
        "manifest_hash": _manifest_hash(seed, "recommendation_eval", len(selected)),
        "creation_date": date.today().isoformat(),
        "seed": seed,
        "total_cue_candidates": len(cue_samples),
        "total_rule_candidates": len(rule_samples),
        "total_insufficient_candidates": len(insufficient_samples),
        "selected_count": len(selected),
        "eval_type_distribution": dict(type_stats),
        "samples": selected,
    }


def build_case_study_set(
    radiology_facts: list[dict],
    smoking_results: list[dict] | None = None,
    target_size: int = 16,
    seed: int = 42,
) -> dict:
    coverage_targets = {
        "solid": False,
        "ground_glass": False,
        "part_solid": False,
        "multiple_nodules": False,
        "size_missing": False,
        "smoking_unknown": False,
        "recommendation_cue_present": False,
        "recommendation_cue_absent": False,
        "high_confidence": False,
        "low_confidence": False,
        "change_present": False,
        "calcified": False,
    }

    smoking_by_subject = {}
    if smoking_results:
        for sr in smoking_results:
            sid = sr.get("subject_id")
            if sid is not None:
                smoking_by_subject[sid] = sr

    categorized = []
    for fact in radiology_facts:
        tags = set()
        nodules = fact.get("nodules", [])
        if not nodules:
            continue

        for nodule in nodules:
            density = nodule.get("density_category")
            if density == "solid":
                tags.add("solid")
            elif density == "ground_glass":
                tags.add("ground_glass")
            elif density == "part_solid":
                tags.add("part_solid")
            elif density == "calcified":
                tags.add("calcified")

            if nodule.get("size_mm") is None:
                tags.add("size_missing")
            if nodule.get("recommendation_cue"):
                tags.add("recommendation_cue_present")
            else:
                tags.add("recommendation_cue_absent")
            if nodule.get("confidence") == "high":
                tags.add("high_confidence")
            if nodule.get("confidence") == "low":
                tags.add("low_confidence")
            if nodule.get("change_status") not in (None, "unclear"):
                tags.add("change_present")

        if fact.get("nodule_count", 0) > 1:
            tags.add("multiple_nodules")

        smoking = smoking_by_subject.get(fact["subject_id"])
        if smoking and smoking.get("smoking_status_norm") == "unknown":
            tags.add("smoking_unknown")

        categorized.append({
            "note_id": fact["note_id"],
            "subject_id": fact["subject_id"],
            "tags": sorted(tags),
            "nodule_count": fact.get("nodule_count", 0),
        })

    selected = []
    used_note_ids = set()

    for target_tag in coverage_targets:
        for item in categorized:
            if item["note_id"] in used_note_ids:
                continue
            if target_tag in item["tags"]:
                selected.append(item)
                used_note_ids.add(item["note_id"])
                coverage_targets[target_tag] = True
                break

    import random
    rng = random.Random(seed)
    remaining = [c for c in categorized if c["note_id"] not in used_note_ids]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= target_size:
            break
        selected.append(item)
        used_note_ids.add(item["note_id"])

    selected.sort(key=lambda x: x["note_id"])

    covered = [tag for tag, hit in coverage_targets.items() if hit]
    uncovered = [tag for tag, hit in coverage_targets.items() if not hit]

    return {
        "subset_name": "case_study_set",
        "manifest_version": "phase4_v1",
        "manifest_hash": _manifest_hash(seed, "case_study_set", len(selected)),
        "creation_date": date.today().isoformat(),
        "seed": seed,
        "total_candidates": len(categorized),
        "selected_count": len(selected),
        "coverage_targets_met": covered,
        "coverage_targets_missed": uncovered,
        "samples": selected,
    }


def save_manifest(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
