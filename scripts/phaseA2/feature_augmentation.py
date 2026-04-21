#!/usr/bin/env python3
"""Cue-feature augmentation for learned Plan B models.

The helpers in this file do not make deterministic predictions. They encode
rule-like cues as additional text features, then the neural classifier remains
the only decision maker.
"""
from __future__ import annotations

import re
from typing import Any

NUMBER = r"(?:\d+(?:\.\d+)?)"
UNIT = r"(?:mm|millimeters?|cm|centimeters?)"
SEP = r"(?:x|X|×|by)"

NUMERIC_UNIT_RE = re.compile(rf"\b{NUMBER}\s*{UNIT}\b", re.IGNORECASE)
TWO_D_RE = re.compile(rf"\b{NUMBER}\s*{SEP}\s*{NUMBER}\s*(?:{UNIT})?\b", re.IGNORECASE)
THREE_D_RE = re.compile(rf"\b{NUMBER}\s*{SEP}\s*{NUMBER}\s*{SEP}\s*{NUMBER}\s*(?:{UNIT})?\b", re.IGNORECASE)
RANGE_RE = re.compile(
    rf"\b(?:{NUMBER}\s*(?:-|to)\s*{NUMBER}\s*{UNIT}|between\s+{NUMBER}\s+and\s+{NUMBER}\s*{UNIT})\b",
    re.IGNORECASE,
)
SIZE_CONTEXT_RE = re.compile(
    r"\b(?:measur(?:e|es|ed|ing)|diameter|dimension|size|nodule[s]?\s+(?:is|are|measures?))\b",
    re.IGNORECASE,
)

LOCATION_ABBREVIATIONS = {
    "RUL": re.compile(r"\b(?:RUL|right\s+upper\s+lobe)\b", re.IGNORECASE),
    "RML": re.compile(r"\b(?:RML|right\s+middle\s+lobe)\b", re.IGNORECASE),
    "RLL": re.compile(r"\b(?:RLL|right\s+lower\s+lobe)\b", re.IGNORECASE),
    "LUL": re.compile(r"\b(?:LUL|left\s+upper\s+lobe)\b", re.IGNORECASE),
    "LLL": re.compile(r"\b(?:LLL|left\s+lower\s+lobe)\b", re.IGNORECASE),
}
LOCATION_WORDS = {
    "lingula": re.compile(r"\blingula(?:r)?\b", re.IGNORECASE),
    "bilateral": re.compile(r"\b(?:bilateral|both\s+lungs|both\s+sides)\b", re.IGNORECASE),
    "right": re.compile(r"\bright\b", re.IGNORECASE),
    "left": re.compile(r"\bleft\b", re.IGNORECASE),
    "upper": re.compile(r"\bupper\b", re.IGNORECASE),
    "middle": re.compile(r"\bmiddle\b", re.IGNORECASE),
    "lower": re.compile(r"\blower\b", re.IGNORECASE),
    "lobe": re.compile(r"\blobe\b", re.IGNORECASE),
    "apical": re.compile(r"\bapic(?:al|es?)\b", re.IGNORECASE),
    "basal": re.compile(r"\b(?:basal|base|bases)\b", re.IGNORECASE),
}


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3plus"


def size_cue_features(mention_text: str) -> dict[str, str]:
    """Extract size cues as symbolic text features."""
    numeric_units = NUMERIC_UNIT_RE.findall(mention_text or "")
    return {
        "size_unit_mm_cm": _yes_no(bool(re.search(rf"\b{UNIT}\b", mention_text or "", re.IGNORECASE))),
        "size_numeric_unit": _yes_no(bool(numeric_units)),
        "size_numeric_unit_count": _count_bucket(len(numeric_units)),
        "size_2d_pattern": _yes_no(bool(TWO_D_RE.search(mention_text or ""))),
        "size_3d_pattern": _yes_no(bool(THREE_D_RE.search(mention_text or ""))),
        "size_range_pattern": _yes_no(bool(RANGE_RE.search(mention_text or ""))),
        "size_context_word": _yes_no(bool(SIZE_CONTEXT_RE.search(mention_text or ""))),
    }


def location_cue_features(mention_text: str) -> dict[str, str]:
    """Extract location cues as symbolic text features."""
    text = mention_text or ""
    features: dict[str, str] = {}
    for label, pattern in LOCATION_ABBREVIATIONS.items():
        features[f"loc_{label.lower()}"] = _yes_no(bool(pattern.search(text)))
    for label, pattern in LOCATION_WORDS.items():
        features[f"loc_{label}"] = _yes_no(bool(pattern.search(text)))
    abbreviation_hits = sum(1 for pattern in LOCATION_ABBREVIATIONS.values() if pattern.search(text))
    directional_hits = sum(1 for pattern in LOCATION_WORDS.values() if pattern.search(text))
    features["loc_abbreviation_count"] = _count_bucket(abbreviation_hits)
    features["loc_directional_count"] = _count_bucket(directional_hits)
    return features


def format_feature_block(task: str, features: dict[str, str]) -> str:
    ordered = " ; ".join(f"{key}={value}" for key, value in sorted(features.items()))
    return f"[{task.upper()} CUE FEATURES] {ordered}"


def add_cue_augmented_text(
    row: dict[str, Any],
    *,
    task: str,
    source_field: str = "section_aware_text",
    output_field: str = "cue_augmented_text",
) -> dict[str, Any]:
    """Return a copy of row with cue features prepended to the learned input."""
    out = dict(row)
    mention = str(out.get("mention_text") or "")
    base_text = str(out.get(source_field) or out.get("mention_text") or "")
    if task == "size":
        features = size_cue_features(mention)
    elif task == "location":
        features = location_cue_features(mention)
    else:
        raise ValueError(f"Unsupported augmentation task: {task}")
    out[output_field] = f"{format_feature_block(task, features)}\n{base_text}".strip()
    out["cue_feature_source"] = "mention_text"
    out["cue_feature_mode"] = f"{task}_text_feature_augmentation"
    return out


def add_size_cue_augmented_text(row: dict[str, Any]) -> dict[str, Any]:
    return add_cue_augmented_text(row, task="size")


def add_location_cue_augmented_text(row: dict[str, Any]) -> dict[str, Any]:
    return add_cue_augmented_text(row, task="location")
