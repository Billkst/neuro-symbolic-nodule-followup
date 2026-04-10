"""Size labeling functions for weak supervision."""

from __future__ import annotations

import re

from src.weak_supervision.base import ABSTAIN, LFOutput, MentionRecord


STANDARD_SIZE_PATTERNS = (
    ("range_mm", re.compile(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*mm\b", re.IGNORECASE)),
    (
        "three_dim_mm",
        re.compile(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*mm\b", re.IGNORECASE),
    ),
    ("two_dim_mm", re.compile(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*mm\b", re.IGNORECASE)),
    (
        "three_dim_cm",
        re.compile(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*cm\b", re.IGNORECASE),
    ),
    ("two_dim_cm", re.compile(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*cm\b", re.IGNORECASE)),
    ("single_mm", re.compile(r"(\d+\.?\d*)\s*mm\b", re.IGNORECASE)),
    ("single_cm", re.compile(r"(\d+\.?\d*)\s*cm\b", re.IGNORECASE)),
)

TOLERANT_SIZE_PATTERNS = (
    ("missing_space_mm", re.compile(r"(\d+)mm\b", re.IGNORECASE)),
    ("typo_mmm", re.compile(r"(\d+)mmm\b", re.IGNORECASE)),
    ("single_m_typo", re.compile(r"(\d+)\s*m\b", re.IGNORECASE)),
    ("measuring_without_unit", re.compile(r"measuring\s+(\d+\.?\d*)", re.IGNORECASE)),
    ("concatenated_mmsince", re.compile(r"(\d+)mmsince\b", re.IGNORECASE)),
    ("concatenated_mmand", re.compile(r"(\d+)mmand\b", re.IGNORECASE)),
)

SIZE_CONTEXT_WORDS = (
    "measuring",
    "measures",
    "diameter",
    "size",
    "sized",
    "dimension",
    "largest",
    "smallest",
    "approximately",
)
SIZE_CONTEXT_RE = re.compile(r"\b(?:" + "|".join(SIZE_CONTEXT_WORDS) + r")\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")

QUALITATIVE_SIZE_PATTERNS = (
    "subcentimeter",
    "sub-centimeter",
    r"sub\s+centimeter",
    "tiny",
    "minute",
    "punctate",
    "large",
    "massive",
    "giant",
    "millimetric",
    "submillimeter",
)
QUALITATIVE_SIZE_RE = re.compile(r"\b(?:" + "|".join(QUALITATIVE_SIZE_PATTERNS) + r")\b", re.IGNORECASE)


def _abstain(lf_name: str) -> LFOutput:
    return LFOutput(lf_name=lf_name, label=ABSTAIN, confidence=0.0)


def _sentence_candidates(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[\.!?;])\s+|\n+", text) if part and part.strip()]


def lf_size_regex_standard(record: MentionRecord) -> LFOutput:
    mention_text = record.get("mention_text") or ""
    lf_name = "LF-S1"

    for pattern_name, pattern in STANDARD_SIZE_PATTERNS:
        match = pattern.search(mention_text)
        if match:
            return LFOutput(
                lf_name=lf_name,
                label="true",
                confidence=1.0,
                evidence_span=match.group(0),
                source_meta={"pattern": pattern_name},
            )

    return _abstain(lf_name)


def lf_size_regex_tolerant(record: MentionRecord) -> LFOutput:
    mention_text = record.get("mention_text") or ""
    lf_name = "LF-S2"

    for pattern_name, pattern in TOLERANT_SIZE_PATTERNS:
        match = pattern.search(mention_text)
        if match:
            return LFOutput(
                lf_name=lf_name,
                label="true",
                confidence=0.9,
                evidence_span=match.group(0),
                source_meta={"pattern": pattern_name},
            )

    return _abstain(lf_name)


def lf_size_numeric_context(record: MentionRecord) -> LFOutput:
    mention_text = record.get("mention_text") or ""
    lf_name = "LF-S3"

    for sentence in _sentence_candidates(mention_text):
        if NUMBER_RE.search(sentence) and SIZE_CONTEXT_RE.search(sentence):
            return LFOutput(
                lf_name=lf_name,
                label="true",
                confidence=0.8,
                evidence_span=sentence,
                source_meta={"cue_type": "numeric_context"},
            )

    if NUMBER_RE.search(mention_text) and SIZE_CONTEXT_RE.search(mention_text):
        return LFOutput(
            lf_name=lf_name,
            label="true",
            confidence=0.8,
            evidence_span=mention_text.strip() or None,
            source_meta={"cue_type": "numeric_context"},
        )

    return _abstain(lf_name)


def lf_size_subcentimeter_cue(record: MentionRecord) -> LFOutput:
    mention_text = record.get("mention_text") or ""
    lf_name = "LF-S4"

    match = QUALITATIVE_SIZE_RE.search(mention_text)
    if not match:
        return _abstain(lf_name)

    return LFOutput(
        lf_name=lf_name,
        label="true",
        confidence=0.7,
        evidence_span=match.group(0),
        source_meta={"cue_word": match.group(0).lower()},
    )


def lf_size_no_size_negative(record: MentionRecord) -> LFOutput:
    mention_text = record.get("mention_text") or ""
    lf_name = "LF-S5"

    has_digits = bool(re.search(r"\d", mention_text))
    has_qualitative_size = bool(QUALITATIVE_SIZE_RE.search(mention_text))
    has_size_context = bool(SIZE_CONTEXT_RE.search(mention_text))

    if not has_digits and not has_qualitative_size and not has_size_context:
        return LFOutput(
            lf_name=lf_name,
            label="false",
            confidence=0.9,
            evidence_span=mention_text.strip() or None,
            source_meta={"rule": "no_size_evidence"},
        )

    return _abstain(lf_name)


SIZE_LFS = [
    lf_size_regex_standard,
    lf_size_regex_tolerant,
    lf_size_numeric_context,
    lf_size_subcentimeter_cue,
    lf_size_no_size_negative,
]
