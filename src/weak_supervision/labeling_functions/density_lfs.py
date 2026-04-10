"""Density labeling functions for weak supervision."""

from __future__ import annotations

import re

from src.weak_supervision.base import ABSTAIN, LFOutput, MentionRecord


_PRIMARY_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "part_solid",
        "primary_exact",
        re.compile(
            r"\b(part\s*[- ]?solid|partially\s+solid|semi\s*[- ]?solid|semisolid)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ground_glass",
        "primary_exact",
        re.compile(r"\b(ground\s*[- ]?glass|ggo|ggn)\b", re.IGNORECASE),
    ),
    (
        "calcified",
        "primary_exact",
        re.compile(r"\b(calcified|calcification)\b", re.IGNORECASE),
    ),
    (
        "solid",
        "primary_exact",
        re.compile(r"\bsolid\b", re.IGNORECASE),
    ),
]

_FUZZY_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "part_solid",
        "fuzzy_subsolid",
        re.compile(r"\b(sub\s*[- ]?solid)\b", re.IGNORECASE),
    ),
    (
        "ground_glass",
        "fuzzy_hazy_frosted",
        re.compile(r"\b(hazy\s+opacity|frosted\s+glass|hazy|frosted)\b", re.IGNORECASE),
    ),
    (
        "ground_glass",
        "fuzzy_attenuation",
        re.compile(r"\b(low\s*[- ]?attenuation|attenuating)\b", re.IGNORECASE),
    ),
    (
        "solid",
        "fuzzy_soft_tissue",
        re.compile(
            r"\b(soft\s*[- ]?tissue\s+(?:density|attenuation))\b",
            re.IGNORECASE,
        ),
    ),
]

_NODULE_KEYWORD_RE = re.compile(
    r"\b(nodule|nodules|opacity|lesion|ggo|ggn|ground\s+glass|granuloma)\b",
    re.IGNORECASE,
)
_DENSITY_TOKEN_RE = re.compile(
    r"\b(part\s*[- ]?solid|partially\s+solid|semi\s*[- ]?solid|semisolid|sub\s*[- ]?solid|"
    r"ground\s*[- ]?glass|ggo|ggn|calcified|calcification|solid|hazy\s+opacity|frosted\s+glass|"
    r"hazy|frosted|attenuating|low\s*[- ]?attenuation|soft\s*[- ]?tissue\s+(?:density|attenuation)|"
    r"hyperdense|dense)\b",
    re.IGNORECASE,
)
_NEGATION_PATTERNS = [
    "no evidence of",
    "negative for",
    "not definitely",
    "without",
    "unlikely",
    "absent",
    "not",
    "no",
    "non",
]
_PART_SOLID_EXCLUSION_RE = re.compile(
    r"\b(part\s*[- ]?solid|partially\s+solid|semi\s*[- ]?solid|semisolid|sub\s*[- ]?solid)\b",
    re.IGNORECASE,
)


def _abstain(lf_name: str) -> LFOutput:
    return LFOutput(lf_name=lf_name, label=ABSTAIN)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[\.!?;])\s+|\n+", text) if part.strip()]


def _match_dense_calcification(text: str) -> re.Match[str] | None:
    for sentence in _split_sentences(text):
        if not re.search(r"\b(calcification|calcified)\b", sentence, re.IGNORECASE):
            continue
        match = re.search(r"\b(hyperdense|dense)\b", sentence, re.IGNORECASE)
        if match:
            return match
    return None


def _scan_patterns(text: str, patterns: list[tuple[str, str, re.Pattern[str]]]) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    exclusion_spans = [match.span() for match in _PART_SOLID_EXCLUSION_RE.finditer(text or "")]
    for label, pattern_name, pattern in patterns:
        for match in pattern.finditer(text or ""):
            if label == "solid" and any(start <= match.start() and match.end() <= end for start, end in exclusion_spans):
                continue
            matches.append((label, pattern_name, match.group(0)))
    dense_match = _match_dense_calcification(text or "")
    if dense_match:
        matches.append(("calcified", "fuzzy_dense_calcification", dense_match.group(0)))
    return matches


def _find_first_priority_match(text: str, patterns: list[tuple[str, str, re.Pattern[str]]]) -> tuple[str, str, str] | None:
    scanned = _scan_patterns(text, patterns)
    priority = {"part_solid": 0, "ground_glass": 1, "calcified": 2, "solid": 3}
    if not scanned:
        return None
    scanned.sort(key=lambda item: priority[item[0]])
    return scanned[0]


def _extract_impression_text(full_text: str) -> str:
    if not full_text:
        return ""
    match = re.search(r"IMPRESSION\s*:\s*(.*)", full_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def lf_density_keyword_exact(record: MentionRecord) -> LFOutput:
    text = record.get("mention_text", "") or ""
    match = _find_first_priority_match(text, _PRIMARY_PATTERNS)
    if not match:
        return _abstain("LF-D1")

    label, pattern_name, evidence_span = match
    return LFOutput(
        lf_name="LF-D1",
        label=label,
        confidence=1.0,
        evidence_span=evidence_span,
        source_meta={"pattern": pattern_name, "source_field": "mention_text"},
    )


def lf_density_keyword_fuzzy(record: MentionRecord) -> LFOutput:
    text = record.get("mention_text", "") or ""
    match = _find_first_priority_match(text, _FUZZY_PATTERNS)
    if not match:
        return _abstain("LF-D2")

    label, pattern_name, evidence_span = match
    return LFOutput(
        lf_name="LF-D2",
        label=label,
        confidence=0.75,
        evidence_span=evidence_span,
        source_meta={"pattern": pattern_name, "source_field": "mention_text"},
    )


def lf_density_negation_aware(record: MentionRecord) -> LFOutput:
    text = record.get("mention_text", "") or ""
    if not text:
        return _abstain("LF-D3")

    for negation in _NEGATION_PATTERNS:
        negation_text = re.escape(negation).replace(r"\ ", r"\s+")
        negation_pattern = re.compile(
            rf"\b{negation_text}\b(?:\W+\w+){{0,3}}\W+{_DENSITY_TOKEN_RE.pattern}",
            re.IGNORECASE,
        )
        match = negation_pattern.search(text)
        if match:
            return LFOutput(
                lf_name="LF-D3",
                label="unclear",
                confidence=0.85,
                evidence_span=match.group(0),
                source_meta={"negation_pattern": negation, "source_field": "mention_text"},
            )

    noncalcified_re = re.compile(r"\b(non\s*[- ]?calcified|noncalcified)\b", re.IGNORECASE)
    match = noncalcified_re.search(text)
    if match:
        return LFOutput(
            lf_name="LF-D3",
            label="unclear",
            confidence=0.85,
            evidence_span=match.group(0),
            source_meta={"negation_pattern": "noncalcified", "source_field": "mention_text"},
        )

    return _abstain("LF-D3")


def lf_density_multi_density(record: MentionRecord) -> LFOutput:
    text = record.get("mention_text", "") or ""
    matches = _scan_patterns(text, _PRIMARY_PATTERNS + _FUZZY_PATTERNS)
    distinct_labels: list[str] = []
    evidence_terms: list[str] = []

    for label, _, evidence in matches:
        if label not in distinct_labels:
            distinct_labels.append(label)
        if evidence not in evidence_terms:
            evidence_terms.append(evidence)

    if len(distinct_labels) < 2:
        return _abstain("LF-D4")

    return LFOutput(
        lf_name="LF-D4",
        label="unclear",
        confidence=0.9,
        evidence_span=" | ".join(evidence_terms),
        source_meta={"distinct_labels": distinct_labels, "source_field": "mention_text"},
    )


def lf_density_impression_cue(record: MentionRecord) -> LFOutput:
    section = (record.get("section", "") or "").lower()
    if section == "impression":
        return _abstain("LF-D5")
    if section != "findings":
        return _abstain("LF-D5")

    impression_text = _extract_impression_text(record.get("full_text", "") or "")
    if not impression_text:
        return _abstain("LF-D5")

    for sentence in _split_sentences(impression_text):
        if not _NODULE_KEYWORD_RE.search(sentence):
            continue
        match = _find_first_priority_match(sentence, _PRIMARY_PATTERNS)
        if not match:
            continue
        label, pattern_name, evidence_span = match
        return LFOutput(
            lf_name="LF-D5",
            label=label,
            confidence=0.85,
            evidence_span=evidence_span,
            source_meta={
                "pattern": pattern_name,
                "source_field": "full_text.impression",
                "matched_sentence": sentence,
            },
        )

    return _abstain("LF-D5")


DENSITY_LFS = [
    lf_density_keyword_exact,
    lf_density_keyword_fuzzy,
    lf_density_negation_aware,
    lf_density_multi_density,
    lf_density_impression_cue,
]
