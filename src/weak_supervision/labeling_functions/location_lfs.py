"""Location labeling functions for weak supervision."""

from __future__ import annotations

import re

from src.weak_supervision.base import LFOutput, ABSTAIN, MentionRecord


_L1_PATTERNS = [
    (re.compile(r"\bbilateral\b", re.IGNORECASE), "bilateral"),
    (re.compile(r"\bboth\s+lungs\b", re.IGNORECASE), "bilateral"),
    (re.compile(r"\b(right\s+upper\s+lobe|rul)\b", re.IGNORECASE), "RUL"),
    (re.compile(r"\b(right\s+middle\s+lobe|rml)\b", re.IGNORECASE), "RML"),
    (re.compile(r"\b(right\s+lower\s+lobe|rll)\b", re.IGNORECASE), "RLL"),
    (re.compile(r"\b(left\s+upper\s+lobe|lul)\b", re.IGNORECASE), "LUL"),
    (re.compile(r"\b(left\s+lower\s+lobe|lll)\b", re.IGNORECASE), "LLL"),
    (re.compile(r"\blingula\b", re.IGNORECASE), "lingula"),
    (re.compile(r"\b(right\s+lung|left\s+lung)\b", re.IGNORECASE), "unclear"),
]

_SPECIFIC_LOBE_PATTERNS = [
    (re.compile(r"\b(right\s+upper\s+lobe|rul)\b", re.IGNORECASE), "RUL", "right"),
    (re.compile(r"\b(right\s+middle\s+lobe|rml)\b", re.IGNORECASE), "RML", "right"),
    (re.compile(r"\b(right\s+lower\s+lobe|rll)\b", re.IGNORECASE), "RLL", "right"),
    (re.compile(r"\b(left\s+upper\s+lobe|lul)\b", re.IGNORECASE), "LUL", "left"),
    (re.compile(r"\b(left\s+lower\s+lobe|lll)\b", re.IGNORECASE), "LLL", "left"),
    (re.compile(r"\blingula\b", re.IGNORECASE), "lingula", "left"),
]

_BILATERAL_KEYWORD_PATTERNS = [
    re.compile(r"\bbilateral\b", re.IGNORECASE),
    re.compile(r"\bbilaterally\b", re.IGNORECASE),
    re.compile(r"\bboth\s+lungs\b", re.IGNORECASE),
    re.compile(r"\bboth\s+lung\s+fields\b", re.IGNORECASE),
    re.compile(r"\bscattered\s+throughout\s+both\b", re.IGNORECASE),
    re.compile(r"\bscattered\s+throughout\b", re.IGNORECASE),
    re.compile(r"\bdiffuse\b", re.IGNORECASE),
    re.compile(r"\bdiffusely\b", re.IGNORECASE),
    re.compile(r"\bthroughout\s+the\s+lungs\b", re.IGNORECASE),
    re.compile(r"\bmultiple\s+bilateral\b", re.IGNORECASE),
]

_LATERALITY_PATTERNS = [
    (re.compile(r"\bright\s+lung\b", re.IGNORECASE), "right"),
    (re.compile(r"\bleft\s+lung\b", re.IGNORECASE), "left"),
    (re.compile(r"\bright\s*[- ]sided\b", re.IGNORECASE), "right"),
    (re.compile(r"\bleft\s*[- ]sided\b", re.IGNORECASE), "left"),
    (re.compile(r"\bright\s+(?:pulmonary|lung|hemithorax)\b", re.IGNORECASE), "right"),
    (re.compile(r"\bleft\s+(?:pulmonary|lung|hemithorax)\b", re.IGNORECASE), "left"),
]


def _abstain(lf_name: str) -> LFOutput:
    return LFOutput(lf_name=lf_name, label=ABSTAIN, confidence=0.0)


def _text(record: MentionRecord, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _match_l1_pattern(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    for pattern, label in _L1_PATTERNS:
        match = pattern.search(text)
        if match:
            return label, match.group(0)
    return None, None


def _has_specific_lobe(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern, _, _ in _SPECIFIC_LOBE_PATTERNS)


def _collect_specific_lobes(text: str) -> list[dict[str, str | int]]:
    matches = []
    for pattern, label, side in _SPECIFIC_LOBE_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append(
                {
                    "label": label,
                    "side": side,
                    "text": match.group(0),
                    "start": match.start(),
                }
            )
    return sorted(matches, key=lambda item: int(item["start"]))


def lf_location_lobe_exact(record: MentionRecord) -> LFOutput:
    mention_text = _text(record, "mention_text")
    label, evidence = _match_l1_pattern(mention_text)
    if not label:
        return _abstain("LF-L1")
    return LFOutput(
        lf_name="LF-L1",
        label=label,
        confidence=1.0,
        evidence_span=evidence,
    )


def lf_location_multi_lobe(record: MentionRecord) -> LFOutput:
    mention_text = _text(record, "mention_text")
    matches = _collect_specific_lobes(mention_text)
    if len(matches) <= 1:
        return _abstain("LF-L2")

    sides = {str(match["side"]) for match in matches}
    evidence = ", ".join(str(match["text"]) for match in matches)
    if len(sides) >= 2:
        return LFOutput(
            lf_name="LF-L2",
            label="bilateral",
            confidence=0.85,
            evidence_span=evidence,
        )

    return LFOutput(
        lf_name="LF-L2",
        label=str(matches[0]["label"]),
        confidence=0.7,
        evidence_span=evidence,
    )


def lf_location_bilateral_keyword(record: MentionRecord) -> LFOutput:
    mention_text = _text(record, "mention_text")
    for pattern in _BILATERAL_KEYWORD_PATTERNS:
        match = pattern.search(mention_text)
        if match:
            return LFOutput(
                lf_name="LF-L3",
                label="bilateral",
                confidence=0.9,
                evidence_span=match.group(0),
            )
    return _abstain("LF-L3")


def lf_location_laterality_inference(record: MentionRecord) -> LFOutput:
    mention_text = _text(record, "mention_text")
    if _has_specific_lobe(mention_text):
        return _abstain("LF-L4")

    for pattern, laterality in _LATERALITY_PATTERNS:
        match = pattern.search(mention_text)
        if match:
            return LFOutput(
                lf_name="LF-L4",
                label="unclear",
                confidence=0.75,
                evidence_span=match.group(0),
                source_meta={"laterality": laterality},
            )
    return _abstain("LF-L4")


def lf_location_context_window(record: MentionRecord) -> LFOutput:
    mention_text = _text(record, "mention_text")
    full_text = _text(record, "full_text")
    mention_label, _ = _match_l1_pattern(mention_text)
    if mention_label:
        return _abstain("LF-L5")
    if not mention_text or not full_text:
        return _abstain("LF-L5")

    mention_start = full_text.lower().find(mention_text.lower())
    if mention_start < 0:
        return _abstain("LF-L5")

    mention_end = mention_start + len(mention_text)
    window_start = max(0, mention_start - 200)
    window_end = min(len(full_text), mention_end + 200)
    context_text = full_text[window_start:mention_start] + full_text[mention_end:window_end]

    label, evidence = _match_l1_pattern(context_text)
    if not label:
        return _abstain("LF-L5")
    return LFOutput(
        lf_name="LF-L5",
        label=label,
        confidence=0.7,
        evidence_span=evidence,
        source_meta={"source": "context_window"},
    )


LOCATION_LFS = [
    lf_location_lobe_exact,
    lf_location_multi_lobe,
    lf_location_bilateral_keyword,
    lf_location_laterality_inference,
    lf_location_context_window,
]
