import re
from datetime import datetime, timezone


_EXTRACTOR_VERSION = "smoking_extractor_v1.1"
_MODEL_NAME = "regex_rule_based"


def _to_float(num_text: str | None) -> float | None:
    if not num_text:
        return None
    t = num_text.strip().lower()
    if t in {"half", "a half", "half a"}:
        return 0.5
    if t == "1/2":
        return 0.5
    try:
        return float(t)
    except ValueError:
        return None


def _is_deidentified(text: str | None) -> bool:
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return re.fullmatch(r"[_\s\-*.]+", stripped) is not None


def find_social_history_section(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    start_match = re.search(r"(?im)^\s*social history\s*:\s*", text)
    if not start_match:
        return None, None

    start = start_match.end()
    tail = text[start:]
    next_header = re.search(
        r"(?im)^\s*(?:family history|review of systems|physical exam|past medical history|medications|allergies|assessment|plan|discharge medications|disposition|hospital course|history of present illness|hpi)\s*:\s*",
        tail,
    )
    end = start + next_header.start() if next_header else len(text)
    section_text = text[start:end].strip()
    return "Social History", section_text if section_text else None


def extract_smoking_status(text: str) -> tuple[str | None, str]:
    if _is_deidentified(text):
        return None, "unknown"

    never_patterns = [
        r"\bnever smoked\b",
        r"\bnon[-\s]?smoker\b",
        r"\bdenies smoking\b",
        r"\bdenies tobacco\b",
        r"\bno tobacco\b",
        r"\btobacco\s*:\s*never\b",
        r"\bno history of smoking\b",
    ]
    former_patterns = [
        r"\bformer smoker\b",
        r"\bex[-\s]?smoker\b",
        r"\bquit smoking\b",
        r"\bstopped smoking\b",
        r"\bused to smoke\b",
        r"\bprior smoker\b",
        r"\bformer tobacco use\b",
        r"\btobacco\s*:\s*former\b",
        r"\bquit\s+\d+(?:\.\d+)?\s+years?\s+(?:ago|prior)\b",
        r"\bhistory of smoking\b",
    ]
    current_patterns = [
        r"\bcurrent smoker\b",
        r"\bcurrently smokes?\b",
        r"\bactive smoker\b",
        r"\bsmokes?\s+\d+(?:\.\d+)?\s*(?:packs?|ppd)\b",
        r"\btobacco use\s*:\s*yes\b",
        r"\bsmokes\b",
        r"\bsmoker\b",
    ]

    for pat in never_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0), "never_smoker"

    for pat in former_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0), "former_smoker"

    for pat in current_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            raw = m.group(0)
            if re.search(r"\bformer\s+smoker\b|\bex[-\s]?smoker\b", text, flags=re.IGNORECASE):
                continue
            return raw, "current_smoker"

    m = re.search(r"\bsignificant\s+for\s+tobacco\b|\btobacco\s+use\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(0), "current_smoker"

    m = re.search(r"\btobacco\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(0), "unknown"
    return None, "unknown"


def _has_tobacco_context(text: str) -> bool:
    return (
        re.search(r"\b(smok|smoker|tobacco|cigarette|pack[-\s]?year|ppd|packs?\s+per\s+day)\b", text, re.IGNORECASE)
        is not None
    )


_SMOKING_SENTENCE_PATTERN = re.compile(
    r"\b(smok(?:e[drs]?|ing)|tobacco|cigarette|pack[-\s]?year|pk[-\s]?yr"
    r"|ppd|packs?\s+per\s+day|nicotine|quit\s+smoking|former\s+smoker"
    r"|current\s+smoker|non[-\s]?smoker|never\s+smoked|denies\s+(?:smoking|tobacco))\b",
    re.IGNORECASE,
)

_PPD_EXCLUDE_FULLTEXT = re.compile(
    r"\b(postpartum|delivery|ppd\s*test|tuberculin|\btb\b|induration|purified\s+protein)\b",
    re.IGNORECASE,
)


def _extract_smoking_sentences(full_text: str) -> str | None:
    """从全文中提取含烟草语义的句子，拼接返回。排除 PPD 歧义上下文。"""
    sentences = re.split(r"(?<=[.!?\n])\s+", full_text)
    relevant = []
    for sent in sentences:
        if not _SMOKING_SENTENCE_PATTERN.search(sent):
            continue
        if _PPD_EXCLUDE_FULLTEXT.search(sent) and not re.search(
            r"\b(smok|tobacco|cigarette|pack[-\s]?year|nicotine)\b", sent, re.IGNORECASE
        ):
            continue
        relevant.append(sent.strip())
    if not relevant:
        return None
    return " ".join(relevant)


def extract_pack_years(text: str) -> tuple[str | None, float | None]:
    if not text:
        return None, None

    explicit = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:pack[-\s]?years?|pk[-\s]?yrs?|pk[-\s]?yr|py)\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(0), float(explicit.group(1))

    computed = re.search(
        r"((?:\d+(?:\.\d+)?)|1/2|half)\s*(?:ppd|packs?\s*(?:per|/)\s*day)\s*(?:x|for)\s*(\d+(?:\.\d+)?)\+?\s*years?",
        text,
        flags=re.IGNORECASE,
    )
    if computed and _has_tobacco_context(computed.group(0)):
        ppd = _to_float(computed.group(1))
        years = _to_float(computed.group(2))
        if ppd is not None and years is not None:
            return computed.group(0), ppd * years

    return None, None


def extract_ppd(text: str) -> tuple[str | None, float | None]:
    if not text:
        return None, None

    exclude_pat = r"\b(postpartum|delivery|ppd\s*test|tuberculin|\btb\b|read at\s*\d+\s*mm|induration)\b"
    patterns = [
        r"((?:\d+(?:\.\d+)?)|1/2|half)\s*ppd\b",
        r"((?:\d+(?:\.\d+)?)|1/2|half)\s*packs?\s*(?:per|/)\s*day\b",
        r"half\s+a\s+pack\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            ctx = text[start:end]
            if re.search(exclude_pat, ctx, flags=re.IGNORECASE):
                continue
            if not _has_tobacco_context(ctx) and not re.search(r"\b(x|for)\s*\d+\+?\s*years\b", ctx, flags=re.IGNORECASE):
                continue

            raw = m.group(0)
            if "half a pack" in raw.lower():
                return raw, 0.5
            val = _to_float(m.group(1))
            return raw, val

    return None, None


def extract_years_smoked(text: str) -> tuple[str | None, float | None]:
    if not text:
        return None, None

    patterns = [
        r"smoked\s+for\s+(\d+(?:\.\d+)?)\+?\s*years?",
        r"x\s*(\d+(?:\.\d+)?)\+?\s*years?",
        r"(\d+(?:\.\d+)?)\+?[-\s]?year\s+history\s+of\s+smoking",
        r"for\s+(\d+(?:\.\d+)?)\+?\s*years?",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            ctx = text[start:end]
            if not _has_tobacco_context(ctx):
                continue
            return m.group(0), float(m.group(1))

    return None, None


def extract_quit_years(text: str) -> tuple[str | None, float | None]:
    if not text:
        return None, None

    couple = re.search(r"(a\s+couple\s+of\s+years\s+ago)", text, flags=re.IGNORECASE)
    if couple:
        return couple.group(1), 2.0

    y = re.search(
        r"(quit(?:\s+smoking)?\s+(\d+(?:\.\d+)?)\s+years?\s+(?:ago|prior))",
        text,
        flags=re.IGNORECASE,
    )
    if y:
        return y.group(1), float(y.group(2))

    yr = re.search(r"(stopped\s+smoking\s+in\s+\d{4})", text, flags=re.IGNORECASE)
    if yr:
        return yr.group(1), None

    return None, None


def determine_eligibility(
    smoking_status_norm: str,
    pack_year_value: float | None,
    quit_years_value: float | None,
) -> tuple[str, str | None, str | None]:
    criteria = "USPSTF_2021"

    if smoking_status_norm == "never_smoker":
        return "not_eligible", criteria, "Never smoker based on note evidence."

    if smoking_status_norm == "current_smoker":
        if pack_year_value is None:
            return "unknown", criteria, "Current smoker but pack-year exposure is missing."
        if pack_year_value >= 20:
            return "eligible", criteria, "Current smoker with >=20 pack-years; age still needs external confirmation."
        return "not_eligible", criteria, "Current smoker but documented pack-years are below 20."

    if smoking_status_norm == "former_smoker":
        if pack_year_value is None or quit_years_value is None:
            return "unknown", criteria, "Former smoker but pack-years or years since quitting are incomplete."
        if pack_year_value >= 20 and quit_years_value <= 15:
            return "eligible", criteria, "Former smoker with >=20 pack-years and quit within 15 years; age still needs external confirmation."
        return "not_eligible", criteria, "Former smoker but pack-years <20 or quit >15 years ago."

    return "unknown", criteria, "Smoking history is unknown or insufficient for eligibility determination."


def assess_evidence_quality(
    smoking_status_raw: str | None,
    pack_year_value: float | None,
    ppd_value: float | None,
    years_smoked_value: float | None,
    social_history_text: str | None,
) -> str:
    if _is_deidentified(social_history_text):
        return "none"

    if smoking_status_raw and (pack_year_value is not None or (ppd_value is not None and years_smoked_value is not None)):
        return "high"

    if smoking_status_raw:
        if re.search(r"\bsignificant for tobacco\b|\btobacco\b", smoking_status_raw, flags=re.IGNORECASE):
            return "low"
        return "medium"

    if pack_year_value is not None or ppd_value is not None or years_smoked_value is not None:
        return "low"

    return "none"


def extract_smoking_eligibility(subject_id: int, note_id: str, text: str) -> dict:
    source_section, social_history_text = find_social_history_section(text or "")

    use_fallback = False
    if social_history_text is None or _is_deidentified(social_history_text):
        fallback_text = _extract_smoking_sentences(text or "")
        if fallback_text:
            target_text = fallback_text
            source_section = "full_text_fallback"
            use_fallback = True
        else:
            target_text = ""
    else:
        target_text = social_history_text

    smoking_status_raw, smoking_status_norm = extract_smoking_status(target_text)
    pack_year_text, pack_year_value = extract_pack_years(target_text)
    ppd_text, ppd_value = extract_ppd(target_text)
    years_smoked_text, years_smoked_value = extract_years_smoked(target_text)
    quit_years_text, quit_years_value = extract_quit_years(target_text)

    computed_pack_year = False
    if pack_year_value is None and ppd_value is not None and years_smoked_value is not None:
        pack_year_value = ppd_value * years_smoked_value
        computed_pack_year = True

    eligibility, criteria, reason = determine_eligibility(
        smoking_status_norm=smoking_status_norm,
        pack_year_value=pack_year_value,
        quit_years_value=quit_years_value,
    )

    ever_smoker_flag: bool | None
    if smoking_status_norm in {"current_smoker", "former_smoker"}:
        ever_smoker_flag = True
    elif smoking_status_norm == "never_smoker":
        ever_smoker_flag = False
    else:
        ever_smoker_flag = None

    evidence_span = smoking_status_raw or pack_year_text or ppd_text or years_smoked_text or quit_years_text
    evidence_quality = assess_evidence_quality(
        smoking_status_raw=smoking_status_raw,
        pack_year_value=pack_year_value,
        ppd_value=ppd_value,
        years_smoked_value=years_smoked_value,
        social_history_text=target_text if use_fallback else social_history_text,
    )
    if use_fallback and evidence_quality in {"high", "medium"}:
        evidence_quality = "low"

    extraction_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    missing_flags = []
    missing_candidates = {
        "source_section": source_section,
        "smoking_status_raw": smoking_status_raw,
        "pack_year_value": pack_year_value,
        "pack_year_text": pack_year_text,
        "ppd_value": ppd_value,
        "ppd_text": ppd_text,
        "years_smoked_value": years_smoked_value,
        "years_smoked_text": years_smoked_text,
        "quit_years_value": quit_years_value,
        "quit_years_text": quit_years_text,
        "evidence_span": evidence_span,
        "ever_smoker_flag": ever_smoker_flag,
        "eligibility_criteria_applied": criteria,
        "eligibility_reason": reason,
        "data_quality_notes": None,
    }
    for key, value in missing_candidates.items():
        if value is None:
            missing_flags.append(key)

    notes = []
    if use_fallback:
        notes.append("Social History missing or de-identified; used full-text fallback.")
    elif source_section is None:
        notes.append("Social History section not found.")
    elif _is_deidentified(social_history_text):
        notes.append("Social History appears de-identified.")
    if computed_pack_year:
        notes.append("pack_year_value computed from ppd_value and years_smoked_value.")

    data_quality_notes = " ".join(notes) if notes else None
    if data_quality_notes is not None and "data_quality_notes" in missing_flags:
        missing_flags.remove("data_quality_notes")

    return {
        "subject_id": subject_id,
        "note_id": note_id,
        "source_section": source_section,
        "smoking_status_raw": smoking_status_raw,
        "smoking_status_norm": smoking_status_norm,
        "pack_year_value": pack_year_value,
        "pack_year_text": pack_year_text,
        "ppd_value": ppd_value,
        "ppd_text": ppd_text,
        "years_smoked_value": years_smoked_value,
        "years_smoked_text": years_smoked_text,
        "quit_years_value": quit_years_value,
        "quit_years_text": quit_years_text,
        "evidence_span": evidence_span,
        "ever_smoker_flag": ever_smoker_flag,
        "eligible_for_high_risk_screening": eligibility,
        "eligibility_criteria_applied": criteria,
        "eligibility_reason": reason,
        "evidence_quality": evidence_quality,
        "extraction_metadata": {
            "extractor_version": _EXTRACTOR_VERSION,
            "extraction_timestamp": extraction_timestamp,
            "model_name": _MODEL_NAME,
        },
        "missing_flags": missing_flags,
        "data_quality_notes": data_quality_notes,
    }
