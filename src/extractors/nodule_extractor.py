import re


NODULE_KEYWORD_RE = re.compile(
    r"\b(nodule|nodules|pulmonary\s+nodule|granuloma|ggo|ggn|ground\s+glass|opacity|lesion)\b",
    re.IGNORECASE,
)

CONTINUATION_RE = re.compile(
    r"\b(measuring|measures|size|sized|diameter|largest|smaller|stable|new|increased|decreased|resolved|"
    r"right|left|lobe|lingula|spicul|calcif|part\s*[- ]?solid|ground\s*[- ]?glass|ggo|ggn)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw_parts = re.split(r"(?<=[\.!?;])\s+|\n+", text)
    return [part.strip() for part in raw_parts if part and part.strip()]


def segment_nodule_mentions(text: str) -> list[str]:
    sentences = split_sentences(text)
    candidates = []
    current = []

    for sentence in sentences:
        has_nodule = bool(NODULE_KEYWORD_RE.search(sentence))
        if has_nodule:
            if current:
                candidates.append(" ".join(current).strip())
            current = [sentence]
            continue

        if current and CONTINUATION_RE.search(sentence):
            current.append(sentence)
            continue

        if current:
            candidates.append(" ".join(current).strip())
            current = []

    if current:
        candidates.append(" ".join(current).strip())

    return candidates


def extract_size(text: str) -> tuple[float | None, str | None]:
    if not text:
        return None, None

    range_mm = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*mm\b", text, re.IGNORECASE)
    if range_mm:
        low = float(range_mm.group(1))
        high = float(range_mm.group(2))
        return (low + high) / 2.0, range_mm.group(0)

    match = re.search(
        r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*mm\b",
        text,
        re.IGNORECASE,
    )
    if match:
        values = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        return max(values), match.group(0)

    match = re.search(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*mm\b", text, re.IGNORECASE)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return max(values), match.group(0)

    match = re.search(
        r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*cm\b",
        text,
        re.IGNORECASE,
    )
    if match:
        values = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        return max(values) * 10.0, match.group(0)

    match = re.search(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*cm\b", text, re.IGNORECASE)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return max(values) * 10.0, match.group(0)

    match = re.search(r"(\d+\.?\d*)\s*[- ]?mm\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), match.group(0)

    match = re.search(r"(\d+\.?\d*)\s*[- ]?cm\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 10.0, match.group(0)

    return None, None


def extract_density(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    match = re.search(
        r"\b(part\s*[- ]?solid|partially\s+solid|semi\s*[- ]?solid|semisolid|subsolid)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return "part_solid", match.group(0)

    match = re.search(
        r"\b(ground\s*[- ]?glass|ggo|ggn|non\s*[- ]?solid|nonsolid)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return "ground_glass", match.group(0)

    match = re.search(r"\b(calcified|calcification)\b", text, re.IGNORECASE)
    if match and not re.search(r"\b(non\s*[- ]?calcified|not\s+definitely\s+calcified)\b", text, re.IGNORECASE):
        return "calcified", match.group(0)

    match = re.search(r"\bsolid\b", text, re.IGNORECASE)
    if match and not re.search(r"\b(part\s*[- ]?solid|semi\s*[- ]?solid|subsolid)\b", text, re.IGNORECASE):
        return "solid", match.group(0)

    return "unclear", None


def extract_location(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    patterns = [
        (r"\bbilateral\b", "bilateral"),
        (r"\b(right\s+upper\s+lobe|rul)\b", "RUL"),
        (r"\b(right\s+middle\s+lobe|rml)\b", "RML"),
        (r"\b(right\s+lower\s+lobe|rll)\b", "RLL"),
        (r"\b(left\s+upper\s+lobe|lul)\b", "LUL"),
        (r"\b(left\s+lower\s+lobe|lll)\b", "LLL"),
        (r"\blingula\b", "lingula"),
        (r"\b(right\s+lung|left\s+lung)\b", "unclear"),
    ]

    for pattern, category in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return category, match.group(0)

    return None, None


def extract_change_status(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    patterns = [
        (r"\b(stable|unchanged|no\s+change|no\s+significant\s+change)\b", "stable"),
        (r"\b(new|newly|new\s+finding)\b", "new"),
        (r"\b(increased|growing|larger|enlarging|interval\s+increase)\b", "increased"),
        (r"\b(decreased|smaller|diminished|interval\s+decrease)\b", "decreased"),
        (r"\b(resolved|no\s+longer\s+seen|disappeared)\b", "resolved"),
    ]

    for pattern, status in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return status, match.group(0)

    return None, None


def extract_morphology(text: str) -> dict:
    if not text:
        return {
            "calcification": False,
            "spiculation": False,
            "lobulation": False,
            "cavitation": False,
            "perifissural": False,
        }

    calcified = bool(re.search(r"\b(calcified|calcification)\b", text, re.IGNORECASE))
    non_calcified = bool(re.search(r"\b(non\s*[- ]?calcified|not\s+definitely\s+calcified)\b", text, re.IGNORECASE))

    return {
        "calcification": calcified and not non_calcified,
        "spiculation": bool(re.search(r"\b(spiculated|spiculation)\b", text, re.IGNORECASE)),
        "lobulation": bool(re.search(r"\b(lobulated|lobulation)\b", text, re.IGNORECASE)),
        "cavitation": bool(re.search(r"\b(cavitary|cavitation|cavitating)\b", text, re.IGNORECASE)),
        "perifissural": bool(re.search(r"\b(perifissural|fissural)\b", text, re.IGNORECASE)),
    }


def extract_recommendation_cue(text: str) -> str | None:
    if not text:
        return None

    cues = []
    for sentence in split_sentences(text):
        if re.search(
            r"\b(recommend|follow\s*[- ]?up|suggested|advised|should\s+be|lung\s*[- ]?rads|screening)\b",
            sentence,
            re.IGNORECASE,
        ):
            cues.append(sentence)

    if not cues:
        return None
    return " ".join(cues)


def extract_lung_rads(text: str) -> str | None:
    if not text:
        return None

    match = re.search(
        r"lung\s*[- ]?rads(?:\s*category)?\s*[:\-]?\s*(0|1|2|3|4A|4B|4X|S)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).upper()
