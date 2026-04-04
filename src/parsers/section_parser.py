import re


SECTION_HEADERS = [
    "EXAMINATION",
    "INDICATION",
    "TECHNIQUE",
    "COMPARISON",
    "FINDINGS",
    "IMPRESSION",
    "RECOMMENDATION",
    "RECOMMENDATION(S)",
]

HEADER_TO_KEY = {
    "indication": "indication",
    "technique": "technique",
    "comparison": "comparison",
    "findings": "findings",
    "impression": "impression",
    "recommendation": "recommendation",
    "recommendation(s)": "recommendation",
}

HEADER_PATTERN = re.compile(
    r"^\s*(EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION|RECOMMENDATION(?:\(S\))?)\s*(?::\s*|\n+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_sections(text: str) -> dict:
    result = {
        "indication": None,
        "technique": None,
        "comparison": None,
        "findings": None,
        "impression": None,
    }
    if not isinstance(text, str) or not text.strip():
        return result

    matches = list(HEADER_PATTERN.finditer(text))
    if not matches:
        return result

    extracted = {}
    for idx, match in enumerate(matches):
        header = match.group(1).strip().lower()
        key = HEADER_TO_KEY.get(header)
        if key is None:
            continue

        section_start = match.end()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[section_start:section_end].strip()
        extracted[key] = content or None

    for key in ("indication", "technique", "comparison", "findings", "impression"):
        if key in extracted:
            result[key] = extracted[key]

    if result["impression"] is None and extracted.get("recommendation"):
        result["impression"] = extracted["recommendation"]

    return result
