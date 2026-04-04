import re


def classify_modality(exam_name: str) -> str:
    text = (exam_name or "").lower()

    if "low dose" in text and ("ct" in text or "lung screening" in text):
        return "LDCT"
    if re.search(r"\bcta\b", text):
        return "CTA"
    if re.search(r"\bpet\b", text):
        return "PET_CT"
    if re.search(r"\b(mr|mri)\b", text):
        return "MRI"
    if "ultrasound" in text or re.search(r"\bus\b", text):
        return "ultrasound"
    if re.search(r"\bct\b", text):
        return "CT"
    if (
        "x-ray" in text
        or re.search(r"\b(cxr|radiograph|portable ap|pa and lat)\b", text)
        or "chest (" in text
    ):
        return "X-ray"
    return "other"


def classify_body_site(exam_name: str) -> str:
    text = (exam_name or "").lower()

    has_chest = bool(re.search(r"\b(chest|lung|thorax)\b", text))
    has_abdomen = bool(re.search(r"\b(abd|abdomen|abdominal)\b", text))
    has_pelvis = bool(re.search(r"\bpelvis|pelvic\b", text))

    if has_chest and (has_abdomen or has_pelvis):
        return "chest_abdomen_pelvis"
    if has_abdomen and has_pelvis:
        return "chest_abdomen_pelvis"
    if has_chest:
        return "chest"
    if has_abdomen:
        return "abdomen"
    if has_pelvis:
        return "pelvis"
    return "other"
