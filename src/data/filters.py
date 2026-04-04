import re


CHEST_CT_PATTERN = re.compile(
    r"CT\s+CHEST|CTA\s+CHEST|CT\s+LOW\s+DOSE\s+LUNG|CT\s+LUNG|CHEST\s+CT",
    flags=re.IGNORECASE,
)
NODULE_PATTERN = re.compile(r"\bnodul(?:e|es|ar)\b", flags=re.IGNORECASE)
NEGATION_NEARBY_PATTERN = re.compile(r"(?:\bno\b|\bwithout\b)[\w\s,;:/\-]{0,20}$", flags=re.IGNORECASE)


def filter_chest_ct(radiology_df, detail_df):
    exam_rows = detail_df[
        detail_df["field_name"].astype(str).str.lower().eq("exam_name")
    ].copy()
    exam_rows = exam_rows[
        exam_rows["field_value"].astype(str).str.contains(CHEST_CT_PATTERN, na=False)
    ].copy()
    exam_rows = exam_rows[["note_id", "field_value"]].drop_duplicates(subset=["note_id"])
    exam_rows = exam_rows.rename(columns={"field_value": "exam_name"})
    return radiology_df.merge(exam_rows, on="note_id", how="inner")


def _has_non_negated_nodule_mention(text):
    if not isinstance(text, str):
        return False
    matches = list(NODULE_PATTERN.finditer(text))
    if not matches:
        return False

    all_negated = True
    for match in matches:
        start = match.start()
        left_window = text[max(0, start - 20) : start]
        if not NEGATION_NEARBY_PATTERN.search(left_window):
            all_negated = False
            break
    return not all_negated


def filter_nodule_reports(df):
    mask = df["text"].apply(_has_non_negated_nodule_mention)
    return df[mask].copy()
