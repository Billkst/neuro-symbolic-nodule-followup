import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filters import filter_chest_ct, filter_nodule_reports
from src.data.loader import load_radiology, load_radiology_detail
from src.extractors.radiology_extractor import extract_radiology_facts
from src.parsers.section_parser import parse_sections


NOTE_ID_PATTERN = re.compile(r"^\d+-RR-\d+$")


def _log(msg, log_fp):
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_note_id(subject_id: int, note_seq, note_id_raw) -> str:
    raw = str(note_id_raw or "").strip()
    if NOTE_ID_PATTERN.match(raw):
        return raw
    seq = _to_int(note_seq, 0)
    if seq > 0:
        return f"{subject_id}-RR-{seq}"
    fallback = _to_int(note_id_raw, 0)
    return f"{subject_id}-RR-{fallback if fallback > 0 else 0}"


def _load_candidates_from_jsonl(input_path: Path):
    candidates = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj["_line_idx"] = line_idx
            candidates.append(obj)
    return candidates


def _load_candidates_from_raw(nrows, limit):
    radiology_df = load_radiology(nrows=nrows)
    detail_df = load_radiology_detail(nrows=nrows)
    chest_ct_df = filter_chest_ct(radiology_df, detail_df)
    nodule_df = filter_nodule_reports(chest_ct_df)
    return nodule_df.head(limit).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="预处理好的 candidates JSONL，若不提供则从原始数据加载")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="outputs/radiology_facts.jsonl")
    args = parser.parse_args()

    log_path = Path("logs/run_radiology_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_radiology_baseline", log_fp)
        _log(f"[Config] input={args.input} nrows={args.nrows} limit={args.limit} output={args.output}", log_fp)

        candidates_jsonl = None
        candidates_df = None
        if args.input:
            input_path = Path(args.input)
            candidates_jsonl = _load_candidates_from_jsonl(input_path)
            _log(f"[Data] from_input_jsonl rows={len(candidates_jsonl)}", log_fp)
        else:
            candidates_df = _load_candidates_from_raw(args.nrows, args.limit)
            _log(f"[Data] from_raw_candidates rows={len(candidates_df)}", log_fp)

        nodule_count_counter = Counter()
        density_counter = Counter()
        location_counter = Counter()
        processed = 0

        with output_path.open("w", encoding="utf-8", buffering=1) as out_fp:
            if candidates_jsonl is not None:
                for idx, item in enumerate(candidates_jsonl, start=1):
                    subject_id = _to_int(item.get("subject_id"), 0)
                    note_id = _normalize_note_id(subject_id, item.get("note_seq"), item.get("note_id"))
                    exam_name = str(item.get("exam_name") or "")
                    text = str(item.get("text") or "")
                    sections = item.get("sections") if isinstance(item.get("sections"), dict) else parse_sections(text)

                    result = extract_radiology_facts(
                        note_id=note_id,
                        subject_id=subject_id,
                        exam_name=exam_name,
                        report_text=text,
                        sections=sections,
                    )
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fp.flush()

                    processed += 1
                    nodule_count_counter[result.get("nodule_count", 0)] += 1
                    for nodule in result.get("nodules", []):
                        density_counter[str(nodule.get("density_category") or "null")] += 1
                        location_counter[str(nodule.get("location_lobe") or "null")] += 1

                    if idx % 50 == 0:
                        _log(f"[Progress] processed={idx}", log_fp)
            else:
                if candidates_df is None:
                    raise ValueError("raw candidates 加载失败")
                for idx, (_, row) in enumerate(candidates_df.iterrows(), start=1):
                    subject_id = _to_int(row.get("subject_id"), 0)
                    note_id = _normalize_note_id(subject_id, row.get("note_seq"), row.get("note_id"))
                    exam_name = str(row.get("exam_name") or "")
                    text = str(row.get("text") or "")
                    sections = parse_sections(text)

                    result = extract_radiology_facts(
                        note_id=note_id,
                        subject_id=subject_id,
                        exam_name=exam_name,
                        report_text=text,
                        sections=sections,
                    )
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fp.flush()

                    processed += 1
                    nodule_count_counter[result.get("nodule_count", 0)] += 1
                    for nodule in result.get("nodules", []):
                        density_counter[str(nodule.get("density_category") or "null")] += 1
                        location_counter[str(nodule.get("location_lobe") or "null")] += 1

                    if idx % 50 == 0:
                        _log(f"[Progress] processed={idx}", log_fp)

        _log(f"[Summary] total_processed={processed}", log_fp)
        _log("[Summary] nodule_count_distribution", log_fp)
        for k, v in sorted(nodule_count_counter.items(), key=lambda x: x[0]):
            _log(f"  - {k}: {v}", log_fp)

        _log("[Summary] density_distribution", log_fp)
        for k, v in density_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Summary] location_distribution", log_fp)
        for k, v in location_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Done] run_radiology_baseline completed", log_fp)


if __name__ == "__main__":
    main()
