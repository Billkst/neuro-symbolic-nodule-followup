import argparse
import json
from pathlib import Path

from src.data.filters import filter_chest_ct, filter_nodule_reports
from src.data.loader import load_radiology, load_radiology_detail
from src.parsers.section_parser import parse_sections


def _log(message, log_file):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def _to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--output", default="outputs/radiology_candidates.jsonl")
    parser.add_argument("--log", default="logs/export_radiology_candidates.log")
    args = parser.parse_args()

    output_path = Path(args.output)
    log_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
        _log("开始加载 radiology.csv.gz", log_file)
        radiology_df = load_radiology(nrows=args.nrows)
        _log(f"radiology 行数: {len(radiology_df)}", log_file)

        _log("开始加载 radiology_detail.csv.gz", log_file)
        detail_df = load_radiology_detail(nrows=args.nrows)
        _log(f"radiology_detail 行数: {len(detail_df)}", log_file)

        chest_ct_df = filter_chest_ct(radiology_df, detail_df)
        _log(f"胸部CT候选行数: {len(chest_ct_df)}", log_file)

        nodule_df = filter_nodule_reports(chest_ct_df)
        _log(f"结节关键词候选行数: {len(nodule_df)}", log_file)

        export_df = nodule_df.head(args.limit).copy()
        _log(f"导出行数: {len(export_df)}", log_file)

        with open(output_path, "w", encoding="utf-8") as out_file:
            for count, (_, row) in enumerate(export_df.iterrows(), start=1):
                text_value = row.get("text", "")
                if not isinstance(text_value, str):
                    text_value = ""

                sections = parse_sections(text_value)
                record = {
                    "note_id": _to_jsonable(row.get("note_id")),
                    "subject_id": _to_jsonable(row.get("subject_id")),
                    "exam_name": _to_jsonable(row.get("exam_name")),
                    "text": text_value,
                    "sections": sections,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if count % 50 == 0:
                    _log(f"已处理 {count} 条", log_file)

        _log(f"导出完成: {output_path}", log_file)


if __name__ == "__main__":
    main()
