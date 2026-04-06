from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_PULMONARY_COUNT = 62
OUTPUT_FIELDS = [
    "sample_id",
    "subject_id",
    "note_id",
    "mention_text",
    "text_window",
    "full_text",
    "gold_density_category",
    "gold_has_size",
    "gold_size_mm",
    "gold_location_lobe",
    "silver_density_category",
    "silver_has_size",
    "silver_size_mm",
    "silver_location_lobe",
    "annotation_confidence",
]


def _normalize_text(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_optional_float(value: str | None) -> float | None:
    text = _normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return None
    number = float(text)
    if math.isnan(number):
        return None
    return number


def _parse_required_bool(value: str | None, true_values: set[str], false_values: set[str], field_name: str) -> bool:
    text = _normalize_text(value).lower()
    if text in true_values:
        return True
    if text in false_values:
        return False
    raise ValueError(f"无法解析 {field_name}: {value!r}")


def _parse_optional_int(value: str | None) -> int | str | None:
    text = _normalize_text(value)
    if not text:
        return None
    return int(text) if text.isdigit() else text


def load_gold_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_jsonl_by_sample_id(path: Path) -> dict[str, dict]:
    rows_by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"JSONL 第 {line_number} 行缺少有效 sample_id")
            if sample_id in rows_by_id:
                raise ValueError(f"JSONL 中存在重复 sample_id: {sample_id}")
            rows_by_id[sample_id] = item
    return rows_by_id


def filter_pulmonary_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()

    for row in rows:
        sample_id = _normalize_text(row.get("sample_id"))
        if not sample_id:
            raise ValueError("Gold CSV 中存在空 sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Gold CSV 中存在重复 sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        is_target = _normalize_text(row.get("gold_is_pulmonary_target")).lower() == "yes"
        needs_review = _normalize_text(row.get("needs_review")) == "0"
        if is_target and needs_review:
            included.append(row)
        else:
            excluded.append(row)
    return included, excluded


def build_manifest_row(gold_row: dict[str, str], test_row: dict) -> dict:
    subject_id = _parse_optional_int(gold_row.get("subject_id"))
    note_id = _normalize_text(gold_row.get("note_id"))
    full_text = test_row.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError(f"sample_id={gold_row.get('sample_id')} 缺少有效 full_text")

    return {
        "sample_id": _normalize_text(gold_row.get("sample_id")),
        "subject_id": subject_id,
        "note_id": note_id,
        "mention_text": _normalize_text(gold_row.get("mention_text")),
        "text_window": _normalize_text(gold_row.get("text_window")),
        "full_text": full_text,
        "gold_density_category": _normalize_text(gold_row.get("gold_density_category")),
        "gold_has_size": _parse_required_bool(
            gold_row.get("gold_has_size"),
            true_values={"yes"},
            false_values={"no"},
            field_name="gold_has_size",
        ),
        "gold_size_mm": _parse_optional_float(gold_row.get("gold_size_mm")),
        "gold_location_lobe": _normalize_text(gold_row.get("gold_location_lobe")),
        "silver_density_category": _normalize_text(gold_row.get("silver_density_category")),
        "silver_has_size": _parse_required_bool(
            gold_row.get("silver_has_size"),
            true_values={"true"},
            false_values={"false"},
            field_name="silver_has_size",
        ),
        "silver_size_mm": _parse_optional_float(gold_row.get("silver_size_mm")),
        "silver_location_lobe": _normalize_text(gold_row.get("silver_location_lobe")),
        "annotation_confidence": _normalize_text(gold_row.get("annotation_confidence")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 5.1 gold evaluation manifest JSONL.")
    parser.add_argument(
        "--gold-csv",
        default=str(PROJECT_ROOT / "data" / "gold_eval_candidates_v1_final_usable_gold.csv"),
        help="Gold CSV 路径",
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(PROJECT_ROOT / "outputs" / "phase5" / "datasets" / "density_test.jsonl"),
        help="Phase 5 test JSONL 路径",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "phase5_1" / "gold_eval_manifest.jsonl"),
        help="输出 manifest JSONL 路径",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_PULMONARY_COUNT,
        help="期望最终 manifest 行数",
    )
    args = parser.parse_args()

    gold_csv_path = Path(args.gold_csv)
    test_jsonl_path = Path(args.test_jsonl)
    output_path = Path(args.output)

    print(f"[Input] gold_csv={gold_csv_path}", flush=True)
    print(f"[Input] test_jsonl={test_jsonl_path}", flush=True)
    print(f"[Output] manifest={output_path}", flush=True)

    print("[Step 1/4] 读取 gold CSV ...", flush=True)
    gold_rows = load_gold_rows(gold_csv_path)
    print(f"  total_gold_rows={len(gold_rows)}", flush=True)

    print("[Step 2/4] 过滤 pulmonary target 样本 ...", flush=True)
    pulmonary_rows, excluded_rows = filter_pulmonary_rows(gold_rows)
    print(f"  filtered_pulmonary={len(pulmonary_rows)}", flush=True)
    print(f"  excluded={len(excluded_rows)}", flush=True)

    print("[Step 3/4] 加载 Phase 5 test JSONL 并对齐 ...", flush=True)
    test_rows_by_sample_id = load_jsonl_by_sample_id(test_jsonl_path)
    matched_rows: list[dict] = []
    unmatched_ids: list[str] = []
    for gold_row in pulmonary_rows:
        sample_id = _normalize_text(gold_row.get("sample_id"))
        test_row = test_rows_by_sample_id.get(sample_id)
        if test_row is None:
            unmatched_ids.append(sample_id)
            continue
        matched_rows.append(build_manifest_row(gold_row, test_row))

    print(f"  matched_with_jsonl={len(matched_rows)}", flush=True)
    print(f"  unmatched={len(unmatched_ids)}", flush=True)
    if unmatched_ids:
        print(
            "WARNING: 以下 sample_id 未在 test JSONL 中匹配到: " + ", ".join(sorted(unmatched_ids)),
            flush=True,
        )

    print("[Step 4/4] 写出 manifest ...", flush=True)
    write_jsonl(output_path, matched_rows)
    print(f"  final_count={len(matched_rows)}", flush=True)

    if len(pulmonary_rows) != args.expected_count:
        print(
            f"WARNING: filtered_pulmonary={len(pulmonary_rows)} 与 expected_count={args.expected_count} 不一致",
            flush=True,
        )
    if len(matched_rows) != args.expected_count:
        print(
            f"WARNING: final_count={len(matched_rows)} 与 expected_count={args.expected_count} 不一致",
            flush=True,
        )

    print("[Done] gold manifest 构建完成", flush=True)


if __name__ == "__main__":
    main()
