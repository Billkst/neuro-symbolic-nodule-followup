import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filters import filter_chest_ct, filter_nodule_reports
from src.data.loader import load_radiology, load_radiology_detail
from src.extractors.nodule_extractor import (
    extract_density,
    extract_location,
    extract_size,
    segment_nodule_mentions,
)
from src.parsers.section_parser import parse_sections


SEED = 42
EXPECTED_OUTPUTS = [
    "density_train.jsonl",
    "density_val.jsonl",
    "density_test.jsonl",
    "size_train.jsonl",
    "size_val.jsonl",
    "size_test.jsonl",
    "location_train.jsonl",
    "location_val.jsonl",
    "location_test.jsonl",
    "split_manifest.json",
    "label_stats.json",
]
SPLITS = ("train", "val", "test")


def _log(message: str, log_fp) -> None:
    print(message, flush=True)
    log_fp.write(message + "\n")
    log_fp.flush()


def _to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_density(label: str | None) -> str:
    if label in {"solid", "part_solid", "ground_glass", "calcified", "unclear"}:
        return label
    return "unclear"


def _build_full_text(findings: str | None, impression: str | None) -> str:
    parts = []
    if isinstance(findings, str) and findings.strip():
        parts.append(f"FINDINGS:\n{findings.strip()}")
    if isinstance(impression, str) and impression.strip():
        parts.append(f"IMPRESSION:\n{impression.strip()}")
    return "\n\n".join(parts)


def _infer_label_quality(
    density_label: str,
    density_text: str | None,
    size_mm: float | None,
    size_text: str | None,
    location_label: str | None,
    location_text: str | None,
) -> str:
    has_explicit_density = density_label != "unclear" and density_text is not None
    has_explicit_size = size_mm is not None and size_text is not None
    has_explicit_location = location_label is not None and location_text is not None

    if has_explicit_density and has_explicit_size and has_explicit_location:
        return "explicit"
    if density_text is not None or size_text is not None or location_text is not None:
        return "silver"
    return "weak"


def _build_subject_split(subject_ids: list[int], seed: int = SEED) -> dict[int, str]:
    unique_subjects = sorted({int(subject_id) for subject_id in subject_ids})
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)

    total = len(unique_subjects)
    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

    subject_to_split = {}
    for idx, subject_id in enumerate(unique_subjects):
        if idx < train_count:
            split = "train"
        elif idx < train_count + val_count:
            split = "val"
        else:
            split = "test"
        subject_to_split[subject_id] = split
    return subject_to_split


def _size_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
    }


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_label_stats(records_by_split: dict[str, list[dict]]) -> dict:
    stats = {
        "density": {"splits": {}, "quality_counts": {}, "quality_counts_by_split": {}},
        "size": {"splits": {}, "quality_counts": {}, "quality_counts_by_split": {}},
        "location": {"splits": {}, "quality_counts": {}, "quality_counts_by_split": {}},
    }

    overall_quality = Counter()
    quality_by_split = defaultdict(Counter)

    for split, rows in records_by_split.items():
        quality_counter = Counter(row["label_quality"] for row in rows)
        overall_quality.update(quality_counter)
        quality_by_split[split].update(quality_counter)

        density_counter = Counter(row["density_label"] for row in rows)
        location_counter = Counter(
            row["location_label"] if row["location_label"] is not None else "no_location"
            for row in rows
        )
        has_size_counter = Counter(str(bool(row["has_size"])).lower() for row in rows)
        size_values = [row["size_label"] for row in rows if row["size_label"] is not None]

        stats["density"]["splits"][split] = {
            "sample_count": len(rows),
            "label_distribution": dict(sorted(density_counter.items())),
        }
        stats["size"]["splits"][split] = {
            "sample_count": len(rows),
            "label_distribution": dict(sorted(has_size_counter.items())),
            "size_value_stats": _size_summary(size_values),
        }
        stats["location"]["splits"][split] = {
            "sample_count": len(rows),
            "label_distribution": dict(sorted(location_counter.items())),
        }

    for task_name in stats:
        stats[task_name]["quality_counts"] = dict(sorted(overall_quality.items()))
        stats[task_name]["quality_counts_by_split"] = {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(quality_by_split.items())
        }

    return stats


def _prepare_task_rows(records_by_split: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    task_rows = {
        "density": {split: [] for split in SPLITS},
        "size": {split: [] for split in SPLITS},
        "location": {split: [] for split in SPLITS},
    }

    for split, rows in records_by_split.items():
        for row in rows:
            density_row = dict(row)
            size_row = dict(row)
            location_row = dict(row)
            location_row["location_label"] = row["location_label"] if row["location_label"] is not None else "no_location"

            task_rows["density"][split].append(density_row)
            task_rows["size"][split].append(size_row)
            task_rows["location"][split].append(location_row)

    return task_rows


def _outputs_ready(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in EXPECTED_OUTPUTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None, help="调试用，只加载前 N 行")
    parser.add_argument("--force", action="store_true", help="强制重建数据集")
    parser.add_argument("--output-dir", default="outputs/phase5/datasets")
    parser.add_argument("--log", default="logs/build_phase5_datasets.log")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] build_phase5_datasets", log_fp)
        _log(f"[Config] seed={SEED} nrows={args.nrows} force={args.force}", log_fp)
        _log(f"[Paths] output_dir={output_dir} log={log_path}", log_fp)

        if _outputs_ready(output_dir) and not args.force:
            _log("[Cache] 检测到完整数据集输出，跳过重建。使用 --force 可强制重建。", log_fp)
            return

        _log("[Step 1/6] 加载 radiology.csv.gz", log_fp)
        radiology_df = load_radiology(nrows=args.nrows)
        _log(f"  radiology rows={len(radiology_df)}", log_fp)

        _log("[Step 2/6] 加载 radiology_detail.csv.gz", log_fp)
        detail_df = load_radiology_detail(nrows=args.nrows)
        _log(f"  radiology_detail rows={len(detail_df)}", log_fp)

        _log("[Step 3/6] 过滤胸部 CT", log_fp)
        chest_ct_df = filter_chest_ct(radiology_df, detail_df)
        _log(f"  chest_ct rows={len(chest_ct_df)}", log_fp)

        _log("[Step 4/6] 过滤含结节报告", log_fp)
        nodule_df = filter_nodule_reports(chest_ct_df)
        _log(f"  nodule_report rows={len(nodule_df)}", log_fp)

        _log("[Step 5/6] 解析 sections 并构建 mention-centered 样本", log_fp)
        records = []
        skipped_empty_text = 0
        skipped_empty_sections = 0
        skipped_empty_mentions = 0
        reports_with_mentions = 0

        for row_idx, row in enumerate(nodule_df.itertuples(index=False), start=1):
            text_value = getattr(row, "text", "")
            if not isinstance(text_value, str) or not text_value.strip():
                skipped_empty_text += 1
                continue

            sections = parse_sections(text_value)
            findings = sections.get("findings")
            impression = sections.get("impression")
            full_text = _build_full_text(findings, impression)
            if not full_text.strip():
                skipped_empty_sections += 1
                continue

            mention_idx = 0
            note_had_mentions = False
            for section_name, section_text in (("findings", findings), ("impression", impression)):
                if not isinstance(section_text, str) or not section_text.strip():
                    continue

                mentions = segment_nodule_mentions(section_text)
                for mention in mentions:
                    mention_text = mention.strip() if isinstance(mention, str) else ""
                    if not mention_text:
                        skipped_empty_mentions += 1
                        continue

                    density_label, density_text = extract_density(mention_text)
                    size_mm, size_text = extract_size(mention_text)
                    location_label, location_text = extract_location(mention_text)

                    density_label = _normalize_density(density_label)
                    subject_id = _to_jsonable(getattr(row, "subject_id"))
                    note_id = str(_to_jsonable(getattr(row, "note_id")))
                    exam_name = _to_jsonable(getattr(row, "exam_name", ""))
                    mention_idx += 1
                    note_had_mentions = True

                    records.append(
                        {
                            "sample_id": f"{note_id}__{mention_idx}",
                            "note_id": note_id,
                            "subject_id": subject_id,
                            "exam_name": exam_name,
                            "section": section_name,
                            "mention_text": mention_text,
                            "full_text": full_text,
                            "density_label": density_label,
                            "size_label": float(size_mm) if size_mm is not None else None,
                            "size_text": size_text,
                            "has_size": size_mm is not None,
                            "location_label": location_label,
                            "has_location": location_label is not None,
                            "label_quality": _infer_label_quality(
                                density_label=density_label,
                                density_text=density_text,
                                size_mm=size_mm,
                                size_text=size_text,
                                location_label=location_label,
                                location_text=location_text,
                            ),
                        }
                    )

            if note_had_mentions:
                reports_with_mentions += 1

            if row_idx % 1000 == 0:
                _log(
                    f"  processed_reports={row_idx} built_mentions={len(records)} reports_with_mentions={reports_with_mentions}",
                    log_fp,
                )

        if not records:
            raise RuntimeError("未构建出任何 mention 样本，请检查输入数据或过滤条件。")

        _log(
            f"  mention samples={len(records)} | reports_with_mentions={reports_with_mentions} | "
            f"skipped_empty_text={skipped_empty_text} | skipped_empty_sections={skipped_empty_sections} | "
            f"skipped_empty_mentions={skipped_empty_mentions}",
            log_fp,
        )

        _log("[Step 6/6] 按 subject_id 去重切分并写出数据集", log_fp)
        subject_to_split = _build_subject_split([int(record["subject_id"]) for record in records], seed=SEED)
        records_by_split = {split: [] for split in SPLITS}
        subjects_by_split = {split: set() for split in SPLITS}

        for record in records:
            subject_id = int(record["subject_id"])
            split = subject_to_split[subject_id]
            records_by_split[split].append(record)
            subjects_by_split[split].add(subject_id)

        task_rows = _prepare_task_rows(records_by_split)
        for task_name, split_rows in task_rows.items():
            for split, rows in split_rows.items():
                path = output_dir / f"{task_name}_{split}.jsonl"
                _write_jsonl(path, rows)
                _log(f"  wrote {task_name}_{split}.jsonl rows={len(rows)}", log_fp)

        overlap_sizes = {
            "train_val": len(subjects_by_split["train"] & subjects_by_split["val"]),
            "train_test": len(subjects_by_split["train"] & subjects_by_split["test"]),
            "val_test": len(subjects_by_split["val"] & subjects_by_split["test"]),
        }
        split_manifest = {
            "seed": SEED,
            "nrows": args.nrows,
            "total_mentions": len(records),
            "total_subjects": len(subject_to_split),
            "reports_with_mentions": reports_with_mentions,
            "skipped": {
                "empty_text_reports": skipped_empty_text,
                "empty_findings_impression_reports": skipped_empty_sections,
                "empty_mentions": skipped_empty_mentions,
            },
            "splits": {
                split: {
                    "mention_count": len(records_by_split[split]),
                    "subject_count": len(subjects_by_split[split]),
                }
                for split in SPLITS
            },
            "subject_overlap_check": {
                "train_val_overlap": overlap_sizes["train_val"],
                "train_test_overlap": overlap_sizes["train_test"],
                "val_test_overlap": overlap_sizes["val_test"],
                "disjoint": all(size == 0 for size in overlap_sizes.values()),
            },
        }
        _write_json(output_dir / "split_manifest.json", split_manifest)
        _log(f"  wrote split_manifest.json total_mentions={len(records)} total_subjects={len(subject_to_split)}", log_fp)

        label_stats = _build_label_stats(records_by_split)
        _write_json(output_dir / "label_stats.json", label_stats)
        _log("  wrote label_stats.json", log_fp)

        _log("[Summary]", log_fp)
        for split in SPLITS:
            _log(
                f"  {split}: mentions={len(records_by_split[split])} subjects={len(subjects_by_split[split])}",
                log_fp,
            )
        _log(f"  overlap_free={split_manifest['subject_overlap_check']['disjoint']}", log_fp)
        _log(f"  output_dir={output_dir}", log_fp)
        _log("[Done] build_phase5_datasets completed", log_fp)


if __name__ == "__main__":
    main()
