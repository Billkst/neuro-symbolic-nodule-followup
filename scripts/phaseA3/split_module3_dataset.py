#!/usr/bin/env python3
"""Create deterministic case-level or patient-level splits for Module 3."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "dimension", "label", "count", "fraction_within_split"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _label_key(row: dict[str, Any]) -> str:
    if row.get("abstention_reason"):
        return f"abstain:{row.get('abstention_reason')}"
    category = row.get("lung_rads_category") or "None"
    return f"category:{category}"


def _group_key(row: dict[str, Any]) -> str:
    patient_id = row.get("patient_id")
    if patient_id is not None:
        return f"patient:{patient_id}"
    subject_id = row.get("subject_id")
    if subject_id is not None:
        return f"subject:{subject_id}"
    return f"case:{row.get('case_id')}"


def _allocate_label_groups(
    groups: list[list[dict[str, Any]]],
    *,
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[list[dict[str, Any]]]]:
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": shuffled, "val": [], "test": []}
    if n == 2:
        return {"train": shuffled[:1], "val": [], "test": shuffled[1:]}

    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1
    test_n = n - train_n - val_n
    if test_n <= 0:
        test_n = 1
        if train_n > val_n:
            train_n -= 1
        else:
            val_n -= 1

    return {
        "train": shuffled[:train_n],
        "val": shuffled[train_n : train_n + val_n],
        "test": shuffled[train_n + val_n :],
    }


def _flatten(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group]


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "0.000000"
    return f"{count / total:.6f}"


def _distribution_for_split(split_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    dimensions = {
        "recommendation_level": Counter(str(row.get("recommendation_level")) for row in rows),
        "lung_rads_category": Counter(str(row.get("lung_rads_category")) for row in rows),
        "abstention_reason": Counter(str(row.get("abstention_reason")) for row in rows),
    }
    output: list[dict[str, Any]] = []
    for dimension, counter in dimensions.items():
        for label, count in sorted(counter.items()):
            output.append(
                {
                    "split": split_name,
                    "dimension": dimension,
                    "label": label,
                    "count": count,
                    "fraction_within_split": _rate(count, total),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Module 3 strong-silver dataset.")
    parser.add_argument("--input", default="outputs/phaseA3/datasets/module3_strong_silver.jsonl")
    parser.add_argument("--output-dir", default="outputs/phaseA3/datasets")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module3_split_summary.csv")
    parser.add_argument(
        "--label-distribution",
        default="outputs/phaseA3/tables/module3_split_label_distribution.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.input))
    grouped_by_label: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    patient_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patient_groups[_group_key(row)].append(row)

    for group_id, group_rows in patient_groups.items():
        labels = Counter(_label_key(row) for row in group_rows)
        label = labels.most_common(1)[0][0]
        grouped_by_label[label][group_id] = group_rows

    rng = random.Random(args.seed)
    split_groups: dict[str, list[list[dict[str, Any]]]] = {"train": [], "val": [], "test": []}
    for label, groups_by_id in sorted(grouped_by_label.items()):
        allocated = _allocate_label_groups(
            list(groups_by_id.values()),
            rng=rng,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        for split_name, groups in allocated.items():
            split_groups[split_name].extend(groups)

    splits = {split_name: _flatten(groups) for split_name, groups in split_groups.items()}
    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "module3_train.jsonl", splits["train"])
    _write_jsonl(output_dir / "module3_val.jsonl", splits["val"])
    _write_jsonl(output_dir / "module3_test.jsonl", splits["test"])

    group_to_split: dict[str, str] = {}
    leakage_count = 0
    for split_name, split_rows in splits.items():
        for row in split_rows:
            key = _group_key(row)
            previous = group_to_split.get(key)
            if previous is not None and previous != split_name:
                leakage_count += 1
            group_to_split[key] = split_name

    total = len(rows)
    label_counts = Counter(_label_key(row) for row in rows)
    actionable = [row for row in rows if not row.get("abstention_reason")]
    actionable_category_counts = Counter(str(row.get("lung_rads_category")) for row in actionable)
    min_actionable_category_count = min(actionable_category_counts.values()) if actionable_category_counts else 0
    imbalance_warning = (
        "actionable_labels_are_small_or_imbalanced"
        if len(actionable) < 50 or min_actionable_category_count < 5
        else "none"
    )

    summary_rows: list[dict[str, Any]] = [
        {"metric": "total_samples", "value": total},
        {"metric": "total_patient_or_case_groups", "value": len(patient_groups)},
        {"metric": "split_unit", "value": "patient_id_or_subject_id_else_case_id"},
        {"metric": "seed", "value": args.seed},
        {"metric": "train_samples", "value": len(splits["train"])},
        {"metric": "val_samples", "value": len(splits["val"])},
        {"metric": "test_samples", "value": len(splits["test"])},
        {"metric": "train_fraction", "value": _rate(len(splits["train"]), total)},
        {"metric": "val_fraction", "value": _rate(len(splits["val"]), total)},
        {"metric": "test_fraction", "value": _rate(len(splits["test"]), total)},
        {"metric": "split_leakage_count", "value": leakage_count},
        {"metric": "actionable_samples", "value": len(actionable)},
        {"metric": "min_actionable_category_count", "value": min_actionable_category_count},
        {"metric": "imbalance_warning", "value": imbalance_warning},
    ]
    for label, count in sorted(label_counts.items()):
        summary_rows.append({"metric": f"label_key.{label}", "value": count})

    distribution_rows: list[dict[str, Any]] = []
    for split_name, split_rows in splits.items():
        distribution_rows.extend(_distribution_for_split(split_name, split_rows))

    _write_summary(Path(args.summary), summary_rows)
    _write_distribution(Path(args.label_distribution), distribution_rows)

    print(
        json.dumps(
            {
                "total_samples": total,
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "split_leakage_count": leakage_count,
                "imbalance_warning": imbalance_warning,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
