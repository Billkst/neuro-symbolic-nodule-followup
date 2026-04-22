#!/usr/bin/env python3
"""Build a non-test Phase5-like calibration/dev split for Has-size."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import size_cue_features
from scripts.phaseA2.train_mws_cfe_common import load_jsonl


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "has_size"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def cue_bucket(row: dict[str, Any]) -> tuple[str, str, str, str]:
    features = size_cue_features(str(row.get("mention_text") or ""))
    return (
        "pos" if bool_value(row.get("has_size")) else "neg",
        features.get("size_numeric_unit", "no"),
        features.get("size_2d_pattern", "no"),
        features.get("size_range_pattern", "no"),
    )


def stratified_sample(rows: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or len(rows) <= max_samples:
        return list(rows)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[cue_bucket(row)].append(row)
    selected: list[dict[str, Any]] = []
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        take = max(1, round(max_samples * len(bucket_rows) / len(rows)))
        selected.extend(bucket_rows[:take])
    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]
    elif len(selected) < max_samples:
        used = {id(row) for row in selected}
        remaining = [row for row in rows if id(row) not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    selected.sort(key=lambda row: str(row.get("sample_id") or row.get("note_id") or row.get("mention_text") or ""))
    return selected


def candidate_sources(phase5_dir: Path, ws_dir: Path) -> list[tuple[str, Path, str]]:
    preferred = [
        ("phase5_dev", phase5_dir / "size_dev.jsonl", "existing Phase5 dev split"),
        ("phase5_val", phase5_dir / "size_val.jsonl", "existing Phase5 validation split"),
        ("phase5_train", phase5_dir / "size_train.jsonl", "existing Phase5 train split"),
    ]
    fallback = [
        ("ws_val", ws_dir / "size_val_ws.jsonl", "WS validation fallback; no Phase5 non-test split found"),
        ("ws_train_g2", ws_dir / "size_train_ws_g2.jsonl", "WS G2 train fallback; no Phase5 non-test split found"),
        ("ws_test", ws_dir / "size_test_ws.jsonl", "WS test fallback; no Phase5 non-test split found"),
    ]
    return preferred + fallback


def label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(1 for row in rows if bool_value(row.get("has_size")))
    return {
        "n": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
        "positive_rate": positives / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Has-size Phase5-like calibration/dev split")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--ws-data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--output", default="outputs/phaseA2_planB/size_wave4/size_phase5_like_dev.jsonl")
    parser.add_argument("--metadata", default="outputs/phaseA2_planB/size_wave4/size_phase5_like_dev_meta.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="auto", help="auto or explicit JSONL path")
    args = parser.parse_args()

    phase5_dir = Path(args.phase5_data_dir)
    ws_dir = Path(args.ws_data_dir)
    source_name = "custom"
    source_reason = "explicit source path"

    if args.source != "auto":
        source_path = Path(args.source)
    else:
        source_path = None
        for name, path, reason in candidate_sources(phase5_dir, ws_dir):
            if path.exists():
                source_name = name
                source_path = path
                source_reason = reason
                break
        if source_path is None:
            raise FileNotFoundError("No size calibration/dev source split found")

    if source_path.name == "size_test.jsonl" and "phase5" in str(source_path):
        raise ValueError("Refusing to build calibration/dev directly from phase5 size_test.jsonl")

    source_rows = load_jsonl(source_path)
    output_rows = stratified_sample(source_rows, args.max_samples, args.seed)
    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)

    meta = {
        "output": str(output_path),
        "source_name": source_name,
        "source_path": str(source_path),
        "source_reason": source_reason,
        "phase5_test_used_for_threshold": False,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "source_summary": label_summary(source_rows),
        "output_summary": label_summary(output_rows),
        "note": "Use this non-test calibration/dev split as --selection-split for tune_size_threshold_v2.py.",
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
