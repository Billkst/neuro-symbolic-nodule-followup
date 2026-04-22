#!/usr/bin/env python3
"""Audit Has-size distribution shift across WS and Phase5 splits."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import size_cue_features
from scripts.phaseA2.train_mws_cfe_common import load_jsonl

TOKEN_RE = re.compile(r"\S+")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "has_size"}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize_rows(name: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [bool_value(row.get("has_size")) for row in rows]
    mentions = [str(row.get("mention_text") or "") for row in rows]
    char_lengths = [len(text) for text in mentions]
    token_lengths = [len(TOKEN_RE.findall(text)) for text in mentions]
    cue_rows = [size_cue_features(text) for text in mentions]
    n = len(rows)
    positives = sum(1 for label in labels if label)
    summary: dict[str, Any] = {
        "split": name,
        "path": str(path),
        "n": n,
        "positive": positives,
        "negative": n - positives,
        "positive_rate": positives / n if n else 0.0,
        "mention_chars_mean": sum(char_lengths) / n if n else 0.0,
        "mention_chars_p50": quantile([float(v) for v in char_lengths], 0.5),
        "mention_chars_p90": quantile([float(v) for v in char_lengths], 0.9),
        "mention_tokens_mean": sum(token_lengths) / n if n else 0.0,
        "mention_tokens_p50": quantile([float(v) for v in token_lengths], 0.5),
        "mention_tokens_p90": quantile([float(v) for v in token_lengths], 0.9),
    }
    cue_keys = [
        "size_unit_mm_cm",
        "size_numeric_unit",
        "size_2d_pattern",
        "size_3d_pattern",
        "size_range_pattern",
        "size_context_word",
    ]
    for key in cue_keys:
        hits = sum(1 for features in cue_rows if features.get(key) == "yes")
        pos_hits = sum(1 for features, label in zip(cue_rows, labels, strict=False) if label and features.get(key) == "yes")
        neg_hits = sum(1 for features, label in zip(cue_rows, labels, strict=False) if (not label) and features.get(key) == "yes")
        summary[f"{key}_rate"] = hits / n if n else 0.0
        summary[f"{key}_positive_coverage"] = pos_hits / positives if positives else 0.0
        summary[f"{key}_negative_coverage"] = neg_hits / (n - positives) if (n - positives) else 0.0
    return summary


def distribution_distance(row: dict[str, Any], target: dict[str, Any]) -> float:
    keys = [
        "positive_rate",
        "mention_chars_mean",
        "mention_tokens_mean",
        "size_unit_mm_cm_rate",
        "size_numeric_unit_rate",
        "size_2d_pattern_rate",
        "size_3d_pattern_rate",
        "size_range_pattern_rate",
        "size_context_word_rate",
    ]
    distance = 0.0
    for key in keys:
        rv = float(row.get(key) or 0.0)
        tv = float(target.get(key) or 0.0)
        scale = max(abs(tv), 1.0) if "mean" in key else 1.0
        distance += abs(rv - tv) / scale
    return distance


def default_paths(phase5_dir: Path, ws_dir: Path) -> dict[str, Path]:
    paths = {
        "ws_train_g2": ws_dir / "size_train_ws_g2.jsonl",
        "ws_val": ws_dir / "size_val_ws.jsonl",
        "ws_test": ws_dir / "size_test_ws.jsonl",
        "phase5_test": phase5_dir / "size_test.jsonl",
    }
    for candidate in sorted(phase5_dir.glob("size_*.jsonl")):
        split = candidate.stem.replace("size_", "phase5_")
        paths.setdefault(split, candidate)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Has-size split distributions")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--ws-data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--extra", action="append", default=[], help="Extra split as name=path or glob")
    parser.add_argument("--output-json", default="outputs/phaseA2_planB/size_wave4/size_distribution_audit.json")
    parser.add_argument("--output-csv", default="outputs/phaseA2_planB/size_wave4/size_distribution_audit.csv")
    args = parser.parse_args()

    phase5_dir = Path(args.phase5_data_dir)
    ws_dir = Path(args.ws_data_dir)
    paths = default_paths(phase5_dir, ws_dir)
    for item in args.extra:
        if "=" in item:
            name, path = item.split("=", 1)
            paths[name] = Path(path)
        else:
            for match in glob.glob(item):
                paths[Path(match).stem] = Path(match)

    rows = []
    for name, path in sorted(paths.items()):
        if not path.exists():
            continue
        rows.append(summarize_rows(name, path, load_jsonl(path)))

    target = next((row for row in rows if row["split"] == "phase5_test"), None)
    if target:
        for row in rows:
            row["distance_to_phase5_test"] = 0.0 if row is target else distribution_distance(row, target)
        non_target = [row for row in rows if row["split"] != "phase5_test"]
        best = min(non_target, key=lambda item: float(item.get("distance_to_phase5_test", float("inf"))), default=None)
    else:
        best = None

    payload = {
        "rows": rows,
        "closest_to_phase5_test": best,
        "note": "Distance is an audit diagnostic only; phase5_test must not be used for threshold selection.",
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
