#!/usr/bin/env python3
"""Select the Plan B density Stage 2 configuration using validation results.

The selector only reads result JSON files and ranks candidate configurations by
validation Macro-F1. It is intended for smoke-test triage before launching the
full five-seed matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULT_RE = re.compile(r"^mws_cfe_density_stage2_results_(?P<tag>.+)_seed(?P<seed>\d+)\.json$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, std


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    return f"{mean * 100.0:.2f} +/- {(std or 0.0) * 100.0:.2f}"


def infer_family(tag: str) -> str:
    if tag == "planb_full":
        return "main_g2_len128"
    if tag.startswith("p2_"):
        return "quality_gate"
    if tag.startswith("p1_"):
        return "max_length"
    return "other"


def infer_setting(tag: str) -> str:
    if tag == "planb_full":
        return "G2 / max_length=128"
    if tag == "p2_g1":
        return "G1"
    if tag == "p2_g3":
        return "G3"
    if tag == "p2_g4":
        return "G4"
    if tag == "p2_g5":
        return "G5"
    if tag.startswith("p1_len"):
        return "max_length=" + tag.replace("p1_len", "")
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Plan B Stage 2 config by validation Macro-F1")
    parser.add_argument("--results-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/sota_sprint")
    parser.add_argument(
        "--candidate-tags",
        default="planb_full,p2_g3,p1_len192",
        help="Comma-separated tags for the minimal G2/G3 and 128/192 sprint decision.",
    )
    parser.add_argument("--required-seeds", default=None, help="Optional comma-separated seed list")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [item.strip() for item in args.candidate_tags.split(",") if item.strip()]
    required_seeds = None
    if args.required_seeds:
        required_seeds = {int(item.strip()) for item in args.required_seeds.split(",") if item.strip()}

    grouped: dict[str, list[tuple[int, Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(results_dir.glob("mws_cfe_density_stage2_results_*_seed*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        tag = match.group("tag")
        seed = int(match.group("seed"))
        if tag not in candidates:
            continue
        if required_seeds is not None and seed not in required_seeds:
            continue
        grouped[tag].append((seed, path, load_json(path)))

    rows: list[dict[str, Any]] = []
    for tag in candidates:
        records = grouped.get(tag, [])
        val_values = [
            float(record["ws_val_results"]["macro_f1"])
            for _, _, record in records
            if isinstance(record.get("ws_val_results", {}).get("macro_f1"), (int, float))
        ]
        test_values = [
            float(record["phase5_test_results"]["macro_f1"])
            for _, _, record in records
            if isinstance(record.get("phase5_test_results", {}).get("macro_f1"), (int, float))
        ]
        val_mean, val_std = mean_std(val_values)
        test_mean, test_std = mean_std(test_values)
        rows.append(
            {
                "tag": tag,
                "family": infer_family(tag),
                "setting": infer_setting(tag),
                "n": len(records),
                "seeds": ",".join(str(seed) for seed, _, _ in sorted(records)),
                "val_macro_f1": fmt(val_mean, val_std),
                "val_macro_f1_mean": "" if val_mean is None else f"{val_mean:.8f}",
                "phase5_macro_f1": fmt(test_mean, test_std),
                "phase5_macro_f1_mean": "" if test_mean is None else f"{test_mean:.8f}",
            }
        )

    rows.sort(key=lambda row: float(row["val_macro_f1_mean"] or "-1"), reverse=True)
    best = rows[0] if rows and rows[0]["val_macro_f1_mean"] else None

    csv_path = output_dir / "stage2_config_selection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["tag"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "stage2_config_selection.json"
    json_path.write_text(
        json.dumps(
            {
                "selection_metric": "ws_val_results.macro_f1",
                "test_metric_for_audit_only": "phase5_test_results.macro_f1",
                "candidate_tags": candidates,
                "required_seeds": sorted(required_seeds) if required_seeds is not None else None,
                "best_by_validation": best,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Saved] {csv_path}", flush=True)
    print(f"[Saved] {json_path}", flush=True)
    if best:
        print(f"[Best] {best['tag']} {best['setting']} val_macro_f1={best['val_macro_f1']}", flush=True)


if __name__ == "__main__":
    main()
