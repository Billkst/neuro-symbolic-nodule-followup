#!/usr/bin/env python3
"""Aggregate Plan B module-2 reconstruction results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_RE = re.compile(
    r"^(?P<method>.+)_(?P<task>density_stage1|density_stage2|size|location)_results_(?P<tag>.+)_seed(?P<seed>\d+)\.json$"
)
METHODS = ["regex_cue", "tfidf_lr", "tfidf_svm", "tfidf_mlp", "vanilla_pubmedbert", "mws_cfe"]
METHOD_DISPLAY = {
    "regex_cue": "Regex / cue-only",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_mlp": "TF-IDF + MLP",
    "vanilla_pubmedbert": "Vanilla PubMedBERT",
    "mws_cfe": "MWS-CFE (Ours)",
}
TASK_METRICS = {
    "density_stage1": [("auprc", "AUPRC"), ("auroc", "AUROC"), ("f1", "F1")],
    "density_stage2": [("macro_f1", "Macro-F1"), ("accuracy", "Acc")],
    "size": [("f1", "F1"), ("accuracy", "Acc")],
    "location": [("macro_f1", "Macro-F1"), ("accuracy", "Acc")],
}
TASK_DISPLAY = {
    "density_stage1": "Density Stage 1",
    "density_stage2": "Density Stage 2",
    "size": "Has_size",
    "location": "Location",
}
EXPECTED_SEEDS = [13, 42, 87, 3407, 31415]
ABLATION_SPECS = [
    ("Full", "planb_full"),
    ("w/o quality gate", "ab_wo_quality_gate"),
    ("w/o weighted aggregation", "ab_wo_weighted_aggregation"),
    ("w/o confidence-aware training", "ab_wo_confidence"),
    ("w/o section strategy", "ab_wo_section"),
    ("w/o multi-source supervision", "ab_wo_multisource"),
]
PARAMETER_SPECS = {
    "P1 max_seq_length": [
        ("64", "p1_len64"),
        ("96", "p1_len96"),
        ("128", "planb_full"),
        ("160", "p1_len160"),
        ("192", "p1_len192"),
    ],
    "P2 quality_gate": [
        ("G1", "p2_g1"),
        ("G2", "planb_full"),
        ("G3", "p2_g3"),
        ("G4", "p2_g4"),
        ("G5", "p2_g5"),
    ],
    "P3 section/input strategy": [
        ("mention_text", "p3_mention_text"),
        ("section_aware_text", "planb_full"),
        ("findings_text", "p3_findings_text"),
        ("impression_text", "p3_impression_text"),
        ("findings_impression_text", "p3_findings_impression_text"),
        ("full_text", "p3_full_text"),
    ],
}


@dataclass(frozen=True)
class ResultRecord:
    method: str
    task: str
    tag: str
    seed: int
    path: Path
    data: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def parse_expected_seeds(value: str | None) -> list[int]:
    if not value:
        return list(EXPECTED_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_records(results_dir: Path) -> tuple[list[ResultRecord], list[str]]:
    records: list[ResultRecord] = []
    unmatched: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            unmatched.append(path.name)
            continue
        method = match.group("method")
        if method not in METHODS:
            unmatched.append(path.name)
            continue
        records.append(
            ResultRecord(
                method=method,
                task=match.group("task"),
                tag=match.group("tag"),
                seed=int(match.group("seed")),
                path=path,
                data=load_json(path),
            )
        )
    return records, unmatched


def group_records(records: list[ResultRecord]) -> dict[tuple[str, str, str], list[ResultRecord]]:
    grouped: dict[tuple[str, str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.task, record.tag)].append(record)
    return {key: sorted(value, key=lambda item: item.seed) for key, value in grouped.items()}


def nested_get(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(clean) / len(clean)
    std = 0.0 if len(clean) <= 1 else math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))
    return {"n": len(clean), "mean": mean, "std": std}


def collect(records: list[ResultRecord], metric: str) -> dict[str, float | int | None]:
    values = []
    for record in records:
        value = nested_get(record.data, f"phase5_test_results.{metric}")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean_std(values)


def fmt(summary: dict[str, float | int | None], percent: bool = True) -> str:
    if not summary or summary.get("n", 0) == 0 or summary.get("mean") is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{float(summary['mean']) * scale:.2f} +/- {float(summary['std'] or 0.0) * scale:.2f}"


def build_main_table(grouped: dict[tuple[str, str, str], list[ResultRecord]], main_tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for method in METHODS:
        row: dict[str, Any] = {"Method": METHOD_DISPLAY[method]}
        raw[method] = {}
        for task, metrics in TASK_METRICS.items():
            records = grouped.get((method, task, main_tag), [])
            raw[method][task] = {"seeds": [record.seed for record in records]}
            for metric, display in metrics:
                summary = collect(records, metric)
                raw[method][task][metric] = summary
                row[f"{TASK_DISPLAY[task]} {display}"] = fmt(summary)
            row[f"{TASK_DISPLAY[task]} N"] = len(records)
        rows.append(row)
    return rows, raw


def build_ablation_table(grouped: dict[tuple[str, str, str], list[ResultRecord]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for variant, tag in ABLATION_SPECS:
        raw[variant] = {}
        row: dict[str, Any] = {"Variant": variant, "Tag": tag}
        for task, metric in (("density_stage1", "auprc"), ("density_stage2", "macro_f1")):
            records = grouped.get(("mws_cfe", task, tag), [])
            summary = collect(records, metric)
            raw[variant][task] = {metric: summary, "seeds": [record.seed for record in records]}
            row[f"{TASK_DISPLAY[task]} {metric}"] = fmt(summary)
            row[f"{TASK_DISPLAY[task]} N"] = len(records)
        rows.append(row)
    return rows, raw


def build_parameter_tables(grouped: dict[tuple[str, str, str], list[ResultRecord]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for parameter, variants in PARAMETER_SPECS.items():
        raw[parameter] = {}
        for value, tag in variants:
            row: dict[str, Any] = {"Parameter": parameter, "Value": value, "Tag": tag}
            raw[parameter][value] = {}
            for task, metric in (("density_stage1", "auprc"), ("density_stage2", "macro_f1")):
                records = grouped.get(("mws_cfe", task, tag), [])
                summary = collect(records, metric)
                raw[parameter][value][task] = {metric: summary, "seeds": [record.seed for record in records]}
                row[f"{TASK_DISPLAY[task]} {metric}"] = fmt(summary)
                row[f"{TASK_DISPLAY[task]} N"] = len(records)
            rows.append(row)
    return rows, raw


def expected_manifest(grouped: dict[tuple[str, str, str], list[ResultRecord]], expected_seeds: list[int], main_tag: str) -> list[dict[str, Any]]:
    expectations: list[tuple[str, str, str, str]] = []
    for method in METHODS:
        for task in ("density_stage1", "density_stage2", "size", "location"):
            expectations.append(("main_table", method, task, main_tag))
    for _, tag in ABLATION_SPECS:
        for task in ("density_stage1", "density_stage2"):
            expectations.append(("ablation", "mws_cfe", task, tag))
    for parameter, variants in PARAMETER_SPECS.items():
        for _, tag in variants:
            for task in ("density_stage1", "density_stage2"):
                expectations.append((parameter, "mws_cfe", task, tag))

    rows = []
    seen = set()
    for scope, method, task, tag in expectations:
        key = (method, task, tag)
        if (scope, key) in seen:
            continue
        seen.add((scope, key))
        records = grouped.get(key, [])
        present = sorted(record.seed for record in records)
        missing = [seed for seed in expected_seeds if seed not in present]
        rows.append(
            {
                "scope": scope,
                "method": method,
                "task": task,
                "tag": tag,
                "expected_seeds": expected_seeds,
                "present_seeds": present,
                "missing_seeds": missing,
                "complete": not missing,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Plan B module2 results")
    parser.add_argument("--results-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/tables")
    parser.add_argument("--main-tag", default="planb_full")
    parser.add_argument("--expected-seeds", default=",".join(str(seed) for seed in EXPECTED_SEEDS))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir
    expected_seeds = parse_expected_seeds(args.expected_seeds)
    records, unmatched = load_records(results_dir)
    grouped = group_records(records)

    main_rows, main_raw = build_main_table(grouped, args.main_tag)
    ablation_rows, ablation_raw = build_ablation_table(grouped)
    parameter_rows, parameter_raw = build_parameter_tables(grouped)
    manifest_rows = expected_manifest(grouped, expected_seeds, args.main_tag)

    write_csv(output_dir / "planb_main_table.csv", main_rows)
    write_json(output_dir / "planb_main_table.json", {"rows": main_rows, "raw": main_raw})
    write_csv(output_dir / "planb_ablation_table.csv", ablation_rows)
    write_json(output_dir / "planb_ablation_table.json", {"rows": ablation_rows, "raw": ablation_raw})
    write_csv(output_dir / "planb_parameter_summary.csv", parameter_rows)
    write_json(output_dir / "planb_parameter_summary.json", {"rows": parameter_rows, "raw": parameter_raw})
    manifest = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "expected_seeds": expected_seeds,
        "records": len(records),
        "unmatched_files": unmatched,
        "manifest": manifest_rows,
    }
    write_json(output_dir / "planb_manifest_report.json", manifest)

    incomplete = [row for row in manifest_rows if not row["complete"]]
    print(f"[Saved] {output_dir}", flush=True)
    print(f"[Manifest] records={len(records)} incomplete={len(incomplete)} unmatched={len(unmatched)}", flush=True)
    if args.strict and incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
