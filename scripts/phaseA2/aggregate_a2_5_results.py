#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Phase A2.5 5-seed results.

This script is intentionally independent from build_main_table.py. It only
reads result JSON files already produced under outputs/phaseA2/results and
generates fair-table summaries for the current multi-source WS protocol.
"""
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
    r"^(?P<method>.+)_(?P<task>density|size|location)_results_(?P<tag>.+)_seed(?P<seed>\d+)\.json$"
)

MAIN_METHODS = ["tfidf_lr", "tfidf_svm", "vanilla_pubmedbert", "mws_cfe"]
METHOD_DISPLAY = {
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "vanilla_pubmedbert": "Vanilla PubMedBERT",
    "mws_cfe": "MWS-CFE (Ours)",
}
TASKS = ["density", "size", "location"]
TASK_DISPLAY = {
    "density": "Density",
    "size": "Has_size",
    "location": "Location",
}
TASK_MAIN_METRIC = {
    "density": ("macro_f1", "Macro-F1"),
    "size": ("f1", "F1"),
    "location": ("macro_f1", "Macro-F1"),
}
EXPECTED_SEEDS = [13, 42, 87, 3407, 31415]
LOCAL_MODEL_MARKER = "outputs/phase5/hf_models/biomedbert_base_safe"


@dataclass(frozen=True)
class ResultRecord:
    method: str
    task: str
    tag: str
    seed: int
    path: Path
    data: dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)


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


def load_records(results_dir: Path) -> tuple[list[ResultRecord], list[str], list[str]]:
    records: list[ResultRecord] = []
    unmatched: list[str] = []
    ignored_methods: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            unmatched.append(path.name)
            continue
        method = match.group("method")
        if method not in MAIN_METHODS:
            ignored_methods.append(path.name)
            continue
        task = match.group("task")
        tag = match.group("tag")
        seed = int(match.group("seed"))
        records.append(ResultRecord(method=method, task=task, tag=tag, seed=seed, path=path, data=load_json(path)))
    return records, unmatched, ignored_methods


def group_records(records: list[ResultRecord]) -> dict[tuple[str, str, str], list[ResultRecord]]:
    grouped: dict[tuple[str, str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.task, record.tag)].append(record)
    for key in grouped:
        grouped[key].sort(key=lambda item: item.seed)
    return dict(grouped)


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
    if len(clean) <= 1:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))
    return {"n": len(clean), "mean": mean, "std": std}


def collect_metric(records: list[ResultRecord], metric_path: str) -> dict[str, float | int | None]:
    values = []
    for record in records:
        value = nested_get(record.data, metric_path)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean_std(values)


def format_summary(summary: dict[str, float | int | None], percent: bool = False, decimals: int = 2) -> str:
    if not summary or summary.get("n", 0) == 0 or summary.get("mean") is None:
        return "—"
    scale = 100.0 if percent else 1.0
    mean = float(summary["mean"]) * scale
    std = float(summary["std"] or 0.0) * scale
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def n_for(grouped: dict[tuple[str, str, str], list[ResultRecord]], method: str, task: str, tag: str) -> int:
    return len(grouped.get((method, task, tag), []))


def build_main_table(grouped: dict[tuple[str, str, str], list[ResultRecord]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for method in MAIN_METHODS:
        row: dict[str, Any] = {"Method": METHOD_DISPLAY[method]}
        raw[method] = {}
        for task in TASKS:
            records = grouped.get((method, task, "main_g2"), [])
            raw[method][task] = {
                "accuracy": collect_metric(records, "phase5_test_results.accuracy"),
                TASK_MAIN_METRIC[task][0]: collect_metric(records, f"phase5_test_results.{TASK_MAIN_METRIC[task][0]}"),
                "seeds": [record.seed for record in records],
            }
            task_name = TASK_DISPLAY[task]
            metric_key, metric_name = TASK_MAIN_METRIC[task]
            row[f"{task_name} Acc"] = format_summary(raw[method][task]["accuracy"], percent=True)
            row[f"{task_name} {metric_name}"] = format_summary(raw[method][task][metric_key], percent=True)
            row[f"{task_name} N"] = n_for(grouped, method, task, "main_g2")
        rows.append(row)
    return rows, raw


def aggregate_numeric_field(records: list[ResultRecord], field: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for record in records:
        value = nested_get(record.data, field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean_std(values)


def build_efficiency_table(grouped: dict[tuple[str, str, str], list[ResultRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in MAIN_METHODS:
        for task in TASKS:
            records = grouped.get((method, task, "main_g2"), [])
            train_samples = aggregate_numeric_field(records, "train_samples")
            train_time = aggregate_numeric_field(records, "train_time_seconds")
            eval_time = aggregate_numeric_field(records, "eval_time_seconds")
            best_epoch = aggregate_numeric_field(records, "best_epoch")
            peak_gpu = aggregate_numeric_field(records, "peak_gpu_memory_gb")
            rows.append(
                {
                    "Method": METHOD_DISPLAY[method],
                    "Task": TASK_DISPLAY[task],
                    "Tag": "main_g2",
                    "N": len(records),
                    "Train Samples": format_summary(train_samples, decimals=1),
                    "Train Time / seed (s)": format_summary(train_time, decimals=1),
                    "Eval Time / seed (s)": format_summary(eval_time, decimals=1),
                    "Best Epoch": format_summary(best_epoch, decimals=1),
                    "Peak GPU Memory (GB)": format_summary(peak_gpu, decimals=2),
                }
            )
    return rows


def build_experiment_rows(
    grouped: dict[tuple[str, str, str], list[ResultRecord]],
    experiment: str,
    variant_specs: list[tuple[str, str, list[str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for variant, tag, tasks in variant_specs:
        raw[variant] = {}
        for task in tasks:
            records = grouped.get(("mws_cfe", task, tag), [])
            metric_key, metric_name = TASK_MAIN_METRIC[task]
            accuracy = collect_metric(records, "phase5_test_results.accuracy")
            main_metric = collect_metric(records, f"phase5_test_results.{metric_key}")
            raw[variant][task] = {
                "accuracy": accuracy,
                metric_key: main_metric,
                "seeds": [record.seed for record in records],
            }
            rows.append(
                {
                    "Experiment": experiment,
                    "Variant": variant,
                    "Task": TASK_DISPLAY[task],
                    "Tag": tag,
                    "N": len(records),
                    "Accuracy": format_summary(accuracy, percent=True),
                    "Main Metric": metric_name,
                    "Main Metric Value": format_summary(main_metric, percent=True),
                }
            )
    return rows, raw


def expected_manifest(grouped: dict[tuple[str, str, str], list[ResultRecord]], expected_seeds: list[int]) -> list[dict[str, Any]]:
    expectations: list[tuple[str, str, str, str]] = []
    for method in MAIN_METHODS:
        for task in TASKS:
            expectations.append(("main_table", method, task, "main_g2"))

    for task in ("density", "location"):
        for tag in ("aqg_g1", "main_g2", "aqg_g3", "aqg_g4", "aqg_g5"):
            expectations.append(("a2_quality_gate", "mws_cfe", task, tag))
        for tag in ("main_g2", "aagg_uniform"):
            expectations.append(("a3_aggregation", "mws_cfe", task, tag))

    for tag in ("p1_len64", "p1_len96", "main_g2", "p1_len160", "p1_len192"):
        expectations.append(("p1_max_seq_length", "mws_cfe", "density", tag))

    for tag in ("main_g2", "p3_findings", "p3_impression", "p3_findings_impression", "p3_fulltext"):
        expectations.append(("p3_section_input", "mws_cfe", "density", tag))

    rows: list[dict[str, Any]] = []
    seen = set()
    for scope, method, task, tag in expectations:
        key = (method, task, tag)
        if (scope, key) in seen:
            continue
        seen.add((scope, key))
        records = grouped.get(key, [])
        present = sorted(record.seed for record in records)
        missing = [seed for seed in expected_seeds if seed not in present]
        extra = [seed for seed in present if seed not in expected_seeds]
        rows.append(
            {
                "scope": scope,
                "method": method,
                "task": task,
                "tag": tag,
                "expected_seeds": expected_seeds,
                "present_seeds": present,
                "missing_seeds": missing,
                "extra_seeds": extra,
                "complete": not missing,
            }
        )
    return rows


def metadata_warnings(records: list[ResultRecord]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for record in records:
        if record.method not in {"mws_cfe", "vanilla_pubmedbert"}:
            continue
        model_name = str(record.data.get("model_name", ""))
        if LOCAL_MODEL_MARKER not in model_name:
            warnings.append(
                {
                    "file": str(record.path),
                    "method": record.method,
                    "task": record.task,
                    "tag": record.tag,
                    "seed": record.seed,
                    "warning": "model_name does not record the local safetensors path",
                    "model_name": model_name,
                }
            )
        protocol = record.data.get("a2_5_protocol", {})
        if record.method != "mws_cfe" and isinstance(protocol, dict) and protocol.get("phase5_single_source_reused"):
            warnings.append(
                {
                    "file": str(record.path),
                    "warning": "result metadata says old Phase 5 single-source data was reused",
                }
            )
    return warnings


def table_payload(rows: list[dict[str, Any]], raw: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"rows": rows, "raw": raw or {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase A2.5 5-seed results")
    parser.add_argument("--results-dir", default="outputs/phaseA2/results")
    parser.add_argument("--output-dir", default="outputs/phaseA2/tables")
    parser.add_argument("--expected-seeds", default=",".join(str(seed) for seed in EXPECTED_SEEDS))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if required manifest entries are incomplete")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = parse_expected_seeds(args.expected_seeds)

    records, unmatched_files, ignored_method_files = load_records(results_dir)
    grouped = group_records(records)
    log(f"[Load] records={len(records)} unmatched={len(unmatched_files)} ignored_methods={len(ignored_method_files)}")

    main_rows, main_raw = build_main_table(grouped)
    efficiency_rows = build_efficiency_table(grouped)

    a2_rows, a2_raw = build_experiment_rows(
        grouped,
        "A2 quality-gate",
        [
            ("G1", "aqg_g1", ["density", "location"]),
            ("G2 / main", "main_g2", ["density", "location"]),
            ("G3", "aqg_g3", ["density", "location"]),
            ("G4", "aqg_g4", ["density", "location"]),
            ("G5", "aqg_g5", ["density", "location"]),
        ],
    )
    a3_rows, a3_raw = build_experiment_rows(
        grouped,
        "A3 aggregation",
        [
            ("Weighted vote / main", "main_g2", ["density", "location"]),
            ("Uniform vote", "aagg_uniform", ["density", "location"]),
        ],
    )
    p1_rows, p1_raw = build_experiment_rows(
        grouped,
        "P1 max_seq_length",
        [
            ("64", "p1_len64", ["density"]),
            ("96", "p1_len96", ["density"]),
            ("128 / main", "main_g2", ["density"]),
            ("160", "p1_len160", ["density"]),
            ("192", "p1_len192", ["density"]),
        ],
    )
    p3_rows, p3_raw = build_experiment_rows(
        grouped,
        "P3 section/input",
        [
            ("mention_text / main", "main_g2", ["density"]),
            ("findings", "p3_findings", ["density"]),
            ("impression", "p3_impression", ["density"]),
            ("findings_impression", "p3_findings_impression", ["density"]),
            ("full_text", "p3_fulltext", ["density"]),
        ],
    )

    manifest_rows = expected_manifest(grouped, expected_seeds)
    manifest = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "expected_seeds": expected_seeds,
        "record_count": len(records),
        "manifest": manifest_rows,
        "unmatched_files": unmatched_files,
        "ignored_method_files": ignored_method_files,
        "metadata_warnings": metadata_warnings(records),
    }

    write_csv(output_dir / "a2_5_main_table.csv", main_rows)
    write_json(output_dir / "a2_5_main_table.json", table_payload(main_rows, main_raw))
    write_csv(output_dir / "a2_5_efficiency_table.csv", efficiency_rows)
    write_json(output_dir / "a2_5_efficiency_table.json", table_payload(efficiency_rows))
    write_csv(output_dir / "a2_quality_gate_summary.csv", a2_rows)
    write_json(output_dir / "a2_quality_gate_summary.json", table_payload(a2_rows, a2_raw))
    write_csv(output_dir / "a3_aggregation_summary.csv", a3_rows)
    write_json(output_dir / "a3_aggregation_summary.json", table_payload(a3_rows, a3_raw))
    write_csv(output_dir / "p1_max_length_summary.csv", p1_rows)
    write_json(output_dir / "p1_max_length_summary.json", table_payload(p1_rows, p1_raw))
    write_csv(output_dir / "p3_section_input_summary.csv", p3_rows)
    write_json(output_dir / "p3_section_input_summary.json", table_payload(p3_rows, p3_raw))
    write_json(output_dir / "a2_5_manifest_report.json", manifest)

    incomplete = [row for row in manifest_rows if not row["complete"]]
    log(f"[Saved] tables -> {output_dir}")
    log(f"[Manifest] incomplete_entries={len(incomplete)} metadata_warnings={len(manifest['metadata_warnings'])}")
    if args.strict and incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

