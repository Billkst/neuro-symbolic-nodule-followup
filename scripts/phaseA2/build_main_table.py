#!/usr/bin/env python3
"""整合所有方法的结果，生成主结果表和对比表。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHODS = ["regex", "tfidf_lr", "tfidf_svm", "vanilla_pubmedbert", "mws_cfe"]
TASKS = ["density", "has_size", "location"]
TASK_PRIMARY_METRIC = {
    "density": ("macro_f1", "Macro-F1"),
    "has_size": ("f1", "F1"),
    "location": ("macro_f1", "Macro-F1"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_phase5_baselines(phase5_results_dir: Path) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}

    baselines = load_json(phase5_results_dir / "baselines_summary.json")

    results["regex"] = {
        "density": {"accuracy": baselines["density"]["regex"]["test_accuracy"], "macro_f1": baselines["density"]["regex"]["test_macro_f1"]},
        "has_size": {"accuracy": baselines["size"]["regex"]["test_accuracy"], "f1": baselines["size"]["regex"]["test_f1"]},
        "location": {"accuracy": baselines["location"]["regex"]["test_accuracy"], "macro_f1": baselines["location"]["regex"]["test_macro_f1"]},
    }

    results["tfidf_lr"] = {
        "density": {"accuracy": baselines["density"]["ml_lr"]["test_accuracy"], "macro_f1": baselines["density"]["ml_lr"]["test_macro_f1"]},
        "has_size": {"accuracy": baselines["size"]["ml_lr"]["test_accuracy"], "f1": baselines["size"]["ml_lr"]["test_f1"]},
        "location": {"accuracy": baselines["location"]["ml_lr"]["test_accuracy"], "macro_f1": baselines["location"]["ml_lr"]["test_macro_f1"]},
    }

    results["tfidf_svm"] = {
        "density": {"accuracy": baselines["density"]["ml_svm"]["test_accuracy"], "macro_f1": baselines["density"]["ml_svm"]["test_macro_f1"]},
        "has_size": {"accuracy": baselines["size"]["ml_svm"]["test_accuracy"], "f1": baselines["size"]["ml_svm"]["test_f1"]},
        "location": {"accuracy": baselines["location"]["ml_svm"]["test_accuracy"], "macro_f1": baselines["location"]["ml_svm"]["test_macro_f1"]},
    }

    for task_key, p5_task in [("density", "density"), ("has_size", "size"), ("location", "location")]:
        p5_result = load_json(phase5_results_dir / f"pubmedbert_{p5_task}_results.json")
        test_r = p5_result["test_results"]
        if task_key == "density":
            results.setdefault("vanilla_pubmedbert", {})[task_key] = {"accuracy": test_r["accuracy"], "macro_f1": test_r["macro_f1"]}
        elif task_key == "has_size":
            results.setdefault("vanilla_pubmedbert", {})[task_key] = {"accuracy": test_r["accuracy"], "f1": test_r["f1"]}
        elif task_key == "location":
            results.setdefault("vanilla_pubmedbert", {})[task_key] = {"accuracy": test_r["accuracy"], "macro_f1": test_r["macro_f1"]}

    return results


def extract_mws_cfe_results(phaseA2_results_dir: Path, gate: str = "g2") -> dict[str, dict]:
    results: dict[str, dict] = {}

    for task_key, file_task in [("density", "density"), ("has_size", "size"), ("location", "location")]:
        result_file = phaseA2_results_dir / f"mws_cfe_{file_task}_results_{gate}.json"
        if not result_file.exists():
            log(f"[Warning] {result_file} not found")
            continue
        data = load_json(result_file)
        p5_test = data.get("phase5_test_results", {})
        if task_key == "density":
            results[task_key] = {"accuracy": p5_test.get("accuracy", 0), "macro_f1": p5_test.get("macro_f1", 0)}
        elif task_key == "has_size":
            results[task_key] = {"accuracy": p5_test.get("accuracy", 0), "f1": p5_test.get("f1", 0)}
        elif task_key == "location":
            results[task_key] = {"accuracy": p5_test.get("accuracy", 0), "macro_f1": p5_test.get("macro_f1", 0)}

    return results


def extract_gold_results(phase5_1_dir: Path, phaseA2_results_dir: Path, gate: str = "g2") -> dict[str, dict[str, dict]]:
    gold_results: dict[str, dict[str, dict]] = {}

    old_gold = load_json(phase5_1_dir / "gold_eval_metrics.json")
    method_map = {
        "regex": "regex",
        "tfidf_lr": "ml_lr",
        "tfidf_svm": "ml_svm",
        "vanilla_pubmedbert": "pubmedbert",
    }
    for our_name, old_name in method_map.items():
        if old_name in old_gold:
            gold_results[our_name] = {}
            for task_key, old_task in [("density", "density"), ("has_size", "has_size"), ("location", "location")]:
                if old_task in old_gold[old_name]:
                    gold_results[our_name][task_key] = old_gold[old_name][old_task]

    mws_gold_path = phaseA2_results_dir / f"gold_eval_mws_{gate}.json"
    if mws_gold_path.exists():
        mws_gold = load_json(mws_gold_path)
        gold_results["mws_cfe"] = {}
        if "density" in mws_gold:
            gold_results["mws_cfe"]["density"] = mws_gold["density"]
        if "has_size" in mws_gold:
            gold_results["mws_cfe"]["has_size"] = mws_gold["has_size"]
        if "location" in mws_gold:
            gold_results["mws_cfe"]["location"] = mws_gold["location"]

    return gold_results


def format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}"


def build_main_table(
    silver_results: dict[str, dict[str, dict]],
    gold_results: dict[str, dict[str, dict]],
) -> list[dict[str, str]]:
    rows = []
    method_display = {
        "regex": "Regex",
        "tfidf_lr": "TF-IDF + LR",
        "tfidf_svm": "TF-IDF + SVM",
        "vanilla_pubmedbert": "Vanilla PubMedBERT",
        "mws_cfe": "MWS-CFE (Ours)",
    }

    for method in METHODS:
        row = {"Method": method_display.get(method, method)}
        for task in TASKS:
            metric_key, metric_name = TASK_PRIMARY_METRIC[task]
            silver_val = silver_results.get(method, {}).get(task, {}).get(metric_key)
            gold_val = gold_results.get(method, {}).get(task, {}).get(metric_key)
            acc_silver = silver_results.get(method, {}).get(task, {}).get("accuracy")

            col_prefix = {"density": "Density", "has_size": "Size", "location": "Location"}[task]
            row[f"{col_prefix} {metric_name} (Silver)"] = format_pct(silver_val)
            row[f"{col_prefix} Acc (Silver)"] = format_pct(acc_silver)
            row[f"{col_prefix} {metric_name} (Gold)"] = format_pct(gold_val)

        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase A2 main result tables")
    parser.add_argument("--output-dir", type=str, default="outputs/phaseA2")
    parser.add_argument("--gate", type=str, default="g2")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"

    phase5_results = PROJECT_ROOT / "outputs" / "phase5" / "results"
    phase5_1_dir = PROJECT_ROOT / "outputs" / "phase5_1"

    log("[Start] Building main result tables")

    silver_results = extract_phase5_baselines(phase5_results)
    mws_results = extract_mws_cfe_results(results_dir, args.gate)
    if mws_results:
        silver_results["mws_cfe"] = mws_results

    gold_results = extract_gold_results(phase5_1_dir, results_dir, args.gate)

    table_rows = build_main_table(silver_results, gold_results)

    csv_path = tables_dir / "main_results.csv"
    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_rows)
        log(f"[Saved] {csv_path}")

    json_path = tables_dir / "main_results.json"
    all_data = {
        "silver_results": silver_results,
        "gold_results": gold_results,
        "table": table_rows,
    }
    json_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"[Saved] {json_path}")

    log("\n[Main Result Table - Silver Test]")
    log(f"{'Method':<25} {'Density MF1':>12} {'Size F1':>10} {'Location MF1':>13}")
    log("-" * 65)
    for row in table_rows:
        method = row["Method"]
        d = row.get("Density Macro-F1 (Silver)", "—")
        s = row.get("Size F1 (Silver)", "—")
        l = row.get("Location Macro-F1 (Silver)", "—")
        log(f"{method:<25} {d:>12} {s:>10} {l:>13}")

    log("\n[Gold Sanity-Check Table (N=62)]")
    log(f"{'Method':<25} {'Density MF1':>12} {'Size F1':>10} {'Location MF1':>13}")
    log("-" * 65)
    for row in table_rows:
        method = row["Method"]
        d = row.get("Density Macro-F1 (Gold)", "—")
        s = row.get("Size F1 (Gold)", "—")
        l = row.get("Location Macro-F1 (Gold)", "—")
        log(f"{method:<25} {d:>12} {s:>10} {l:>13}")

    log("\n[Done]")


if __name__ == "__main__":
    main()
