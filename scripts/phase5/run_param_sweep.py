from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5.analysis_common import (
    Logger,
    ensure_dir,
    format_seconds,
    load_task_splits,
    prepare_hf_environment,
    primary_metric_name,
    primary_metric_value,
    run_svm_experiment,
    save_json,
)


def summarize_result(task: str, result: dict) -> dict:
    metrics = result["metrics"]
    payload: dict[str, Any] = {
        "primary_metric_name": primary_metric_name(task),
        "primary_metric": primary_metric_value(task, metrics),
        "accuracy": float(metrics["accuracy"]),
    }
    if task == "size":
        payload["f1"] = float(metrics["f1"])
        payload["macro_f1"] = float(metrics["macro_f1"])
    else:
        payload["macro_f1"] = float(metrics["macro_f1"])
    return payload


def run_sweep(tasks, logger: Logger, sweep_name: str, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        logger.log(f"[{sweep_name}] {idx}/{len(variants)} config={variant}")
        config_start = time.perf_counter()
        task_metrics: dict[str, dict[str, Any]] = {}
        for task in ("density", "size", "location"):
            ngram_value = variant.get("ngram_range", (1, 2))
            if not isinstance(ngram_value, (tuple, list)) or len(ngram_value) != 2:
                raise ValueError(f"非法 ngram_range: {ngram_value}")
            result = run_svm_experiment(
                task,
                tasks[task]["train"],
                tasks[task]["test"],
                max_features=int(variant.get("max_features", 10000)),
                ngram_range=(int(ngram_value[0]), int(ngram_value[1])),
                c_value=float(variant.get("C", 1.0)),
            )
            task_metrics[task] = summarize_result(task, result)
        outputs.append(
            {
                "config": {
                    "max_features": int(variant.get("max_features", 10000)),
                    "ngram_range": list(variant.get("ngram_range", (1, 2))),
                    "C": float(variant.get("C", 1.0)),
                },
                "tasks": task_metrics,
                "elapsed_seconds": round(time.perf_counter() - config_start, 4),
            }
        )
        logger.log(f"[{sweep_name}] done config={variant} | elapsed={format_seconds(time.perf_counter() - config_start)}")
    return outputs


def pick_best_by_task(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for task in ("density", "size", "location"):
        ordered = sorted(results, key=lambda item: item["tasks"][task]["primary_metric"], reverse=True)
        best[task] = ordered[0]
    return best


def main() -> None:
    dataset_dir = PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    output_dir = PROJECT_ROOT / "outputs" / "phase5" / "results" / "param_sweep"
    log_path = PROJECT_ROOT / "logs" / "run_param_sweep.log"
    ensure_dir(output_dir)

    with Logger(log_path) as logger:
        total_start = time.perf_counter()
        logger.log("[Start] run_param_sweep")
        logger.log(f"[Paths] dataset_dir={dataset_dir} output_dir={output_dir}")
        prepare_hf_environment(logger)
        task_splits = {task: load_task_splits(task, dataset_dir) for task in ("density", "size", "location")}

        logger.log("[Step 1/4] P1 max_features sweep")
        max_feature_results = run_sweep(
            task_splits,
            logger,
            "P1",
            [
                {"max_features": 5000, "ngram_range": (1, 2), "C": 1.0},
                {"max_features": 10000, "ngram_range": (1, 2), "C": 1.0},
                {"max_features": 20000, "ngram_range": (1, 2), "C": 1.0},
            ],
        )
        save_json(output_dir / "sweep_max_features.json", max_feature_results)

        logger.log("[Step 2/4] P2 ngram_range sweep")
        ngram_results = run_sweep(
            task_splits,
            logger,
            "P2",
            [
                {"max_features": 10000, "ngram_range": (1, 1), "C": 1.0},
                {"max_features": 10000, "ngram_range": (1, 2), "C": 1.0},
                {"max_features": 10000, "ngram_range": (1, 3), "C": 1.0},
            ],
        )
        save_json(output_dir / "sweep_ngram_range.json", ngram_results)

        logger.log("[Step 3/4] P3 regularization C sweep")
        regularization_results = run_sweep(
            task_splits,
            logger,
            "P3",
            [
                {"max_features": 10000, "ngram_range": (1, 2), "C": 0.1},
                {"max_features": 10000, "ngram_range": (1, 2), "C": 1.0},
                {"max_features": 10000, "ngram_range": (1, 2), "C": 10.0},
            ],
        )
        save_json(output_dir / "sweep_regularization.json", regularization_results)

        logger.log("[Step 4/4] 写出 param_sweep_summary.json")
        summary = {
            "max_features": {
                "results": max_feature_results,
                "best_by_task": pick_best_by_task(max_feature_results),
            },
            "ngram_range": {
                "results": ngram_results,
                "best_by_task": pick_best_by_task(ngram_results),
            },
            "regularization": {
                "results": regularization_results,
                "best_by_task": pick_best_by_task(regularization_results),
            },
            "takeaways": [
                "P1 用 TF-IDF max_features 近似特征空间大小，对应论文中的输入容量/长度讨论。",
                "P2 比较 unigram 到 trigram，讨论文本粒度是否带来增益。",
                "P3 比较不同 C 值，讨论 SVM 正则化强度对泛化的影响。",
            ],
        }
        save_json(output_dir / "param_sweep_summary.json", summary)
        logger.log(f"[Done] run_param_sweep completed in {format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
