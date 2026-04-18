#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Phase A2.5 TF-IDF baselines on multi-source WS data.

This script replaces the old Phase 5 baseline path for the A2 fair main table.
It reads Phase A1 multi-source weak-supervision labels, evaluates on the same
WS val/test files, and also evaluates on the full Phase 5 test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import load_jsonl, to_jsonable
from src.phase5.evaluation.metrics import evaluate_density, evaluate_location, evaluate_size_detection


DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified", "unclear"]
LOCATION_MODEL_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"]
LOCATION_EVAL_LABELS = LOCATION_MODEL_LABELS + ["no_location"]
TASKS = ["density", "size", "location"]
METHODS = ["tfidf_lr", "tfidf_svm"]


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def _parse_csv(value: str | None, choices: list[str], name: str) -> list[str]:
    if not value:
        return list(choices)
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(f"{name} contains invalid values: {invalid}; choices={choices}")
    return items


def _task_label_field(task: str) -> str:
    if task == "density":
        return "density_label"
    if task == "location":
        return "location_label"
    return "has_size"


def _primary_metric(task: str) -> str:
    return "f1" if task == "size" else "macro_f1"


def _texts(rows: list[dict[str, Any]], input_field: str) -> list[str]:
    return [str(row.get(input_field) or row.get("mention_text") or "") for row in rows]


def _density_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        label = str(row.get("density_label", "unclear"))
        labels.append(label if label in DENSITY_LABELS else "unclear")
    return labels


def _size_labels(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if bool(row.get("has_size")) else 0 for row in rows]


def _location_model_label(label: Any) -> str:
    if label in LOCATION_MODEL_LABELS:
        return str(label)
    return "unclear"


def _location_train_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [_location_model_label(row.get("location_label")) for row in rows]


def _build_pipeline(method: str, seed: int) -> Pipeline:
    if method == "tfidf_lr":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=seed,
        )
    elif method == "tfidf_svm":
        classifier = LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            C=1.0,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


def _classifier_param_count(pipeline: Pipeline) -> int | None:
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None
    total = 0
    for attr in ("coef_", "intercept_"):
        value = getattr(clf, attr, None)
        if value is not None:
            total += int(np.asarray(value).size)
    return total


def _evaluate_predictions(task: str, rows: list[dict[str, Any]], predictions: list[Any]) -> dict[str, Any]:
    if task == "density":
        y_true = _density_labels(rows)
        y_pred = [str(pred) if str(pred) in DENSITY_LABELS else "unclear" for pred in predictions]
        return evaluate_density(y_true, y_pred, DENSITY_LABELS)

    if task == "size":
        y_true = _size_labels(rows)
        y_pred = [int(pred) for pred in predictions]
        return evaluate_size_detection(y_true, y_pred)

    y_true: list[str] = []
    y_pred: list[str] = []
    for row, pred in zip(rows, predictions, strict=False):
        true_label = row.get("location_label")
        if true_label == "no_location":
            y_true.append("no_location")
            y_pred.append("no_location")
            continue
        y_true.append(_location_model_label(true_label))
        y_pred.append(_location_model_label(pred))
    return evaluate_location(y_true, y_pred, LOCATION_EVAL_LABELS)


def _predict_for_eval(task: str, pipeline: Pipeline, rows: list[dict[str, Any]], input_field: str) -> list[Any]:
    if task != "location":
        return pipeline.predict(_texts(rows, input_field)).tolist()

    # Match the current MWS-CFE protocol: no_location samples are handled by a
    # deterministic fallback and the classifier only predicts location-bearing rows.
    has_location_rows = [row for row in rows if row.get("location_label") != "no_location"]
    predictions = pipeline.predict(_texts(has_location_rows, input_field)).tolist() if has_location_rows else []
    merged: list[Any] = []
    pred_idx = 0
    for row in rows:
        if row.get("location_label") == "no_location":
            merged.append("no_location")
        else:
            merged.append(predictions[pred_idx])
            pred_idx += 1
    return merged


def _load_splits(task: str, gate: str, ws_data_dir: Path, phase5_data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": load_jsonl(ws_data_dir / f"{task}_train_ws_{gate}.jsonl"),
        "val": load_jsonl(ws_data_dir / f"{task}_val_ws.jsonl"),
        "test_ws": load_jsonl(ws_data_dir / f"{task}_test_ws.jsonl"),
        "test_phase5": load_jsonl(phase5_data_dir / f"{task}_test.jsonl"),
    }


def _limit(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    return rows if n is None else rows[:n]


def run_one(args: argparse.Namespace, method: str, task: str, log_fp) -> Path:
    ws_data_dir = Path(args.ws_data_dir) if args.ws_data_dir else PROJECT_ROOT / "outputs" / "phaseA1" / task
    phase5_data_dir = Path(args.phase5_data_dir) if args.phase5_data_dir else PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    output_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2"
    results_dir = output_base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"main_{args.gate}_seed{args.seed}"
    splits = _load_splits(task, args.gate, ws_data_dir, phase5_data_dir)
    train_rows = _limit(splits["train"], args.max_train_samples)
    val_rows = _limit(splits["val"], args.max_val_samples)
    test_ws_rows = _limit(splits["test_ws"], args.max_test_samples)
    phase5_test_rows = splits["test_phase5"]

    log(f"[Start] method={method} task={task} gate={args.gate} seed={args.seed} tag={tag}", log_fp)
    log(f"[Config] input_field={args.input_field} data=Phase A1 multi-source WS", log_fp)
    log(f"[Paths] ws_data_dir={ws_data_dir} phase5_data_dir={phase5_data_dir}", log_fp)
    log(
        f"[Data] train={len(train_rows)} val={len(val_rows)} "
        f"test_ws={len(test_ws_rows)} test_phase5={len(phase5_test_rows)}",
        log_fp,
    )

    pipeline = _build_pipeline(method, args.seed)
    train_texts = _texts(train_rows, args.input_field)
    if task == "density":
        train_labels: list[Any] = _density_labels(train_rows)
    elif task == "size":
        train_labels = _size_labels(train_rows)
    else:
        train_labels = _location_train_labels(train_rows)

    train_start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    train_time = time.perf_counter() - train_start
    log(f"[TrainDone] method={method} task={task} train_time={train_time:.2f}s", log_fp)

    eval_start = time.perf_counter()
    ws_val_predictions = _predict_for_eval(task, pipeline, val_rows, args.input_field)
    ws_test_predictions = _predict_for_eval(task, pipeline, test_ws_rows, args.input_field)
    phase5_predictions = _predict_for_eval(task, pipeline, phase5_test_rows, args.input_field)

    ws_val_results = _evaluate_predictions(task, val_rows, ws_val_predictions)
    ws_test_results = _evaluate_predictions(task, test_ws_rows, ws_test_predictions)
    phase5_test_results = _evaluate_predictions(task, phase5_test_rows, phase5_predictions)
    eval_time = time.perf_counter() - eval_start

    log(
        f"[Metric] phase5_{_primary_metric(task)}="
        f"{phase5_test_results.get(_primary_metric(task), 0.0):.4f} "
        f"phase5_accuracy={phase5_test_results.get('accuracy', 0.0):.4f}",
        log_fp,
    )

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    result = {
        "method": method,
        "task": task,
        "seed": args.seed,
        "gate": args.gate,
        "tag": tag,
        "input_field": args.input_field,
        "label_field": _task_label_field(task),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_ws_samples": len(test_ws_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "ws_val_results": to_jsonable(ws_val_results),
        "ws_test_results": to_jsonable(ws_test_results),
        "phase5_test_results": to_jsonable(phase5_test_results),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "best_epoch": None,
        "peak_gpu_memory_gb": 0.0,
        "model_config": {
            "vectorizer": {
                "class": "TfidfVectorizer",
                "max_features": vectorizer.max_features,
                "ngram_range": list(vectorizer.ngram_range),
                "sublinear_tf": vectorizer.sublinear_tf,
                "vocabulary_size": len(vectorizer.vocabulary_),
            },
            "classifier": classifier.__class__.__name__,
            "classifier_params": classifier.get_params(deep=False),
            "classifier_parameter_count": _classifier_param_count(pipeline),
        },
        "runtime_environment": {
            "device": "cpu",
            "phase5_single_source_reused": False,
        },
        "a2_5_protocol": {
            "training_data": "Phase A1 multi-source WS",
            "phase5_single_source_reused": False,
            "gate": args.gate,
            "phase5_test_for_main_table": True,
        },
    }

    result_path = results_dir / f"{method}_{task}_results_{tag}.json"
    result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[Saved] {result_path}", log_fp)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase A2.5 TF-IDF LR/SVM baselines")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma separated: tfidf_lr,tfidf_svm")
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma separated: density,size,location")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate", type=str, default="g2")
    parser.add_argument("--ws-data-dir", type=str, default=None, help="Override one task WS data dir; use with a single task")
    parser.add_argument("--phase5-data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--input-field", type=str, default="mention_text")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    methods = _parse_csv(args.methods, METHODS, "methods")
    tasks = _parse_csv(args.tasks, TASKS, "tasks")
    if args.ws_data_dir and len(tasks) != 1:
        raise ValueError("--ws-data-dir override is only valid with exactly one task")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else log_dir / f"train_tfidf_baselines_{args.gate}_seed{args.seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] train_tfidf_baselines", log_fp)
        log(f"[Config] methods={methods} tasks={tasks} seed={args.seed} gate={args.gate}", log_fp)
        written = []
        for task in tasks:
            for method in methods:
                written.append(str(run_one(args, method, task, log_fp)))
        log(f"[Done] wrote {len(written)} result files", log_fp)
        for path in written:
            log(f"  {path}", log_fp)


if __name__ == "__main__":
    main()

