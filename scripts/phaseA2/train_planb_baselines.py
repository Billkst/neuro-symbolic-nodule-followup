#!/usr/bin/env python3
"""Train/evaluate Plan B non-PLM baselines.

Methods:
- regex_cue: cue-only rules, no learned parameters.
- tfidf_lr: TF-IDF + Logistic Regression.
- tfidf_svm: TF-IDF + Linear SVM.
- tfidf_mlp: TF-IDF + shallow MLP.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import load_jsonl, to_jsonable
from src.extractors.nodule_extractor import extract_density, extract_location, extract_size
from src.phase5.evaluation.metrics import (
    evaluate_binary_detection,
    evaluate_density,
    evaluate_location,
    evaluate_size_detection,
)


METHODS = ["regex_cue", "tfidf_lr", "tfidf_svm", "tfidf_mlp"]
TASKS = ["density_stage1", "density_stage2", "size", "location"]
EXPLICIT_DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified"]
STAGE1_LABELS = ["explicit_density", "unclear_or_no_evidence"]
LOCATION_MODEL_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"]
LOCATION_EVAL_LABELS = LOCATION_MODEL_LABELS + ["no_location"]


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def parse_csv(value: str | None, choices: list[str], name: str) -> list[str]:
    if not value:
        return list(choices)
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(f"{name} contains invalid values: {invalid}; choices={choices}")
    return items


def labels_for_task(task: str) -> list[str]:
    if task == "density_stage1":
        return STAGE1_LABELS
    if task == "density_stage2":
        return EXPLICIT_DENSITY_LABELS
    if task == "size":
        return ["no_size", "has_size"]
    if task == "location":
        return LOCATION_EVAL_LABELS
    raise ValueError(f"Unsupported task: {task}")


def label_field_for_task(task: str) -> str:
    if task == "density_stage1":
        return "density_stage1_label"
    if task == "density_stage2":
        return "density_stage2_label"
    if task == "size":
        return "has_size"
    if task == "location":
        return "location_label"
    raise ValueError(f"Unsupported task: {task}")


def primary_metric(task: str) -> str:
    if task == "density_stage1":
        return "auprc"
    if task == "size":
        return "f1"
    return "macro_f1"


def default_task_data_dir(task: str) -> Path:
    if task in {"density_stage1", "density_stage2"}:
        return PROJECT_ROOT / "outputs" / "phaseA2_planB" / task
    return PROJECT_ROOT / "outputs" / "phaseA1" / task


def texts(rows: list[dict[str, Any]], input_field: str) -> list[str]:
    return [str(row.get(input_field) or row.get("mention_text") or "") for row in rows]


def task_labels(task: str, rows: list[dict[str, Any]]) -> list[Any]:
    if task == "density_stage1":
        return [str(row.get("density_stage1_label") or "unclear_or_no_evidence") for row in rows]
    if task == "density_stage2":
        return [str(row.get("density_stage2_label") or "solid") for row in rows]
    if task == "size":
        return [1 if bool(row.get("has_size")) else 0 for row in rows]
    if task == "location":
        labels = []
        for row in rows:
            label = row.get("location_label")
            labels.append(str(label) if label in LOCATION_EVAL_LABELS else "no_location")
        return labels
    raise ValueError(f"Unsupported task: {task}")


def train_sample_weights(rows: list[dict[str, Any]], enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    return np.asarray([float(row.get("ws_confidence", 1.0) or 0.0) for row in rows], dtype=float)


def build_pipeline(method: str, seed: int) -> Pipeline:
    if method == "tfidf_lr":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=seed)
    elif method == "tfidf_svm":
        classifier = LinearSVC(class_weight="balanced", max_iter=5000, C=1.0, random_state=seed)
    elif method == "tfidf_mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            early_stopping=True,
            max_iter=50,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported learned method: {method}")
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


def majority_label(task: str, train_rows: list[dict[str, Any]]) -> Any:
    labels = task_labels(task, train_rows)
    return Counter(labels).most_common(1)[0][0]


def regex_predict(task: str, rows: list[dict[str, Any]], fallback: Any) -> tuple[list[Any], list[float] | None]:
    predictions: list[Any] = []
    scores: list[float] = []
    for row in rows:
        mention = str(row.get("mention_text") or "")
        if task == "density_stage1":
            density, _ = extract_density(mention)
            is_explicit = density in EXPLICIT_DENSITY_LABELS
            predictions.append("explicit_density" if is_explicit else "unclear_or_no_evidence")
            scores.append(1.0 if is_explicit else 0.0)
        elif task == "density_stage2":
            density, _ = extract_density(mention)
            predictions.append(density if density in EXPLICIT_DENSITY_LABELS else fallback)
        elif task == "size":
            size_value, _ = extract_size(mention)
            predictions.append(1 if size_value is not None else 0)
        elif task == "location":
            label, _ = extract_location(mention)
            predictions.append(str(label) if label in LOCATION_EVAL_LABELS else "no_location")
        else:
            raise ValueError(f"Unsupported task: {task}")
    return predictions, scores if task == "density_stage1" else None


def positive_scores_from_pipeline(task: str, pipeline: Pipeline, rows: list[dict[str, Any]], input_field: str) -> list[float] | None:
    if task != "density_stage1":
        return None
    x = texts(rows, input_field)
    clf = pipeline.named_steps["clf"]
    classes = list(getattr(clf, "classes_", []))
    if "explicit_density" not in classes:
        return [0.0 for _ in rows]
    pos_idx = classes.index("explicit_density")
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x)[:, pos_idx].astype(float).tolist()
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x)
        scores_arr = np.asarray(scores, dtype=float)
        if scores_arr.ndim == 1:
            signed = scores_arr if pos_idx == 1 else -scores_arr
        else:
            signed = scores_arr[:, pos_idx]
        return (1.0 / (1.0 + np.exp(-signed))).astype(float).tolist()
    return None


def evaluate_task(task: str, rows: list[dict[str, Any]], predictions: list[Any], scores: list[float] | None = None) -> dict[str, Any]:
    if task == "density_stage1":
        y_true = task_labels(task, rows)
        y_pred = [str(pred) if str(pred) in STAGE1_LABELS else "unclear_or_no_evidence" for pred in predictions]
        return evaluate_binary_detection(
            y_true,
            y_pred,
            STAGE1_LABELS,
            positive_label="explicit_density",
            positive_scores=scores,
        )
    if task == "density_stage2":
        y_true = task_labels(task, rows)
        y_pred = [str(pred) if str(pred) in EXPLICIT_DENSITY_LABELS else "solid" for pred in predictions]
        return evaluate_density(y_true, y_pred, EXPLICIT_DENSITY_LABELS)
    if task == "size":
        y_true = task_labels(task, rows)
        y_pred = [int(pred) for pred in predictions]
        return evaluate_size_detection(y_true, y_pred)
    if task == "location":
        y_true = task_labels(task, rows)
        y_pred = [str(pred) if str(pred) in LOCATION_EVAL_LABELS else "no_location" for pred in predictions]
        return evaluate_location(y_true, y_pred, LOCATION_EVAL_LABELS)
    raise ValueError(f"Unsupported task: {task}")


def load_splits(task: str, gate: str, task_data_dir: Path, phase5_data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    if task in {"density_stage1", "density_stage2"}:
        return {
            "train": load_jsonl(task_data_dir / f"{task}_train_ws_{gate}.jsonl"),
            "val": load_jsonl(task_data_dir / f"{task}_val_ws.jsonl"),
            "test_ws": load_jsonl(task_data_dir / f"{task}_test_ws.jsonl"),
            "test_phase5": load_jsonl(task_data_dir / f"{task}_test.jsonl"),
        }
    return {
        "train": load_jsonl(task_data_dir / f"{task}_train_ws_{gate}.jsonl"),
        "val": load_jsonl(task_data_dir / f"{task}_val_ws.jsonl"),
        "test_ws": load_jsonl(task_data_dir / f"{task}_test_ws.jsonl"),
        "test_phase5": load_jsonl(phase5_data_dir / f"{task}_test.jsonl"),
    }


def limit_rows(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    return rows if n is None else rows[:n]


def classifier_param_count(pipeline: Pipeline | None) -> int | None:
    if pipeline is None:
        return 0
    total = 0
    clf = pipeline.named_steps["clf"]
    for attr in ("coef_", "intercept_", "coefs_", "intercepts_"):
        value = getattr(clf, attr, None)
        if value is None:
            continue
        if isinstance(value, list):
            total += sum(int(np.asarray(item).size) for item in value)
        else:
            total += int(np.asarray(value).size)
    return total


def run_one(args: argparse.Namespace, method: str, task: str, log_fp) -> Path:
    task_data_dir = Path(args.task_data_dir) if args.task_data_dir else default_task_data_dir(task)
    if not task_data_dir.is_absolute():
        task_data_dir = PROJECT_ROOT / task_data_dir
    phase5_data_dir = PROJECT_ROOT / args.phase5_data_dir
    output_base = PROJECT_ROOT / args.output_dir
    results_dir = output_base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"main_{args.gate}_seed{args.seed}"

    splits = load_splits(task, args.gate, task_data_dir, phase5_data_dir)
    train_rows = limit_rows(splits["train"], args.max_train_samples)
    val_rows = limit_rows(splits["val"], args.max_val_samples)
    test_ws_rows = limit_rows(splits["test_ws"], args.max_test_samples)
    phase5_test_rows = splits["test_phase5"]

    log(f"[Start] method={method} task={task} gate={args.gate} tag={tag}", log_fp)
    log(f"[Config] input_field={args.input_field} confidence_weight={args.use_confidence_weight}", log_fp)
    log(f"[Paths] task_data_dir={task_data_dir} phase5_data_dir={phase5_data_dir}", log_fp)
    log(
        f"[Data] train={len(train_rows)} val={len(val_rows)} "
        f"test_ws={len(test_ws_rows)} test_phase5={len(phase5_test_rows)}",
        log_fp,
    )

    train_start = time.perf_counter()
    pipeline: Pipeline | None = None
    sample_weight_used = False
    fallback = majority_label(task, train_rows)
    if method == "regex_cue":
        train_time = 0.0
    else:
        pipeline = build_pipeline(method, args.seed)
        fit_kwargs: dict[str, Any] = {}
        weights = train_sample_weights(train_rows, args.use_confidence_weight)
        if weights is not None:
            fit_kwargs["clf__sample_weight"] = weights
        try:
            pipeline.fit(texts(train_rows, args.input_field), task_labels(task, train_rows), **fit_kwargs)
            sample_weight_used = bool(fit_kwargs)
        except TypeError:
            pipeline.fit(texts(train_rows, args.input_field), task_labels(task, train_rows))
            sample_weight_used = False
        train_time = time.perf_counter() - train_start
    log(f"[TrainDone] method={method} task={task} train_time={train_time:.2f}s", log_fp)

    def predict(rows: list[dict[str, Any]]) -> tuple[list[Any], list[float] | None]:
        if method == "regex_cue":
            return regex_predict(task, rows, fallback)
        assert pipeline is not None
        preds = pipeline.predict(texts(rows, args.input_field)).tolist()
        scores = positive_scores_from_pipeline(task, pipeline, rows, args.input_field)
        return preds, scores

    eval_start = time.perf_counter()
    ws_val_pred, ws_val_scores = predict(val_rows)
    ws_test_pred, ws_test_scores = predict(test_ws_rows)
    phase5_pred, phase5_scores = predict(phase5_test_rows)
    ws_val_results = evaluate_task(task, val_rows, ws_val_pred, ws_val_scores)
    ws_test_results = evaluate_task(task, test_ws_rows, ws_test_pred, ws_test_scores)
    phase5_test_results = evaluate_task(task, phase5_test_rows, phase5_pred, phase5_scores)
    eval_time = time.perf_counter() - eval_start

    log(
        f"[Metric] phase5_{primary_metric(task)}={phase5_test_results.get(primary_metric(task), 0.0) or 0.0:.4f} "
        f"phase5_accuracy={phase5_test_results.get('accuracy', 0.0):.4f}",
        log_fp,
    )

    result = {
        "method": method,
        "task": task,
        "seed": args.seed,
        "gate": args.gate,
        "tag": tag,
        "input_field": args.input_field,
        "label_field": label_field_for_task(task),
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
            "method": method,
            "fallback_label": fallback,
            "classifier_parameter_count": classifier_param_count(pipeline),
            "sample_weight_used": sample_weight_used,
        },
        "planb_protocol": {
            "task_definition": "two-stage density" if task.startswith("density_stage") else "retained module2 task",
            "training_data": str(task_data_dir),
            "phase5_test_for_main_table": True,
        },
    }
    if pipeline is not None:
        vectorizer = pipeline.named_steps["tfidf"]
        result["model_config"]["vectorizer"] = {
            "class": "TfidfVectorizer",
            "max_features": vectorizer.max_features,
            "ngram_range": list(vectorizer.ngram_range),
            "sublinear_tf": vectorizer.sublinear_tf,
            "vocabulary_size": len(vectorizer.vocabulary_),
        }
        result["model_config"]["classifier"] = pipeline.named_steps["clf"].__class__.__name__

    result_path = results_dir / f"{method}_{task}_results_{tag}.json"
    result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[Saved] {result_path}", log_fp)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Plan B regex/TF-IDF/MLP baselines")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate", default="g2")
    parser.add_argument("--task-data-dir", default=None, help="Override data dir; valid with a single task")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--input-field", default="section_aware_text")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--use-confidence-weight", action="store_true")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    methods = parse_csv(args.methods, METHODS, "methods")
    tasks = parse_csv(args.tasks, TASKS, "tasks")
    if args.task_data_dir and len(tasks) != 1:
        raise ValueError("--task-data-dir override is only valid with exactly one task")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else log_dir / f"train_planb_baselines_{args.gate}_seed{args.seed}.log"
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] train_planb_baselines", log_fp)
        log(f"[Config] methods={methods} tasks={tasks} seed={args.seed}", log_fp)
        written = []
        for task in tasks:
            for method in methods:
                written.append(str(run_one(args, method, task, log_fp)))
        log(f"[Done] wrote {len(written)} result files", log_fp)
        for path in written:
            log(f"  {path}", log_fp)


if __name__ == "__main__":
    main()
