#!/usr/bin/env python3
"""Evaluate rule-first + model-fallback hybrids for Plan B size/location.

The script emits result JSON files compatible with the Plan B aggregator:

    mws_cfe_size_results_<tag>_seed<seed>.json
    mws_cfe_location_results_<tag>_seed<seed>.json

Rules are applied first. A trained neural classifier can be used only on
rule-unresolved rows with a validation-selected confidence threshold. If no
model directory is supplied, the fallback is deterministic negative/no-location.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import LazyMentionDataset, load_jsonl, to_jsonable
from src.extractors.nodule_extractor import extract_location, extract_size
from src.phase5.evaluation.metrics import evaluate_location, evaluate_size_detection

SIZE_LABELS = ["no_size", "has_size"]
LOCATION_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"]
LOCATION_EVAL_LABELS = LOCATION_LABELS + ["no_location"]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def make_label_encoder(task: str):
    if task == "size":
        return lambda row: 1 if bool(row.get("has_size")) else 0
    label_to_id = {label: idx for idx, label in enumerate(LOCATION_LABELS)}
    return lambda row: label_to_id.get(str(row.get("location_label")), 0)


def input_field_for_row(row: dict[str, Any], input_field: str) -> str:
    return str(row.get(input_field) or row.get("mention_text") or "")


def predict_model(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    task: str,
    input_field: str,
    max_length: int,
) -> tuple[list[str], list[float]]:
    if not rows:
        return [], []
    labels = SIZE_LABELS if task == "size" else LOCATION_LABELS
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(labels))
    dataset = LazyMentionDataset(rows, tokenizer, make_label_encoder(task), max_length, input_field)
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    output = trainer.predict(dataset, metric_key_prefix="predict")
    probs = softmax(np.asarray(output.predictions, dtype=np.float64))
    pred_ids = probs.argmax(axis=1).tolist()
    confs = probs.max(axis=1).astype(float).tolist()
    return [labels[idx] for idx in pred_ids], confs


def rule_prediction(task: str, row: dict[str, Any]) -> str | None:
    mention = str(row.get("mention_text") or "")
    if task == "size":
        size_value, _ = extract_size(mention)
        return "has_size" if size_value is not None else None
    label, _ = extract_location(mention)
    return str(label) if label in LOCATION_EVAL_LABELS and label != "no_location" else None


def true_labels(task: str, rows: list[dict[str, Any]]) -> list[Any]:
    if task == "size":
        return [1 if bool(row.get("has_size")) else 0 for row in rows]
    labels = []
    for row in rows:
        label = row.get("location_label")
        labels.append(str(label) if label in LOCATION_EVAL_LABELS else "no_location")
    return labels


def build_hybrid_predictions(
    task: str,
    rows: list[dict[str, Any]],
    *,
    model_dir: Path | None,
    input_field: str,
    max_length: int,
    fallback_threshold: float,
) -> tuple[list[Any], dict[str, int]]:
    predictions: list[Any] = [None] * len(rows)
    unresolved: list[dict[str, Any]] = []
    unresolved_indices: list[int] = []
    rule_hits = 0

    for idx, row in enumerate(rows):
        rule_label = rule_prediction(task, row)
        if rule_label is not None:
            rule_hits += 1
            predictions[idx] = 1 if task == "size" else rule_label
        else:
            unresolved.append(row)
            unresolved_indices.append(idx)

    model_used = 0
    default_used = 0
    if model_dir is not None and unresolved:
        model_preds, model_confs = predict_model(model_dir, unresolved, task=task, input_field=input_field, max_length=max_length)
        for idx, pred, conf in zip(unresolved_indices, model_preds, model_confs, strict=False):
            if conf >= fallback_threshold:
                model_used += 1
                predictions[idx] = 1 if task == "size" and pred == "has_size" else pred
            else:
                default_used += 1
                predictions[idx] = 0 if task == "size" else "no_location"
    else:
        for idx in unresolved_indices:
            default_used += 1
            predictions[idx] = 0 if task == "size" else "no_location"

    return predictions, {
        "rows": len(rows),
        "rule_hits": rule_hits,
        "unresolved": len(unresolved),
        "model_fallback_used": model_used,
        "default_fallback_used": default_used,
    }


def evaluate(task: str, rows: list[dict[str, Any]], predictions: list[Any]) -> dict[str, Any]:
    y_true = true_labels(task, rows)
    if task == "size":
        pred_int = [int(pred) for pred in predictions]
        result = evaluate_size_detection(y_true, pred_int)
        return {
            "accuracy": float(result["accuracy"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1": float(result["f1"]),
        }
    pred_str = [str(pred) if str(pred) in LOCATION_EVAL_LABELS else "no_location" for pred in predictions]
    result = evaluate_location(y_true, pred_str, LOCATION_EVAL_LABELS)
    return {
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "per_class_f1": result.get("per_class_f1", {}),
        "confusion_matrix": result.get("confusion_matrix", {}),
    }


def tune_fallback_threshold(
    task: str,
    val_rows: list[dict[str, Any]],
    *,
    model_dir: Path | None,
    input_field: str,
    max_length: int,
) -> tuple[float, dict[str, Any], dict[str, int]]:
    if model_dir is None:
        preds, stats = build_hybrid_predictions(
            task,
            val_rows,
            model_dir=None,
            input_field=input_field,
            max_length=max_length,
            fallback_threshold=1.0,
        )
        return 1.0, evaluate(task, val_rows, preds), stats

    candidates = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.01]
    best_threshold = candidates[0]
    best_result: dict[str, Any] | None = None
    best_stats: dict[str, int] | None = None
    metric_name = "f1" if task == "size" else "macro_f1"
    for threshold in candidates:
        preds, stats = build_hybrid_predictions(
            task,
            val_rows,
            model_dir=model_dir,
            input_field=input_field,
            max_length=max_length,
            fallback_threshold=threshold,
        )
        result = evaluate(task, val_rows, preds)
        if best_result is None or float(result[metric_name]) > float(best_result[metric_name]):
            best_threshold = threshold
            best_result = result
            best_stats = stats
    assert best_result is not None and best_stats is not None
    return best_threshold, best_result, best_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Plan B rule-first + model-fallback hybrid")
    parser.add_argument("--task", choices=["size", "location"], required=True)
    parser.add_argument("--model-dir", default=None, help="Optional trained MWS-CFE model directory for fallback")
    parser.add_argument("--data-dir", default=None, help="WS data directory; defaults to outputs/phaseA1/<task>")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--input-field", default="mention_text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    task = args.task
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "outputs" / "phaseA1" / task
    phase5_data_dir = Path(args.phase5_data_dir)
    model_dir = Path(args.model_dir) if args.model_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = load_jsonl(data_dir / f"{task}_val_ws.jsonl")
    ws_test_rows = load_jsonl(data_dir / f"{task}_test_ws.jsonl")
    phase5_test_rows = load_jsonl(phase5_data_dir / f"{task}_test.jsonl")

    threshold, val_results, val_stats = tune_fallback_threshold(
        task,
        val_rows,
        model_dir=model_dir,
        input_field=args.input_field,
        max_length=args.max_length,
    )
    ws_test_preds, ws_test_stats = build_hybrid_predictions(
        task,
        ws_test_rows,
        model_dir=model_dir,
        input_field=args.input_field,
        max_length=args.max_length,
        fallback_threshold=threshold,
    )
    phase5_preds, phase5_stats = build_hybrid_predictions(
        task,
        phase5_test_rows,
        model_dir=model_dir,
        input_field=args.input_field,
        max_length=args.max_length,
        fallback_threshold=threshold,
    )

    output = {
        "method": "mws_cfe",
        "task": task,
        "seed": args.seed,
        "tag": args.tag,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "input_field": args.input_field,
        "label_field": "has_size" if task == "size" else "location_label",
        "train_samples": None,
        "val_samples": len(val_rows),
        "test_ws_samples": len(ws_test_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "hybrid_policy": {
            "type": "rule_first_model_fallback",
            "rule_source": "src.extractors.nodule_extractor",
            "fallback_threshold_selected_on": f"{task}_val_ws",
            "fallback_threshold": threshold,
            "test_set_used_for_threshold": False,
        },
        "hybrid_stats": {
            "ws_val": val_stats,
            "ws_test": ws_test_stats,
            "phase5_test": phase5_stats,
        },
        "ws_val_results": evaluate(task, val_rows, build_hybrid_predictions(
            task,
            val_rows,
            model_dir=model_dir,
            input_field=args.input_field,
            max_length=args.max_length,
            fallback_threshold=threshold,
        )[0]),
        "ws_test_results": evaluate(task, ws_test_rows, ws_test_preds),
        "phase5_test_results": evaluate(task, phase5_test_rows, phase5_preds),
        "eval_time_seconds": time.time() - t0,
        "planb_protocol": {
            "task_definition": "hybrid retained module2 task",
            "phase5_test_for_main_table": True,
        },
    }

    result_path = output_dir / f"mws_cfe_{task}_results_{args.tag}_seed{args.seed}.json"
    result_path.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {result_path}", flush=True)


if __name__ == "__main__":
    main()
