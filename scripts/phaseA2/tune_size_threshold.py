#!/usr/bin/env python3
"""Tune Has-size decision threshold on validation data only.

This is eval-only. It loads a trained size model, optionally fits Platt scaling
on WS validation logits, chooses the validation-F1-optimal threshold, and then
evaluates the fixed threshold on WS test and Phase5 test.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import add_size_cue_augmented_mention_text
from scripts.phaseA2.train_mws_cfe_common import LazyMentionDataset, load_jsonl, to_jsonable
from src.phase5.evaluation.metrics import evaluate_size_detection

LABELS = ["no_size", "has_size"]
POSITIVE_ID = 1


def maybe_transform_rows(rows: list[dict[str, Any]], size_input_mode: str) -> list[dict[str, Any]]:
    if size_input_mode == "cue_augmented_mention_text":
        return [add_size_cue_augmented_mention_text(row) for row in rows]
    return rows


def label_encoder(row: dict[str, Any]) -> int:
    return 1 if bool(row.get("has_size")) else 0


def labels(rows: list[dict[str, Any]]) -> list[int]:
    return [label_encoder(row) for row in rows]


def predict_logits(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    input_field: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
    dataset = LazyMentionDataset(rows, tokenizer, label_encoder, max_length, input_field)
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "outputs" / "phaseA2_planB" / "tmp" / "size_threshold_predict"),
        per_device_eval_batch_size=max(1, int(batch_size)),
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    output = trainer.predict(dataset, metric_key_prefix="predict")
    return np.asarray(output.predictions, dtype=np.float64)


def raw_positive_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, POSITIVE_ID]


def platt_calibrate(val_logits: np.ndarray, val_y: np.ndarray, target_logits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    val_margin = (val_logits[:, 1] - val_logits[:, 0]).reshape(-1, 1)
    target_margin = (target_logits[:, 1] - target_logits[:, 0]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(val_margin, val_y)
    probs = calibrator.predict_proba(target_margin)[:, 1]
    return probs, {
        "type": "platt_scaling",
        "coef": calibrator.coef_.astype(float).ravel().tolist(),
        "intercept": calibrator.intercept_.astype(float).ravel().tolist(),
        "classes": calibrator.classes_.astype(int).tolist(),
    }


def evaluate_threshold(rows: list[dict[str, Any]], probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = labels(rows)
    y_pred = [1 if float(prob) >= threshold else 0 for prob in probs]
    result = evaluate_size_detection(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result["confusion_matrix"] = {
        "labels": LABELS,
        "matrix": matrix.astype(int).tolist(),
        "rows": {
            "no_size": {"no_size": int(matrix[0, 0]), "has_size": int(matrix[0, 1])},
            "has_size": {"no_size": int(matrix[1, 0]), "has_size": int(matrix[1, 1])},
        },
    }
    if len(set(y_true)) == 2:
        result["auprc"] = float(average_precision_score(y_true, probs))
        result["auroc"] = float(roc_auc_score(y_true, probs))
    else:
        result["auprc"] = None
        result["auroc"] = None
    return result


def tune_threshold(rows: list[dict[str, Any]], probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    y_true = np.asarray(labels(rows), dtype=np.int64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        threshold = 0.5
    else:
        precision, recall, thresholds = precision_recall_curve(y_true, probs, pos_label=1)
        if thresholds.size == 0:
            threshold = 0.5
        else:
            precision = precision[:-1]
            recall = recall[:-1]
            denom = precision + recall
            f1 = np.divide(
                2.0 * precision * recall,
                denom,
                out=np.zeros_like(denom, dtype=np.float64),
                where=denom > 0,
            )
            threshold = float(thresholds[int(np.argmax(f1))])
    return threshold, evaluate_threshold(rows, probs, threshold)


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(result["accuracy"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f1"]),
        "auprc": float(result["auprc"]) if result.get("auprc") is not None else None,
        "auroc": float(result["auroc"]) if result.get("auroc") is not None else None,
        "confusion_matrix": result.get("confusion_matrix", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Has-size threshold on WS validation data")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-input-mode", choices=["mention_text", "cue_augmented_mention_text"], default="mention_text")
    parser.add_argument("--input-field", default=None)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration", choices=["platt", "none"], default="platt")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    input_field = args.input_field or args.size_input_mode
    data_dir = Path(args.data_dir)
    phase5_data_dir = Path(args.phase5_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = maybe_transform_rows(load_jsonl(data_dir / "size_val_ws.jsonl"), args.size_input_mode)
    ws_test_rows = maybe_transform_rows(load_jsonl(data_dir / "size_test_ws.jsonl"), args.size_input_mode)
    phase5_test_rows = maybe_transform_rows(load_jsonl(phase5_data_dir / "size_test.jsonl"), args.size_input_mode)

    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
    if args.max_test_samples:
        ws_test_rows = ws_test_rows[: args.max_test_samples]
        phase5_test_rows = phase5_test_rows[: args.max_test_samples]

    val_logits = predict_logits(Path(args.model_dir), val_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)
    ws_test_logits = predict_logits(Path(args.model_dir), ws_test_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)
    phase5_logits = predict_logits(Path(args.model_dir), phase5_test_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)

    val_y = np.asarray(labels(val_rows), dtype=np.int64)
    raw_val_probs = raw_positive_probs(val_logits)
    base_val = evaluate_threshold(val_rows, raw_val_probs, 0.5)

    if args.calibration == "platt":
        val_probs, calibration = platt_calibrate(val_logits, val_y, val_logits)
        ws_test_probs, _ = platt_calibrate(val_logits, val_y, ws_test_logits)
        phase5_probs, _ = platt_calibrate(val_logits, val_y, phase5_logits)
    else:
        val_probs = raw_val_probs
        ws_test_probs = raw_positive_probs(ws_test_logits)
        phase5_probs = raw_positive_probs(phase5_logits)
        calibration = {"type": "none"}

    selected_threshold, tuned_val = tune_threshold(val_rows, val_probs)
    ws_test_result = evaluate_threshold(ws_test_rows, ws_test_probs, selected_threshold)
    phase5_result = evaluate_threshold(phase5_test_rows, phase5_probs, selected_threshold)

    output = {
        "method": "mws_cfe",
        "task": "size",
        "seed": args.seed,
        "tag": args.tag,
        "model_dir": str(args.model_dir),
        "input_field": input_field,
        "size_input_mode": args.size_input_mode,
        "label_field": "has_size",
        "train_samples": None,
        "val_samples": len(val_rows),
        "test_ws_samples": len(ws_test_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "threshold_tuning": {
            "selection_split": "size_val_ws",
            "test_set_used_for_threshold": False,
            "base_threshold": 0.5,
            "base_val_results": result_summary(base_val),
            "calibration": calibration,
            "selected_threshold": float(selected_threshold),
            "tuned_val_results": result_summary(tuned_val),
        },
        "ws_val_results": result_summary(tuned_val),
        "ws_test_results": result_summary(ws_test_result),
        "phase5_test_results": result_summary(phase5_result),
        "eval_time_seconds": time.time() - t0,
        "method_components": {
            "learned_model": True,
            "deterministic_rule_override": False,
            "decision_layer": f"{args.calibration}_validation_threshold_tuning",
        },
    }

    result_path = output_dir / f"mws_cfe_size_results_{args.tag}_seed{args.seed}.json"
    result_path.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {result_path}", flush=True)


if __name__ == "__main__":
    main()
