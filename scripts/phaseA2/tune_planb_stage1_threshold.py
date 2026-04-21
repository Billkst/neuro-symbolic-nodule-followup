#!/usr/bin/env python3
"""Tune Plan B density Stage 1 decision threshold on validation data.

This script never uses the test set to choose a threshold. It loads a trained
MWS-CFE density_stage1 model, fits Platt scaling on validation logits, chooses
the F1-optimal threshold on validation probabilities, and then evaluates the
fixed calibrated threshold on WS test and Phase5 test.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import LazyMentionDataset, load_jsonl, to_jsonable
from src.phase5.evaluation.metrics import evaluate_binary_detection

LABELS = ["explicit_density", "unclear_or_no_evidence"]
POSITIVE = "explicit_density"


def label_encoder(row: dict[str, Any]) -> int:
    return 0 if str(row.get("density_stage1_label")) == POSITIVE else 1


def label_strings(rows: list[dict[str, Any]]) -> list[str]:
    return [POSITIVE if label_encoder(row) == 0 else "unclear_or_no_evidence" for row in rows]


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
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    output = trainer.predict(dataset, metric_key_prefix="predict")
    return np.asarray(output.predictions, dtype=np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def raw_positive_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 0]


def platt_calibrate(val_logits: np.ndarray, val_y: np.ndarray, target_logits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    val_margin = (val_logits[:, 0] - val_logits[:, 1]).reshape(-1, 1)
    target_margin = (target_logits[:, 0] - target_logits[:, 1]).reshape(-1, 1)
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
    y_true = label_strings(rows)
    y_pred = [POSITIVE if float(prob) >= threshold else "unclear_or_no_evidence" for prob in probs]
    return evaluate_binary_detection(
        y_true,
        y_pred,
        LABELS,
        positive_label=POSITIVE,
        positive_scores=probs,
    )


def tune_threshold(rows: list[dict[str, Any]], probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = sorted({0.0, 0.5, 1.0, *[float(x) for x in probs]})
    best_threshold = 0.5
    best_result: dict[str, Any] | None = None
    for threshold in candidates:
        result = evaluate_threshold(rows, probs, threshold)
        if best_result is None or float(result["f1"]) > float(best_result["f1"]):
            best_threshold = threshold
            best_result = result
    assert best_result is not None
    return best_threshold, best_result


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(result["accuracy"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f1"]),
        "macro_f1": float(result["macro_f1"]),
        "auprc": float(result["auprc"]) if result.get("auprc") is not None else None,
        "auroc": float(result["auroc"]) if result.get("auroc") is not None else None,
        "confusion_matrix": result.get("confusion_matrix", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Plan B Stage 1 threshold on validation data")
    parser.add_argument("--model-dir", required=True, help="Trained density_stage1 model directory")
    parser.add_argument("--data-dir", default="outputs/phaseA2_planB/density_stage1")
    parser.add_argument("--phase5-data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-field", default="section_aware_text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    t0 = time.time()
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    phase5_data_dir = Path(args.phase5_data_dir) if args.phase5_data_dir else data_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = load_jsonl(data_dir / "density_stage1_val_ws.jsonl")
    ws_test_rows = load_jsonl(data_dir / "density_stage1_test_ws.jsonl")
    phase5_test_rows = load_jsonl(phase5_data_dir / "density_stage1_test.jsonl")

    val_logits = predict_logits(model_dir, val_rows, input_field=args.input_field, max_length=args.max_length, batch_size=args.batch_size)
    ws_test_logits = predict_logits(model_dir, ws_test_rows, input_field=args.input_field, max_length=args.max_length, batch_size=args.batch_size)
    phase5_logits = predict_logits(model_dir, phase5_test_rows, input_field=args.input_field, max_length=args.max_length, batch_size=args.batch_size)

    val_y = np.asarray([1 if label == POSITIVE else 0 for label in label_strings(val_rows)], dtype=np.int64)
    val_raw_probs = raw_positive_probs(val_logits)
    base_val = evaluate_threshold(val_rows, val_raw_probs, 0.5)

    val_cal_probs, calibration = platt_calibrate(val_logits, val_y, val_logits)
    ws_test_cal_probs, _ = platt_calibrate(val_logits, val_y, ws_test_logits)
    phase5_cal_probs, _ = platt_calibrate(val_logits, val_y, phase5_logits)

    tuned_threshold, tuned_val = tune_threshold(val_rows, val_cal_probs)
    ws_test_result = evaluate_threshold(ws_test_rows, ws_test_cal_probs, tuned_threshold)
    phase5_result = evaluate_threshold(phase5_test_rows, phase5_cal_probs, tuned_threshold)

    output = {
        "method": "mws_cfe",
        "task": "density_stage1",
        "seed": args.seed,
        "tag": args.tag,
        "model_dir": str(model_dir),
        "input_field": args.input_field,
        "label_field": "density_stage1_label",
        "train_samples": None,
        "val_samples": len(val_rows),
        "test_ws_samples": len(ws_test_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "threshold_tuning": {
            "selection_split": "density_stage1_val_ws",
            "test_set_used_for_threshold": False,
            "base_threshold": 0.5,
            "base_val_results": result_summary(base_val),
            "calibration": calibration,
            "selected_threshold": float(tuned_threshold),
            "tuned_val_results": result_summary(tuned_val),
        },
        "ws_val_results": result_summary(tuned_val),
        "ws_test_results": result_summary(ws_test_result),
        "phase5_test_results": result_summary(phase5_result),
        "eval_time_seconds": time.time() - t0,
        "planb_protocol": {
            "task_definition": "two-stage density",
            "stage1_decision_layer": "validation_platt_scaling_plus_threshold_tuning",
            "phase5_test_for_main_table": True,
        },
    }

    result_path = output_dir / f"mws_cfe_density_stage1_results_{args.tag}_seed{args.seed}.json"
    result_path.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {result_path}", flush=True)


if __name__ == "__main__":
    main()
