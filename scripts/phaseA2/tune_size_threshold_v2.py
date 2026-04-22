#!/usr/bin/env python3
"""Tune Has-size threshold with an arbitrary non-test selection split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
        output_dir=str(PROJECT_ROOT / "outputs" / "phaseA2_planB" / "tmp" / "size_threshold_v2_predict"),
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


def softmax_positive_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, POSITIVE_ID]


def platt_calibrate(selection_logits: np.ndarray, selection_y: np.ndarray, target_logits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    selection_margin = (selection_logits[:, 1] - selection_logits[:, 0]).reshape(-1, 1)
    target_margin = (target_logits[:, 1] - target_logits[:, 0]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(selection_margin, selection_y)
    probs = calibrator.predict_proba(target_margin)[:, 1]
    return probs, {
        "type": "platt_scaling",
        "coef": calibrator.coef_.astype(float).ravel().tolist(),
        "intercept": calibrator.intercept_.astype(float).ravel().tolist(),
        "classes": calibrator.classes_.astype(int).tolist(),
    }


def binary_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)).mean())


def temperature_calibrate(selection_logits: np.ndarray, selection_y: np.ndarray, target_logits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = np.concatenate(
        [
            np.linspace(0.05, 0.95, 19),
            np.linspace(1.0, 5.0, 17),
            np.linspace(5.5, 12.0, 14),
        ]
    )
    losses = []
    for temperature in candidates:
        losses.append(binary_nll(selection_y, softmax_positive_probs(selection_logits, float(temperature))))
    best_idx = int(np.argmin(np.asarray(losses)))
    best_temperature = float(candidates[best_idx])
    return softmax_positive_probs(target_logits, best_temperature), {
        "type": "temperature_scaling_grid",
        "temperature": best_temperature,
        "selection_nll": float(losses[best_idx]),
    }


def calibrate_probs(
    selection_logits: np.ndarray,
    selection_y: np.ndarray,
    target_logits: np.ndarray,
    calibration: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if calibration == "platt":
        return platt_calibrate(selection_logits, selection_y, target_logits)
    if calibration == "temperature":
        return temperature_calibrate(selection_logits, selection_y, target_logits)
    return softmax_positive_probs(target_logits), {"type": "none"}


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


def maybe_limit(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    return rows if n is None else rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Has-size threshold on a non-test selection split")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--selection-split", required=True, help="JSONL path used for calibration and threshold selection")
    parser.add_argument("--selection-source", default="phase5_like_dev")
    parser.add_argument("--ws-test-split", default="outputs/phaseA1/size/size_test_ws.jsonl")
    parser.add_argument("--phase5-test-split", default="outputs/phase5/datasets/size_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-input-mode", choices=["mention_text", "cue_augmented_mention_text"], default="mention_text")
    parser.add_argument("--input-field", default=None)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration", choices=["none", "platt", "temperature"], default="platt")
    parser.add_argument("--max-selection-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    input_field = args.input_field or args.size_input_mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = maybe_transform_rows(maybe_limit(load_jsonl(Path(args.selection_split)), args.max_selection_samples), args.size_input_mode)
    ws_test_rows = maybe_transform_rows(maybe_limit(load_jsonl(Path(args.ws_test_split)), args.max_test_samples), args.size_input_mode)
    phase5_test_rows = maybe_transform_rows(maybe_limit(load_jsonl(Path(args.phase5_test_split)), args.max_test_samples), args.size_input_mode)

    model_dir = Path(args.model_dir)
    selection_logits = predict_logits(model_dir, selection_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)
    ws_test_logits = predict_logits(model_dir, ws_test_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)
    phase5_logits = predict_logits(model_dir, phase5_test_rows, input_field=input_field, max_length=args.max_length, batch_size=args.batch_size)

    selection_y = np.asarray(labels(selection_rows), dtype=np.int64)
    raw_selection_probs = softmax_positive_probs(selection_logits)
    base_selection = evaluate_threshold(selection_rows, raw_selection_probs, 0.5)

    selection_probs, calibration_meta = calibrate_probs(selection_logits, selection_y, selection_logits, args.calibration)
    ws_test_probs, _ = calibrate_probs(selection_logits, selection_y, ws_test_logits, args.calibration)
    phase5_probs, _ = calibrate_probs(selection_logits, selection_y, phase5_logits, args.calibration)

    selected_threshold, tuned_selection = tune_threshold(selection_rows, selection_probs)
    ws_test_result = evaluate_threshold(ws_test_rows, ws_test_probs, selected_threshold)
    phase5_result = evaluate_threshold(phase5_test_rows, phase5_probs, selected_threshold)

    output = {
        "method": "mws_cfe",
        "task": "size",
        "seed": args.seed,
        "tag": args.tag,
        "model_dir": str(model_dir),
        "input_field": input_field,
        "size_input_mode": args.size_input_mode,
        "label_field": "has_size",
        "train_samples": None,
        "val_samples": len(selection_rows),
        "test_ws_samples": len(ws_test_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "threshold_tuning": {
            "selection_split": str(args.selection_split),
            "selection_source": args.selection_source,
            "test_set_used_for_threshold": False,
            "base_threshold": 0.5,
            "base_val_results": result_summary(base_selection),
            "calibration": calibration_meta,
            "selected_threshold": float(selected_threshold),
            "tuned_val_results": result_summary(tuned_selection),
        },
        "ws_val_results": result_summary(tuned_selection),
        "ws_test_results": result_summary(ws_test_result),
        "phase5_test_results": result_summary(phase5_result),
        "eval_time_seconds": time.time() - t0,
        "method_components": {
            "learned_model": True,
            "deterministic_rule_override": False,
            "decision_layer": f"{args.calibration}_phase5_like_dev_threshold_tuning",
        },
    }

    result_path = output_dir / f"mws_cfe_size_results_{args.tag}_seed{args.seed}.json"
    result_path.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {result_path}", flush=True)


if __name__ == "__main__":
    main()
