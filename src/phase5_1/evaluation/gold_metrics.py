from __future__ import annotations

import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    median_absolute_error,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.evaluation.metrics import (
    evaluate_density,
    evaluate_location,
    evaluate_size_detection,
    evaluate_size_regression,
)


def _to_float(value: Any) -> float:
    return float(value)


def _normalize_label_mapping(values: list[Any], label_names: list[str]) -> dict[str, float]:
    return {
        label: _to_float(score)
        for label, score in zip(label_names, values, strict=False)
    }


def _serialize_binary_confusion_matrix(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = {
        str(true_label): {
            str(pred_label): int(matrix[row_index, col_index])
            for col_index, pred_label in enumerate(labels)
        }
        for row_index, true_label in enumerate(labels)
    }
    return {
        "labels": labels,
        "matrix": matrix.astype(int).tolist(),
        "rows": rows,
    }


def _extend_multiclass_metrics(
    base_result: dict[str, Any],
    y_true: list[Any],
    y_pred: list[Any],
    label_names: list[str],
) -> dict[str, Any]:
    precision_values = precision_score(
        y_true,
        y_pred,
        labels=label_names,
        average=cast(Any, None),
        zero_division=cast(Any, 0),
    )
    recall_values = recall_score(
        y_true,
        y_pred,
        labels=label_names,
        average=cast(Any, None),
        zero_division=cast(Any, 0),
    )
    class_distribution = Counter(y_true)

    return {
        **base_result,
        "weighted_f1": _to_float(
            f1_score(
                y_true,
                y_pred,
                labels=label_names,
                average="weighted",
                zero_division=cast(Any, 0),
            )
        ),
        "per_class_precision": _normalize_label_mapping(
            np.asarray(precision_values, dtype=float).tolist(),
            label_names,
        ),
        "per_class_recall": _normalize_label_mapping(
            np.asarray(recall_values, dtype=float).tolist(),
            label_names,
        ),
        "sample_count": int(len(y_true)),
        "class_distribution": {
            label: int(class_distribution.get(label, 0)) for label in label_names
        },
    }


def evaluate_density_gold(y_true, y_pred, label_names):
    base_result = evaluate_density(y_true, y_pred, label_names)
    return _extend_multiclass_metrics(base_result, list(y_true), list(y_pred), list(label_names))


def evaluate_location_gold(y_true, y_pred, label_names):
    base_result = evaluate_location(y_true, y_pred, label_names)
    return _extend_multiclass_metrics(base_result, list(y_true), list(y_pred), list(label_names))


def evaluate_has_size_gold(y_true, y_pred):
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    base_result = evaluate_size_detection(y_true_list, y_pred_list)
    return {
        **base_result,
        "confusion_matrix": _serialize_binary_confusion_matrix(y_true_list, y_pred_list),
        "sample_count": int(len(y_true_list)),
    }


def evaluate_size_regression_gold(y_true, y_pred, tolerance=1.0):
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    base_result = evaluate_size_regression(y_true_list, y_pred_list, tolerance=tolerance)
    y_true_array = np.asarray(y_true_list, dtype=float)
    y_pred_array = np.asarray(y_pred_list, dtype=float)

    if y_true_array.size == 0:
        return {
            **base_result,
            "median_absolute_error": 0.0,
            "max_error": 0.0,
            "sample_count": 0,
            "per_sample_errors": [],
        }

    absolute_error = np.abs(y_true_array - y_pred_array)
    return {
        **base_result,
        "median_absolute_error": _to_float(
            median_absolute_error(y_true_array, y_pred_array)
        ),
        "max_error": _to_float(np.max(absolute_error)),
        "sample_count": int(y_true_array.size),
        "per_sample_errors": [
            {
                "true": _to_float(true_value),
                "pred": _to_float(pred_value),
                "error": _to_float(error_value),
            }
            for true_value, pred_value, error_value in zip(
                y_true_array,
                y_pred_array,
                absolute_error,
                strict=False,
            )
        ],
    }


def compute_silver_vs_gold_agreement(silver_labels, gold_labels):
    silver_list = list(silver_labels)
    gold_list = list(gold_labels)
    disagreements = [
        {"index": int(index), "silver": silver_value, "gold": gold_value}
        for index, (silver_value, gold_value) in enumerate(
            zip_longest(silver_list, gold_list, fillvalue=None)
        )
        if silver_value != gold_value
    ]
    total = max(len(silver_list), len(gold_list))
    agreement_count = total - len(disagreements)
    agreement_rate = 0.0 if total == 0 else _to_float(agreement_count / total)
    return {
        "agreement_rate": agreement_rate,
        "disagreement_count": int(len(disagreements)),
        "total": int(total),
        "disagreements": disagreements,
    }


def _extract_prediction_value(prediction: Any, pred_field: str | None) -> Any:
    if pred_field is not None and isinstance(prediction, dict):
        return prediction.get(pred_field)
    return prediction


def _extract_silver_value(row: dict[str, Any], gold_field: str) -> Any:
    silver_keys = [
        "silver",
        "silver_label",
        f"silver_{gold_field}",
        f"{gold_field}_silver",
    ]
    for key in silver_keys:
        if key in row:
            return row.get(key)
    return None


def build_error_cases(manifest_rows, predictions, gold_field, pred_field=None):
    error_cases = []
    for row, prediction in zip_longest(manifest_rows, predictions, fillvalue=None):
        if row is None:
            continue
        gold_value = row.get(gold_field)
        predicted_value = _extract_prediction_value(prediction, pred_field)
        if predicted_value == gold_value:
            continue
        error_cases.append(
            {
                "sample_id": row.get("sample_id"),
                "mention_text": row.get("mention_text"),
                "gold": gold_value,
                "predicted": predicted_value,
                "silver": _extract_silver_value(row, gold_field),
                "annotation_confidence": row.get("annotation_confidence"),
            }
        )
    return error_cases
