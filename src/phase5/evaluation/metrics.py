from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def _to_serializable_float(value: float | np.floating[Any]) -> float:
    return float(value)


def _serialize_confusion_matrix(
    y_true: list[Any] | np.ndarray,
    y_pred: list[Any] | np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    rows = {
        true_label: {
            pred_label: int(matrix[row_idx, col_idx])
            for col_idx, pred_label in enumerate(label_names)
        }
        for row_idx, true_label in enumerate(label_names)
    }
    return {
        "labels": list(label_names),
        "matrix": matrix.astype(int).tolist(),
        "rows": rows,
    }


def evaluate_density(y_true, y_pred, label_names):
    per_class_scores = np.asarray(
        f1_score(
            y_true,
            y_pred,
            labels=label_names,
            average=cast(Any, None),
            zero_division=cast(Any, 0),
        ),
        dtype=float,
    ).tolist()
    return {
        "accuracy": _to_serializable_float(accuracy_score(y_true, y_pred)),
        "macro_f1": _to_serializable_float(
            f1_score(
                y_true,
                y_pred,
                labels=label_names,
                average="macro",
                zero_division=cast(Any, 0),
            )
        ),
        "per_class_f1": {
            label: _to_serializable_float(score)
            for label, score in zip(label_names, per_class_scores, strict=False)
        },
        "confusion_matrix": _serialize_confusion_matrix(y_true, y_pred, label_names),
    }


def evaluate_size_detection(y_true, y_pred):
    positive_label = 1
    return {
        "accuracy": _to_serializable_float(accuracy_score(y_true, y_pred)),
        "precision": _to_serializable_float(
            precision_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=cast(Any, 0),
            )
        ),
        "recall": _to_serializable_float(
            recall_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=cast(Any, 0),
            )
        ),
        "f1": _to_serializable_float(
            f1_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=cast(Any, 0),
            )
        ),
    }


def evaluate_size_regression(y_true, y_pred, tolerance=1.0):
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.size == 0:
        return {
            "mae": 0.0,
            "exact_match_rate": 0.0,
            "within_tolerance_rate": 0.0,
        }

    absolute_error = np.abs(y_true_array - y_pred_array)
    return {
        "mae": _to_serializable_float(mean_absolute_error(y_true_array, y_pred_array)),
        "exact_match_rate": _to_serializable_float(np.mean(absolute_error == 0.0)),
        "within_tolerance_rate": _to_serializable_float(np.mean(absolute_error <= tolerance)),
    }


def evaluate_location(y_true, y_pred, label_names):
    per_class_scores = np.asarray(
        f1_score(
            y_true,
            y_pred,
            labels=label_names,
            average=cast(Any, None),
            zero_division=cast(Any, 0),
        ),
        dtype=float,
    ).tolist()
    return {
        "accuracy": _to_serializable_float(accuracy_score(y_true, y_pred)),
        "macro_f1": _to_serializable_float(
            f1_score(
                y_true,
                y_pred,
                labels=label_names,
                average="macro",
                zero_division=cast(Any, 0),
            )
        ),
        "per_class_f1": {
            label: _to_serializable_float(score)
            for label, score in zip(label_names, per_class_scores, strict=False)
        },
        "confusion_matrix": _serialize_confusion_matrix(y_true, y_pred, label_names),
    }


def format_results_table(results_dict):
    rows = ["| 指标 | 数值 |", "|---|---|"]

    for key, value in results_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    rows.append(f"| {key}.{sub_key} | `{sub_value}` |")
                elif isinstance(sub_value, float):
                    rows.append(f"| {key}.{sub_key} | {sub_value:.4f} |")
                else:
                    rows.append(f"| {key}.{sub_key} | {sub_value} |")
            continue

        if isinstance(value, float):
            rows.append(f"| {key} | {value:.4f} |")
        else:
            rows.append(f"| {key} | {value} |")

    return "\n".join(rows)
