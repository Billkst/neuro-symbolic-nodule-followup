#!/usr/bin/env python3
"""Train Wave5 Has-size learned stacked/calibrated heads.

The meta-classifier is learned from non-test validation / Phase5-like dev
examples. Inputs may include lexical expert probability, BERT probability, and
symbolic cue features. No deterministic rule override is applied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import size_cue_features
from scripts.phaseA2.train_mws_cfe_common import load_jsonl, to_jsonable
from scripts.phaseA2.tune_size_threshold_v2 import predict_logits, softmax_positive_probs
from src.phase5.evaluation.metrics import evaluate_size_detection

LABELS = ["no_size", "has_size"]
CANDIDATES = {
    "lexical_bert_lr": {
        "uses_lexical_probability": True,
        "uses_bert_probability": True,
        "uses_cue_features": False,
    },
    "lexical_bert_cue_lr": {
        "uses_lexical_probability": True,
        "uses_bert_probability": True,
        "uses_cue_features": True,
    },
    "lexical_cue_lr": {
        "uses_lexical_probability": True,
        "uses_bert_probability": False,
        "uses_cue_features": True,
    },
}
CUE_FEATURE_NAMES = [
    "size_unit_mm_cm",
    "size_numeric_unit",
    "size_numeric_unit_count",
    "size_2d_pattern",
    "size_3d_pattern",
    "size_range_pattern",
    "size_context_word",
]


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def bool_label(row: dict[str, Any]) -> int:
    return 1 if bool(row.get("has_size")) else 0


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool_label(row) for row in rows], dtype=np.int64)


def resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def limit_rows(rows: list[dict[str, Any]], n: int | None) -> tuple[list[dict[str, Any]], bool]:
    if n is None or len(rows) <= n:
        return rows, False
    return rows[:n], True


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probability_file(path: Path) -> np.ndarray:
    probs: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            probs.append(float(json.loads(line)["prob_has_size"]))
    return np.asarray(probs, dtype=np.float64)


def lexical_probs_from_manifest(manifest: dict[str, Any], split: str, expected_n: int) -> np.ndarray:
    outputs = manifest.get("probability_outputs", {})
    if split not in outputs:
        raise KeyError(f"Lexical probability manifest does not contain split={split}")
    prob_path = resolve_path(str(outputs[split]["path"]), PROJECT_ROOT / str(outputs[split]["path"]))
    probs = load_probability_file(prob_path)
    if len(probs) != expected_n:
        raise ValueError(
            f"Lexical probability count mismatch for {split}: probs={len(probs)} rows={expected_n}. "
            "Regenerate the lexical expert probabilities with matching selection/val truncation settings."
        )
    return probs


def cue_vector(row: dict[str, Any]) -> list[float]:
    features = size_cue_features(str(row.get("mention_text") or ""))
    vector: list[float] = []
    for name in CUE_FEATURE_NAMES:
        value = features.get(name, "no")
        if name == "size_numeric_unit_count":
            vector.append({"0": 0.0, "1": 1.0, "2": 2.0, "3plus": 3.0}.get(str(value), 0.0))
        else:
            vector.append(1.0 if str(value).lower() == "yes" else 0.0)
    return vector


def build_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    lexical_probs: np.ndarray,
    bert_probs: np.ndarray | None,
    candidate: str,
) -> tuple[np.ndarray, list[str]]:
    spec = CANDIDATES[candidate]
    columns: list[str] = []
    parts: list[np.ndarray] = []

    if spec["uses_lexical_probability"]:
        parts.append(lexical_probs.reshape(-1, 1))
        columns.append("lexical_prob_has_size")
    if spec["uses_bert_probability"]:
        if bert_probs is None:
            raise ValueError(f"Candidate {candidate} requires BERT probabilities")
        parts.append(bert_probs.reshape(-1, 1))
        columns.append("bert_prob_has_size")
    if spec["uses_cue_features"]:
        cue = np.asarray([cue_vector(row) for row in rows], dtype=np.float64)
        parts.append(cue)
        columns.extend(CUE_FEATURE_NAMES)

    if not parts:
        raise ValueError(f"Candidate {candidate} produced no features")
    return np.concatenate(parts, axis=1).astype(np.float64), columns


def evaluate_probs(rows: list[dict[str, Any]], probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = labels(rows)
    y_pred = (probs >= float(threshold)).astype(np.int64)
    result = evaluate_size_detection(y_true.tolist(), y_pred.tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result["confusion_matrix"] = {
        "labels": LABELS,
        "matrix": matrix.astype(int).tolist(),
        "rows": {
            "no_size": {"no_size": int(matrix[0, 0]), "has_size": int(matrix[0, 1])},
            "has_size": {"no_size": int(matrix[1, 0]), "has_size": int(matrix[1, 1])},
        },
    }
    if len(set(y_true.tolist())) == 2:
        result["auprc"] = float(average_precision_score(y_true, probs))
        result["auroc"] = float(roc_auc_score(y_true, probs))
    else:
        result["auprc"] = None
        result["auroc"] = None
    return result


def tune_threshold(rows: list[dict[str, Any]], probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    y_true = labels(rows)
    if len(rows) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        threshold = 0.5
    else:
        precision, recall, thresholds = precision_recall_curve(y_true, probs, pos_label=1)
        if thresholds.size == 0:
            threshold = 0.5
        else:
            precision = precision[:-1]
            recall = recall[:-1]
            denom = precision + recall
            f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
            threshold = float(thresholds[int(np.argmax(f1))])
    return threshold, evaluate_probs(rows, probs, threshold)


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


def split_meta_indices(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(y), dtype=np.int64)
    if len(indices) < 4 or val_fraction <= 0:
        return indices, indices, "selection_full_no_holdout"
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) == 2 and int(counts.min()) >= 2
    stratify = y if can_stratify else None
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return indices, indices, "selection_full_split_failed"
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return indices, indices, "selection_full_single_class_fallback"
    split_type = "stratified_holdout" if can_stratify else "random_holdout"
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), split_type


def train_meta_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c: float,
    max_iter: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    if len(np.unique(y_train)) < 2:
        raise ValueError("Meta-classifier training split must contain both has_size classes")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(x_scaled, y_train)
    return scaler, clf


def predict_meta_probs(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    classes = list(getattr(clf, "classes_", []))
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=np.float64)
    pos_idx = classes.index(1)
    return np.asarray(clf.predict_proba(scaler.transform(x))[:, pos_idx], dtype=np.float64)


def bert_probs_for_split(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    input_field: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    logits = predict_logits(model_dir, rows, input_field=input_field, max_length=max_length, batch_size=batch_size)
    return softmax_positive_probs(logits)


def classifier_param_count(clf: LogisticRegression) -> int:
    total = 0
    for attr in ("coef_", "intercept_"):
        value = getattr(clf, attr, None)
        if value is not None:
            total += int(np.asarray(value).size)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wave5 Has-size stacked LR head")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate", default="g2")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--lexical-prob-manifest", required=True)
    parser.add_argument("--bert-model-dir", default=None)
    parser.add_argument("--bert-input-field", default="mention_text")
    parser.add_argument("--bert-max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-split", required=True)
    parser.add_argument("--selection-source", default="phase5_like_dev")
    parser.add_argument("--ws-val-split", default="outputs/phaseA1/size/size_val_ws.jsonl")
    parser.add_argument("--ws-test-split", default="outputs/phaseA1/size/size_test_ws.jsonl")
    parser.add_argument("--phase5-test-split", default="outputs/phase5/datasets/size_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-selection-samples", type=int, default=None)
    parser.add_argument("--meta-val-fraction", type=float, default=0.4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_base = resolve_path(args.output_dir, PROJECT_ROOT / "outputs/phaseA2_planB")
    results_dir = output_base / "results"
    models_dir = output_base / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log_path = resolve_path(args.log, PROJECT_ROOT / "logs" / f"train_size_stacked_head_{args.tag}_seed{args.seed}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    selection_path = resolve_path(args.selection_split, PROJECT_ROOT / args.selection_split)
    ws_val_path = resolve_path(args.ws_val_split, PROJECT_ROOT / "outputs/phaseA1/size/size_val_ws.jsonl")
    ws_test_path = resolve_path(args.ws_test_split, PROJECT_ROOT / "outputs/phaseA1/size/size_test_ws.jsonl")
    phase5_test_path = resolve_path(args.phase5_test_split, PROJECT_ROOT / "outputs/phase5/datasets/size_test.jsonl")
    manifest_path = resolve_path(args.lexical_prob_manifest, PROJECT_ROOT / args.lexical_prob_manifest)
    manifest = load_manifest(manifest_path)
    spec = CANDIDATES[args.candidate]

    bert_model_dir = resolve_path(args.bert_model_dir, PROJECT_ROOT / "") if args.bert_model_dir else None
    if spec["uses_bert_probability"]:
        if bert_model_dir is None:
            raise ValueError(f"--bert-model-dir is required for candidate={args.candidate}")
        if not bert_model_dir.exists():
            raise FileNotFoundError(f"BERT model dir not found: {bert_model_dir}")

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] train_size_stacked_head", log_fp)
        log(
            f"[Config] candidate={args.candidate} tag={args.tag} seed={args.seed} "
            f"selection_source={args.selection_source} meta_val_fraction={args.meta_val_fraction}",
            log_fp,
        )
        log("[Progress] sklearn LogisticRegression is non-epoch training; metrics are emitted after fit.", log_fp)
        log(f"[Paths] lexical_manifest={manifest_path}", log_fp)
        log(f"[Paths] selection={selection_path}", log_fp)
        log(f"[Paths] ws_val={ws_val_path}", log_fp)
        log(f"[Paths] ws_test={ws_test_path}", log_fp)
        log(f"[Paths] phase5_test={phase5_test_path}", log_fp)
        if bert_model_dir is not None:
            log(f"[Paths] bert_model_dir={bert_model_dir}", log_fp)

        selection_rows_raw = load_jsonl(selection_path)
        selection_rows, selection_truncated = limit_rows(selection_rows_raw, args.max_selection_samples)
        ws_val_rows_raw = load_jsonl(ws_val_path)
        ws_val_rows, ws_val_truncated = limit_rows(ws_val_rows_raw, args.max_val_samples)
        ws_test_rows = load_jsonl(ws_test_path)
        phase5_test_rows = load_jsonl(phase5_test_path)

        log(
            f"[Data] selection={len(selection_rows)} ws_val={len(ws_val_rows)} "
            f"test_ws={len(ws_test_rows)} phase5_test={len(phase5_test_rows)}",
            log_fp,
        )
        log(
            f"[Truncation] selection={selection_truncated} ws_val={ws_val_truncated} "
            f"ws_test=false phase5_test=false",
            log_fp,
        )

        split_rows = {
            "selection": selection_rows,
            "ws_val": ws_val_rows,
            "ws_test": ws_test_rows,
            "phase5_test": phase5_test_rows,
        }
        lexical = {
            split: lexical_probs_from_manifest(manifest, split, len(rows))
            for split, rows in split_rows.items()
        }

        bert: dict[str, np.ndarray | None] = {split: None for split in split_rows}
        if spec["uses_bert_probability"]:
            assert bert_model_dir is not None
            for split, rows in split_rows.items():
                log(f"[BERT] Predicting probabilities for {split} n={len(rows)}", log_fp)
                bert[split] = bert_probs_for_split(
                    bert_model_dir,
                    rows,
                    input_field=args.bert_input_field,
                    max_length=args.bert_max_length,
                    batch_size=args.batch_size,
                )

        feature_matrices: dict[str, np.ndarray] = {}
        feature_columns: list[str] | None = None
        for split, rows in split_rows.items():
            x, columns = build_feature_matrix(
                rows,
                lexical_probs=lexical[split],
                bert_probs=bert[split],
                candidate=args.candidate,
            )
            feature_matrices[split] = x
            feature_columns = columns
        assert feature_columns is not None

        y_selection = labels(selection_rows)
        train_idx, val_idx, split_type = split_meta_indices(y_selection, args.meta_val_fraction, args.seed)
        x_train = feature_matrices["selection"][train_idx]
        y_train = y_selection[train_idx]
        x_threshold = feature_matrices["selection"][val_idx]
        threshold_rows = [selection_rows[int(idx)] for idx in val_idx.tolist()]

        train_start = time.perf_counter()
        scaler, clf = train_meta_lr(x_train, y_train, c=args.c, max_iter=args.max_iter, seed=args.seed)
        train_time = time.perf_counter() - train_start
        log(
            f"[TrainDone] train_time={train_time:.2f}s meta_train={len(train_idx)} "
            f"threshold_selection={len(val_idx)} split_type={split_type}",
            log_fp,
        )

        eval_start = time.perf_counter()
        threshold_probs = predict_meta_probs(scaler, clf, x_threshold)
        selected_threshold, threshold_result = tune_threshold(threshold_rows, threshold_probs)
        selection_probs = predict_meta_probs(scaler, clf, feature_matrices["selection"])
        ws_val_probs = predict_meta_probs(scaler, clf, feature_matrices["ws_val"])
        ws_test_probs = predict_meta_probs(scaler, clf, feature_matrices["ws_test"])
        phase5_probs = predict_meta_probs(scaler, clf, feature_matrices["phase5_test"])

        selection_result = evaluate_probs(selection_rows, selection_probs, selected_threshold)
        ws_val_result = evaluate_probs(ws_val_rows, ws_val_probs, selected_threshold)
        ws_test_result = evaluate_probs(ws_test_rows, ws_test_probs, selected_threshold)
        phase5_result = evaluate_probs(phase5_test_rows, phase5_probs, selected_threshold)
        eval_time = time.perf_counter() - eval_start

        model_path = models_dir / f"size_wave5_stacked_head_{args.tag}_seed{args.seed}.joblib"
        joblib.dump(
            {
                "candidate": args.candidate,
                "feature_columns": feature_columns,
                "scaler": scaler,
                "classifier": clf,
                "selected_threshold": selected_threshold,
                "selection_source": args.selection_source,
            },
            model_path,
        )

        result = {
            "method": "mws_cfe",
            "task": "size",
            "seed": args.seed,
            "gate": args.gate,
            "tag": args.tag,
            "input_field": "wave5_meta_features",
            "label_field": "has_size",
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_ws_samples": len(ws_test_rows),
            "test_phase5_samples": len(phase5_test_rows),
            "test_truncated": False,
            "test_sample_count": len(phase5_test_rows),
            "chosen_threshold": float(selected_threshold),
            "selection_split_source": args.selection_source,
            "ws_val_results": result_summary(ws_val_result),
            "ws_test_results": result_summary(ws_test_result),
            "phase5_test_results": result_summary(phase5_result),
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "best_epoch": None,
            "peak_gpu_memory_gb": 0.0,
            "model_path": str(model_path),
            "threshold_tuning": {
                "selection_split": str(selection_path),
                "selection_source": args.selection_source,
                "selection_samples": len(selection_rows),
                "selection_truncated": selection_truncated,
                "meta_train_samples": int(len(train_idx)),
                "threshold_selection_samples": int(len(val_idx)),
                "threshold_split_type": split_type,
                "test_set_used_for_threshold": False,
                "selected_threshold": float(selected_threshold),
                "tuned_val_results": result_summary(threshold_result),
                "full_selection_results": result_summary(selection_result),
            },
            "model_config": {
                "meta_classifier": "LogisticRegression",
                "classifier_params": clf.get_params(deep=False),
                "classifier_parameter_count": classifier_param_count(clf),
                "feature_columns": feature_columns,
                "scaler": "StandardScaler",
                "lexical_probability_manifest": str(manifest_path),
                "bert_model_dir": str(bert_model_dir) if bert_model_dir is not None else None,
                "bert_input_field": args.bert_input_field if spec["uses_bert_probability"] else None,
                "bert_max_length": args.bert_max_length if spec["uses_bert_probability"] else None,
            },
            "method_components": {
                "learned_model": True,
                "deterministic_rule_override": False,
                "direct_rule_label_override": False,
                "decision_layer": "wave5_stacked_logistic_regression",
                "uses_lexical_probability": bool(spec["uses_lexical_probability"]),
                "uses_bert_probability": bool(spec["uses_bert_probability"]),
                "uses_cue_features": bool(spec["uses_cue_features"]),
            },
            "wave5_protocol": {
                "candidate": args.candidate,
                "phase5_test_for_final_eval": True,
                "phase5_test_truncated": False,
                "phase5_test_used_for_threshold": False,
                "selection_split_source": args.selection_source,
                "test_sample_count": len(phase5_test_rows),
            },
            "runtime_environment": {
                "device": "cpu_meta_classifier",
                "elapsed_seconds": time.perf_counter() - t0,
            },
        }

        log(
            f"[Metric] threshold_val_f1={threshold_result.get('f1', 0.0):.4f} "
            f"phase5_f1={phase5_result.get('f1', 0.0):.4f} "
            f"threshold={selected_threshold:.6f} test_truncated=false test_n={len(phase5_test_rows)}",
            log_fp,
        )

        result_path = results_dir / f"mws_cfe_size_results_{args.tag}_seed{args.seed}.json"
        result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[Saved] result={result_path}", log_fp)
        log(f"[Saved] model={model_path}", log_fp)


if __name__ == "__main__":
    main()
