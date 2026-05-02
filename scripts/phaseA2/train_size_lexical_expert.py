#!/usr/bin/env python3
"""Train the Has-size Wave5 lexical expert.

This is a learned TF-IDF + Logistic Regression expert over mention_text. It
does not apply deterministic rule overrides. The script also exports
positive-class probabilities for downstream Wave5 stacking.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import load_jsonl, to_jsonable
from src.phase5.evaluation.metrics import evaluate_size_detection

LABELS = ["no_size", "has_size"]
POSITIVE_ID = 1


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def bool_label(row: dict[str, Any]) -> int:
    return 1 if bool(row.get("has_size")) else 0


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool_label(row) for row in rows], dtype=np.int64)


def texts(rows: list[dict[str, Any]], input_field: str) -> list[str]:
    return [str(row.get(input_field) or row.get("mention_text") or "") for row in rows]


def limit_rows(rows: list[dict[str, Any]], n: int | None) -> tuple[list[dict[str, Any]], bool]:
    if n is None or len(rows) <= n:
        return rows, False
    return rows[:n], True


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    min_df=args.min_df,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )


def sample_weights(rows: list[dict[str, Any]], enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    return np.asarray([float(row.get("ws_confidence", 1.0) or 0.0) for row in rows], dtype=np.float64)


def classifier_param_count(pipeline: Pipeline) -> int:
    clf = pipeline.named_steps["clf"]
    total = 0
    for attr in ("coef_", "intercept_"):
        value = getattr(clf, attr, None)
        if value is not None:
            total += int(np.asarray(value).size)
    return total


def positive_probs(pipeline: Pipeline, rows: list[dict[str, Any]], input_field: str) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    classes = list(getattr(clf, "classes_", []))
    if POSITIVE_ID not in classes:
        return np.zeros(len(rows), dtype=np.float64)
    pos_idx = classes.index(POSITIVE_ID)
    return np.asarray(pipeline.predict_proba(texts(rows, input_field))[:, pos_idx], dtype=np.float64)


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


def row_key(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "row_index": idx,
        "sample_id": row.get("sample_id"),
        "note_id": row.get("note_id"),
        "mention_id": row.get("mention_id"),
        "mention_text": row.get("mention_text"),
    }


def write_probability_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    *,
    split: str,
    source_path: Path,
    truncated: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for idx, (row, prob) in enumerate(zip(rows, probs, strict=False)):
            payload = row_key(row, idx)
            payload.update(
                {
                    "split": split,
                    "source_path": str(source_path),
                    "has_size": bool(row.get("has_size")),
                    "label": int(bool_label(row)),
                    "prob_has_size": float(prob),
                    "truncated": bool(truncated),
                }
            )
            fp.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
            fp.flush()
    return {
        "split": split,
        "path": str(path),
        "source_path": str(source_path),
        "sample_count": len(rows),
        "truncated": bool(truncated),
    }


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


def resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wave5 Has-size TF-IDF/LR lexical expert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate", default="g2")
    parser.add_argument("--ws-data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--selection-split", default=None)
    parser.add_argument("--selection-source", default="ws_val")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--probability-dir", default=None)
    parser.add_argument("--tag", default="size_wave5_lexical_alone")
    parser.add_argument("--input-field", default="mention_text")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-selection-samples", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--use-confidence-weight", action="store_true")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    ws_data_dir = resolve_path(args.ws_data_dir, PROJECT_ROOT / "outputs/phaseA1/size")
    phase5_data_dir = resolve_path(args.phase5_data_dir, PROJECT_ROOT / "outputs/phase5/datasets")
    output_base = resolve_path(args.output_dir, PROJECT_ROOT / "outputs/phaseA2_planB")
    results_dir = output_base / "results"
    models_dir = output_base / "models"
    probability_dir = resolve_path(
        args.probability_dir,
        output_base / "size_wave5" / "probabilities",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    log_path = resolve_path(args.log, PROJECT_ROOT / "logs" / f"train_size_lexical_expert_{args.tag}_seed{args.seed}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = ws_data_dir / f"size_train_ws_{args.gate}.jsonl"
    ws_val_path = ws_data_dir / "size_val_ws.jsonl"
    ws_test_path = ws_data_dir / "size_test_ws.jsonl"
    phase5_test_path = phase5_data_dir / "size_test.jsonl"
    selection_path = resolve_path(args.selection_split, ws_val_path) if args.selection_split else ws_val_path

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] train_size_lexical_expert", log_fp)
        log(
            f"[Config] tag={args.tag} seed={args.seed} gate={args.gate} input_field={args.input_field} "
            f"model=tfidf_lr c={args.c} max_features={args.max_features}",
            log_fp,
        )
        log("[Progress] sklearn LogisticRegression is non-epoch training; metrics are emitted after fit.", log_fp)
        log(f"[Paths] train={train_path}", log_fp)
        log(f"[Paths] ws_val={ws_val_path}", log_fp)
        log(f"[Paths] ws_test={ws_test_path}", log_fp)
        log(f"[Paths] phase5_test={phase5_test_path}", log_fp)
        log(f"[Paths] selection={selection_path} source={args.selection_source}", log_fp)

        train_rows_raw = load_jsonl(train_path)
        ws_val_rows_raw = load_jsonl(ws_val_path)
        ws_test_rows = load_jsonl(ws_test_path)
        phase5_test_rows = load_jsonl(phase5_test_path)
        selection_rows_raw = load_jsonl(selection_path)

        train_rows, train_truncated = limit_rows(train_rows_raw, args.max_train_samples)
        ws_val_rows, ws_val_truncated = limit_rows(ws_val_rows_raw, args.max_val_samples)
        selection_rows, selection_truncated = limit_rows(selection_rows_raw, args.max_selection_samples)

        y_train = labels(train_rows)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("Training split must contain both has_size classes")

        log(
            f"[Data] train={len(train_rows)} val={len(ws_val_rows)} selection={len(selection_rows)} "
            f"test_ws={len(ws_test_rows)} phase5_test={len(phase5_test_rows)}",
            log_fp,
        )
        log(
            f"[Truncation] train={train_truncated} val={ws_val_truncated} "
            f"selection={selection_truncated} ws_test=false phase5_test=false",
            log_fp,
        )

        pipeline = build_pipeline(args)
        fit_kwargs: dict[str, Any] = {}
        weights = sample_weights(train_rows, args.use_confidence_weight)
        if weights is not None:
            fit_kwargs["clf__sample_weight"] = weights

        train_start = time.perf_counter()
        pipeline.fit(texts(train_rows, args.input_field), y_train, **fit_kwargs)
        train_time = time.perf_counter() - train_start
        log(f"[TrainDone] train_time={train_time:.2f}s sample_weight_used={bool(fit_kwargs)}", log_fp)

        eval_start = time.perf_counter()
        split_rows = {
            "ws_val": ws_val_rows,
            "selection": selection_rows,
            "ws_test": ws_test_rows,
            "phase5_test": phase5_test_rows,
        }
        split_paths = {
            "ws_val": ws_val_path,
            "selection": selection_path,
            "ws_test": ws_test_path,
            "phase5_test": phase5_test_path,
        }
        split_truncated = {
            "ws_val": ws_val_truncated,
            "selection": selection_truncated,
            "ws_test": False,
            "phase5_test": False,
        }
        split_probs = {
            split: positive_probs(pipeline, rows, args.input_field)
            for split, rows in split_rows.items()
        }

        selected_threshold, tuned_selection = tune_threshold(selection_rows, split_probs["selection"])
        ws_val_results = evaluate_probs(ws_val_rows, split_probs["ws_val"], selected_threshold)
        ws_test_results = evaluate_probs(ws_test_rows, split_probs["ws_test"], selected_threshold)
        phase5_test_results = evaluate_probs(phase5_test_rows, split_probs["phase5_test"], selected_threshold)
        eval_time = time.perf_counter() - eval_start

        probability_outputs: dict[str, Any] = {}
        for split, probs in split_probs.items():
            prob_path = probability_dir / f"{args.tag}_seed{args.seed}_{split}_probs.jsonl"
            probability_outputs[split] = write_probability_jsonl(
                prob_path,
                split_rows[split],
                probs,
                split=split,
                source_path=split_paths[split],
                truncated=split_truncated[split],
            )

        manifest = {
            "tag": args.tag,
            "seed": args.seed,
            "input_field": args.input_field,
            "probability_outputs": probability_outputs,
            "test_truncated": False,
            "test_sample_count": len(phase5_test_rows),
        }
        manifest_path = probability_dir / f"{args.tag}_seed{args.seed}_manifest.json"
        manifest_path.write_text(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

        model_path = models_dir / f"size_wave5_lexical_expert_{args.tag}_seed{args.seed}.joblib"
        joblib.dump(pipeline, model_path)

        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]
        result = {
            "method": "mws_cfe",
            "task": "size",
            "seed": args.seed,
            "gate": args.gate,
            "tag": args.tag,
            "input_field": args.input_field,
            "label_field": "has_size",
            "train_samples": len(train_rows),
            "val_samples": len(selection_rows),
            "test_ws_samples": len(ws_test_rows),
            "test_phase5_samples": len(phase5_test_rows),
            "test_truncated": False,
            "test_sample_count": len(phase5_test_rows),
            "chosen_threshold": float(selected_threshold),
            "selection_split_source": args.selection_source,
            "ws_val_results": result_summary(ws_val_results),
            "ws_test_results": result_summary(ws_test_results),
            "phase5_test_results": result_summary(phase5_test_results),
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "best_epoch": None,
            "peak_gpu_memory_gb": 0.0,
            "model_path": str(model_path),
            "probability_manifest": str(manifest_path),
            "probability_outputs": probability_outputs,
            "threshold_tuning": {
                "selection_split": str(selection_path),
                "selection_source": args.selection_source,
                "selection_samples": len(selection_rows),
                "selection_truncated": selection_truncated,
                "test_set_used_for_threshold": False,
                "selected_threshold": float(selected_threshold),
                "tuned_val_results": result_summary(tuned_selection),
            },
            "model_config": {
                "vectorizer": {
                    "class": "TfidfVectorizer",
                    "max_features": vectorizer.max_features,
                    "ngram_range": list(vectorizer.ngram_range),
                    "min_df": vectorizer.min_df,
                    "sublinear_tf": vectorizer.sublinear_tf,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                },
                "classifier": classifier.__class__.__name__,
                "classifier_params": classifier.get_params(deep=False),
                "classifier_parameter_count": classifier_param_count(pipeline),
                "sample_weight_used": bool(fit_kwargs),
            },
            "method_components": {
                "learned_model": True,
                "deterministic_rule_override": False,
                "direct_rule_label_override": False,
                "decision_layer": "tfidf_lr_probability_threshold_selected_on_non_test_split",
                "uses_lexical_probability": True,
                "uses_bert_probability": False,
                "uses_cue_features": False,
            },
            "wave5_protocol": {
                "candidate": "lexical_expert_alone",
                "phase5_test_for_final_eval": True,
                "phase5_test_truncated": False,
                "phase5_test_used_for_threshold": False,
                "selection_split_source": args.selection_source,
                "probability_manifest": str(manifest_path),
            },
            "runtime_environment": {
                "device": "cpu",
                "elapsed_seconds": time.perf_counter() - t0,
            },
        }

        log(
            f"[Metric] selection_f1={tuned_selection.get('f1', 0.0):.4f} "
            f"phase5_f1={phase5_test_results.get('f1', 0.0):.4f} "
            f"threshold={selected_threshold:.6f} test_truncated=false test_n={len(phase5_test_rows)}",
            log_fp,
        )

        result_path = results_dir / f"mws_cfe_size_results_{args.tag}_seed{args.seed}.json"
        result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[Saved] result={result_path}", log_fp)
        log(f"[Saved] model={model_path}", log_fp)
        log(f"[Saved] probability_manifest={manifest_path}", log_fp)


if __name__ == "__main__":
    main()
