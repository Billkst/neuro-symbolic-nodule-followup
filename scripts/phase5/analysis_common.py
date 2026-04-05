from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.nodule_extractor import extract_density, extract_location, extract_size
from src.phase5.evaluation.metrics import evaluate_density, evaluate_location, evaluate_size_detection


DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified", "unclear"]
SIZE_LABELS = ["no_size", "has_size"]
LOCATION_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear", "no_location"]
TASK_TO_LABELS = {
    "density": DENSITY_LABELS,
    "size": SIZE_LABELS,
    "location": LOCATION_LABELS,
}
DEFAULT_RANDOM_SEED = 42
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

SIZE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm)\b", re.IGNORECASE)
LOCATION_PATTERNS = {
    "RUL": [r"right upper lobe", r"\brul\b"],
    "RML": [r"right middle lobe", r"\brml\b"],
    "RLL": [r"right lower lobe", r"\brll\b"],
    "LUL": [r"left upper lobe", r"\blul\b"],
    "LLL": [r"left lower lobe", r"\blll\b"],
    "lingula": [r"\blingula\b", r"lingular"],
    "bilateral": [r"\bbilateral\b", r"both lungs", r"both lower lobes", r"both upper lobes"],
}
LOCATION_REGEXES = {
    label: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for label, patterns in LOCATION_PATTERNS.items()
}
TOKEN_SUPPORT = {
    "solid": ["solid", "soft tissue", "attenuation"],
    "part_solid": ["part-solid", "part solid", "subsolid", "sub-solid"],
    "ground_glass": ["ground-glass", "ground glass", "ggo", "crazy paving"],
    "calcified": ["calcified", "calcification", "granuloma"],
}


class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = log_path.open("w", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._fp.write(message + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_hf_environment(logger: Logger | None = None) -> None:
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    cache_root = PROJECT_ROOT / "outputs" / "phase5" / "hf_cache"
    hub_cache = cache_root / "hub"
    ensure_dir(hub_cache)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    for proxy_key in ("ALL_PROXY", "all_proxy"):
        proxy_value = os.environ.get(proxy_key)
        if proxy_value and proxy_value.lower().startswith("socks"):
            os.environ.pop(proxy_key, None)
            if logger is not None:
                logger.log(f"[Network] 已移除不兼容代理 {proxy_key}={proxy_value}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def normalize_location_label(label: str | None) -> str:
    if label in LOCATION_LABELS:
        return str(label)
    if label is None:
        return "no_location"
    return "unclear"


def get_task_paths(task: str, dataset_dir: Path) -> dict[str, Path]:
    return {split: dataset_dir / f"{task}_{split}.jsonl" for split in ("train", "val", "test")}


def load_task_splits(task: str, dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    paths = get_task_paths(task, dataset_dir)
    return {split: load_jsonl(path) for split, path in paths.items()}


def get_gold_labels(task: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    if task == "density":
        return [str(row.get("density_label") or "unclear") for row in rows]
    if task == "size":
        return ["has_size" if bool(row.get("has_size")) else "no_size" for row in rows]
    if task == "location":
        return [normalize_location_label(row.get("location_label")) for row in rows]
    raise ValueError(f"未知任务: {task}")


def get_texts(rows: Sequence[dict[str, Any]], field: str = "mention_text", prepend_exam: bool = False) -> list[str]:
    texts: list[str] = []
    for row in rows:
        text = str(row.get(field) or "")
        if prepend_exam:
            exam_name = str(row.get("exam_name") or "")
            texts.append(f"[EXAM] {exam_name} [TEXT] {text}".strip())
        else:
            texts.append(text)
    return texts


def build_svm_pipeline(max_features: int = 10000, ngram_range: tuple[int, int] = (1, 2), c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", max_iter=5000, C=c_value)),
        ]
    )


def evaluate_task(task: str, y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if task == "density":
        return evaluate_density(y_true_list, y_pred_list, DENSITY_LABELS)
    if task == "location":
        return evaluate_location(y_true_list, y_pred_list, LOCATION_LABELS)
    detection_results = evaluate_size_detection([1 if label == "has_size" else 0 for label in y_true_list], [1 if label == "has_size" else 0 for label in y_pred_list])
    detection_results["macro_f1"] = float(
        f1_score(y_true_list, y_pred_list, labels=SIZE_LABELS, average="macro", zero_division=cast(Any, 0))
    )
    return detection_results


def primary_metric_name(task: str) -> str:
    return "f1" if task == "size" else "macro_f1"


def primary_metric_value(task: str, metrics: dict[str, Any]) -> float:
    return float(metrics[primary_metric_name(task)])


def run_svm_experiment(
    task: str,
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    text_field: str = "mention_text",
    prepend_exam: bool = False,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    c_value: float = 1.0,
) -> dict[str, Any]:
    train_texts = get_texts(train_rows, field=text_field, prepend_exam=prepend_exam)
    test_texts = get_texts(test_rows, field=text_field, prepend_exam=prepend_exam)
    y_train = get_gold_labels(task, train_rows)
    y_test = get_gold_labels(task, test_rows)
    pipeline = build_svm_pipeline(max_features=max_features, ngram_range=ngram_range, c_value=c_value)
    fit_start = time.perf_counter()
    pipeline.fit(train_texts, y_train)
    fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    y_pred = pipeline.predict(test_texts).tolist()
    pred_seconds = time.perf_counter() - pred_start
    metrics = evaluate_task(task, y_test, y_pred)
    return {
        "pipeline": pipeline,
        "predictions": y_pred,
        "y_true": y_test,
        "metrics": metrics,
        "fit_seconds": round(fit_seconds, 4),
        "predict_seconds": round(pred_seconds, 4),
        "config": {
            "text_field": text_field,
            "prepend_exam": prepend_exam,
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "C": c_value,
        },
    }


def predict_regex(task: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        mention_text = str(row.get("mention_text") or "")
        if task == "density":
            label, _ = extract_density(mention_text)
            predictions.append(label if label in DENSITY_LABELS else "unclear")
        elif task == "size":
            size_value, _ = extract_size(mention_text)
            predictions.append("has_size" if size_value is not None else "no_size")
        elif task == "location":
            label, _ = extract_location(mention_text)
            predictions.append(normalize_location_label(label))
        else:
            raise ValueError(f"未知任务: {task}")
    return predictions


@dataclass
class BertPrediction:
    label: str
    confidence: float
    probability_by_label: dict[str, float]


class LocalBertClassifier:
    def __init__(self, task: str, model_dir: Path, batch_size: int = 64) -> None:
        self.task = task
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.labels = TASK_TO_LABELS[task]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict_rows(self, rows: Sequence[dict[str, Any]], text_field: str = "mention_text") -> list[BertPrediction]:
        texts = [str(row.get(text_field) or "") for row in rows]
        outputs: list[BertPrediction] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                for row_probs in probabilities:
                    best_index = int(np.argmax(row_probs))
                    probability_by_label = {
                        label: round(float(row_probs[idx]), 6)
                        for idx, label in enumerate(self.labels)
                    }
                    outputs.append(
                        BertPrediction(
                            label=self.labels[best_index],
                            confidence=round(float(row_probs[best_index]), 6),
                            probability_by_label=probability_by_label,
                        )
                    )
        return outputs


def compute_distribution(values: Iterable[str]) -> dict[str, int]:
    counter = Counter(values)
    return {key: int(counter.get(key, 0)) for key in sorted(counter)}


def support_assessment(task: str, predicted_label: str, mention_text: str) -> dict[str, Any]:
    text = mention_text.lower()
    if task == "density":
        if predicted_label == "unclear":
            return {"assessment": "ambiguous", "reason": "预测为 unclear，不做文本支持判断"}
        for token in TOKEN_SUPPORT.get(predicted_label, []):
            if token in text:
                return {"assessment": "supported", "reason": f"文本含有关键词: {token}"}
        return {"assessment": "not_supported", "reason": "未发现与预测类别直接对应的密度关键词"}

    if task == "size":
        matched = SIZE_PATTERN.search(mention_text)
        if predicted_label == "has_size":
            if matched:
                return {"assessment": "supported", "reason": f"检测到尺寸表达: {matched.group(0)}"}
            return {"assessment": "not_supported", "reason": "文本中未检测到明确尺寸表达"}
        return {"assessment": "ambiguous", "reason": "预测为 no_size，不做支持性抽样"}

    if task == "location":
        if predicted_label in {"no_location", "unclear"}:
            return {"assessment": "ambiguous", "reason": "预测为 no_location/unclear，不做具体位置支持判断"}
        for pattern in LOCATION_REGEXES.get(predicted_label, []):
            matched = pattern.search(mention_text)
            if matched:
                return {"assessment": "supported", "reason": f"文本含有位置线索: {matched.group(0)}"}
        return {"assessment": "not_supported", "reason": "未发现与预测位置直接对应的关键词"}

    raise ValueError(f"未知任务: {task}")


def sample_rows(
    rows: Sequence[dict[str, Any]],
    indices: Sequence[int],
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    selected = list(indices)
    if len(selected) <= sample_size:
        return selected
    return sorted(rng.sample(selected, sample_size))


def confidence_bucket(confidence: float) -> str:
    if confidence >= 0.95:
        return "high"
    if confidence >= 0.80:
        return "mid"
    return "low"


def summarize_confidence(task: str, y_true: Sequence[str], predictions: Sequence[BertPrediction], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[int]] = defaultdict(list)
    paired: list[dict[str, Any]] = []
    for idx, (gold, pred, row) in enumerate(zip(y_true, predictions, rows, strict=False)):
        correct = gold == pred.label
        bucket = confidence_bucket(pred.confidence)
        by_bucket[bucket].append(idx)
        paired.append(
            {
                "sample_id": row.get("sample_id"),
                "mention_text": row.get("mention_text"),
                "gold_label": gold,
                "predicted_label": pred.label,
                "confidence": pred.confidence,
                "correct": correct,
                "probabilities": pred.probability_by_label,
            }
        )

    bucket_summary: dict[str, Any] = {}
    for bucket_name in ("high", "mid", "low"):
        indices = by_bucket.get(bucket_name, [])
        if not indices:
            bucket_summary[bucket_name] = {"count": 0, "accuracy": None}
            continue
        bucket_summary[bucket_name] = {
            "count": len(indices),
            "accuracy": round(
                float(
                    accuracy_score(
                        [y_true[idx] for idx in indices],
                        [predictions[idx].label for idx in indices],
                    )
                ),
                6,
            ),
            "mean_confidence": round(float(np.mean([predictions[idx].confidence for idx in indices])), 6),
        }

    least_certain = sorted(paired, key=lambda item: item["confidence"])[:20]
    incorrect = [item for item in paired if not item["correct"]]
    incorrect_sorted = sorted(incorrect, key=lambda item: item["confidence"], reverse=True)[:20]
    return {
        "task": task,
        "note": "准确率基于 silver label 计算，仅用于置信度分层讨论，不代表人工真值。",
        "bucket_summary": bucket_summary,
        "overall_accuracy": round(float(accuracy_score(y_true, [pred.label for pred in predictions])), 6),
        "least_certain_samples": least_certain,
        "high_confidence_errors": incorrect_sorted,
    }


def exam_group_key(exam_name: str, top_exam_names: set[str]) -> str:
    cleaned = exam_name.strip() if isinstance(exam_name, str) else ""
    return cleaned if cleaned in top_exam_names else "OTHER"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m{remain:.1f}s"


def top_counts(values: Iterable[str], limit: int = 10) -> list[dict[str, Any]]:
    counter = Counter(values)
    return [
        {"value": value, "count": int(count)}
        for value, count in counter.most_common(limit)
    ]


def safe_round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return round(float(value), ndigits)
