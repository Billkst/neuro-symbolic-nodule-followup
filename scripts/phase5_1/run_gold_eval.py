from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.nodule_extractor import extract_density, extract_location, extract_size
from src.phase5_1.evaluation.gold_metrics import (
    build_error_cases,
    compute_silver_vs_gold_agreement,
    evaluate_density_gold,
    evaluate_has_size_gold,
    evaluate_location_gold,
    evaluate_size_regression_gold,
)


DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified", "unclear"]
LOCATION_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear", "no_location"]
METHODS = ["silver", "regex", "ml_lr", "ml_svm", "pubmedbert"]
TASKS = ["density", "has_size", "size_mm", "location"]
TASK_TO_PHASE5_DATASET = {
    "density": "density",
    "has_size": "size",
    "location": "location",
}
PUBMEDBERT_TASK_LABELS: dict[str, list[Any]] = {
    "density": DENSITY_LABELS,
    "size": [0, 1],
    "location": LOCATION_LABELS,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: path={path} line={line_number}") from exc
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_density_label(label: str | None) -> str:
    if label in DENSITY_LABELS:
        return str(label)
    return "unclear"


def normalize_location_label(label: str | None) -> str:
    if label in LOCATION_LABELS:
        return str(label)
    if label is None:
        return "no_location"
    text = str(label).strip()
    if not text:
        return "no_location"
    lowered = text.lower()
    mapping = {
        "rul": "RUL",
        "rml": "RML",
        "rll": "RLL",
        "lul": "LUL",
        "lll": "LLL",
        "lingula": "lingula",
        "bilateral": "bilateral",
        "unclear": "unclear",
        "no_location": "no_location",
        "none": "no_location",
    }
    if lowered in mapping:
        return mapping[lowered]
    return "unclear"


def ensure_pubmedbert_environment() -> None:
    cache_root = PROJECT_ROOT / "outputs" / "phase5" / "hf_cache"
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    for proxy_key in ("ALL_PROXY", "all_proxy"):
        proxy_value = os.environ.get(proxy_key)
        if proxy_value and proxy_value.lower().startswith("socks"):
            os.environ.pop(proxy_key, None)


def predict_silver_density(rows: list[dict[str, Any]]) -> list[str]:
    return [normalize_density_label(row.get("silver_density_category")) for row in rows]


def predict_silver_has_size(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("silver_has_size") else 0 for row in rows]


def predict_silver_size_mm(rows: list[dict[str, Any]]) -> list[float | None]:
    predictions: list[float | None] = []
    for row in rows:
        value = row.get("silver_size_mm")
        predictions.append(float(value) if value is not None else None)
    return predictions


def predict_silver_location(rows: list[dict[str, Any]]) -> list[str]:
    return [normalize_location_label(row.get("silver_location_lobe")) for row in rows]


def predict_regex_density(rows: list[dict[str, Any]]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        label, _ = extract_density(str(row.get("mention_text") or ""))
        predictions.append(normalize_density_label(label))
    return predictions


def predict_regex_has_size(rows: list[dict[str, Any]]) -> list[int]:
    predictions: list[int] = []
    for row in rows:
        size_value, _ = extract_size(str(row.get("mention_text") or ""))
        predictions.append(1 if size_value is not None else 0)
    return predictions


def predict_regex_size_mm(rows: list[dict[str, Any]]) -> list[float | None]:
    predictions: list[float | None] = []
    for row in rows:
        size_value, _ = extract_size(str(row.get("mention_text") or ""))
        predictions.append(float(size_value) if size_value is not None else None)
    return predictions


def predict_regex_location(rows: list[dict[str, Any]]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        label, _ = extract_location(str(row.get("mention_text") or ""))
        predictions.append(normalize_location_label(label))
    return predictions


def train_ml_pipeline(method: str, task: str, train_texts: list[str], train_labels: Sequence[Any]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    if method == "ml_lr":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    elif method == "ml_svm":
        classifier = LinearSVC(class_weight="balanced", max_iter=5000, C=1.0)
    else:
        raise ValueError(f"不支持的 ML 方法: {method}")

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", classifier),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    return pipeline


def predict_ml_density(pipeline, rows: list[dict[str, Any]]) -> list[str]:
    texts = [str(row.get("mention_text") or "") for row in rows]
    return [normalize_density_label(value) for value in pipeline.predict(texts).tolist()]


def predict_ml_has_size(pipeline, rows: list[dict[str, Any]]) -> list[int]:
    texts = [str(row.get("mention_text") or "") for row in rows]
    return [int(value) for value in pipeline.predict(texts).tolist()]


def predict_ml_location(pipeline, rows: list[dict[str, Any]]) -> list[str]:
    texts = [str(row.get("mention_text") or "") for row in rows]
    return [normalize_location_label(value) for value in pipeline.predict(texts).tolist()]


def load_pubmedbert_model(task: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_pubmedbert_environment()
    model_dir = PROJECT_ROOT / "outputs" / "phase5" / "models" / f"{task}_pubmedbert"
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到 PubMedBERT 模型目录: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_pubmedbert(model, tokenizer, rows: list[dict[str, Any]], label_names: Sequence[Any], max_length: int = 128, batch_size: int = 64) -> list[Any]:
    import torch

    texts = [str(row.get("mention_text") or "") for row in rows]
    predictions: list[Any] = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for pred_id in pred_ids:
            predictions.append(label_names[int(pred_id)])
    return predictions


def get_gold_density(rows: list[dict[str, Any]]) -> list[str]:
    return [normalize_density_label(row.get("gold_density_category")) for row in rows]


def get_gold_has_size(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("gold_has_size") else 0 for row in rows]


def get_gold_size_mm(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("gold_has_size") and row.get("gold_size_mm") is not None:
            values.append(float(row["gold_size_mm"]))
    return values


def get_gold_location(rows: list[dict[str, Any]]) -> list[str]:
    return [normalize_location_label(row.get("gold_location_lobe")) for row in rows]


def evaluate_method_on_task(method: str, task: str, predictions: list[Any], gold_labels: list[Any], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if task == "density":
        return evaluate_density_gold(gold_labels, predictions, DENSITY_LABELS)
    if task == "has_size":
        return evaluate_has_size_gold(gold_labels, predictions)
    if task == "size_mm":
        filtered_gold: list[float] = []
        filtered_pred: list[float] = []
        for row, pred in zip(manifest_rows, predictions, strict=False):
            if not row.get("gold_has_size") or row.get("gold_size_mm") is None:
                continue
            filtered_gold.append(float(row["gold_size_mm"]))
            filtered_pred.append(float(pred) if pred is not None else 0.0)
        return evaluate_size_regression_gold(filtered_gold, filtered_pred, tolerance=1.0)
    if task == "location":
        return evaluate_location_gold(gold_labels, predictions, LOCATION_LABELS)
    raise ValueError(f"未知任务: {task}")


def get_primary_metric(task: str, metrics: dict[str, Any]) -> tuple[str, float]:
    if task in {"density", "location"}:
        return "macro_f1", float(metrics["macro_f1"])
    if task == "has_size":
        return "f1", float(metrics["f1"])
    return "within_tolerance_rate", float(metrics["within_tolerance_rate"])


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def format_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def confusion_matrix_markdown(confusion: dict[str, Any]) -> str:
    labels = [str(label) for label in confusion.get("labels", [])]
    headers = ["gold\\pred", *labels]
    rows: list[list[Any]] = []
    row_map = confusion.get("rows", {})
    for gold_label in labels:
        row_values = row_map.get(gold_label, {})
        rows.append([gold_label, *[row_values.get(pred_label, 0) for pred_label in labels]])
    return markdown_table(headers, rows)


def summarize_disagreement_patterns(agreement: dict[str, Any]) -> list[tuple[str, int]]:
    pattern_counter: Counter[str] = Counter()
    for item in agreement.get("disagreements", []):
        pattern = f"silver={item.get('silver')} -> gold={item.get('gold')}"
        pattern_counter[pattern] += 1
    return pattern_counter.most_common(5)


def build_size_error_cases(rows: list[dict[str, Any]], predictions: list[float | None]) -> list[dict[str, Any]]:
    error_cases: list[dict[str, Any]] = []
    for row, pred in zip(rows, predictions, strict=False):
        if not row.get("gold_has_size") or row.get("gold_size_mm") is None:
            continue
        gold_value = float(row["gold_size_mm"])
        pred_value = float(pred) if pred is not None else 0.0
        if pred_value == gold_value:
            continue
        error_cases.append(
            {
                "sample_id": row.get("sample_id"),
                "mention_text": row.get("mention_text"),
                "gold": gold_value,
                "predicted": pred_value,
                "silver": row.get("silver_size_mm"),
                "annotation_confidence": row.get("annotation_confidence"),
                "absolute_error": abs(pred_value - gold_value),
            }
        )
    error_cases.sort(key=lambda item: float(item.get("absolute_error", 0.0)), reverse=True)
    return error_cases


def build_task_error_cases(task: str, rows: list[dict[str, Any]], predictions: list[Any]) -> list[dict[str, Any]]:
    if task == "density":
        return build_error_cases(rows, predictions, "gold_density_category")
    if task == "has_size":
        wrapped_predictions = [{"pred": int(value)} for value in predictions]
        normalized_rows = []
        for row in rows:
            copied = dict(row)
            copied["gold_has_size"] = 1 if row.get("gold_has_size") else 0
            copied["silver_gold_has_size"] = 1 if row.get("silver_has_size") else 0
            normalized_rows.append(copied)
        return build_error_cases(normalized_rows, wrapped_predictions, "gold_has_size", pred_field="pred")
    if task == "size_mm":
        return build_size_error_cases(rows, predictions)
    if task == "location":
        return build_error_cases(rows, predictions, "gold_location_lobe")
    raise ValueError(f"未知任务: {task}")


def build_prediction_records(rows: list[dict[str, Any]], task_predictions_by_method: dict[str, dict[str, list[Any]]]) -> dict[str, list[dict[str, Any]]]:
    records_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in task_predictions_by_method}
    for row_index, row in enumerate(rows):
        gold_payload = {
            "density": normalize_density_label(row.get("gold_density_category")),
            "has_size": 1 if row.get("gold_has_size") else 0,
            "size_mm": float(row["gold_size_mm"]) if row.get("gold_size_mm") is not None else None,
            "location": normalize_location_label(row.get("gold_location_lobe")),
        }
        silver_payload = {
            "density": normalize_density_label(row.get("silver_density_category")),
            "has_size": 1 if row.get("silver_has_size") else 0,
            "size_mm": float(row["silver_size_mm"]) if row.get("silver_size_mm") is not None else None,
            "location": normalize_location_label(row.get("silver_location_lobe")),
        }
        for method, task_predictions in task_predictions_by_method.items():
            predictions_payload = {
                task: task_predictions[task][row_index] for task in TASKS
            }
            records_by_method[method].append(
                {
                    "sample_id": row.get("sample_id"),
                    "subject_id": row.get("subject_id"),
                    "note_id": row.get("note_id"),
                    "mention_text": row.get("mention_text"),
                    "annotation_confidence": row.get("annotation_confidence"),
                    "predictions": predictions_payload,
                    "gold": gold_payload,
                    "silver": silver_payload,
                }
            )
    return records_by_method


def write_predictions_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def write_summary_csv(path: Path, all_results: dict[str, dict[str, dict[str, Any]]], methods: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task",
        "primary_metric_name",
        "primary_metric_value",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision",
        "recall",
        "f1",
        "mae",
        "median_absolute_error",
        "exact_match_rate",
        "within_tolerance_rate",
        "sample_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            for task in TASKS:
                metrics = all_results[method][task]
                primary_name, primary_value = get_primary_metric(task, metrics)
                writer.writerow(
                    {
                        "method": method,
                        "task": task,
                        "primary_metric_name": primary_name,
                        "primary_metric_value": format_metric(primary_value),
                        "accuracy": format_metric(metrics.get("accuracy", "")),
                        "macro_f1": format_metric(metrics.get("macro_f1", "")),
                        "weighted_f1": format_metric(metrics.get("weighted_f1", "")),
                        "precision": format_metric(metrics.get("precision", "")),
                        "recall": format_metric(metrics.get("recall", "")),
                        "f1": format_metric(metrics.get("f1", "")),
                        "mae": format_metric(metrics.get("mae", "")),
                        "median_absolute_error": format_metric(metrics.get("median_absolute_error", "")),
                        "exact_match_rate": format_metric(metrics.get("exact_match_rate", "")),
                        "within_tolerance_rate": format_metric(metrics.get("within_tolerance_rate", "")),
                        "sample_count": metrics.get("sample_count", ""),
                    }
                )


def pick_best_method(all_results: dict[str, dict[str, dict[str, Any]]], task: str, methods: Sequence[str]) -> str:
    scored: list[tuple[float, str]] = []
    for method in methods:
        _, value = get_primary_metric(task, all_results[method][task])
        if task == "size_mm":
            score = value
        else:
            score = value
        scored.append((score, method))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def render_task_summary_table(all_results: dict[str, dict[str, dict[str, Any]]], task: str, methods: Sequence[str]) -> str:
    headers_map = {
        "density": ["method", "accuracy", "macro_f1", "weighted_f1", "sample_count"],
        "has_size": ["method", "accuracy", "precision", "recall", "f1", "sample_count"],
        "size_mm": ["method", "mae", "median_absolute_error", "exact_match_rate", "within_tolerance_rate", "sample_count"],
        "location": ["method", "accuracy", "macro_f1", "weighted_f1", "sample_count"],
    }
    headers = headers_map[task]
    rows: list[list[Any]] = []
    for method in methods:
        metrics = all_results[method][task]
        rows.append([method, *[format_metric(metrics.get(header, "")) for header in headers[1:]]])
    return markdown_table(headers, rows)


def render_density_per_class_table(metrics: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    per_class_precision = metrics.get("per_class_precision", {})
    per_class_recall = metrics.get("per_class_recall", {})
    per_class_f1 = metrics.get("per_class_f1", {})
    for label in DENSITY_LABELS:
        rows.append(
            [
                label,
                format_metric(per_class_precision.get(label, 0.0)),
                format_metric(per_class_recall.get(label, 0.0)),
                format_metric(per_class_f1.get(label, 0.0)),
            ]
        )
    return markdown_table(["label", "precision", "recall", "f1"], rows)


def render_location_per_class_table(metrics: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    per_class_f1 = metrics.get("per_class_f1", {})
    for label in LOCATION_LABELS:
        rows.append([label, format_metric(per_class_f1.get(label, 0.0))])
    return markdown_table(["label", "f1"], rows)


def build_failure_mode_summary(error_analysis: dict[str, dict[str, list[dict[str, Any]]]]) -> list[tuple[str, int, dict[str, Any] | None]]:
    counter: Counter[str] = Counter()
    example_by_mode: dict[str, dict[str, Any]] = {}
    for task, method_map in error_analysis.items():
        for method, cases in method_map.items():
            for case in cases:
                mode = f"{task}: gold={case.get('gold')} pred={case.get('predicted')}"
                counter[mode] += 1
                if mode not in example_by_mode:
                    example_by_mode[mode] = {
                        "method": method,
                        "sample_id": case.get("sample_id"),
                        "mention_text": str(case.get("mention_text") or "")[:160],
                    }
    ranked = []
    for mode, count in counter.most_common(5):
        ranked.append((mode, count, example_by_mode.get(mode)))
    return ranked


def generate_report(all_results: dict[str, dict[str, dict[str, Any]]], manifest_rows: list[dict[str, Any]], output_dir: Path, error_analysis: dict[str, dict[str, list[dict[str, Any]]]], agreement_stats: dict[str, Any], methods: Sequence[str]) -> Path:
    report_path = PROJECT_ROOT / "reports" / "phase5_1_gold_eval.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    density_best_method = pick_best_method(all_results, "density", methods)
    location_best_method = pick_best_method(all_results, "location", methods)
    density_best_metrics = all_results[density_best_method]["density"]
    location_best_metrics = all_results[location_best_method]["location"]
    size_positive_count = sum(1 for row in manifest_rows if row.get("gold_has_size") and row.get("gold_size_mm") is not None)
    failure_modes = build_failure_mode_summary(error_analysis)

    agreement_bullets = []
    for field_name, stats in agreement_stats.items():
        patterns = summarize_disagreement_patterns(stats)
        pattern_text = "；".join([f"{name} ({count})" for name, count in patterns[:3]]) if patterns else "无明显分歧模式"
        agreement_bullets.append(
            f"- `{field_name}` 一致率 {stats['agreement_rate']:.4f}，分歧 {stats['disagreement_count']}/{stats['total']}；主要模式：{pattern_text}"
        )

    error_breakdown_lines = []
    for task in TASKS:
        method_counts = ", ".join(
            f"{method}={len(error_analysis[task].get(method, []))}"
            for method in methods
        )
        error_breakdown_lines.append(f"- `{task}`: {method_counts}")

    failure_mode_lines = []
    for mode, count, example in failure_modes:
        if example is None:
            failure_mode_lines.append(f"- {mode}：{count} 次")
            continue
        failure_mode_lines.append(
            f"- {mode}：{count} 次；例如 `{example['sample_id']}`（{example['method']}）— {example['mention_text']}"
        )

    lines = [
        "# Phase 5.1: 人工 Gold 小样本评测报告",
        "",
        "## 1. 背景与动机",
        "- Silver 评测会出现天花板效应，因为 `regex` 产生的 silver 标签本身就来自规则抽取，导致在 silver 上 `regex=1.0` 并不代表真实泛化能力。",
        "- Gold 评测的目标是用人工校正后的 62 条样本重新估计方法在真实任务定义上的有效性，并区分模型问题与标注/定义问题。",
        "",
        "## 2. 数据来源与构造",
        "- Source: `density_test.jsonl`，最初抽取 80 个候选，导出后保留 75 条，最终筛出 62 条 pulmonary targets。",
        "- 采样方式为分层抽样（A/B/C/D strata），保证不同样本类型都有覆盖。",
        "- Gold 文件来源：`data/gold_eval_candidates_v1_final_usable_gold.csv`。",
        "- Exclusion: 共移除 13 条非肺部目标 mention，因此最终 gold manifest 样本数为 62。",
        "",
        "## 3. 评测协议",
        "- 对齐方式：按 `sample_id` 对齐 gold 与预测结果。",
        "- 评测方法：`silver`、`regex`、`ml_lr`、`ml_svm`、`pubmedbert`。",
        "- 评测任务：`density_category`、`has_size`、`size_mm`、`location_lobe`。",
        "- 指标设置：`density_category` 与 `location_lobe` 使用 accuracy / macro-F1 / weighted-F1 / per-class P-R-F1 / confusion matrix；`has_size` 使用 accuracy / precision / recall / F1 / confusion matrix；`size_mm` 使用 MAE / median absolute error / exact match rate / within 1.0 mm tolerance rate。",
        "",
        "## 4. 结果",
        "### 4.1 density_category",
        render_task_summary_table(all_results, "density", methods),
        "",
        f"最佳方法：`{density_best_method}`",
        "",
        render_density_per_class_table(density_best_metrics),
        "",
        confusion_matrix_markdown(density_best_metrics["confusion_matrix"]),
        "",
        "### 4.2 has_size",
        render_task_summary_table(all_results, "has_size", methods),
        "",
        "### 4.3 size_mm",
        render_task_summary_table(all_results, "size_mm", methods),
        "",
        f"- 回归仅在 `gold_has_size=yes` 且 `gold_size_mm` 非空样本上评测，本次样本数为 `{size_positive_count}`。",
        "",
        "### 4.4 location_lobe",
        render_task_summary_table(all_results, "location", methods),
        "",
        f"最佳方法：`{location_best_method}`",
        "",
        render_location_per_class_table(location_best_metrics),
        "",
        "### 4.5 Silver vs Gold 一致性",
        *agreement_bullets,
        "- 关键分歧通常集中在密度混合表述（如 `solid/part_solid/ground_glass` 重叠）、尺寸是否算作当前 mention 的归属，以及 location 是否需要精确到叶级。",
        "",
        "## 5. 错误分析",
        *failure_mode_lines,
        "",
        *error_breakdown_lines,
        "",
        "## 6. 结论与论文建议",
        "- 若 gold 上 `density_category` 与 `location_lobe` 仍有明显性能差距，说明这两个字段存在真实建模价值，而不仅是 silver 对规则的复现。",
        "- 若 `size_mm` 的主要误差来自 mention 边界与尺寸归属，而非纯提取失败，则问题更多来自数据定义与样本构造，而不完全是模型能力不足。",
        "- `has_size` 若在 gold 上接近饱和，可作为高稳定性字段；反之则说明报告表述中存在大量隐式尺寸线索，值得进一步建模。",
        "- 论文可明确写出：silver 评测高分并不等价于真实临床抽取质量，gold 小样本评测揭示了字段难度的真实排序。",
        "- 论文可进一步写出：真正值得投入 Phase 5.2 的方向，应优先放在 gold 上仍存在显著误差且具有决策价值的字段。",
        "- 是否需要 Phase 5.2：若 PubMedBERT 与传统基线在 gold 上仍有系统性错误，且错误集中于可学习模式而非标注歧义，则需要；否则应优先回到任务定义与数据协议修订。",
        "",
        "## 7. 统计局限性声明",
        f"- 本次 gold 评测样本量仅 `N={len(manifest_rows)}`，适合作为方向性验证，不适合作为高精度总体性能估计。",
        "- 小样本下单个样本会显著影响 macro-F1 与 per-class 指标，论文中应同时报告置信区间说明或至少明确这一不确定性。",
        "",
    ]

    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return report_path


def load_phase5_train_rows(phase5_dir: Path, task: str) -> list[dict[str, Any]]:
    dataset_task = TASK_TO_PHASE5_DATASET[task]
    path = phase5_dir / "datasets" / f"{dataset_task}_train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少训练集文件: {path}")
    return load_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.1 gold evaluation one-click runner")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "outputs" / "phase5_1" / "gold_eval_manifest.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "phase5_1"))
    parser.add_argument("--phase5-dir", default=str(PROJECT_ROOT / "outputs" / "phase5"))
    parser.add_argument("--skip-pubmedbert", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    phase5_dir = Path(args.phase5_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    log(f"[Start] manifest={manifest_path}")
    log(f"[Output] output_dir={output_dir}")
    log(f"[Phase5] phase5_dir={phase5_dir}")

    manifest_rows = load_jsonl(manifest_path)
    log(f"[Data] gold_manifest_rows={len(manifest_rows)}")
    if len(manifest_rows) != 62:
        log(f"[Warn] 预期 gold 样本数为 62，当前为 {len(manifest_rows)}")

    evaluated_methods = [method for method in METHODS if not (method == "pubmedbert" and args.skip_pubmedbert)]

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    error_analysis: dict[str, dict[str, list[dict[str, Any]]]] = {task: {} for task in TASKS}
    task_predictions_by_method: dict[str, dict[str, list[Any]]] = {}

    gold_by_task = {
        "density": get_gold_density(manifest_rows),
        "has_size": get_gold_has_size(manifest_rows),
        "size_mm": get_gold_size_mm(manifest_rows),
        "location": get_gold_location(manifest_rows),
    }

    ml_pipelines: dict[tuple[str, str], Any] = {}
    for method in ("ml_lr", "ml_svm"):
        for task in ("density", "has_size", "location"):
            train_rows = load_phase5_train_rows(phase5_dir, task)
            train_texts = [str(row.get("mention_text") or "") for row in train_rows]
            if task == "density":
                train_labels = [normalize_density_label(row.get("density_label")) for row in train_rows]
            elif task == "has_size":
                train_labels = [1 if row.get("has_size") else 0 for row in train_rows]
            else:
                train_labels = [normalize_location_label(row.get("location_label")) for row in train_rows]
            log(f"[Train] method={method} task={task} train_rows={len(train_rows)}")
            fit_start = time.perf_counter()
            ml_pipelines[(method, task)] = train_ml_pipeline(method, task, train_texts, train_labels)
            log(f"[TrainDone] method={method} task={task} seconds={time.perf_counter() - fit_start:.2f}")

    pubmedbert_models: dict[str, tuple[Any, Any]] = {}
    if not args.skip_pubmedbert:
        for task in ("density", "size", "location"):
            log(f"[LoadModel] task={task}")
            load_start = time.perf_counter()
            pubmedbert_models[task] = load_pubmedbert_model(task)
            log(f"[LoadModelDone] task={task} seconds={time.perf_counter() - load_start:.2f}")

    for method in evaluated_methods:

        log(f"[Method] {method}")
        method_results: dict[str, dict[str, Any]] = {}
        method_predictions: dict[str, list[Any]] = {}

        if method == "silver":
            method_predictions["density"] = predict_silver_density(manifest_rows)
            method_predictions["has_size"] = predict_silver_has_size(manifest_rows)
            method_predictions["size_mm"] = predict_silver_size_mm(manifest_rows)
            method_predictions["location"] = predict_silver_location(manifest_rows)
        elif method == "regex":
            method_predictions["density"] = predict_regex_density(manifest_rows)
            method_predictions["has_size"] = predict_regex_has_size(manifest_rows)
            method_predictions["size_mm"] = predict_regex_size_mm(manifest_rows)
            method_predictions["location"] = predict_regex_location(manifest_rows)
        elif method in {"ml_lr", "ml_svm"}:
            method_predictions["density"] = predict_ml_density(ml_pipelines[(method, "density")], manifest_rows)
            method_predictions["has_size"] = predict_ml_has_size(ml_pipelines[(method, "has_size")], manifest_rows)
            method_predictions["size_mm"] = predict_regex_size_mm(manifest_rows)
            method_predictions["location"] = predict_ml_location(ml_pipelines[(method, "location")], manifest_rows)
        elif method == "pubmedbert":
            density_model, density_tokenizer = pubmedbert_models["density"]
            size_model, size_tokenizer = pubmedbert_models["size"]
            location_model, location_tokenizer = pubmedbert_models["location"]
            method_predictions["density"] = [
                normalize_density_label(value)
                for value in predict_pubmedbert(density_model, density_tokenizer, manifest_rows, PUBMEDBERT_TASK_LABELS["density"])
            ]
            method_predictions["has_size"] = [
                int(value)
                for value in predict_pubmedbert(size_model, size_tokenizer, manifest_rows, PUBMEDBERT_TASK_LABELS["size"])
            ]
            method_predictions["size_mm"] = predict_regex_size_mm(manifest_rows)
            method_predictions["location"] = [
                normalize_location_label(value)
                for value in predict_pubmedbert(location_model, location_tokenizer, manifest_rows, PUBMEDBERT_TASK_LABELS["location"])
            ]
        else:
            raise ValueError(f"未知方法: {method}")

        for task in TASKS:
            gold_labels = gold_by_task[task]
            task_predictions = method_predictions[task]
            metrics = evaluate_method_on_task(method, task, task_predictions, gold_labels, manifest_rows)
            method_results[task] = metrics
            error_analysis[task][method] = build_task_error_cases(task, manifest_rows, task_predictions)
            primary_name, primary_value = get_primary_metric(task, metrics)
            log(
                f"[Metric] method={method} task={task} {primary_name}={primary_value:.4f} sample_count={metrics.get('sample_count', len(manifest_rows))}"
            )

        all_results[method] = method_results
        task_predictions_by_method[method] = method_predictions

    prediction_records = build_prediction_records(manifest_rows, task_predictions_by_method)
    for method, rows in prediction_records.items():
        write_predictions_jsonl(output_dir / "predictions" / f"{method}_predictions.jsonl", rows)

    agreement_stats = {
        "density_category": compute_silver_vs_gold_agreement(
            [normalize_density_label(row.get("silver_density_category")) for row in manifest_rows],
            gold_by_task["density"],
        ),
        "has_size": compute_silver_vs_gold_agreement(
            [1 if row.get("silver_has_size") else 0 for row in manifest_rows],
            gold_by_task["has_size"],
        ),
        "size_mm": compute_silver_vs_gold_agreement(
            [float(row["silver_size_mm"]) if row.get("silver_size_mm") is not None else None for row in manifest_rows],
            [float(row["gold_size_mm"]) if row.get("gold_size_mm") is not None else None for row in manifest_rows],
        ),
        "location_lobe": compute_silver_vs_gold_agreement(
            [normalize_location_label(row.get("silver_location_lobe")) for row in manifest_rows],
            gold_by_task["location"],
        ),
    }

    save_json(output_dir / "gold_eval_metrics.json", all_results)
    write_summary_csv(output_dir / "gold_eval_summary.csv", all_results, evaluated_methods)
    save_json(output_dir / "error_analysis.json", error_analysis)
    save_json(output_dir / "silver_vs_gold_agreement.json", agreement_stats)
    report_path = generate_report(all_results, manifest_rows, output_dir, error_analysis, agreement_stats, evaluated_methods)

    log("[SummaryTable]")
    for task in TASKS:
        log(f"  task={task}")
        for method in evaluated_methods:
            if method not in all_results:
                continue
            primary_name, primary_value = get_primary_metric(task, all_results[method][task])
            log(f"    {method:10s} {primary_name}={primary_value:.4f}")

    elapsed = time.perf_counter() - start_time
    log(f"[Done] report={report_path} elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
