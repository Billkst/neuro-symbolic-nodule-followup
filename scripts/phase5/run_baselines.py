import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.extractors.nodule_extractor import extract_density, extract_location, extract_size
from src.phase5.evaluation.metrics import (
    evaluate_density,
    evaluate_location,
    evaluate_size_detection,
    evaluate_size_regression,
)


DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified", "unclear"]
LOCATION_LABELS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear", "no_location"]
DEFAULT_TASKS = ["density", "size", "location"]
DEFAULT_METHODS = ["regex", "ml_lr", "ml_svm"]


def log(message: str, log_fp) -> None:
    print(message, flush=True)
    log_fp.write(message + "\n")
    log_fp.flush()


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def parse_csv_arg(value: str | None, default_values: list[str], field_name: str) -> list[str]:
    if value is None or not value.strip():
        return list(default_values)

    parsed = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in default_values]
    if invalid:
        raise ValueError(f"{field_name} 包含非法值: {invalid}，可选值: {default_values}")
    return parsed


def normalize_location_label(label: str | None) -> str:
    if label is None:
        return "no_location"
    if label in LOCATION_LABELS:
        return label
    return "unclear"


def get_dataset_paths(task: str, dataset_dir: Path) -> dict[str, Path]:
    return {
        split: dataset_dir / f"{task}_{split}.jsonl"
        for split in ("train", "val", "test")
    }


def load_task_splits(task: str, dataset_dir: Path) -> dict[str, list[dict]]:
    paths = get_dataset_paths(task, dataset_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 {task} 数据集文件: {missing}")
    return {split: load_jsonl(path) for split, path in paths.items()}


def get_texts(rows: list[dict]) -> list[str]:
    return [str(row.get("mention_text") or "") for row in rows]


def get_density_labels(rows: list[dict]) -> list[str]:
    labels = []
    for row in rows:
        label = row.get("density_label")
        labels.append(label if label in DENSITY_LABELS else "unclear")
    return labels


def get_size_detection_labels(rows: list[dict]) -> list[int]:
    return [1 if bool(row.get("has_size")) else 0 for row in rows]


def get_location_labels(rows: list[dict]) -> list[str]:
    return [normalize_location_label(row.get("location_label")) for row in rows]


def evaluate_density_split(rows: list[dict], predictions: list[str]) -> dict:
    y_true = get_density_labels(rows)
    return evaluate_density(y_true, predictions, DENSITY_LABELS)


def evaluate_location_split(rows: list[dict], predictions: list[str]) -> dict:
    y_true = get_location_labels(rows)
    return evaluate_location(y_true, predictions, LOCATION_LABELS)


def evaluate_size_split(rows: list[dict], detection_predictions: list[int], size_predictions: list[float | None]) -> dict:
    y_true_detection = get_size_detection_labels(rows)
    detection_metrics = evaluate_size_detection(y_true_detection, detection_predictions)

    y_true_regression: list[float] = []
    y_pred_regression: list[float] = []
    for row, predicted_size in zip(rows, size_predictions, strict=False):
        gold_size = row.get("size_label")
        if gold_size is None:
            continue
        y_true_regression.append(float(gold_size))
        y_pred_regression.append(float(predicted_size) if predicted_size is not None else 0.0)

    regression_metrics = evaluate_size_regression(y_true_regression, y_pred_regression)
    return {
        "accuracy": detection_metrics["accuracy"],
        "precision": detection_metrics["precision"],
        "recall": detection_metrics["recall"],
        "f1": detection_metrics["f1"],
        "detection": detection_metrics,
        "regression": regression_metrics,
        "positive_gold_count": len(y_true_regression),
    }


def predict_regex_density(rows: list[dict]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        label, _ = extract_density(str(row.get("mention_text") or ""))
        predictions.append(label if label in DENSITY_LABELS else "unclear")
    return predictions


def predict_regex_size(rows: list[dict]) -> tuple[list[int], list[float | None]]:
    detection_predictions: list[int] = []
    size_predictions: list[float | None] = []
    for row in rows:
        size_value, _ = extract_size(str(row.get("mention_text") or ""))
        has_size = size_value is not None
        detection_predictions.append(1 if has_size else 0)
        size_predictions.append(float(size_value) if size_value is not None else None)
    return detection_predictions, size_predictions


def predict_regex_location(rows: list[dict]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        label, _ = extract_location(str(row.get("mention_text") or ""))
        predictions.append(normalize_location_label(label))
    return predictions


def build_ml_pipeline(method: str) -> Pipeline:
    if method == "ml_lr":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    elif method == "ml_svm":
        classifier = LinearSVC(class_weight="balanced", max_iter=5000, C=1.0)
    else:
        raise ValueError(f"不支持的 ML 方法: {method}")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True),
            ),
            ("clf", classifier),
        ]
    )


def run_regex_baseline(task: str, splits: dict[str, list[dict]], log_fp) -> dict:
    start_time = time.perf_counter()
    log(f"[Run] method=regex task={task} | train_time=0.00s (规则法无需训练)", log_fp)

    eval_start = time.perf_counter()
    if task == "density":
        val_predictions = predict_regex_density(splits["val"])
        test_predictions = predict_regex_density(splits["test"])
        val_results = evaluate_density_split(splits["val"], val_predictions)
        test_results = evaluate_density_split(splits["test"], test_predictions)
    elif task == "size":
        val_detection, val_sizes = predict_regex_size(splits["val"])
        test_detection, test_sizes = predict_regex_size(splits["test"])
        val_results = evaluate_size_split(splits["val"], val_detection, val_sizes)
        test_results = evaluate_size_split(splits["test"], test_detection, test_sizes)
    elif task == "location":
        val_predictions = predict_regex_location(splits["val"])
        test_predictions = predict_regex_location(splits["test"])
        val_results = evaluate_location_split(splits["val"], val_predictions)
        test_results = evaluate_location_split(splits["test"], test_predictions)
    else:
        raise ValueError(f"未知任务: {task}")

    eval_time = time.perf_counter() - eval_start
    total_time = time.perf_counter() - start_time
    log(f"[Done] method=regex task={task} | eval_time={eval_time:.2f}s total_time={total_time:.2f}s", log_fp)
    return {
        "method": "regex",
        "task": task,
        "val_results": val_results,
        "test_results": test_results,
        "train_time_seconds": 0.0,
        "eval_time_seconds": round(eval_time, 4),
    }


def run_ml_baseline(task: str, method: str, splits: dict[str, list[dict]], log_fp) -> dict:
    train_texts = get_texts(splits["train"])
    val_texts = get_texts(splits["val"])
    test_texts = get_texts(splits["test"])

    pipeline = build_ml_pipeline(method)
    log(
        f"[Run] method={method} task={task} | train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}",
        log_fp,
    )

    if task == "density":
        train_labels = get_density_labels(splits["train"])
    elif task == "size":
        train_labels = get_size_detection_labels(splits["train"])
    elif task == "location":
        train_labels = get_location_labels(splits["train"])
    else:
        raise ValueError(f"未知任务: {task}")

    train_start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    train_time = time.perf_counter() - train_start
    log(f"[Train] method={method} task={task} | train_time={train_time:.2f}s", log_fp)

    eval_start = time.perf_counter()
    val_predictions = pipeline.predict(val_texts).tolist()
    test_predictions = pipeline.predict(test_texts).tolist()

    if task == "density":
        val_results = evaluate_density_split(splits["val"], val_predictions)
        test_results = evaluate_density_split(splits["test"], test_predictions)
    elif task == "size":
        val_detection = [int(value) for value in val_predictions]
        test_detection = [int(value) for value in test_predictions]
        val_results = evaluate_size_split(splits["val"], val_detection, [None] * len(val_detection))
        test_results = evaluate_size_split(splits["test"], test_detection, [None] * len(test_detection))
    else:
        val_results = evaluate_location_split(splits["val"], val_predictions)
        test_results = evaluate_location_split(splits["test"], test_predictions)

    eval_time = time.perf_counter() - eval_start
    log(f"[Done] method={method} task={task} | eval_time={eval_time:.2f}s", log_fp)
    return {
        "method": method,
        "task": task,
        "val_results": val_results,
        "test_results": test_results,
        "train_time_seconds": round(train_time, 4),
        "eval_time_seconds": round(eval_time, 4),
    }


def build_summary_entry(task: str, result: dict) -> dict:
    if task == "size":
        return {
            "val_f1": result["val_results"]["f1"],
            "test_f1": result["test_results"]["f1"],
            "test_accuracy": result["test_results"]["accuracy"],
        }

    return {
        "val_macro_f1": result["val_results"]["macro_f1"],
        "test_macro_f1": result["test_results"]["macro_f1"],
        "test_accuracy": result["test_results"]["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="逗号分隔: density,size,location")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="逗号分隔: regex,ml_lr,ml_svm")
    parser.add_argument("--dataset-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phase5/results")
    parser.add_argument("--log", default="logs/run_baselines.log")
    args = parser.parse_args()

    tasks = parse_csv_arg(args.tasks, DEFAULT_TASKS, "tasks")
    methods = parse_csv_arg(args.methods, DEFAULT_METHODS, "methods")
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] run_baselines", log_fp)
        log(f"[Config] tasks={tasks} methods={methods}", log_fp)
        log(f"[Paths] dataset_dir={dataset_dir} output_dir={output_dir} log={log_path}", log_fp)

        summary: dict[str, dict[str, dict]] = {task: {} for task in tasks}

        for task in tasks:
            splits = load_task_splits(task, dataset_dir)
            log(
                f"[Data] task={task} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
                log_fp,
            )

            for method in methods:
                if method == "regex":
                    result = run_regex_baseline(task, splits, log_fp)
                else:
                    result = run_ml_baseline(task, method, splits, log_fp)

                output_path = output_dir / f"{method}_{task}_results.json"
                save_json(output_path, result)
                summary[task][method] = build_summary_entry(task, result)

                if task == "size":
                    log(
                        "[Metric] "
                        f"task={task} method={method} "
                        f"val_f1={result['val_results']['f1']:.4f} "
                        f"test_f1={result['test_results']['f1']:.4f} "
                        f"test_acc={result['test_results']['accuracy']:.4f}",
                        log_fp,
                    )
                else:
                    log(
                        "[Metric] "
                        f"task={task} method={method} "
                        f"val_macro_f1={result['val_results']['macro_f1']:.4f} "
                        f"test_macro_f1={result['test_results']['macro_f1']:.4f} "
                        f"test_acc={result['test_results']['accuracy']:.4f}",
                        log_fp,
                    )

        summary_path = output_dir / "baselines_summary.json"
        save_json(summary_path, summary)
        log(f"[Summary] wrote={summary_path}", log_fp)
        log(json.dumps(summary, ensure_ascii=False, indent=2), log_fp)
        log("[Done] run_baselines completed", log_fp)


if __name__ == "__main__":
    main()
