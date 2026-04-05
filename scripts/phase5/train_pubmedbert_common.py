from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.nodule_extractor import extract_size
from src.phase5.evaluation.metrics import (
    evaluate_density,
    evaluate_location,
    evaluate_size_detection,
    evaluate_size_regression,
)


MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_MAX_LENGTH = 128
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WORKERS = 4
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_hf_environment(hf_endpoint: str | None) -> None:
    endpoint = hf_endpoint or DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint

    cache_root = PROJECT_ROOT / "outputs" / "phase5" / "hf_cache"
    hub_cache = cache_root / "hub"
    ensure_dir(hub_cache)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)

    for proxy_key in ("all_proxy", "ALL_PROXY"):
        proxy_value = os.environ.get(proxy_key)
        if proxy_value and proxy_value.lower().startswith("socks"):
            os.environ.pop(proxy_key, None)
            log(f"[Network] 已移除不兼容的 {proxy_key}={proxy_value}")

    log(f"[Network] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    log(f"[Network] HF_HOME={os.environ['HF_HOME']}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LazyMentionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        label_encoder: Callable[[dict[str, Any]], int],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["mention_text"],
            truncation=True,
            max_length=self.max_length,
        )
        item = dict(encoding)
        item["labels"] = self.label_encoder(row)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class EpochMetricsLogger(TrainerCallback):
    def __init__(self, total_epochs: int, primary_metric: str, seed: int) -> None:
        self.total_epochs = total_epochs
        self.primary_metric = primary_metric
        self.seed = seed
        self.epoch_start_time: float | None = None
        self.last_train_loss: float | None = None
        self.last_learning_rate: float | None = None
        self.logged_epochs: set[int] = set()

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        self.epoch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if not logs:
            return

        if "loss" in logs:
            self.last_train_loss = float(logs["loss"])
        if "learning_rate" in logs:
            self.last_learning_rate = float(logs["learning_rate"])

        if "eval_loss" not in logs:
            return

        epoch_value = logs.get("epoch", state.epoch)
        if epoch_value is None:
            return

        epoch_idx = int(round(float(epoch_value)))
        if epoch_idx in self.logged_epochs:
            return
        self.logged_epochs.add(epoch_idx)

        elapsed = 0.0
        if self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time

        lr = self.last_learning_rate if self.last_learning_rate is not None else 0.0
        train_loss = self.last_train_loss if self.last_train_loss is not None else float("nan")
        val_loss = float(logs["eval_loss"])
        best_metric = state.best_metric
        current_metric = logs.get(f"eval_{self.primary_metric}")
        if best_metric is None and current_metric is not None:
            best_metric = float(current_metric)
        best_metric_value = float(best_metric) if best_metric is not None else float("nan")

        log(
            "Epoch "
            f"{epoch_idx}/{self.total_epochs} | "
            f"Seed {self.seed} | "
            f"LR {lr:.2e} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"BestVal{self.primary_metric.upper()} {best_metric_value:.4f} | "
            f"Time {elapsed:.1f}s"
        )


@dataclass(frozen=True)
class TaskConfig:
    task: str
    label_field: str
    label_names: list[str]
    model_dir_name: str
    result_file_name: str
    primary_metric: str
    weighted_loss: bool


def build_arg_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Phase 5 PubMedBERT training for {task}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataloader-num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT)
    return parser


def maybe_limit_rows(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None:
        return rows
    return rows[:max_samples]


def get_label_encoder(config: TaskConfig) -> Callable[[dict[str, Any]], int]:
    label_to_id = {label: idx for idx, label in enumerate(config.label_names)}

    if config.task == "size":
        return lambda row: 1 if bool(row["has_size"]) else 0

    return lambda row: label_to_id[str(row[config.label_field])]


def compute_class_weights_from_rows(
    rows: list[dict[str, Any]],
    config: TaskConfig,
    label_encoder: Callable[[dict[str, Any]], int],
) -> torch.Tensor | None:
    if not config.weighted_loss:
        return None
    labels = np.asarray([label_encoder(row) for row in rows], dtype=np.int64)
    present_classes = np.unique(labels)
    present_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels)
    full_weights = np.ones(len(config.label_names), dtype=np.float32)
    for class_id, weight in zip(present_classes.tolist(), present_weights.tolist(), strict=False):
        full_weights[int(class_id)] = float(weight)
    return torch.tensor(full_weights, dtype=torch.float32)


def build_compute_metrics(config: TaskConfig) -> Callable[[Any], dict[str, float]]:
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        if config.task == "density":
            y_true = [config.label_names[int(idx)] for idx in labels]
            y_pred = [config.label_names[int(idx)] for idx in predictions]
            results = evaluate_density(y_true, y_pred, config.label_names)
            return {
                "accuracy": float(results["accuracy"]),
                "macro_f1": float(results["macro_f1"]),
            }

        if config.task == "location":
            y_true = [config.label_names[int(idx)] for idx in labels]
            y_pred = [config.label_names[int(idx)] for idx in predictions]
            results = evaluate_location(y_true, y_pred, config.label_names)
            return {
                "accuracy": float(results["accuracy"]),
                "macro_f1": float(results["macro_f1"]),
            }

        results = evaluate_size_detection(labels.tolist(), predictions.tolist())
        return {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1": float(results["f1"]),
        }

    return compute_metrics


def get_best_epoch(log_history: list[dict[str, Any]], primary_metric: str) -> int | None:
    eval_metric_key = f"eval_{primary_metric}"
    best_score: float | None = None
    best_epoch: int | None = None
    for entry in log_history:
        if eval_metric_key not in entry:
            continue
        score = float(entry[eval_metric_key])
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        epoch_idx = int(round(float(epoch)))
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch_idx
    return best_epoch


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def evaluate_split(
    trainer: Trainer,
    dataset: LazyMentionDataset,
    rows: list[dict[str, Any]],
    config: TaskConfig,
) -> dict[str, Any]:
    prediction_output = trainer.predict(dataset)
    pred_ids = np.argmax(prediction_output.predictions, axis=-1).tolist()

    if config.task == "density":
        y_true = [str(row[config.label_field]) for row in rows]
        y_pred = [config.label_names[idx] for idx in pred_ids]
        return evaluate_density(y_true, y_pred, config.label_names)

    if config.task == "location":
        y_true = [str(row[config.label_field]) for row in rows]
        y_pred = [config.label_names[idx] for idx in pred_ids]
        return evaluate_location(y_true, y_pred, config.label_names)

    y_true = [1 if bool(row["has_size"]) else 0 for row in rows]
    classification_results: dict[str, Any] = evaluate_size_detection(y_true, pred_ids)
    regression_truth = [float(row["size_label"]) for row in rows if bool(row["has_size"]) and row["size_label"] is not None]
    regression_pred = []
    for row in rows:
        if not bool(row["has_size"]) or row["size_label"] is None:
            continue
        predicted_size, _ = extract_size(str(row["mention_text"]))
        regression_pred.append(0.0 if predicted_size is None else float(predicted_size))
    classification_results["regression"] = evaluate_size_regression(regression_truth, regression_pred)
    return classification_results


def run_task(config: TaskConfig) -> None:
    parser = build_arg_parser(config.task)
    args = parser.parse_args()
    set_seed(args.seed)

    prepare_hf_environment(args.hf_endpoint)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    data_dir = PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    model_dir = PROJECT_ROOT / "outputs" / "phase5" / "models" / config.model_dir_name
    results_dir = PROJECT_ROOT / "outputs" / "phase5" / "results"
    ensure_dir(model_dir)
    ensure_dir(results_dir)

    train_path = data_dir / f"{config.task}_train.jsonl"
    val_path = data_dir / f"{config.task}_val.jsonl"
    test_path = data_dir / f"{config.task}_test.jsonl"

    log(f"[Start] train_{config.task}")
    log(f"[Config] model={MODEL_NAME} seed={args.seed} epochs={args.epochs} max_length={args.max_length}")
    log(f"[Paths] train={train_path} val={val_path} test={test_path}")
    log(f"[Output] model_dir={model_dir} results_dir={results_dir}")

    train_rows = maybe_limit_rows(load_jsonl(train_path), args.max_train_samples)
    val_rows = maybe_limit_rows(load_jsonl(val_path), args.max_val_samples)
    test_rows = maybe_limit_rows(load_jsonl(test_path), args.max_test_samples)
    log(f"[Data] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    label_encoder = get_label_encoder(config)

    train_dataset = LazyMentionDataset(train_rows, tokenizer, label_encoder, args.max_length)
    val_dataset = LazyMentionDataset(val_rows, tokenizer, label_encoder, args.max_length)
    test_dataset = LazyMentionDataset(test_rows, tokenizer, label_encoder, args.max_length)
    class_weights = compute_class_weights_from_rows(train_rows, config, label_encoder)

    if class_weights is not None:
        weight_msg = ", ".join(
            f"{label}={weight:.4f}" for label, weight in zip(config.label_names, class_weights.tolist(), strict=False)
        )
        log(f"[ClassWeights] {weight_msg}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(config.label_names),
        use_safetensors=True,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.primary_metric,
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=50,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(config),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
            EpochMetricsLogger(total_epochs=args.epochs, primary_metric=config.primary_metric, seed=args.seed),
        ],
        class_weights=class_weights,
    )

    train_start = time.time()
    train_output = trainer.train()
    train_time_seconds = time.time() - train_start

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    best_epoch = get_best_epoch(trainer.state.log_history, config.primary_metric)
    log(
        f"[TrainDone] train_loss={train_output.training_loss:.4f} "
        f"best_checkpoint={trainer.state.best_model_checkpoint} best_epoch={best_epoch}"
    )

    eval_start = time.time()
    val_results = evaluate_split(trainer, val_dataset, val_rows, config)
    test_results = evaluate_split(trainer, test_dataset, test_rows, config)
    eval_time_seconds = time.time() - eval_start

    results = {
        "method": "pubmedbert",
        "task": config.task,
        "model_name": MODEL_NAME,
        "val_results": val_results,
        "test_results": test_results,
        "train_time_seconds": train_time_seconds,
        "eval_time_seconds": eval_time_seconds,
        "best_epoch": best_epoch,
        "training_args": training_args.to_dict(),
    }

    result_path = results_dir / config.result_file_name
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(results), fp, ensure_ascii=False, indent=2)
    log(f"[ResultSaved] {result_path}")
    log(f"[ValResults] {json.dumps(to_jsonable(val_results), ensure_ascii=False)}")
    log(f"[TestResults] {json.dumps(to_jsonable(test_results), ensure_ascii=False)}")
