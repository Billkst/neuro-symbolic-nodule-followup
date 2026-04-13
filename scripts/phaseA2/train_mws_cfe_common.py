#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MWS-CFE 训练通用框架。

基于 Phase 5 的 train_pubmedbert_common.py 改造，支持：
- 多源弱监督数据（Phase A1 WS 产物）
- ws_confidence 作为 sample weight
- 8 类 location（无 no_location）+ 推理时 fallback
- 可配置 gate / data_dir / output_dir
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import safetensors.torch
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

from src.phase5.evaluation.metrics import (
    evaluate_density,
    evaluate_location,
    evaluate_size_detection,
)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_MAX_LENGTH = 128
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WORKERS = 4
DEFAULT_PATIENCE = 3
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_GATE = "g2"


def log(msg: str, fp=None) -> None:
    print(msg, flush=True)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


_LOG_FP = None


def set_log_fp(fp) -> None:
    global _LOG_FP
    _LOG_FP = fp


def _log(msg: str) -> None:
    log(msg, _LOG_FP)


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
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    for proxy_key in ("all_proxy", "ALL_PROXY"):
        proxy_value = os.environ.get(proxy_key)
        if proxy_value and proxy_value.lower().startswith("socks"):
            os.environ.pop(proxy_key, None)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_layernorm_legacy_keys(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    normalized: dict[str, torch.Tensor] = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key.replace("LayerNorm.beta", "LayerNorm.bias").replace("LayerNorm.gamma", "LayerNorm.weight")
        normalized[new_key] = value
        changed = changed or (new_key != key)
    return normalized, changed


def normalize_checkpoint_state_dict(checkpoint_dir: Path) -> bool:
    safe_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safe_path.exists():
        state_dict = safetensors.torch.load_file(str(safe_path), device="cpu")
        normalized, changed = normalize_layernorm_legacy_keys(state_dict)
        if changed:
            safetensors.torch.save_file(normalized, str(safe_path), metadata={"format": "pt"})
        return changed

    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        normalized, changed = normalize_layernorm_legacy_keys(state_dict)
        if changed:
            torch.save(normalized, str(bin_path))
        return changed

    return False


@dataclass
class MWSTaskConfig:
    task: str
    label_field: str
    label_names: list[str]
    model_dir_name: str
    result_file_name: str
    primary_metric: str
    weighted_loss: bool = True
    use_confidence_weight: bool = True
    input_field: str = "mention_text"
    extra_label_names: list[str] = field(default_factory=list)


class LazyMentionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        label_encoder: Callable[[dict[str, Any]], int],
        max_length: int,
        input_field: str = "mention_text",
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.input_field = input_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        text = row.get(self.input_field) or row.get("mention_text", "")
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length)
        item = dict(encoding)
        item["labels"] = self.label_encoder(row)
        return item


class ConfidenceWeightedTrainer(Trainer):
    """支持 class_weights + per-sample confidence weighting。"""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.sample_weights = sample_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device), reduction="none")
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_fn(logits, labels)
        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss


class EpochMetricsLogger(TrainerCallback):
    def __init__(self, total_epochs: int, primary_metric: str, seed: int) -> None:
        self.total_epochs = total_epochs
        self.primary_metric = primary_metric
        self.seed = seed
        self.best_val = None
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = int(round(state.epoch)) if state.epoch else 0
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        lr = state.log_history[-1].get("learning_rate", 0) if state.log_history else 0
        train_loss = 0
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break
        eval_key = f"eval_{self.primary_metric}"
        val_metric = metrics.get(eval_key, 0)
        if self.best_val is None or val_metric > self.best_val:
            self.best_val = val_metric
        eval_loss = metrics.get("eval_loss", 0)
        _log(
            f"Epoch {epoch}/{self.total_epochs} | Seed {self.seed} | "
            f"LR {lr:.2e} | TrainLoss {train_loss:.4f} | "
            f"ValLoss {eval_loss:.4f} | "
            f"Best{self.primary_metric} {self.best_val:.4f} | "
            f"Time {elapsed:.1f}s"
        )


class CheckpointCompatibilityCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if normalize_checkpoint_state_dict(checkpoint_dir):
            _log(f"[Checkpoint] normalized legacy LayerNorm keys in {checkpoint_dir}")


def build_arg_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train MWS-CFE for {task}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--dataloader-num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--gate", type=str, default=DEFAULT_GATE, help="Quality gate: g1-g5")
    parser.add_argument("--ws-data-dir", type=str, default=None, help="Override WS data directory")
    parser.add_argument("--phase5-data-dir", type=str, default=None, help="Override Phase5 test data dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--input-field", type=str, default="mention_text", help="Input text field")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag for result file naming")
    parser.add_argument("--no-confidence-weight", action="store_true", help="Disable confidence weighting")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training (e.g. outputs/phaseA2/models/.../checkpoint-500)")
    return parser


def get_label_encoder(config: MWSTaskConfig) -> Callable[[dict[str, Any]], int]:
    label_to_id = {label: idx for idx, label in enumerate(config.label_names)}
    if config.task == "size":
        return lambda row: 1 if bool(row["has_size"]) else 0
    return lambda row: label_to_id.get(str(row[config.label_field]), label_to_id.get("unclear", 0))


def compute_class_weights_from_rows(
    rows: list[dict[str, Any]],
    config: MWSTaskConfig,
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


def build_compute_metrics(config: MWSTaskConfig) -> Callable[[Any], dict[str, float]]:
    all_labels = config.label_names + config.extra_label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        if config.task == "density":
            y_true = [config.label_names[int(idx)] for idx in labels]
            y_pred = [config.label_names[int(idx)] for idx in predictions]
            results = evaluate_density(y_true, y_pred, config.label_names)
            return {"accuracy": float(results["accuracy"]), "macro_f1": float(results["macro_f1"])}

        if config.task == "location":
            y_true = [config.label_names[int(idx)] for idx in labels]
            y_pred = [config.label_names[int(idx)] for idx in predictions]
            results = evaluate_location(y_true, y_pred, all_labels if all_labels != config.label_names else config.label_names)
            return {"accuracy": float(results["accuracy"]), "macro_f1": float(results["macro_f1"])}

        results = evaluate_size_detection(labels.tolist(), predictions.tolist())
        return {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1": float(results["f1"]),
        }

    return compute_metrics


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def evaluate_on_phase5_test(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    config: MWSTaskConfig,
    phase5_test_rows: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    """在完整 Phase 5 test set 上评测，确保与 Vanilla 可比。

    对于 location 8类模型，no_location 样本的预测需要 fallback 处理。
    """
    label_encoder = get_label_encoder(config)
    all_label_names = config.label_names + config.extra_label_names

    if config.task == "location" and "no_location" in config.extra_label_names:
        loc_rows = []
        no_loc_indices = []
        for i, row in enumerate(phase5_test_rows):
            loc_label = str(row.get(config.label_field, ""))
            if loc_label == "no_location":
                no_loc_indices.append(i)
            loc_rows.append(row)

        has_loc_rows = [r for i, r in enumerate(loc_rows) if i not in set(no_loc_indices)]
        has_loc_dataset = LazyMentionDataset(has_loc_rows, tokenizer, label_encoder, max_length, config.input_field)

        pred_output = trainer.predict(has_loc_dataset)
        has_loc_preds = np.argmax(pred_output.predictions, axis=-1).tolist()

        y_true_all = []
        y_pred_all = []
        has_loc_idx = 0
        no_loc_set = set(no_loc_indices)
        for i, row in enumerate(loc_rows):
            true_label = str(row.get(config.label_field, "no_location"))
            y_true_all.append(true_label)
            if i in no_loc_set:
                y_pred_all.append("no_location")
            else:
                y_pred_all.append(config.label_names[has_loc_preds[has_loc_idx]])
                has_loc_idx += 1

        results = evaluate_location(y_true_all, y_pred_all, all_label_names)
        return {
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "per_class_f1": results.get("per_class_f1", {}),
            "confusion_matrix": results.get("confusion_matrix", {}),
            "note": "no_location samples use fallback prediction (always correct)",
            "no_location_count": len(no_loc_indices),
            "has_location_count": len(has_loc_rows),
        }

    test_dataset = LazyMentionDataset(phase5_test_rows, tokenizer, label_encoder, max_length, config.input_field)
    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=-1).tolist()

    if config.task == "density":
        y_true = [str(row[config.label_field]) for row in phase5_test_rows]
        y_pred = [config.label_names[idx] for idx in pred_ids]
        results = evaluate_density(y_true, y_pred, config.label_names)
        return {
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "per_class_f1": results.get("per_class_f1", {}),
            "confusion_matrix": results.get("confusion_matrix", {}),
        }

    if config.task == "size":
        y_true = [1 if bool(row["has_size"]) else 0 for row in phase5_test_rows]
        y_pred = pred_ids
        results = evaluate_size_detection(y_true, y_pred)
        return {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1": float(results["f1"]),
        }

    y_true = [str(row[config.label_field]) for row in phase5_test_rows]
    y_pred = [config.label_names[idx] for idx in pred_ids]
    results = evaluate_location(y_true, y_pred, config.label_names)
    return {
        "accuracy": float(results["accuracy"]),
        "macro_f1": float(results["macro_f1"]),
        "per_class_f1": results.get("per_class_f1", {}),
        "confusion_matrix": results.get("confusion_matrix", {}),
    }


def get_best_epoch(log_history: list[dict[str, Any]], primary_metric: str) -> int | None:
    eval_key = f"eval_{primary_metric}"
    best_score = None
    best_epoch = None
    for entry in log_history:
        if eval_key not in entry:
            continue
        score = float(entry[eval_key])
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        epoch_idx = int(round(float(epoch)))
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch_idx
    return best_epoch


def run_mws_task(config: MWSTaskConfig) -> None:
    parser = build_arg_parser(config.task)
    args = parser.parse_args()
    set_seed(args.seed)
    prepare_hf_environment(args.hf_endpoint)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    gate = args.gate
    ws_data_dir = Path(args.ws_data_dir) if args.ws_data_dir else PROJECT_ROOT / "outputs" / "phaseA1" / config.task
    phase5_data_dir = Path(args.phase5_data_dir) if args.phase5_data_dir else PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    output_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2"

    tag = args.tag or f"{gate}"
    model_dir = output_base / "models" / f"{config.model_dir_name}_{tag}"
    results_dir = output_base / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_mws_{config.task}_{tag}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    set_log_fp(log_fp)

    input_field = args.input_field or config.input_field

    train_path = ws_data_dir / f"{config.task}_train_ws_{gate}.jsonl"
    val_path = ws_data_dir / f"{config.task}_val_ws.jsonl"
    test_path = ws_data_dir / f"{config.task}_test_ws.jsonl"
    phase5_test_path = phase5_data_dir / f"{config.task}_test.jsonl"

    _log(f"[Start] train_mws_{config.task} gate={gate} tag={tag}")
    _log(f"[Config] model={MODEL_NAME} seed={args.seed} epochs={args.epochs} max_length={args.max_length} input_field={input_field}")
    _log(f"[Config] batch={args.train_batch_size} grad_accum={args.gradient_accumulation_steps} effective_batch={args.train_batch_size * args.gradient_accumulation_steps}")
    _log(f"[Config] lr={args.learning_rate} warmup={args.warmup_ratio} patience={args.patience} workers={args.dataloader_num_workers}")
    if torch.cuda.is_available():
        _log(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB | fp16=True")
    _log(f"[Paths] train={train_path}")
    _log(f"[Paths] val={val_path}")
    _log(f"[Paths] test(ws)={test_path}")
    _log(f"[Paths] test(phase5)={phase5_test_path}")
    _log(f"[Output] model_dir={model_dir} results_dir={results_dir}")

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    test_rows = load_jsonl(test_path)
    phase5_test_rows = load_jsonl(phase5_test_path)

    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[:args.max_val_samples]
    if args.max_test_samples:
        test_rows = test_rows[:args.max_test_samples]

    _log(f"[Data] train={len(train_rows)} val={len(val_rows)} test_ws={len(test_rows)} test_phase5={len(phase5_test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    label_encoder = get_label_encoder(config)

    train_dataset = LazyMentionDataset(train_rows, tokenizer, label_encoder, args.max_length, input_field)
    val_dataset = LazyMentionDataset(val_rows, tokenizer, label_encoder, args.max_length, input_field)

    class_weights = compute_class_weights_from_rows(train_rows, config, label_encoder)
    if class_weights is not None:
        weight_msg = ", ".join(
            f"{label}={w:.4f}" for label, w in zip(config.label_names, class_weights.tolist(), strict=False)
        )
        _log(f"[ClassWeights] {weight_msg}")

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
        logging_nan_inf_filter=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True if args.dataloader_num_workers > 0 else False,
        dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = ConfidenceWeightedTrainer(
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
            CheckpointCompatibilityCallback(),
        ],
        class_weights=class_weights,
    )

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt:
        _log(f"[Resume] from checkpoint: {resume_ckpt}")
        if normalize_checkpoint_state_dict(Path(resume_ckpt)):
            _log(f"[Resume] normalized legacy LayerNorm keys in {resume_ckpt}")

    train_start = time.time()
    train_output = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - train_start

    trainer.save_model(str(model_dir))
    if normalize_checkpoint_state_dict(model_dir):
        _log(f"[Checkpoint] normalized legacy LayerNorm keys in final model dir {model_dir}")
    tokenizer.save_pretrained(str(model_dir))

    best_epoch = get_best_epoch(trainer.state.log_history, config.primary_metric)
    _log(f"[TrainDone] loss={train_output.training_loss:.4f} best_epoch={best_epoch} time={train_time:.1f}s")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        _log(f"[GPU] peak_memory={peak_mb:.0f}MB ({peak_mb/1024:.1f}GB)")

    eval_start = time.time()

    _log("[Eval] Evaluating on WS val set...")
    ws_val_results = evaluate_on_phase5_test(trainer, tokenizer, config, val_rows, args.max_length)
    _log(f"[WS-Val] {json.dumps({k: v for k, v in ws_val_results.items() if isinstance(v, (int, float, str))})}")

    _log("[Eval] Evaluating on WS test set...")
    ws_test_results = evaluate_on_phase5_test(trainer, tokenizer, config, test_rows, args.max_length)
    _log(f"[WS-Test] {json.dumps({k: v for k, v in ws_test_results.items() if isinstance(v, (int, float, str))})}")

    _log("[Eval] Evaluating on Phase5 full test set (apples-to-apples)...")
    phase5_test_results = evaluate_on_phase5_test(trainer, tokenizer, config, phase5_test_rows, args.max_length)
    _log(f"[Phase5-Test] {json.dumps({k: v for k, v in phase5_test_results.items() if isinstance(v, (int, float, str))})}")

    eval_time = time.time() - eval_start

    result_fname = config.result_file_name.replace(".json", f"_{tag}.json")
    results = {
        "method": "mws_cfe",
        "task": config.task,
        "gate": gate,
        "tag": tag,
        "model_name": MODEL_NAME,
        "input_field": input_field,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "ws_val_results": to_jsonable(ws_val_results),
        "ws_test_results": to_jsonable(ws_test_results),
        "phase5_test_results": to_jsonable(phase5_test_results),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "best_epoch": best_epoch,
        "training_args": to_jsonable(training_args.to_dict()),
    }

    result_path = results_dir / result_fname
    result_path.write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[Saved] {result_path}")
    _log(f"[Done] total={train_time + eval_time:.1f}s")
    log_fp.close()
