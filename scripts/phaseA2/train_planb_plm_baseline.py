#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one extra PLM baseline for Module 2 Plan B.

This runner is intentionally task/model generic so SciBERT and ClinicalBERT
baselines share the same training and evaluation path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_GATE,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_LR,
    DEFAULT_MAX_LENGTH,
    DEFAULT_PATIENCE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    CheckpointCompatibilityCallback,
    LazyMentionDataset,
    MWSTaskConfig,
    build_compute_metrics,
    evaluate_on_phase5_test,
    get_best_epoch,
    get_label_encoder,
    load_jsonl,
    log,
    normalize_checkpoint_state_dict,
    prepare_hf_environment,
    set_log_fp,
    set_seed,
    to_jsonable,
)


MODEL_CHOICES = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

TASK_CHOICES = ("density_stage1", "density_stage2", "size", "location")

TASK_CONFIGS = {
    "density_stage1": MWSTaskConfig(
        task="density_stage1",
        label_field="density_stage1_label",
        label_names=["explicit_density", "unclear_or_no_evidence"],
        model_dir_name="density_stage1_extra_plm",
        result_file_name="extra_plm_density_stage1_results.json",
        primary_metric="auprc",
        weighted_loss=False,
        use_confidence_weight=False,
        input_field="section_aware_text",
    ),
    "density_stage2": MWSTaskConfig(
        task="density_stage2",
        label_field="density_stage2_label",
        label_names=["solid", "part_solid", "ground_glass", "calcified"],
        model_dir_name="density_stage2_extra_plm",
        result_file_name="extra_plm_density_stage2_results.json",
        primary_metric="macro_f1",
        weighted_loss=False,
        use_confidence_weight=False,
        input_field="section_aware_text",
    ),
    "size": MWSTaskConfig(
        task="size",
        label_field="has_size",
        label_names=["no_size", "has_size"],
        model_dir_name="size_extra_plm",
        result_file_name="extra_plm_size_results.json",
        primary_metric="f1",
        weighted_loss=False,
        use_confidence_weight=False,
        input_field="mention_text",
    ),
    "location": MWSTaskConfig(
        task="location",
        label_field="location_label",
        label_names=["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"],
        model_dir_name="location_extra_plm",
        result_file_name="extra_plm_location_results.json",
        primary_metric="macro_f1",
        weighted_loss=False,
        use_confidence_weight=False,
        input_field="mention_text",
        extra_label_names=["no_location"],
    ),
}


class PLMEpochMetricsLogger(TrainerCallback):
    """Plain-text epoch logger that keeps foreground and nohup logs readable."""

    def __init__(self, total_epochs: int, primary_metric: str, seed: int, log_fp) -> None:
        self.total_epochs = total_epochs
        self.primary_metric = primary_metric
        self.seed = seed
        self.log_fp = log_fp
        self.best_val: float | None = None
        self.epoch_start_time: float | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = int(round(state.epoch)) if state.epoch else 0
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        lr = 0.0
        train_loss = 0.0
        for entry in reversed(state.log_history):
            if lr == 0.0 and "learning_rate" in entry:
                lr = float(entry["learning_rate"])
            if "loss" in entry:
                train_loss = float(entry["loss"])
                break
        eval_key = f"eval_{self.primary_metric}"
        val_metric = float(metrics.get(eval_key, 0.0) or 0.0)
        if self.best_val is None or val_metric > self.best_val:
            self.best_val = val_metric
        bits = [
            f"Epoch {epoch}/{self.total_epochs}",
            f"Seed {self.seed}",
            f"LR {lr:.2e}",
            f"TrainLoss {train_loss:.4f}",
            f"ValLoss {float(metrics.get('eval_loss', 0.0) or 0.0):.4f}",
            f"BestVal{self.primary_metric} {(self.best_val or 0.0):.4f}",
        ]
        for key, title in [
            ("eval_auprc", "AUPRC"),
            ("eval_auroc", "AUROC"),
            ("eval_f1", "F1"),
            ("eval_macro_f1", "MacroF1"),
            ("eval_precision", "Precision"),
            ("eval_recall", "Recall"),
            ("eval_accuracy", "Accuracy"),
        ]:
            if key in metrics:
                bits.append(f"{title} {float(metrics[key] or 0.0):.4f}")
        bits.append(f"Time {elapsed:.1f}s")
        log(" | ".join(bits), self.log_fp)


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_task_data_dir(task: str) -> Path:
    if task in {"density_stage1", "density_stage2"}:
        return PROJECT_ROOT / "outputs" / "phaseA2_planB" / task
    return PROJECT_ROOT / "outputs" / "phaseA1" / task


def phase5_test_path_for_task(task: str, task_data_dir: Path, phase5_data_dir: Path) -> Path:
    if task in {"density_stage1", "density_stage2"}:
        return task_data_dir / f"{task}_test.jsonl"
    return phase5_data_dir / f"{task}_test.jsonl"


def runtime_environment(model_name_or_path: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "model_name_or_path": model_name_or_path,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "use_safetensors": args.use_safetensors,
    }


def model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.use_safetensors != "auto":
        kwargs["use_safetensors"] = args.use_safetensors == "true"
    return kwargs


def write_failure_result(
    path: Path,
    *,
    args: argparse.Namespace,
    config: MWSTaskConfig,
    model_name_or_path: str,
    tag: str,
    input_field: str,
    counts: dict[str, int] | None,
    stage: str,
    exc: BaseException,
) -> None:
    payload = {
        "status": "failed",
        "method": args.model_key,
        "task": config.task,
        "seed": args.seed,
        "gate": args.gate,
        "tag": tag,
        "model_name": model_name_or_path,
        "input_field": input_field,
        "train_samples": counts.get("train", 0) if counts else 0,
        "val_samples": counts.get("val", 0) if counts else 0,
        "test_ws_samples": counts.get("test_ws", 0) if counts else 0,
        "test_phase5_samples": counts.get("phase5_test", 0) if counts else 0,
        "test_truncated": False,
        "test_sample_count": counts.get("phase5_test", 0) if counts else 0,
        "failure_stage": stage,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "runtime_environment": runtime_environment(model_name_or_path, args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Plan B extra PLM baseline")
    parser.add_argument("--model-key", choices=sorted(MODEL_CHOICES), required=True)
    parser.add_argument(
        "--model-name",
        default=None,
        help="HF model id or local path. Defaults to the canonical model for --model-key.",
    )
    parser.add_argument("--task", choices=TASK_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-safetensors", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gate", default=DEFAULT_GATE)
    parser.add_argument("--task-data-dir", default=None, help="Override data dir; normally one task at a time")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--input-field", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--max-ws-test-samples",
        "--max-test-samples",
        dest="max_ws_test_samples",
        type=int,
        default=None,
        help="Optional WS-test truncation for smoke only. Phase5 test is never truncated.",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def run_one(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    prepare_hf_environment(args.hf_endpoint)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    config = TASK_CONFIGS[args.task]
    model_name_or_path = args.model_name or MODEL_CHOICES[args.model_key]
    input_field = args.input_field or config.input_field
    tag = args.tag or f"extra_plm_seed{args.seed}"

    task_data_dir = resolve_path(args.task_data_dir, default_task_data_dir(args.task))
    phase5_data_dir = resolve_path(args.phase5_data_dir, PROJECT_ROOT / "outputs" / "phase5" / "datasets")
    output_base = resolve_path(args.output_dir, PROJECT_ROOT / "outputs" / "phaseA2_planB")
    results_dir = output_base / "results"
    model_dir = output_base / "models" / f"{args.model_key}_{config.task}_{tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.model_key}_{config.task}_results_{tag}.json"

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_planb_plm_baseline_{args.model_key}_{config.task}_{tag}.log"
    log_fp = log_path.open("w", encoding="utf-8", buffering=1)
    set_log_fp(log_fp)

    counts: dict[str, int] | None = None
    train_start = time.time()
    try:
        train_path = task_data_dir / f"{config.task}_train_ws_{args.gate}.jsonl"
        val_path = task_data_dir / f"{config.task}_val_ws.jsonl"
        test_path = task_data_dir / f"{config.task}_test_ws.jsonl"
        phase5_test_path = phase5_test_path_for_task(config.task, task_data_dir, phase5_data_dir)

        log(f"[Start] train_planb_plm_baseline model={args.model_key} task={config.task} tag={tag}", log_fp)
        log(f"[Config] model_name_or_path={model_name_or_path}", log_fp)
        log(f"[Config] seed={args.seed} gate={args.gate} epochs={args.epochs} max_length={args.max_length}", log_fp)
        log(f"[Config] input_field={input_field} class_weights=false confidence_weighting=false", log_fp)
        log(
            f"[Config] batch={args.train_batch_size} grad_accum={args.gradient_accumulation_steps} "
            f"effective_batch={args.train_batch_size * args.gradient_accumulation_steps}",
            log_fp,
        )
        log(f"[Config] lr={args.learning_rate} warmup={args.warmup_ratio} patience={args.patience}", log_fp)
        log(f"[Runtime] {json.dumps(runtime_environment(model_name_or_path, args), ensure_ascii=False)}", log_fp)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            log(
                f"[GPU] {torch.cuda.get_device_name(0)} | "
                f"VRAM {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB | fp16=True",
                log_fp,
            )
        log(f"[Paths] train={train_path}", log_fp)
        log(f"[Paths] val={val_path}", log_fp)
        log(f"[Paths] test(ws)={test_path}", log_fp)
        log(f"[Paths] test(phase5)={phase5_test_path}", log_fp)
        log(f"[Output] model_dir={model_dir} results_dir={results_dir}", log_fp)

        train_rows_full = load_jsonl(train_path)
        val_rows_full = load_jsonl(val_path)
        test_rows_full = load_jsonl(test_path)
        phase5_test_rows = load_jsonl(phase5_test_path)

        train_rows = train_rows_full[: args.max_train_samples] if args.max_train_samples else train_rows_full
        val_rows = val_rows_full[: args.max_val_samples] if args.max_val_samples else val_rows_full
        test_rows = test_rows_full[: args.max_ws_test_samples] if args.max_ws_test_samples else test_rows_full
        counts = {
            "train": len(train_rows),
            "val": len(val_rows),
            "test_ws": len(test_rows),
            "phase5_test": len(phase5_test_rows),
        }
        log(
            f"[Data] train={len(train_rows)} val={len(val_rows)} test_ws={len(test_rows)} "
            f"test_phase5={len(phase5_test_rows)} test_truncated=false "
            f"test_sample_count={len(phase5_test_rows)}",
            log_fp,
        )
        log(
            f"[DataLimits] train_truncated={len(train_rows) != len(train_rows_full)} "
            f"val_truncated={len(val_rows) != len(val_rows_full)} "
            f"ws_test_truncated={len(test_rows) != len(test_rows_full)} "
            f"phase5_test_truncated=false",
            log_fp,
        )

        load_kwargs = model_load_kwargs(args)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(config.label_names),
                **load_kwargs,
            )
        except Exception as exc:
            log(
                f"[ModelLoadError] model={model_name_or_path} type={exc.__class__.__name__} error={exc}",
                log_fp,
            )
            write_failure_result(
                result_path,
                args=args,
                config=config,
                model_name_or_path=model_name_or_path,
                tag=tag,
                input_field=input_field,
                counts=counts,
                stage="model_load",
                exc=exc,
            )
            raise

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        label_encoder = get_label_encoder(config)
        train_dataset = LazyMentionDataset(train_rows, tokenizer, label_encoder, args.max_length, input_field)
        val_dataset = LazyMentionDataset(val_rows, tokenizer, label_encoder, args.max_length, input_field)

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=build_compute_metrics(config),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.patience),
                PLMEpochMetricsLogger(args.epochs, config.primary_metric, args.seed, log_fp),
                CheckpointCompatibilityCallback(),
            ],
        )

        if args.resume_from_checkpoint:
            log(f"[Resume] from checkpoint: {args.resume_from_checkpoint}", log_fp)
            if normalize_checkpoint_state_dict(Path(args.resume_from_checkpoint)):
                log(f"[Resume] normalized legacy LayerNorm keys in {args.resume_from_checkpoint}", log_fp)

        train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        train_time = time.time() - train_start

        trainer.save_model(str(model_dir))
        if normalize_checkpoint_state_dict(model_dir):
            log(f"[Checkpoint] normalized legacy LayerNorm keys in final model dir {model_dir}", log_fp)
        tokenizer.save_pretrained(str(model_dir))

        best_epoch = get_best_epoch(trainer.state.log_history, config.primary_metric)
        peak_memory_gb = None
        log(f"[TrainDone] loss={train_output.training_loss:.4f} best_epoch={best_epoch} time={train_time:.1f}s", log_fp)
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            peak_memory_gb = peak_mb / 1024
            log(f"[GPU] peak_memory={peak_mb:.0f}MB ({peak_memory_gb:.1f}GB)", log_fp)

        eval_start = time.time()
        log("[Eval] Evaluating on WS val set...", log_fp)
        ws_val_results = evaluate_on_phase5_test(trainer, tokenizer, config, val_rows, args.max_length)
        log(f"[WS-Val] {json.dumps({k: v for k, v in ws_val_results.items() if isinstance(v, (int, float, str))})}", log_fp)

        log("[Eval] Evaluating on WS test set...", log_fp)
        ws_test_results = evaluate_on_phase5_test(trainer, tokenizer, config, test_rows, args.max_length)
        log(f"[WS-Test] {json.dumps({k: v for k, v in ws_test_results.items() if isinstance(v, (int, float, str))})}", log_fp)

        log("[Eval] Evaluating on Phase5 full test set (not truncated)...", log_fp)
        phase5_test_results = evaluate_on_phase5_test(trainer, tokenizer, config, phase5_test_rows, args.max_length)
        log(
            f"[Phase5-Test] {json.dumps({k: v for k, v in phase5_test_results.items() if isinstance(v, (int, float, str))})}",
            log_fp,
        )
        eval_time = time.time() - eval_start

        result = {
            "status": "ok",
            "method": args.model_key,
            "task": config.task,
            "seed": args.seed,
            "gate": args.gate,
            "tag": tag,
            "model_name": model_name_or_path,
            "canonical_model_name": MODEL_CHOICES[args.model_key],
            "input_field": input_field,
            "label_field": config.label_field,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_ws_samples": len(test_rows),
            "test_phase5_samples": len(phase5_test_rows),
            "test_truncated": False,
            "test_sample_count": len(phase5_test_rows),
            "ws_test_truncated": len(test_rows) != len(test_rows_full),
            "ws_val_results": to_jsonable(ws_val_results),
            "ws_test_results": to_jsonable(ws_test_results),
            "phase5_test_results": to_jsonable(phase5_test_results),
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "best_epoch": best_epoch,
            "peak_gpu_memory_gb": peak_memory_gb,
            "training_args": to_jsonable(training_args.to_dict()),
            "runtime_environment": runtime_environment(model_name_or_path, args),
            "planb_plm_protocol": {
                "training_data": str(task_data_dir),
                "phase5_test_for_main_table": True,
                "phase5_test_truncated": False,
                "test_sample_count": len(phase5_test_rows),
                "hard_labels": True,
                "class_weighting": False,
                "confidence_weighting": False,
                "density_stage1_threshold_tuning": "not_used; reports default classifier F1/AUPRC/AUROC",
            },
        }

        result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[Saved] {result_path}", log_fp)
        log(f"[Done] total={train_time + eval_time:.1f}s", log_fp)
        return result_path
    except Exception as exc:
        log(f"[Error] {exc.__class__.__name__}: {exc}", log_fp)
        log(traceback.format_exc(), log_fp)
        if not result_path.exists():
            write_failure_result(
                result_path,
                args=args,
                config=config,
                model_name_or_path=model_name_or_path,
                tag=tag,
                input_field=input_field,
                counts=counts,
                stage="runtime",
                exc=exc,
            )
        raise
    finally:
        log_fp.close()


def main() -> None:
    args = parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
