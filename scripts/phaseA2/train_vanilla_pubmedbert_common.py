#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vanilla PubMedBERT training entry for Phase A2.5.

This runner uses the current Phase A1 multi-source weak-supervision data and
the Phase A2 evaluation protocol, but trains a plain hard-label PubMedBERT
classifier. It intentionally avoids the old Phase 5 single-source datasets.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import (
    MODEL_NAME,
    CheckpointCompatibilityCallback,
    EpochMetricsLogger,
    LazyMentionDataset,
    MWSTaskConfig,
    build_arg_parser,
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


def _runtime_environment() -> dict[str, Any]:
    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "model_name": MODEL_NAME,
        "use_safetensors": True,
    }


def run_vanilla_task(config: MWSTaskConfig) -> None:
    parser = build_arg_parser(config.task)
    parser.description = f"Train Vanilla PubMedBERT for Phase A2.5 {config.task}"
    args = parser.parse_args()

    set_seed(args.seed)
    prepare_hf_environment(args.hf_endpoint)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    gate = args.gate
    ws_data_dir = Path(args.ws_data_dir) if args.ws_data_dir else PROJECT_ROOT / "outputs" / "phaseA1" / config.task
    phase5_data_dir = Path(args.phase5_data_dir) if args.phase5_data_dir else PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    output_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2"

    tag = args.tag or f"main_{gate}_seed{args.seed}"
    model_dir = output_base / "models" / f"{config.model_dir_name}_{tag}"
    results_dir = output_base / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_vanilla_{config.task}_{tag}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    set_log_fp(log_fp)

    input_field = args.input_field or config.input_field
    train_path = ws_data_dir / f"{config.task}_train_ws_{gate}.jsonl"
    val_path = ws_data_dir / f"{config.task}_val_ws.jsonl"
    test_path = ws_data_dir / f"{config.task}_test_ws.jsonl"
    phase5_test_path = phase5_data_dir / f"{config.task}_test.jsonl"

    log(f"[Start] train_vanilla_pubmedbert task={config.task} gate={gate} tag={tag}", log_fp)
    log(f"[Config] model={MODEL_NAME}", log_fp)
    log(f"[Config] seed={args.seed} epochs={args.epochs} max_length={args.max_length} input_field={input_field}", log_fp)
    log("[Config] hard_labels=true class_weights=false confidence_weighting=false", log_fp)
    log(
        f"[Config] batch={args.train_batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"effective_batch={args.train_batch_size * args.gradient_accumulation_steps}",
        log_fp,
    )
    log(f"[Config] lr={args.learning_rate} warmup={args.warmup_ratio} patience={args.patience}", log_fp)
    log(f"[Runtime] {json.dumps(_runtime_environment(), ensure_ascii=False)}", log_fp)
    if torch.cuda.is_available():
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

    log(
        f"[Data] train={len(train_rows)} val={len(val_rows)} "
        f"test_ws={len(test_rows)} test_phase5={len(phase5_test_rows)}",
        log_fp,
    )

    task_config = MWSTaskConfig(
        task=config.task,
        label_field=config.label_field,
        label_names=config.label_names,
        model_dir_name=config.model_dir_name,
        result_file_name=config.result_file_name,
        primary_metric=config.primary_metric,
        weighted_loss=False,
        use_confidence_weight=False,
        input_field=input_field,
        extra_label_names=config.extra_label_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    label_encoder = get_label_encoder(task_config)
    train_dataset = LazyMentionDataset(train_rows, tokenizer, label_encoder, args.max_length, input_field)
    val_dataset = LazyMentionDataset(val_rows, tokenizer, label_encoder, args.max_length, input_field)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(task_config.label_names),
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
        metric_for_best_model=task_config.primary_metric,
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
        compute_metrics=build_compute_metrics(task_config),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
            EpochMetricsLogger(total_epochs=args.epochs, primary_metric=task_config.primary_metric, seed=args.seed),
            CheckpointCompatibilityCallback(),
        ],
    )

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt:
        log(f"[Resume] from checkpoint: {resume_ckpt}", log_fp)
        if normalize_checkpoint_state_dict(Path(resume_ckpt)):
            log(f"[Resume] normalized legacy LayerNorm keys in {resume_ckpt}", log_fp)

    train_start = time.time()
    train_output = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - train_start

    trainer.save_model(str(model_dir))
    if normalize_checkpoint_state_dict(model_dir):
        log(f"[Checkpoint] normalized legacy LayerNorm keys in final model dir {model_dir}", log_fp)
    tokenizer.save_pretrained(str(model_dir))

    best_epoch = get_best_epoch(trainer.state.log_history, task_config.primary_metric)
    peak_memory_gb = None
    log(f"[TrainDone] loss={train_output.training_loss:.4f} best_epoch={best_epoch} time={train_time:.1f}s", log_fp)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_memory_gb = peak_mb / 1024
        log(f"[GPU] peak_memory={peak_mb:.0f}MB ({peak_memory_gb:.1f}GB)", log_fp)

    eval_start = time.time()
    log("[Eval] Evaluating on WS val set...", log_fp)
    ws_val_results = evaluate_on_phase5_test(trainer, tokenizer, task_config, val_rows, args.max_length)
    log(f"[WS-Val] {json.dumps({k: v for k, v in ws_val_results.items() if isinstance(v, (int, float, str))})}", log_fp)

    log("[Eval] Evaluating on WS test set...", log_fp)
    ws_test_results = evaluate_on_phase5_test(trainer, tokenizer, task_config, test_rows, args.max_length)
    log(f"[WS-Test] {json.dumps({k: v for k, v in ws_test_results.items() if isinstance(v, (int, float, str))})}", log_fp)

    log("[Eval] Evaluating on Phase5 full test set (apples-to-apples)...", log_fp)
    phase5_test_results = evaluate_on_phase5_test(trainer, tokenizer, task_config, phase5_test_rows, args.max_length)
    log(f"[Phase5-Test] {json.dumps({k: v for k, v in phase5_test_results.items() if isinstance(v, (int, float, str))})}", log_fp)
    eval_time = time.time() - eval_start

    result_fname = config.result_file_name.replace(".json", f"_{tag}.json")
    results = {
        "method": "vanilla_pubmedbert",
        "task": config.task,
        "seed": args.seed,
        "gate": gate,
        "tag": tag,
        "model_name": MODEL_NAME,
        "input_field": input_field,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_ws_samples": len(test_rows),
        "test_phase5_samples": len(phase5_test_rows),
        "ws_val_results": to_jsonable(ws_val_results),
        "ws_test_results": to_jsonable(ws_test_results),
        "phase5_test_results": to_jsonable(phase5_test_results),
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "best_epoch": best_epoch,
        "peak_gpu_memory_gb": peak_memory_gb,
        "training_args": to_jsonable(training_args.to_dict()),
        "runtime_environment": _runtime_environment(),
        "a2_5_protocol": {
            "training_data": "Phase A1 multi-source WS",
            "phase5_single_source_reused": False,
            "confidence_weighting": False,
            "class_weighting": False,
            "use_safetensors": True,
        },
    }

    result_path = results_dir / result_fname
    result_path.write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[Saved] {result_path}", log_fp)
    log(f"[Done] total={train_time + eval_time:.1f}s", log_fp)
    log_fp.close()

