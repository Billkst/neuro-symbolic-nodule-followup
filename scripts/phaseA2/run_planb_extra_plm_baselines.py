#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run extra PLM baselines without letting one failure block the rest."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "phaseA2" / "train_planb_plm_baseline.py"

MODEL_DEFAULTS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}
TASKS = ["density_stage1", "density_stage2", "size", "location"]


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def parse_csv(value: str, allowed: list[str] | None = None, name: str = "value") -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if allowed is not None:
        invalid = [item for item in items if item not in allowed]
        if invalid:
            raise ValueError(f"{name} contains invalid values: {invalid}; choices={allowed}")
    return items


def parse_seed_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value, name="seeds")]


def add_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_command(args: argparse.Namespace, model_key: str, task: str, seed: int, tag: str) -> list[str]:
    model_name = args.scibert_model if model_key == "scibert" else args.bioclinicalbert_model
    cmd = [
        args.python,
        "-u",
        str(RUNNER),
        "--model-key",
        model_key,
        "--model-name",
        model_name,
        "--task",
        task,
        "--seed",
        str(seed),
        "--gate",
        args.gate,
        "--tag",
        tag,
        "--output-dir",
        args.output_dir,
        "--phase5-data-dir",
        args.phase5_data_dir,
        "--hf-endpoint",
        args.hf_endpoint,
        "--use-safetensors",
        args.use_safetensors,
    ]
    add_optional(cmd, "--epochs", args.epochs)
    add_optional(cmd, "--max-length", args.max_length)
    add_optional(cmd, "--train-batch-size", args.train_batch_size)
    add_optional(cmd, "--eval-batch-size", args.eval_batch_size)
    add_optional(cmd, "--learning-rate", args.learning_rate)
    add_optional(cmd, "--weight-decay", args.weight_decay)
    add_optional(cmd, "--warmup-ratio", args.warmup_ratio)
    add_optional(cmd, "--patience", args.patience)
    add_optional(cmd, "--dataloader-num-workers", args.dataloader_num_workers)
    add_optional(cmd, "--gradient-accumulation-steps", args.gradient_accumulation_steps)
    add_optional(cmd, "--max-train-samples", args.max_train_samples)
    add_optional(cmd, "--max-val-samples", args.max_val_samples)
    add_optional(cmd, "--max-ws-test-samples", args.max_ws_test_samples)
    add_optional(cmd, "--input-field", args.input_field)
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def run_command(cmd: list[str], log_fp) -> int:
    log(f"[Command] {shlex.join(cmd)}", log_fp)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log(line, log_fp)
    return proc.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SciBERT and BioClinicalBERT Plan B baselines")
    parser.add_argument("--models", default="scibert,bioclinicalbert")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scibert-model", default=MODEL_DEFAULTS["scibert"])
    parser.add_argument("--bioclinicalbert-model", default=MODEL_DEFAULTS["bioclinicalbert"])
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-safetensors", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gate", default="g2")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--tag-template", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--input-field", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-ws-test-samples", type=int, default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    if args.smoke:
        if args.tag_template is None:
            args.tag_template = "smoke_extra_plm_seed{seed}"
        if args.epochs is None:
            args.epochs = 1
        if args.max_train_samples is None:
            args.max_train_samples = 2048
        if args.max_val_samples is None:
            args.max_val_samples = 1024
        if args.max_ws_test_samples is None:
            args.max_ws_test_samples = 1024
    elif args.tag_template is None:
        args.tag_template = "extra_plm_seed{seed}"
    return args


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models, list(MODEL_DEFAULTS), "models")
    tasks = parse_csv(args.tasks, TASKS, "tasks")
    seeds = parse_seed_csv(args.seeds)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if args.smoke else "full"
    log_path = Path(args.log) if args.log else log_dir / f"run_planb_extra_plm_baselines_{mode}_{int(time.time())}.log"
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str | int]] = []
    total = len(models) * len(tasks) * len(seeds)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] run_planb_extra_plm_baselines", log_fp)
        log(f"[Config] mode={mode} models={models} tasks={tasks} seeds={seeds}", log_fp)
        log(f"[Config] scibert_model={args.scibert_model}", log_fp)
        log(f"[Config] bioclinicalbert_model={args.bioclinicalbert_model}", log_fp)
        log("[Protocol] Phase5 test is never truncated by this wrapper; smoke limits train/val/WS-test only.", log_fp)

        idx = 0
        for seed in seeds:
            for model_key in models:
                for task in tasks:
                    idx += 1
                    tag = args.tag_template.format(seed=seed, model=model_key, task=task)
                    log(f"[Run {idx}/{total}] model={model_key} task={task} seed={seed} tag={tag}", log_fp)
                    cmd = build_command(args, model_key, task, seed, tag)
                    rc = run_command(cmd, log_fp)
                    if rc != 0:
                        item = {"model": model_key, "task": task, "seed": seed, "returncode": rc}
                        failures.append(item)
                        log(f"[Failure] {item}; continuing={not args.stop_on_error}", log_fp)
                        if args.stop_on_error:
                            raise SystemExit(rc)
                    else:
                        log(f"[Success] model={model_key} task={task} seed={seed}", log_fp)

        log(f"[Done] total={total} failures={len(failures)} log={log_path}", log_fp)
        for item in failures:
            log(f"[FailedRun] {item}", log_fp)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
