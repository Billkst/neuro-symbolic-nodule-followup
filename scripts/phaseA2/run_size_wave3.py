#!/usr/bin/env python3
"""Run Has-size Wave3 local-input + validation threshold tuning.

Fixed smoke grid:
1. g2 + focal + len192 + mention_text
2. g2 + focal + len192 + cue_augmented_mention_text
3. g2 + ce + len192 + mention_text
4. g2 + ce + len192 + cue_augmented_mention_text

Each combo trains a raw learned model, then emits a tuned JSON compatible with
the Plan B aggregator via tune_size_threshold.py.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/train_mws_size_augmented_v3.py"
TUNE_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/tune_size_threshold.py"

WAVE3_COMBOS = [
    ("g2", "focal", 192, "mention_text"),
    ("g2", "focal", 192, "cue_augmented_mention_text"),
    ("g2", "ce", 192, "mention_text"),
    ("g2", "ce", 192, "cue_augmented_mention_text"),
]


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mode_tag(size_input_mode: str) -> str:
    return "cue_mention" if size_input_mode == "cue_augmented_mention_text" else "mention"


def final_tag(gate: str, loss_type: str, max_length: int, size_input_mode: str, class_weighting: str) -> str:
    weight_tag = "wcls" if class_weighting == "weighted" else "nocls"
    return f"size_wave3_{gate}_{loss_type}_{weight_tag}_len{max_length}_{mode_tag(size_input_mode)}"


def build_train_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    gate: str,
    loss_type: str,
    max_length: int,
    size_input_mode: str,
    tag: str,
) -> list[str]:
    raw_tag = f"{tag}_raw_seed{seed}"
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--size-input-mode",
        size_input_mode,
        "--seed",
        str(seed),
        "--gate",
        gate,
        "--max-length",
        str(max_length),
        "--loss-type",
        loss_type,
        "--focal-gamma",
        str(args.focal_gamma),
        "--output-dir",
        args.output_dir,
        "--tag",
        raw_tag,
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--patience",
        str(args.patience),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
    ]
    if args.ws_data_dir:
        cmd.extend(["--ws-data-dir", args.ws_data_dir])
    if args.phase5_data_dir:
        cmd.extend(["--phase5-data-dir", args.phase5_data_dir])
    if args.gradient_accumulation_steps != 1:
        cmd.extend(["--gradient-accumulation-steps", str(args.gradient_accumulation_steps)])
    if args.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_test_samples is not None:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    if args.class_weighting == "unweighted":
        cmd.append("--no-class-weight")
    if args.no_confidence_weight:
        cmd.append("--no-confidence-weight")
    return cmd


def build_tune_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    max_length: int,
    size_input_mode: str,
    tag: str,
) -> list[str]:
    raw_tag = f"{tag}_raw_seed{seed}"
    model_dir = Path(args.output_dir) / "models" / f"size_mws_cfe_augmented_v3_{raw_tag}"
    cmd = [
        sys.executable,
        "-u",
        str(TUNE_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--data-dir",
        args.ws_data_dir or "outputs/phaseA1/size",
        "--phase5-data-dir",
        args.phase5_data_dir or "outputs/phase5/datasets",
        "--output-dir",
        str(Path(args.output_dir) / "results"),
        "--tag",
        tag,
        "--seed",
        str(seed),
        "--size-input-mode",
        size_input_mode,
        "--max-length",
        str(max_length),
        "--batch-size",
        str(args.eval_batch_size),
        "--calibration",
        args.calibration,
    ]
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_test_samples is not None:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    return cmd


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Has-size Wave3 local-input threshold-tuning grid")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--ws-data-dir", default=None)
    parser.add_argument("--phase5-data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weighting", choices=["unweighted", "weighted"], default="unweighted")
    parser.add_argument("--calibration", choices=["platt", "none"], default="platt")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-confidence-weight", action="store_true")
    parser.add_argument("--steps", default="train,tune", help="Comma-separated subset: train,tune")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds)
    steps = {step.strip() for step in args.steps.split(",") if step.strip()}

    for seed in seeds:
        for gate, loss_type, max_length, size_input_mode in WAVE3_COMBOS:
            tag = final_tag(gate, loss_type, max_length, size_input_mode, args.class_weighting)
            if "train" in steps:
                run_or_print(
                    build_train_cmd(
                        args,
                        seed=seed,
                        gate=gate,
                        loss_type=loss_type,
                        max_length=max_length,
                        size_input_mode=size_input_mode,
                        tag=tag,
                    ),
                    dry_run=args.dry_run,
                )
            if "tune" in steps:
                run_or_print(
                    build_tune_cmd(
                        args,
                        seed=seed,
                        max_length=max_length,
                        size_input_mode=size_input_mode,
                        tag=tag,
                    ),
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
