#!/usr/bin/env python3
"""Run final Plan B density combo candidates.

Candidates:
- G3 + max_length 128 + validation calibration/threshold tuning for Stage 1
- G3 + max_length 192 + validation calibration/threshold tuning for Stage 1

Stage 2 uses the same G3/max_length candidate. The emitted JSON names remain
compatible with aggregate_planb_results.py:

    mws_cfe_density_stage1_results_<tag>_seed<seed>.json
    mws_cfe_density_stage2_results_<tag>_seed<seed>.json

The Stage 1 raw training JSON is kept under a ``*_raw`` tag; the tuned Stage 1
JSON uses the final tag.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEEDS = [13, 42, 87, 3407, 31415]
DEFAULT_LENGTHS = [128, 192]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_base_train_args(args: argparse.Namespace, *, max_length: int, seed: int, tag: str) -> list[str]:
    cmd = [
        "--seed",
        str(seed),
        "--gate",
        args.gate,
        "--max-length",
        str(max_length),
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
        "--output-dir",
        args.output_dir,
        "--input-field",
        args.input_field,
        "--tag",
        tag,
    ]
    if args.gradient_accumulation_steps != 1:
        cmd.extend(["--gradient-accumulation-steps", str(args.gradient_accumulation_steps)])
    if args.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_test_samples is not None:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    if args.no_confidence_weight:
        cmd.append("--no-confidence-weight")
    if args.loss_type != "ce":
        cmd.extend(["--loss-type", args.loss_type, "--focal-gamma", str(args.focal_gamma)])
    return cmd


def density_stage1_train_cmd(args: argparse.Namespace, *, max_length: int, seed: int, raw_tag: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/phaseA2/train_mws_density_stage1.py"),
        *build_base_train_args(args, max_length=max_length, seed=seed, tag=f"{raw_tag}_seed{seed}"),
        "--ws-data-dir",
        str(Path(args.data_dir) / "density_stage1"),
        "--phase5-data-dir",
        str(Path(args.data_dir) / "density_stage1"),
    ]


def density_stage1_tune_cmd(args: argparse.Namespace, *, max_length: int, seed: int, final_tag: str, raw_tag: str) -> list[str]:
    model_dir = Path(args.output_dir) / "models" / f"density_stage1_mws_cfe_{raw_tag}_seed{seed}"
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/phaseA2/tune_planb_stage1_threshold.py"),
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(Path(args.data_dir) / "density_stage1"),
        "--phase5-data-dir",
        str(Path(args.data_dir) / "density_stage1"),
        "--output-dir",
        str(Path(args.output_dir) / "results"),
        "--tag",
        final_tag,
        "--seed",
        str(seed),
        "--input-field",
        args.input_field,
        "--max-length",
        str(max_length),
        "--batch-size",
        str(args.eval_batch_size),
    ]


def density_stage2_train_cmd(args: argparse.Namespace, *, max_length: int, seed: int, final_tag: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/phaseA2/train_mws_density_stage2.py"),
        *build_base_train_args(args, max_length=max_length, seed=seed, tag=f"{final_tag}_seed{seed}"),
        "--ws-data-dir",
        str(Path(args.data_dir) / "density_stage2"),
        "--phase5-data-dir",
        str(Path(args.data_dir) / "density_stage2"),
    ]


def command_label(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def maybe_run(cmd: list[str], *, dry_run: bool) -> None:
    print(command_label(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final density G3+length combo candidates")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--lengths", default=",".join(str(length) for length in DEFAULT_LENGTHS))
    parser.add_argument("--gate", default="g3")
    parser.add_argument("--data-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--input-field", default="section_aware_text")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-confidence-weight", action="store_true")
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--steps",
        default="stage1,stage1_tune,stage2",
        help="Comma-separated subset: stage1,stage1_tune,stage2",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    lengths = parse_int_list(args.lengths)
    steps = {step.strip() for step in args.steps.split(",") if step.strip()}

    for max_length in lengths:
        final_tag = f"density_final_g3_len{max_length}"
        raw_tag = f"{final_tag}_raw"
        for seed in seeds:
            if "stage1" in steps:
                maybe_run(
                    density_stage1_train_cmd(args, max_length=max_length, seed=seed, raw_tag=raw_tag),
                    dry_run=args.dry_run,
                )
            if "stage1_tune" in steps:
                maybe_run(
                    density_stage1_tune_cmd(args, max_length=max_length, seed=seed, final_tag=final_tag, raw_tag=raw_tag),
                    dry_run=args.dry_run,
                )
            if "stage2" in steps:
                maybe_run(
                    density_stage2_train_cmd(args, max_length=max_length, seed=seed, final_tag=final_tag),
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
