#!/usr/bin/env python3
"""Run Has-size Wave4 Phase5-like-dev threshold tuning.

Fixed candidates:
- g2 + focal + len192 + mention_text
- g2 + ce + len192 + mention_text

Each raw model is tuned against a Phase5-like non-test dev split using both
Platt and no calibration by default.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DEV_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/build_size_phase5_like_dev.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/train_mws_size_augmented_v3.py"
TUNE_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/tune_size_threshold_v2.py"

WAVE4_COMBOS = [
    ("g2", "focal", 192, "mention_text"),
    ("g2", "ce", 192, "mention_text"),
]


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def raw_tag(gate: str, loss_type: str, max_length: int, seed: int, class_weighting: str) -> str:
    weight_tag = "wcls" if class_weighting == "weighted" else "nocls"
    return f"size_wave4_{gate}_{loss_type}_{weight_tag}_len{max_length}_mention_raw_seed{seed}"


def final_tag(gate: str, loss_type: str, max_length: int, calibration: str, class_weighting: str) -> str:
    weight_tag = "wcls" if class_weighting == "weighted" else "nocls"
    return f"size_wave4_{gate}_{loss_type}_{weight_tag}_len{max_length}_mention_phase5dev_{calibration}"


def build_dev_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(BUILD_DEV_SCRIPT),
        "--phase5-data-dir",
        args.phase5_data_dir,
        "--ws-data-dir",
        args.ws_data_dir,
        "--output",
        args.selection_split,
        "--metadata",
        args.selection_metadata,
        "--seed",
        str(args.seed_for_dev),
    ]
    if args.dev_max_samples is not None:
        cmd.extend(["--max-samples", str(args.dev_max_samples)])
    return cmd


def train_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    gate: str,
    loss_type: str,
    max_length: int,
) -> list[str]:
    tag = raw_tag(gate, loss_type, max_length, seed, args.class_weighting)
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--size-input-mode",
        "mention_text",
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
        tag,
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
        "--ws-data-dir",
        args.ws_data_dir,
        "--phase5-data-dir",
        args.phase5_data_dir,
    ]
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


def tune_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    gate: str,
    loss_type: str,
    max_length: int,
    calibration: str,
) -> list[str]:
    raw = raw_tag(gate, loss_type, max_length, seed, args.class_weighting)
    model_dir = Path(args.output_dir) / "models" / f"size_mws_cfe_augmented_v3_{raw}"
    cmd = [
        sys.executable,
        "-u",
        str(TUNE_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--selection-split",
        args.selection_split,
        "--selection-source",
        args.selection_source,
        "--ws-test-split",
        str(Path(args.ws_data_dir) / "size_test_ws.jsonl"),
        "--phase5-test-split",
        str(Path(args.phase5_data_dir) / "size_test.jsonl"),
        "--output-dir",
        str(Path(args.output_dir) / "results"),
        "--tag",
        final_tag(gate, loss_type, max_length, calibration, args.class_weighting),
        "--seed",
        str(seed),
        "--size-input-mode",
        "mention_text",
        "--max-length",
        str(max_length),
        "--batch-size",
        str(args.eval_batch_size),
        "--calibration",
        calibration,
    ]
    if args.max_selection_samples is not None:
        cmd.extend(["--max-selection-samples", str(args.max_selection_samples)])
    if args.max_test_samples is not None:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    return cmd


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Has-size Wave4 Phase5-like-dev tuning")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--calibrations", default="platt,none")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--ws-data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--selection-split", default="outputs/phaseA2_planB/size_wave4/size_phase5_like_dev.jsonl")
    parser.add_argument("--selection-metadata", default="outputs/phaseA2_planB/size_wave4/size_phase5_like_dev_meta.json")
    parser.add_argument("--selection-source", default="phase5_like_dev")
    parser.add_argument("--seed-for-dev", type=int, default=42)
    parser.add_argument("--dev-max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weighting", choices=["unweighted", "weighted"], default="unweighted")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-selection-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-confidence-weight", action="store_true")
    parser.add_argument("--steps", default="build_dev,train,tune", help="Comma-separated subset: build_dev,train,tune")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    steps = {step.strip() for step in args.steps.split(",") if step.strip()}
    seeds = parse_int_csv(args.seeds)
    calibrations = parse_csv(args.calibrations)

    if "build_dev" in steps:
        run_or_print(build_dev_cmd(args), dry_run=args.dry_run)

    for seed in seeds:
        for gate, loss_type, max_length, _input_mode in WAVE4_COMBOS:
            if "train" in steps:
                run_or_print(
                    train_cmd(args, seed=seed, gate=gate, loss_type=loss_type, max_length=max_length),
                    dry_run=args.dry_run,
                )
            if "tune" in steps:
                for calibration in calibrations:
                    run_or_print(
                        tune_cmd(
                            args,
                            seed=seed,
                            gate=gate,
                            loss_type=loss_type,
                            max_length=max_length,
                            calibration=calibration,
                        ),
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    main()
