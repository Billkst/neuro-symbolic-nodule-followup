#!/usr/bin/env python3
"""Run Wave2 Has-size augmented learned-model grid.

The runner executes or prints a compact smoke/full grid. All candidates remain
learned models: cue features are appended to the input, and predictions are
made by the classifier.
"""
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/train_mws_size_augmented_v2.py"

DEFAULT_SMOKE_COMBOS = [
    ("g2", "ce", 128),
    ("g3", "ce", 128),
    ("g2", "focal", 128),
    ("g3", "focal", 128),
    ("g2", "ce", 192),
    ("g3", "ce", 192),
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def build_combos(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    if args.preset == "smoke":
        return list(DEFAULT_SMOKE_COMBOS)
    gates = parse_csv(args.gates)
    losses = parse_csv(args.loss_types)
    lengths = parse_int_csv(args.lengths)
    return [(gate, loss, length) for gate, loss, length in itertools.product(gates, losses, lengths)]


def build_tag(gate: str, loss_type: str, max_length: int, weighting: str, seed: int) -> str:
    weight_tag = "wcls" if weighting == "weighted" else "nocls"
    return f"size_aug_v2_{gate}_{loss_type}_{weight_tag}_len{max_length}_seed{seed}"


def build_command(
    args: argparse.Namespace,
    *,
    seed: int,
    gate: str,
    loss_type: str,
    max_length: int,
    weighting: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
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
        build_tag(gate, loss_type, max_length, weighting, seed),
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
    if weighting == "unweighted":
        cmd.append("--no-class-weight")
    if args.no_confidence_weight:
        cmd.append("--no-confidence-weight")
    return cmd


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wave2 Has-size augmented learned-model grid")
    parser.add_argument("--preset", choices=["smoke", "custom"], default="smoke")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gates", default="g2,g3")
    parser.add_argument("--loss-types", default="ce,focal")
    parser.add_argument("--lengths", default="128,192")
    parser.add_argument("--class-weighting", choices=["weighted", "unweighted", "both"], default="weighted")
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
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-confidence-weight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds)
    combos = build_combos(args)
    weightings = ["weighted", "unweighted"] if args.class_weighting == "both" else [args.class_weighting]

    for seed in seeds:
        for gate, loss_type, max_length in combos:
            for weighting in weightings:
                cmd = build_command(
                    args,
                    seed=seed,
                    gate=gate,
                    loss_type=loss_type,
                    max_length=max_length,
                    weighting=weighting,
                )
                run_or_print(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
