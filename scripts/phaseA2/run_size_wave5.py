#!/usr/bin/env python3
"""Run Has-size Wave5 learned lexical-calibrated smoke candidates.

Wave5 only targets Has-size. Phase5 test is always evaluated in full; this
runner intentionally has no max-test-samples argument.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DEV_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/build_size_phase5_like_dev.py"
LEXICAL_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/train_size_lexical_expert.py"
STACKED_SCRIPT = PROJECT_ROOT / "scripts/phaseA2/train_size_stacked_head.py"

STACKED_CANDIDATES = [
    ("lexical_bert_lr", "size_wave5_lexical_bert_lr"),
    ("lexical_bert_cue_lr", "size_wave5_lexical_bert_cue_lr"),
    ("lexical_cue_lr", "size_wave5_lexical_cue_lr"),
]


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_bert_model_dir(output_dir: str, seed: int) -> str:
    return str(
        Path(output_dir)
        / "models"
        / f"size_mws_cfe_augmented_v3_size_wave4_g2_ce_nocls_len192_mention_raw_seed{seed}"
    )


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


def lexical_tag(args: argparse.Namespace) -> str:
    return args.lexical_tag


def lexical_manifest_path(args: argparse.Namespace, seed: int) -> str:
    return str(Path(args.probability_dir) / f"{lexical_tag(args)}_seed{seed}_manifest.json")


def lexical_cmd(args: argparse.Namespace, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(LEXICAL_SCRIPT),
        "--seed",
        str(seed),
        "--gate",
        args.gate,
        "--ws-data-dir",
        args.ws_data_dir,
        "--phase5-data-dir",
        args.phase5_data_dir,
        "--selection-split",
        args.selection_split,
        "--selection-source",
        args.selection_source,
        "--output-dir",
        args.output_dir,
        "--probability-dir",
        args.probability_dir,
        "--tag",
        lexical_tag(args),
        "--input-field",
        "mention_text",
        "--max-features",
        str(args.lexical_max_features),
        "--ngram-max",
        str(args.lexical_ngram_max),
        "--c",
        str(args.lexical_c),
        "--max-iter",
        str(args.max_iter),
    ]
    if args.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_selection_samples is not None:
        cmd.extend(["--max-selection-samples", str(args.max_selection_samples)])
    if args.use_confidence_weight:
        cmd.append("--use-confidence-weight")
    return cmd


def stacked_cmd(args: argparse.Namespace, *, seed: int, candidate: str, tag: str) -> list[str]:
    bert_model = args.bert_model_dir or default_bert_model_dir(args.output_dir, seed)
    cmd = [
        sys.executable,
        "-u",
        str(STACKED_SCRIPT),
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--gate",
        args.gate,
        "--tag",
        tag,
        "--lexical-prob-manifest",
        lexical_manifest_path(args, seed),
        "--selection-split",
        args.selection_split,
        "--selection-source",
        args.selection_source,
        "--ws-val-split",
        str(Path(args.ws_data_dir) / "size_val_ws.jsonl"),
        "--ws-test-split",
        str(Path(args.ws_data_dir) / "size_test_ws.jsonl"),
        "--phase5-test-split",
        str(Path(args.phase5_data_dir) / "size_test.jsonl"),
        "--output-dir",
        args.output_dir,
        "--meta-val-fraction",
        str(args.meta_val_fraction),
        "--c",
        str(args.meta_c),
        "--max-iter",
        str(args.max_iter),
        "--bert-input-field",
        "mention_text",
        "--bert-max-length",
        str(args.bert_max_length),
        "--batch-size",
        str(args.eval_batch_size),
    ]
    if "bert" in candidate:
        cmd.extend(["--bert-model-dir", bert_model])
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_selection_samples is not None:
        cmd.extend(["--max-selection-samples", str(args.max_selection_samples)])
    return cmd


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Has-size Wave5 learned lexical-calibrated candidates")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gate", default="g2")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--ws-data-dir", default="outputs/phaseA1/size")
    parser.add_argument("--selection-split", default="outputs/phaseA2_planB/size_wave5/size_phase5_like_dev.jsonl")
    parser.add_argument("--selection-metadata", default="outputs/phaseA2_planB/size_wave5/size_phase5_like_dev_meta.json")
    parser.add_argument("--selection-source", default="phase5_like_dev")
    parser.add_argument("--seed-for-dev", type=int, default=42)
    parser.add_argument("--dev-max-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-selection-samples", type=int, default=None)
    parser.add_argument("--lexical-tag", default="size_wave5_lexical_alone")
    parser.add_argument("--probability-dir", default="outputs/phaseA2_planB/size_wave5/probabilities")
    parser.add_argument("--lexical-max-features", type=int, default=20000)
    parser.add_argument("--lexical-ngram-max", type=int, default=2)
    parser.add_argument("--lexical-c", type=float, default=1.0)
    parser.add_argument("--use-confidence-weight", action="store_true")
    parser.add_argument("--bert-model-dir", default=None)
    parser.add_argument("--bert-max-length", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--meta-val-fraction", type=float, default=0.4)
    parser.add_argument("--meta-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--candidates", default="lexical_alone,lexical_bert_lr,lexical_bert_cue_lr,lexical_cue_lr")
    parser.add_argument("--steps", default="build_dev,lexical,stacked", help="Comma-separated subset: build_dev,lexical,stacked")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if hasattr(args, "max_test_samples"):
        raise ValueError("Wave5 runner must not accept max-test-samples")

    seeds = parse_int_csv(args.seeds)
    steps = set(parse_csv(args.steps))
    requested_candidates = set(parse_csv(args.candidates))
    valid_candidates = {"lexical_alone", *[name for name, _tag in STACKED_CANDIDATES]}
    invalid = sorted(requested_candidates - valid_candidates)
    if invalid:
        raise ValueError(f"Invalid candidates: {invalid}; valid={sorted(valid_candidates)}")

    if "build_dev" in steps:
        run_or_print(build_dev_cmd(args), dry_run=args.dry_run)

    for seed in seeds:
        if "lexical" in steps:
            run_or_print(lexical_cmd(args, seed), dry_run=args.dry_run)
        elif "stacked" in steps:
            manifest = Path(lexical_manifest_path(args, seed))
            if not args.dry_run and not (PROJECT_ROOT / manifest).exists() and not manifest.exists():
                raise FileNotFoundError(
                    f"Missing lexical manifest for stacked candidates: {manifest}. "
                    "Run with --steps lexical first or include lexical_alone in --candidates."
                )

        if "stacked" in steps:
            for candidate, tag in STACKED_CANDIDATES:
                if candidate not in requested_candidates:
                    continue
                run_or_print(stacked_cmd(args, seed=seed, candidate=candidate, tag=tag), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
