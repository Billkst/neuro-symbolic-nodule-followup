#!/usr/bin/env python3
"""Train Wave3 Has-size learned models with local inputs.

Supported input modes:
- mention_text
- cue_augmented_mention_text

Cue augmentation only appends symbolic features to the local mention text. The
classifier remains the sole decision maker; no deterministic rule override is
used.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import add_size_cue_augmented_mention_text
from scripts.phaseA2.train_mws_cfe_common import MWSTaskConfig, run_mws_task

INPUT_MODES = {"mention_text", "cue_augmented_mention_text"}


def pop_size_input_mode(argv: list[str]) -> tuple[str, list[str]]:
    mode = "mention_text"
    filtered: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--size-input-mode":
            if idx + 1 >= len(argv):
                raise SystemExit("--size-input-mode requires a value")
            mode = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--size-input-mode="):
            mode = arg.split("=", 1)[1]
            idx += 1
            continue
        filtered.append(arg)
        idx += 1
    if mode not in INPUT_MODES:
        raise SystemExit(f"--size-input-mode must be one of {sorted(INPUT_MODES)}")
    return mode, filtered


if __name__ == "__main__":
    size_input_mode, remaining = pop_size_input_mode(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    row_transform = None
    if size_input_mode == "cue_augmented_mention_text":
        row_transform = add_size_cue_augmented_mention_text

    run_mws_task(
        MWSTaskConfig(
            task="size",
            label_field="has_size",
            label_names=["no_size", "has_size"],
            model_dir_name="size_mws_cfe_augmented_v3",
            result_file_name="mws_cfe_size_results.json",
            primary_metric="f1",
            weighted_loss=True,
            input_field=size_input_mode,
            row_transform=row_transform,
        )
    )
