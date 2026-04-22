#!/usr/bin/env python3
"""Train Wave2 cue-feature augmented MWS-CFE for Has-size.

This is still a learned classifier. Size cues are encoded into the input text;
no deterministic rule override is used at prediction time.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import add_size_cue_augmented_text
from scripts.phaseA2.train_mws_cfe_common import MWSTaskConfig, run_mws_task


if __name__ == "__main__":
    run_mws_task(
        MWSTaskConfig(
            task="size",
            label_field="has_size",
            label_names=["no_size", "has_size"],
            model_dir_name="size_mws_cfe_augmented_v2",
            result_file_name="mws_cfe_size_results.json",
            primary_metric="f1",
            weighted_loss=True,
            input_field="cue_augmented_text",
            row_transform=add_size_cue_augmented_text,
        )
    )
