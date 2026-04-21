#!/usr/bin/env python3
"""Train cue-feature augmented MWS-CFE for Location.

This remains a learned model: location cues are encoded as input features and
the classifier predicts the label. No deterministic rule override is applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.feature_augmentation import add_location_cue_augmented_text
from scripts.phaseA2.train_mws_cfe_common import MWSTaskConfig, run_mws_task


if __name__ == "__main__":
    run_mws_task(
        MWSTaskConfig(
            task="location",
            label_field="location_label",
            label_names=["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"],
            model_dir_name="location_mws_cfe_augmented",
            result_file_name="mws_cfe_location_results.json",
            primary_metric="macro_f1",
            weighted_loss=True,
            input_field="cue_augmented_text",
            extra_label_names=["no_location"],
            row_transform=add_location_cue_augmented_text,
        )
    )
