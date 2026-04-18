#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import MWSTaskConfig
from scripts.phaseA2.train_vanilla_pubmedbert_common import run_vanilla_task

if __name__ == "__main__":
    run_vanilla_task(
        MWSTaskConfig(
            task="location",
            label_field="location_label",
            label_names=["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"],
            model_dir_name="location_vanilla_pubmedbert",
            result_file_name="vanilla_pubmedbert_location_results.json",
            primary_metric="macro_f1",
            weighted_loss=False,
            use_confidence_weight=False,
            extra_label_names=["no_location"],
        )
    )

