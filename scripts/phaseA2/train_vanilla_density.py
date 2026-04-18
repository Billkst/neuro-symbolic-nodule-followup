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
            task="density",
            label_field="density_label",
            label_names=["solid", "part_solid", "ground_glass", "calcified", "unclear"],
            model_dir_name="density_vanilla_pubmedbert",
            result_file_name="vanilla_pubmedbert_density_results.json",
            primary_metric="macro_f1",
            weighted_loss=False,
            use_confidence_weight=False,
        )
    )

