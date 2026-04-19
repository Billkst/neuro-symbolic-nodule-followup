#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import MWSTaskConfig, run_mws_task


if __name__ == "__main__":
    run_mws_task(
        MWSTaskConfig(
            task="density_stage2",
            label_field="density_stage2_label",
            label_names=["solid", "part_solid", "ground_glass", "calcified"],
            model_dir_name="density_stage2_mws_cfe",
            result_file_name="mws_cfe_density_stage2_results.json",
            primary_metric="macro_f1",
            weighted_loss=True,
            use_confidence_weight=True,
            input_field="section_aware_text",
        )
    )
