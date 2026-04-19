#!/usr/bin/env python3
from __future__ import annotations

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
            task="density_stage1",
            label_field="density_stage1_label",
            label_names=["explicit_density", "unclear_or_no_evidence"],
            model_dir_name="density_stage1_vanilla_pubmedbert",
            result_file_name="vanilla_pubmedbert_density_stage1_results.json",
            primary_metric="auprc",
            weighted_loss=False,
            use_confidence_weight=False,
            input_field="section_aware_text",
        )
    )
