from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "scripts" / "phase5" / "train_pubmedbert_common.py"
SPEC = importlib.util.spec_from_file_location("_phase5_train_pubmedbert_common", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载模块: {MODULE_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TaskConfig = MODULE.TaskConfig
run_task = MODULE.run_task
