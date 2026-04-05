import sys
from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMMON_PATH = PROJECT_ROOT / "scripts" / "phase5" / "train_pubmedbert_common.py"
COMMON_SPEC = importlib.util.spec_from_file_location("_phase5_location_common", COMMON_PATH)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise ImportError(f"无法加载模块: {COMMON_PATH}")
COMMON_MODULE = importlib.util.module_from_spec(COMMON_SPEC)
sys.modules[COMMON_SPEC.name] = COMMON_MODULE
COMMON_SPEC.loader.exec_module(COMMON_MODULE)

TaskConfig = COMMON_MODULE.TaskConfig
run_task = COMMON_MODULE.run_task


if __name__ == "__main__":
    run_task(
        TaskConfig(
            task="location",
            label_field="location_label",
            label_names=["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear", "no_location"],
            model_dir_name="location_pubmedbert",
            result_file_name="pubmedbert_location_results.json",
            primary_metric="macro_f1",
            weighted_loss=True,
        )
    )
