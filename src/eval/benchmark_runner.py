import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _log(msg, log_fp=None):
    print(msg, flush=True)
    if log_fp:
        log_fp.write(msg + "\n")
        log_fp.flush()


def run_step(cmd: list[str], step_name: str, log_fp=None) -> dict:
    _log(f"[Step] {step_name}: {' '.join(cmd)}", log_fp)
    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    elapsed = time.time() - start
    success = result.returncode == 0

    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            _log(f"  {line}", log_fp)
    if not success and result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            _log(f"  [stderr] {line}", log_fp)

    _log(f"  -> {'OK' if success else 'FAILED'} ({elapsed:.1f}s)", log_fp)
    return {
        "step": step_name,
        "command": " ".join(cmd),
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
        "returncode": result.returncode,
    }


def load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_results(results_dir: Path, comparisons_dir: Path) -> dict:
    collected = {
        "radiology": {},
        "smoking": {},
        "recommendation": {},
        "comparisons": {},
    }

    for f in sorted(results_dir.glob("radiology_metrics_*.json")):
        variant = f.stem.replace("radiology_metrics_", "")
        collected["radiology"][variant] = load_json_safe(f)

    for f in sorted(results_dir.glob("smoking_metrics_*.json")):
        variant = f.stem.replace("smoking_metrics_", "")
        collected["smoking"][variant] = load_json_safe(f)

    for f in sorted(results_dir.glob("recommendation_metrics_*.json")):
        variant = f.stem.replace("recommendation_metrics_", "")
        collected["recommendation"][variant] = load_json_safe(f)

    for f in sorted(comparisons_dir.glob("*.json")):
        collected["comparisons"][f.stem] = load_json_safe(f)

    return collected


def _fmt_val(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val) if val is not None else "N/A"


def _table_section(title: str, variants: list[str], metrics_map: dict, keys: list[str]) -> list[str]:
    lines = []
    lines.append(f"\n## {title}")
    if not variants:
        lines.append("  (no results)")
        return lines

    header = f"{'Metric':<40}"
    for v in variants:
        header += f" | {v:<25}"
    lines.append(header)
    lines.append("-" * len(header))

    for key in keys:
        row = f"{key:<40}"
        for v in variants:
            m = metrics_map.get(v) or {}
            row += f" | {_fmt_val(m.get(key, 'N/A')):<25}"
        lines.append(row)

    return lines


def generate_summary_table(collected: dict) -> str:
    lines = ["=" * 80, "PHASE 4 BENCHMARK SUMMARY", "=" * 80]

    rad_variants = list(collected["radiology"].keys())
    lines.extend(_table_section(
        "Radiology Baseline Results", rad_variants, collected["radiology"],
        ["nodule_detect_rate", "size_mm_extract_rate", "density_category_extract_rate",
         "location_lobe_extract_rate", "change_status_extract_rate",
         "recommendation_cue_extract_rate", "total_nodules", "avg_nodules_per_note"],
    ))

    smk_variants = list(collected["smoking"].keys())
    lines.extend(_table_section(
        "Smoking Baseline Results", smk_variants, collected["smoking"],
        ["non_unknown_rate", "ever_smoker_rate", "eligible_rate",
         "fallback_trigger_rate", "pack_year_parse_rate"],
    ))

    rec_variants = list(collected["recommendation"].keys())
    lines.extend(_table_section(
        "Recommendation Baseline Results", rec_variants, collected["recommendation"],
        ["actionable_rate", "monitoring_rate", "insufficient_data_rate",
         "guideline_anchor_presence_rate", "reasoning_path_nonempty_rate"],
    ))

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
