import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.eval.benchmark_runner import (
    collect_results,
    generate_summary_table,
    load_json_safe,
    run_step,
)


def _log(msg, log_fp):
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_config.yaml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    results_dir = Path(config["output"]["eval_results_dir"])
    comparisons_dir = Path(config["output"]["comparisons_dir"])
    manifests_dir = Path(config["output"]["manifests_dir"])
    phase4_dir = Path(config["output"]["phase4_dir"])

    for d in [results_dir, comparisons_dir, manifests_dir, phase4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs/run_phase4_benchmark.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    total_start = time.time()
    step_results = []

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_phase4_benchmark", log_fp)
        _log(f"[Config] config={args.config} nrows={args.nrows} skip_build={args.skip_build}", log_fp)
        _log(f"[Time] {datetime.now().isoformat()}", log_fp)

        if not args.skip_build:
            _log("\n" + "=" * 60, log_fp)
            _log("[Phase 1/5] Building eval set manifests...", log_fp)
            _log("=" * 60, log_fp)
            cmd = [python, "-u", "scripts/build_phase4_eval_sets.py", "--config", args.config]
            if args.nrows:
                cmd.extend(["--nrows", str(args.nrows)])
            step_results.append(run_step(cmd, "build_eval_sets", log_fp))

            if not step_results[-1]["success"]:
                _log("[FATAL] Eval set build failed. Aborting.", log_fp)
                return
        else:
            _log("[Skip] Eval set build skipped (--skip-build)", log_fp)

        _log("\n" + "=" * 60, log_fp)
        _log("[Phase 2/5] Running radiology evaluation...", log_fp)
        _log("=" * 60, log_fp)
        cmd = [python, "-u", "scripts/eval_radiology_baseline.py", "--config", args.config]
        step_results.append(run_step(cmd, "eval_radiology", log_fp))

        _log("\n" + "=" * 60, log_fp)
        _log("[Phase 3/5] Running smoking evaluation...", log_fp)
        _log("=" * 60, log_fp)
        cmd = [python, "-u", "scripts/eval_smoking_baseline.py", "--config", args.config]
        step_results.append(run_step(cmd, "eval_smoking", log_fp))

        _log("\n" + "=" * 60, log_fp)
        _log("[Phase 4/5] Running recommendation evaluation...", log_fp)
        _log("=" * 60, log_fp)
        cmd = [python, "-u", "scripts/eval_recommendation_baseline.py", "--config", args.config]
        step_results.append(run_step(cmd, "eval_recommendation", log_fp))

        _log("\n" + "=" * 60, log_fp)
        _log("[Phase 5/5] Collecting results and generating summary...", log_fp)
        _log("=" * 60, log_fp)

        collected = collect_results(results_dir, comparisons_dir)

        summary_path = phase4_dir / "benchmark_summary.json"
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": args.config,
                "steps": step_results,
                "results": collected,
            }, fp, ensure_ascii=False, indent=2, default=str)

        summary_text = generate_summary_table(collected)
        _log("\n" + summary_text, log_fp)

        summary_txt_path = phase4_dir / "benchmark_summary.txt"
        with summary_txt_path.open("w", encoding="utf-8") as fp:
            fp.write(summary_text)

        total_elapsed = time.time() - total_start
        success_count = sum(1 for s in step_results if s["success"])
        total_count = len(step_results)

        _log(f"\n[Summary] {success_count}/{total_count} steps succeeded", log_fp)
        _log(f"[Summary] Total time: {total_elapsed:.1f}s", log_fp)
        _log(f"[Summary] Results: {summary_path}", log_fp)
        _log(f"[Summary] Text summary: {summary_txt_path}", log_fp)

        manifest_files = list(manifests_dir.glob("*.json"))
        _log(f"[Summary] Manifests: {len(manifest_files)} files in {manifests_dir}", log_fp)
        for mf in manifest_files:
            m = load_json_safe(mf)
            if m:
                _log(f"  - {mf.name}: {m.get('selected_count', '?')} samples", log_fp)

        _log("[Done] run_phase4_benchmark completed", log_fp)


if __name__ == "__main__":
    main()
