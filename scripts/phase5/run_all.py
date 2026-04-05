"""Phase 5 全流程串联脚本。

按顺序执行：
1. 数据集构建
2. Regex + ML baselines
3. PubMedBERT 训练（density → size → location）
4. 深度评测分析
5. 消融实验
6. 参数讨论实验
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "phase5"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("build_datasets", "build_datasets.py", []),
    ("run_baselines", "run_baselines.py", []),
    ("train_density", "train_density.py", []),
    ("train_size", "train_size.py", []),
    ("train_location", "train_location.py", []),
    ("run_deep_analysis", "run_deep_analysis.py", []),
    ("run_ablation", "run_ablation.py", []),
    ("run_param_sweep", "run_param_sweep.py", []),
]


def run_step(name: str, script: str, extra_args: list[str]) -> bool:
    script_path = SCRIPTS_DIR / script
    log_path = LOGS_DIR / f"{name}.log"
    cmd = [sys.executable, "-u", str(script_path)] + extra_args

    print(f"\n{'='*60}", flush=True)
    print(f"[run_all] START: {name}", flush=True)
    print(f"[run_all] Script: {script_path}", flush=True)
    print(f"[run_all] Log: {log_path}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    with open(log_path, "w", buffering=1) as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
        proc.wait()

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"[run_all] {name}: {status} ({elapsed:.1f}s)", flush=True)
    return ok


def main() -> None:
    print(f"[run_all] Phase 5 全流程开始", flush=True)
    print(f"[run_all] 共 {len(STEPS)} 个步骤", flush=True)

    t_total = time.time()
    results: list[tuple[str, bool, float]] = []

    for name, script, args in STEPS:
        t0 = time.time()
        ok = run_step(name, script, args)
        elapsed = time.time() - t0
        results.append((name, ok, elapsed))
        if not ok:
            print(f"\n[run_all] ABORT: {name} 失败，终止后续步骤", flush=True)
            break

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"[run_all] Phase 5 全流程完成 ({total_elapsed:.1f}s)", flush=True)
    print(f"{'='*60}", flush=True)
    for name, ok, elapsed in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {elapsed:.1f}s", flush=True)

    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n[run_all] {len(failed)} 个步骤失败", flush=True)
        sys.exit(1)
    else:
        print(f"\n[run_all] 全部通过", flush=True)


if __name__ == "__main__":
    main()
