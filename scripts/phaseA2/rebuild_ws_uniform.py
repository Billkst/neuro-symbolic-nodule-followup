#!/usr/bin/env python3
"""用 uniform weights 重新聚合 WS 数据，用于 A-agg 消融。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weak_supervision.aggregation import weighted_majority_vote
from src.weak_supervision.quality_gate import evaluate_gate
from src.weak_supervision.base import ABSTAIN


def log(msg: str) -> None:
    print(msg, flush=True)


def rebuild_with_uniform_weights(ws_path: Path, output_path: Path, task: str) -> int:
    """读取完整 WS 记录，用 uniform weights 重新聚合。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    lf_names_by_task = {
        "density": ["LF-D1", "LF-D2", "LF-D3", "LF-D4", "LF-D5"],
        "size": ["LF-S1", "LF-S2", "LF-S3", "LF-S4", "LF-S5"],
        "location": ["LF-L1", "LF-L2", "LF-L3", "LF-L4", "LF-L5"],
    }
    lf_names = lf_names_by_task[task]
    uniform_weights = {name: 1.0 for name in lf_names}

    with ws_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            lf_details = row.get("lf_details", [])
            if not lf_details:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                continue

            votes = {}
            for lf in lf_details:
                label = lf.get("label", ABSTAIN)
                if label != ABSTAIN:
                    votes[lf["lf_name"]] = label

            if not votes:
                row["ws_confidence"] = 0.0
                row["gate_level"] = "REJECTED"
                row["passed_gates"] = []
            else:
                agg = weighted_majority_vote(
                    {name: votes[name] for name in votes},
                    {name: 1.0 for name in votes},
                )
                row["ws_confidence"] = agg.confidence
                gate_result = evaluate_gate(agg)
                row["gate_level"] = gate_result.gate_level
                row["passed_gates"] = gate_result.passed_gates

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild WS data with uniform weights for A-agg ablation")
    parser.add_argument("--gate", type=str, default="g2")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ws_base = PROJECT_ROOT / "outputs" / "phaseA1"
    out_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2" / "ws_uniform"

    for task in ["density", "size", "location"]:
        task_ws_dir = ws_base / task
        task_out_dir = out_base / task
        task_out_dir.mkdir(parents=True, exist_ok=True)

        ws_full = task_ws_dir / "ws_train.jsonl"
        if not ws_full.exists():
            log(f"[Skip] {ws_full} not found")
            continue

        log(f"[Rebuild] {task} with uniform weights...")
        out_path = task_out_dir / f"{task}_train_ws_{args.gate}.jsonl"

        gate_filter = args.gate.upper()
        temp_all = task_out_dir / f"{task}_train_ws_all.jsonl"
        n = rebuild_with_uniform_weights(ws_full, temp_all, task)
        log(f"[Rebuild] {task}: {n} total records")

        count = 0
        with temp_all.open("r", encoding="utf-8") as fin, \
             out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                row = json.loads(line.strip())
                if gate_filter in row.get("passed_gates", []):
                    fout.write(line)
                    count += 1
        log(f"[Filter] {task} {gate_filter}: {count} samples -> {out_path}")

        for split in ["val", "test"]:
            src = task_ws_dir / f"{task}_{split}_ws.jsonl"
            dst = task_out_dir / f"{task}_{split}_ws.jsonl"
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
                log(f"[Copy] {task}/{split} -> {dst}")

        temp_all.unlink(missing_ok=True)

    log("[Done]")


if __name__ == "__main__":
    main()
