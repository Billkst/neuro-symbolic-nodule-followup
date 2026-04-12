#!/usr/bin/env python3
"""按 section 过滤 WS 训练数据，用于 A-section 消融。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["density", "size", "location"]


def log(msg: str) -> None:
    print(msg, flush=True)


def filter_by_section(input_path: Path, output_path: Path, allowed_sections: set[str]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("section", "").lower() in allowed_sections:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter WS data by section for A-section ablation")
    parser.add_argument("--section", type=str, required=True,
                        choices=["findings", "impression", "findings_impression"],
                        help="Section filter strategy")
    parser.add_argument("--gate", type=str, default="g2")
    parser.add_argument("--ws-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    section_map = {
        "findings": {"findings"},
        "impression": {"impression"},
        "findings_impression": {"findings", "impression"},
    }
    allowed = section_map[args.section]

    ws_base = Path(args.ws_dir) if args.ws_dir else PROJECT_ROOT / "outputs" / "phaseA1"
    out_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2" / f"ws_{args.section}"

    for task in TASKS:
        task_ws_dir = ws_base / task
        task_out_dir = out_base / task
        task_out_dir.mkdir(parents=True, exist_ok=True)

        train_in = task_ws_dir / f"{task}_train_ws_{args.gate}.jsonl"
        train_out = task_out_dir / f"{task}_train_ws_{args.gate}.jsonl"
        if train_in.exists():
            n = filter_by_section(train_in, train_out, allowed)
            log(f"[Filter] {task}/train {args.section}: {n} samples -> {train_out}")

        for split in ["val", "test"]:
            src = task_ws_dir / f"{task}_{split}_ws.jsonl"
            dst = task_out_dir / f"{task}_{split}_ws.jsonl"
            if src.exists():
                n = filter_by_section(src, dst, allowed)
                log(f"[Filter] {task}/{split} {args.section}: {n} samples -> {dst}")

    log("[Done]")


if __name__ == "__main__":
    main()
