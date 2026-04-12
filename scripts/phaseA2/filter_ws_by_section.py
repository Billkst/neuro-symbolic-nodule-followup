#!/usr/bin/env python3
"""按 section 过滤 WS 训练数据，用于 A-section 消融。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["density", "size", "location"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def filter_by_section(input_path: Path, output_path: Path, allowed_sections: set[str]) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if row.get("section", "").lower() in allowed_sections:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count, total


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

    t_total = time.time()
    _log(f"[Start] filter_ws_by_section section={args.section} gate={args.gate}")
    _log(f"[Config] allowed_sections={allowed} ws_dir={ws_base} output_dir={out_base}")

    for task in TASKS:
        t0 = time.time()
        task_ws_dir = ws_base / task
        task_out_dir = out_base / task
        task_out_dir.mkdir(parents=True, exist_ok=True)

        train_in = task_ws_dir / f"{task}_train_ws_{args.gate}.jsonl"
        train_out = task_out_dir / f"{task}_train_ws_{args.gate}.jsonl"
        if train_in.exists():
            kept, total = filter_by_section(train_in, train_out, allowed)
            _log(f"[Filter] {task}/train: {kept}/{total} kept ({kept/total*100:.1f}%) -> {train_out}")
        else:
            _log(f"[Skip] {train_in} not found")

        for split in ["val", "test"]:
            src = task_ws_dir / f"{task}_{split}_ws.jsonl"
            dst = task_out_dir / f"{task}_{split}_ws.jsonl"
            if src.exists():
                kept, total = filter_by_section(src, dst, allowed)
                _log(f"[Filter] {task}/{split}: {kept}/{total} kept ({kept/total*100:.1f}%) -> {dst}")

        _log(f"  {task} done in {time.time() - t0:.1f}s")

    _log(f"[Done] total={time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
