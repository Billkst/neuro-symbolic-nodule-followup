#!/usr/bin/env python3
"""Phase A2 消融实验与参数讨论统一入口。

支持 5 组消融 (A-lf, A-qg, A-window, A-agg, A-section) 和 3 组参数讨论 (P1, P2, P3)。
每个实验通过 --experiment 参数指定，内部调用 train_mws_cfe_common 的训练框架。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "a_lf": {
        "description": "A-lf: single-source vs multi-source weak supervision",
        "note": "single-source = Phase5 旧数据 (Regex teacher), multi-source = Phase A1 WS G2",
    },
    "a_qg": {
        "description": "A-qg: quality gate strength (G1-G5)",
        "variants": ["g1", "g2", "g3", "g4", "g5"],
    },
    "a_window": {
        "description": "A-window: mention_text vs full_text input",
        "variants": ["mention_text", "full_text"],
    },
    "a_agg": {
        "description": "A-agg: weighted vote vs uniform vote",
        "note": "需要先用 uniform vote 重新生成 WS 数据",
    },
    "a_section": {
        "description": "A-section: section-aware vs no section filtering",
        "variants": ["findings_only", "impression_only", "findings_impression", "full_text"],
    },
    "p1_max_seq_length": {
        "description": "P1: max_seq_length sensitivity",
        "variants": [64, 96, 128, 160, 192],
    },
    "p2_section_strategy": {
        "description": "P2: section strategy",
        "note": "与 A-section 共享部分变体",
    },
    "p3_quality_gate": {
        "description": "P3: quality gate strength (与 A-qg 共享)",
        "note": "与 A-qg 结果复用",
    },
}


def build_train_cmd(
    task_script: str,
    gate: str = "g2",
    tag: str = "",
    max_length: int = 128,
    input_field: str = "mention_text",
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        "python", "-u",
        str(PROJECT_ROOT / "scripts" / "phaseA2" / task_script),
        "--gate", gate,
        "--max-length", str(max_length),
        "--input-field", input_field,
    ]
    if tag:
        cmd.extend(["--tag", tag])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def print_experiment_commands(experiment: str, tasks: list[str] | None = None) -> None:
    if tasks is None:
        tasks = ["density", "size", "location"]

    task_scripts = {
        "density": "train_mws_density.py",
        "size": "train_mws_size.py",
        "location": "train_mws_location.py",
    }

    print(f"\n{'='*80}")
    print(f"Experiment: {experiment}")
    print(f"Description: {EXPERIMENTS.get(experiment, {}).get('description', 'N/A')}")
    print(f"{'='*80}\n")

    if experiment == "a_lf":
        print("## A-lf: single-source 对照已由 Phase 5 Vanilla PubMedBERT 提供（直接复用）")
        print("## multi-source = MWS-CFE 主结果 (G2)")
        print("## 无需额外训练，只需在结果表中对比\n")
        return

    if experiment == "a_qg":
        for gate in ["g1", "g2", "g3", "g4", "g5"]:
            for task in tasks:
                cmd = build_train_cmd(task_scripts[task], gate=gate, tag=f"aqg_{gate}")
                print(f"# {task} gate={gate}")
                print(" ".join(cmd))
                print()

    elif experiment == "a_window":
        for input_field in ["mention_text", "full_text"]:
            tag = f"awin_{input_field}"
            for task in tasks:
                cmd = build_train_cmd(task_scripts[task], tag=tag, input_field=input_field)
                print(f"# {task} input={input_field}")
                print(" ".join(cmd))
                print()

    elif experiment == "a_agg":
        print("## A-agg: 需要先用 uniform vote 重新生成 WS 数据")
        print("## Step 1: 修改 build_ws_datasets.py 中的权重为全 1.0，重新运行")
        print("## Step 2: 用新数据训练")
        print("## 或者：直接在 build_ws_datasets.py 中加 --uniform-weights 参数\n")
        for task in tasks:
            cmd = build_train_cmd(
                task_scripts[task],
                tag="aagg_uniform",
                extra_args=["--ws-data-dir", str(PROJECT_ROOT / "outputs" / "phaseA2" / "ws_uniform" / task)],
            )
            print(f"# {task} aggregation=uniform (需先生成 uniform WS 数据)")
            print(" ".join(cmd))
            print()

    elif experiment == "a_section":
        print("## A-section: 需要按 section 过滤训练数据")
        print("## 使用 filter_ws_by_section.py 预处理\n")
        for section in ["findings", "impression"]:
            for task in tasks:
                cmd = build_train_cmd(
                    task_scripts[task],
                    tag=f"asec_{section}",
                    extra_args=["--ws-data-dir", str(PROJECT_ROOT / "outputs" / "phaseA2" / f"ws_{section}" / task)],
                )
                print(f"# {task} section={section}")
                print(" ".join(cmd))
                print()

    elif experiment == "p1_max_seq_length":
        for length in [64, 96, 128, 160, 192]:
            for task in tasks:
                cmd = build_train_cmd(task_scripts[task], tag=f"p1_len{length}", max_length=length)
                print(f"# {task} max_length={length}")
                print(" ".join(cmd))
                print()

    elif experiment == "p2_section_strategy":
        print("## P2 与 A-section 共享变体，加上 report-intent filtered")
        print("## 复用 A-section 的 findings / impression 结果")
        print("## 额外需要: findings+impression 合并 和 full_text\n")

    elif experiment == "p3_quality_gate":
        print("## P3 与 A-qg 完全共享，直接复用 A-qg 结果\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A2 ablation & parameter sweep command generator")
    parser.add_argument("--experiment", type=str, choices=list(EXPERIMENTS.keys()) + ["all"],
                        default="all", help="Which experiment to generate commands for")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Subset of tasks (density, size, location)")
    args = parser.parse_args()

    if args.experiment == "all":
        for exp in EXPERIMENTS:
            print_experiment_commands(exp, args.tasks)
    else:
        print_experiment_commands(args.experiment, args.tasks)


if __name__ == "__main__":
    main()
