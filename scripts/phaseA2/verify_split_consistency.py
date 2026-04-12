#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A2 Task A: Split 一致性核验。

验证 Phase A1 WS 数据与 Phase 5 旧数据的 subject-level split 一致性。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS = ["density", "size", "location"]
SPLITS = ["train", "val", "test"]
GATES = ["g1", "g2", "g3", "g4", "g5"]


def log(msg: str, fp=None) -> None:
    print(msg, flush=True)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


def load_subject_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(json.loads(line)["subject_id"]))
    return ids


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A2 split consistency verification")
    parser.add_argument("--output-dir", type=str, default="outputs/phaseA2")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_dir / "verify_split_consistency.log", "w", encoding="utf-8", buffering=1)

    start = time.perf_counter()
    log(f"[Start] Phase A2 split consistency verification @ {datetime.now(timezone.utc).isoformat()}", log_fp)

    phase5_dir = PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    ws_dir = PROJECT_ROOT / "outputs" / "phaseA1"

    # --- 1. Phase 5 split manifest ---
    manifest_path = phase5_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log(f"[Phase5] split_manifest loaded: {manifest['splits']}", log_fp)

    phase5_subjects: dict[str, dict[str, set[int]]] = {}
    for task in TASKS:
        phase5_subjects[task] = {}
        for split in SPLITS:
            path = phase5_dir / f"{task}_{split}.jsonl"
            phase5_subjects[task][split] = load_subject_ids(path)
            log(f"[Phase5] {task}/{split}: {len(phase5_subjects[task][split])} subjects", log_fp)

    # Verify Phase 5 itself is disjoint
    for task in TASKS:
        s = phase5_subjects[task]
        assert s["train"] & s["val"] == set(), f"Phase5 {task} train/val overlap!"
        assert s["train"] & s["test"] == set(), f"Phase5 {task} train/test overlap!"
        assert s["val"] & s["test"] == set(), f"Phase5 {task} val/test overlap!"
    log("[Phase5] All tasks: train/val/test subject-disjoint ✓", log_fp)

    # --- 2. WS split consistency ---
    ws_consistency: dict[str, dict] = {}
    ws_subjects: dict[str, dict[str, set[int]]] = {}
    for task in TASKS:
        ws_subjects[task] = {}
        task_dir = ws_dir / task
        result: dict[str, bool | int] = {}
        for split in SPLITS:
            ws_path = task_dir / f"{task}_{split}_ws.jsonl"
            ws_subs = load_subject_ids(ws_path)
            ws_subjects[task][split] = ws_subs
            p5_subs = phase5_subjects[task][split]

            is_subset = ws_subs.issubset(p5_subs)
            is_exact = ws_subs == p5_subs
            extra = ws_subs - p5_subs

            result[f"{split}_ws_count"] = len(ws_subs)
            result[f"{split}_phase5_count"] = len(p5_subs)
            result[f"{split}_subjects_subset"] = is_subset
            result[f"{split}_subjects_exact_match"] = is_exact
            result[f"{split}_extra_subjects"] = len(extra)

            status = "✓ exact" if is_exact else ("✓ subset" if is_subset else "✗ LEAK")
            log(f"[WS] {task}/{split}: ws={len(ws_subs)} p5={len(p5_subs)} {status}", log_fp)

        ws_consistency[task] = result

    # --- 3. Cross-split leakage in WS data ---
    leakage: dict[str, int | bool] = {}
    all_disjoint = True
    for task in TASKS:
        s = ws_subjects[task]
        tv = s["train"] & s["val"]
        tt = s["train"] & s["test"]
        vt = s["val"] & s["test"]
        leakage[f"{task}_train_val_overlap"] = len(tv)
        leakage[f"{task}_train_test_overlap"] = len(tt)
        leakage[f"{task}_val_test_overlap"] = len(vt)
        if tv or tt or vt:
            all_disjoint = False
            log(f"[LEAK] {task}: train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}", log_fp)
        else:
            log(f"[WS] {task}: cross-split disjoint ✓", log_fp)
    leakage["all_disjoint"] = all_disjoint

    # --- 4. Gate consistency ---
    gate_consistency: dict[str, dict] = {}
    for task in TASKS:
        task_dir = ws_dir / task
        gate_consistency[task] = {}
        train_subs = phase5_subjects[task]["train"]
        for gate in GATES:
            gate_path = task_dir / f"{task}_train_ws_{gate}.jsonl"
            if not gate_path.exists():
                gate_consistency[task][gate] = {"exists": False}
                continue
            gate_subs = load_subject_ids(gate_path)
            gate_count = count_lines(gate_path)
            is_subset = gate_subs.issubset(train_subs)
            gate_consistency[task][gate] = {
                "exists": True,
                "sample_count": gate_count,
                "subject_count": len(gate_subs),
                "subjects_subset_of_train": is_subset,
            }
            status = "✓" if is_subset else "✗ LEAK"
            log(f"[Gate] {task}/{gate}: {gate_count} samples, {len(gate_subs)} subjects {status}", log_fp)

    # --- 5. Verdict ---
    all_pass = True
    issues: list[str] = []

    if not all_disjoint:
        all_pass = False
        issues.append("Cross-split subject leakage detected in WS data")

    for task in TASKS:
        for split in SPLITS:
            if not ws_consistency[task].get(f"{split}_subjects_subset", True):
                all_pass = False
                issues.append(f"{task}/{split}: WS subjects not subset of Phase5")

    for task in TASKS:
        for gate in GATES:
            gc = gate_consistency[task].get(gate, {})
            if gc.get("exists") and not gc.get("subjects_subset_of_train", True):
                all_pass = False
                issues.append(f"{task}/{gate}: gate subjects not subset of train")

    verdict = "PASS" if all_pass else "FAIL"
    log(f"\n[Verdict] {verdict}", log_fp)
    if issues:
        for issue in issues:
            log(f"  ✗ {issue}", log_fp)

    # --- 6. Save report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase5_split_manifest": {
            "train_subjects": manifest["splits"]["train"]["subject_count"],
            "val_subjects": manifest["splits"]["val"]["subject_count"],
            "test_subjects": manifest["splits"]["test"]["subject_count"],
            "disjoint": manifest["subject_overlap_check"]["disjoint"],
        },
        "ws_split_consistency": ws_consistency,
        "cross_split_leakage": leakage,
        "gate_consistency": gate_consistency,
        "verdict": verdict,
        "issues": issues,
    }

    report_path = output_dir / "split_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[Saved] {report_path}", log_fp)

    elapsed = time.perf_counter() - start
    log(f"[Done] {elapsed:.1f}s", log_fp)
    log_fp.close()


if __name__ == "__main__":
    main()
