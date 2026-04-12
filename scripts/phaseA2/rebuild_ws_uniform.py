#!/usr/bin/env python3
"""用 uniform weights 重新聚合已有 WS 数据，用于 A-agg 消融。

从 ws_train.jsonl（含 lf_details）读取原始 LF 输出，
用 uniform weights (全 1.0) 重新调用 weighted_majority_vote，
重新计算 ws_label / ws_confidence / gate_level / passed_gates，
然后按指定 gate 过滤输出训练文件。

val/test 直接复制（不受聚合权重影响，因为它们的标签来自同一 LF 输出）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weak_supervision.aggregation import weighted_majority_vote
from src.weak_supervision.base import ABSTAIN, LFOutput
from src.weak_supervision.quality_gate import evaluate_gate


TASKS = ["density", "size", "location"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def rebuild_with_uniform_weights(ws_path: Path, task: str) -> list[dict]:
    uniform_weights: dict[str, float] = {}

    rebuilt: list[dict] = []
    with ws_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            lf_details = row.get("lf_details", [])
            lf_outputs: list[LFOutput] = []
            for d in lf_details:
                lf_outputs.append(LFOutput(
                    lf_name=d["lf_name"],
                    label=d["label"],
                    confidence=d.get("confidence", 1.0),
                    evidence_span=d.get("evidence_span"),
                ))
                if d["lf_name"] not in uniform_weights:
                    uniform_weights[d["lf_name"]] = 1.0

            agg = weighted_majority_vote(lf_outputs, weights=uniform_weights)
            gate = evaluate_gate(agg)

            row["ws_label"] = agg.label
            row["ws_confidence"] = round(agg.confidence, 4)
            row["lf_coverage"] = agg.lf_coverage
            row["lf_agreement"] = round(agg.lf_agreement, 4)
            row["supporting_lfs"] = agg.supporting_lfs
            row["all_votes"] = {k: round(v, 4) for k, v in agg.all_votes.items()}
            row["evidence_spans"] = agg.evidence_spans
            row["gate_level"] = gate.gate_level
            row["passed_gates"] = gate.passed_gates

            rebuilt.append(row)

    return rebuilt


def build_training_record(ws_record: dict, task: str) -> dict:
    rec = {
        "sample_id": ws_record["sample_id"],
        "note_id": ws_record["note_id"],
        "subject_id": ws_record["subject_id"],
        "exam_name": ws_record.get("exam_name", ""),
        "section": ws_record.get("section", ""),
        "mention_text": ws_record["mention_text"],
        "full_text": ws_record.get("full_text", ""),
        "label_quality": ws_record.get("label_quality", ""),
        "ws_confidence": ws_record["ws_confidence"],
        "lf_coverage": ws_record["lf_coverage"],
        "gate_level": ws_record["gate_level"],
        "passed_gates": ws_record["passed_gates"],
    }
    if task == "density":
        rec["density_label"] = ws_record["ws_label"]
    elif task == "size":
        rec["has_size"] = ws_record["ws_label"] == "true"
        rec["size_label"] = None
        rec["size_text"] = None
    elif task == "location":
        rec["location_label"] = ws_record["ws_label"]
        rec["has_location"] = ws_record["ws_label"] not in ("no_location", ABSTAIN)
    return rec


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild WS data with uniform weights for A-agg ablation")
    parser.add_argument("--gate", type=str, default="g2")
    parser.add_argument("--ws-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ws_base = Path(args.ws_dir) if args.ws_dir else PROJECT_ROOT / "outputs" / "phaseA1"
    out_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2" / "ws_uniform"
    gate_filter = args.gate.upper()

    t_total = time.time()
    _log(f"[Start] rebuild_ws_uniform gate={args.gate}")

    for task in TASKS:
        t0 = time.time()
        task_ws_dir = ws_base / task
        task_out_dir = out_base / task
        task_out_dir.mkdir(parents=True, exist_ok=True)

        ws_full = task_ws_dir / "ws_train.jsonl"
        if not ws_full.exists():
            _log(f"[Skip] {ws_full} not found")
            continue

        _log(f"[Rebuild] {task} with uniform weights from {ws_full}")
        rebuilt = rebuild_with_uniform_weights(ws_full, task)
        _log(f"  total records: {len(rebuilt)}")

        non_abstain = [r for r in rebuilt if r["ws_label"] != ABSTAIN]
        training_records = [build_training_record(r, task) for r in non_abstain]
        _log(f"  non-ABSTAIN: {len(training_records)}")

        gated = [r for r in training_records if gate_filter in r.get("passed_gates", [])]
        out_path = task_out_dir / f"{task}_train_ws_{args.gate}.jsonl"
        write_jsonl(out_path, gated)
        _log(f"  {gate_filter} filtered: {len(gated)} -> {out_path}")

        for split in ["val", "test"]:
            src = task_ws_dir / f"{task}_{split}_ws.jsonl"
            dst = task_out_dir / f"{task}_{split}_ws.jsonl"
            if src.exists():
                shutil.copy2(src, dst)
                _log(f"  copied {split} -> {dst}")

        _log(f"  {task} done in {time.time() - t0:.1f}s")

    _log(f"[Done] total={time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
