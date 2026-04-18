#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild Phase A1 WS data for A1 supervision-source ablation.

This script stays inside the current multi-source weak-supervision framework:
it reads Phase A1 ws_*.jsonl files with lf_details, keeps selected labeling
function sources, re-aggregates votes, reapplies quality gates, and writes a
WS directory that can be passed to the existing Phase A2 MWS-CFE trainers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA1.build_ws_datasets import DEFAULT_WEIGHTS
from src.weak_supervision.aggregation import weighted_majority_vote
from src.weak_supervision.base import ABSTAIN, LFOutput
from src.weak_supervision.quality_gate import GATE_ORDER, evaluate_gate


TASKS = ["density", "size", "location"]
SPLITS = ["train", "val", "test"]


def log(message: str, log_fp=None) -> None:
    print(message, flush=True)
    if log_fp:
        log_fp.write(message + "\n")
        log_fp.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_csv(value: str | None, choices: list[str], name: str) -> list[str]:
    if not value:
        return list(choices)
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(f"{name} contains invalid values: {invalid}; choices={choices}")
    return items


def safe_name(value: str) -> str:
    value = value.strip().lower().replace("+", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def lf_outputs_from_details(row: dict[str, Any], sources: set[str]) -> list[LFOutput]:
    outputs: list[LFOutput] = []
    for detail in row.get("lf_details", []):
        lf_name = str(detail.get("lf_name", ""))
        if lf_name not in sources:
            continue
        outputs.append(
            LFOutput(
                lf_name=lf_name,
                label=detail.get("label", ABSTAIN),
                confidence=float(detail.get("confidence", 1.0)),
                evidence_span=detail.get("evidence_span"),
            )
        )
    return outputs


def build_training_record(ws_record: dict[str, Any], task: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
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
        "supervision_sources": ws_record["supervision_sources"],
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
    else:
        raise ValueError(f"Unknown task: {task}")
    return rec


def rebuild_row(row: dict[str, Any], task: str, sources: list[str], weights: dict[str, float]) -> dict[str, Any]:
    lf_outputs = lf_outputs_from_details(row, set(sources))
    agg = weighted_majority_vote(lf_outputs, weights=weights)
    gate = evaluate_gate(agg)

    rebuilt = {
        "sample_id": row["sample_id"],
        "note_id": row["note_id"],
        "subject_id": row["subject_id"],
        "exam_name": row.get("exam_name", ""),
        "section": row.get("section", ""),
        "mention_text": row["mention_text"],
        "full_text": row.get("full_text", ""),
        "ws_label": agg.label,
        "ws_confidence": round(agg.confidence, 4),
        "lf_coverage": agg.lf_coverage,
        "lf_agreement": round(agg.lf_agreement, 4),
        "supporting_lfs": agg.supporting_lfs,
        "all_votes": {k: round(v, 4) for k, v in agg.all_votes.items()},
        "evidence_spans": agg.evidence_spans,
        "gate_level": gate.gate_level,
        "passed_gates": gate.passed_gates,
        "lf_details": [
            {
                "lf_name": output.lf_name,
                "label": output.label,
                "confidence": output.confidence,
                "evidence_span": output.evidence_span,
            }
            for output in lf_outputs
        ],
        "original_label": row.get("original_label", ""),
        "label_quality": row.get("label_quality", ""),
        "supervision_sources": sources,
        "a2_5_protocol": {
            "source_ablation": True,
            "phase5_single_source_reused": False,
            "rebuilt_from": "Phase A1 ws_*.jsonl lf_details",
        },
    }
    return rebuilt


def rebuild_split(
    rows: list[dict[str, Any]],
    task: str,
    sources: list[str],
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ws_records = [rebuild_row(row, task, sources, weights) for row in rows]
    non_abstain = [row for row in ws_records if row["ws_label"] != ABSTAIN]
    training_records = [build_training_record(row, task) for row in non_abstain]

    gate_counts = Counter(row["gate_level"] for row in ws_records)
    label_counts = Counter(row["ws_label"] for row in ws_records)
    stats = {
        "total_records": len(ws_records),
        "non_abstain_records": len(non_abstain),
        "label_distribution": dict(label_counts.most_common()),
        "gate_distribution": {gate: gate_counts.get(gate, 0) for gate in GATE_ORDER + ["REJECTED"]},
        "gate_retention": {
            gate: sum(1 for row in training_records if gate in row.get("passed_gates", []))
            for gate in GATE_ORDER
        },
    }
    return ws_records, training_records, stats


def task_source_variants(task: str, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    available = list(DEFAULT_WEIGHTS[task].keys())
    if args.all_single_sources:
        return [(safe_name(source), [source]) for source in available]
    if not args.sources:
        raise ValueError("Either --sources or --all-single-sources is required")
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    invalid = [source for source in sources if source not in available]
    if invalid:
        raise ValueError(f"Invalid sources for task={task}: {invalid}; available={available}")
    source_name = args.source_name or safe_name("_".join(sources))
    return [(source_name, sources)]


def rebuild_variant(args: argparse.Namespace, task: str, source_name: str, sources: list[str], log_fp) -> Path:
    ws_base = Path(args.ws_dir) if args.ws_dir else PROJECT_ROOT / "outputs" / "phaseA1"
    out_base = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "phaseA2" / "ws_source"
    task_in_dir = ws_base / task
    task_out_dir = out_base / source_name / task
    task_out_dir.mkdir(parents=True, exist_ok=True)

    weights = {source: DEFAULT_WEIGHTS[task][source] for source in sources}
    if args.uniform_weights:
        weights = {source: 1.0 for source in sources}

    log(f"[Variant] task={task} source_name={source_name} sources={sources} gate={args.gate}", log_fp)
    log(f"[Weights] {weights}", log_fp)
    variant_stats: dict[str, Any] = {
        "task": task,
        "source_name": source_name,
        "sources": sources,
        "gate": args.gate,
        "weights": weights,
        "splits": {},
    }

    for split in SPLITS:
        src_path = task_in_dir / f"ws_{split}.jsonl"
        rows = load_jsonl(src_path)
        if args.nrows:
            rows = rows[:args.nrows]
        ws_records, training_records, split_stats = rebuild_split(rows, task, sources, weights)

        ws_path = task_out_dir / f"ws_{split}.jsonl"
        write_jsonl(ws_path, ws_records)

        train_path = task_out_dir / f"{task}_{split}_ws.jsonl"
        write_jsonl(train_path, training_records)
        log(
            f"[Split] {task}/{source_name}/{split}: total={split_stats['total_records']} "
            f"non_abstain={split_stats['non_abstain_records']} -> {train_path}",
            log_fp,
        )

        if split == "train":
            gate_name = args.gate.upper()
            gated = [row for row in training_records if gate_name in row.get("passed_gates", [])]
            gated_path = task_out_dir / f"{task}_train_ws_{args.gate}.jsonl"
            write_jsonl(gated_path, gated)
            split_stats["selected_gate"] = gate_name
            split_stats["selected_gate_records"] = len(gated)
            split_stats["selected_gate_path"] = str(gated_path)
            log(f"[Gate] {task}/{source_name}/{gate_name}: {len(gated)} -> {gated_path}", log_fp)

        variant_stats["splits"][split] = split_stats

    stats_path = task_out_dir / "source_stats.json"
    write_json(stats_path, variant_stats)
    log(f"[Saved] {stats_path}", log_fp)
    return task_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild WS data by supervision source for Phase A2.5")
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma separated: density,size,location")
    parser.add_argument("--sources", type=str, default=None, help="Comma separated LF names; use with a single task")
    parser.add_argument("--source-name", type=str, default=None)
    parser.add_argument("--all-single-sources", action="store_true", help="Build one variant for each LF source")
    parser.add_argument("--gate", type=str, default="g2")
    parser.add_argument("--ws-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--uniform-weights", action="store_true")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    tasks = parse_csv(args.tasks, TASKS, "tasks")
    if args.sources and len(tasks) != 1:
        raise ValueError("--sources is task-specific and requires exactly one task")
    if args.source_name and args.all_single_sources:
        raise ValueError("--source-name cannot be combined with --all-single-sources")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else log_dir / f"rebuild_ws_by_source_{args.gate}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] rebuild_ws_by_source", log_fp)
        log(f"[Config] tasks={tasks} gate={args.gate} all_single_sources={args.all_single_sources}", log_fp)
        written = []
        for task in tasks:
            for source_name, sources in task_source_variants(task, args):
                written.append(str(rebuild_variant(args, task, source_name, sources, log_fp)))
        log(f"[Done] variants={len(written)} total_time={time.time() - t0:.1f}s", log_fp)
        for path in written:
            log(f"  {path}", log_fp)


if __name__ == "__main__":
    main()

