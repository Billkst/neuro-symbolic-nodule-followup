#!/usr/bin/env python3
"""Build Plan B two-stage density datasets.

Stage 1: explicit_density vs unclear_or_no_evidence.
Stage 2: subtype classification on explicit-density rows only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phaseA2.train_mws_cfe_common import load_jsonl
from src.parsers.section_parser import parse_sections
from src.weak_supervision.base import ABSTAIN
from src.weak_supervision.quality_gate import GATE_ORDER


EXPLICIT_DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified"]
STAGE1_LABELS = ["explicit_density", "unclear_or_no_evidence"]


def log(message: str, fp=None) -> None:
    print(message, flush=True)
    if fp:
        fp.write(message + "\n")
        fp.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _join_sections(findings: str, impression: str) -> str:
    parts = []
    if findings:
        parts.append("FINDINGS: " + findings)
    if impression:
        parts.append("IMPRESSION: " + impression)
    return "\n\n".join(parts)


def add_input_strategy_fields(row: dict[str, Any], section_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    full_text = str(out.get("full_text") or "")
    cache_key = str(out.get("note_id") or full_text[:256])
    sections = section_cache.get(cache_key)
    if sections is None:
        sections = parse_sections(full_text)
        section_cache[cache_key] = sections
    findings = str(sections.get("findings") or "")
    impression = str(sections.get("impression") or "")
    mention = str(out.get("mention_text") or "")
    section = str(out.get("section") or "unknown")
    exam = str(out.get("exam_name") or "")
    out["findings_text"] = findings
    out["impression_text"] = impression
    out["findings_impression_text"] = _join_sections(findings, impression)
    out["section_aware_text"] = (
        f"[SECTION] {section}\n"
        f"[EXAM] {exam}\n"
        f"[MENTION] {mention}\n"
        f"{out['findings_impression_text']}"
    ).strip()
    return out


def base_training_record(row: dict[str, Any], section_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "note_id",
        "subject_id",
        "exam_name",
        "section",
        "mention_text",
        "full_text",
        "label_quality",
        "lf_coverage",
        "gate_level",
        "passed_gates",
    ]
    out = {key: row.get(key) for key in keys if key in row}
    return add_input_strategy_fields(out, section_cache)


def label_from_lf(row: dict[str, Any], lf_name: str) -> tuple[str, float]:
    for detail in row.get("lf_details", []):
        if detail.get("lf_name") != lf_name:
            continue
        label = str(detail.get("label") or ABSTAIN)
        confidence = float(detail.get("confidence") or 0.0)
        return label, confidence
    return ABSTAIN, 0.0


def source_density_label(row: dict[str, Any], source_mode: str, single_lf_name: str) -> tuple[str, float]:
    if source_mode == "single_lf":
        return label_from_lf(row, single_lf_name)
    return str(row.get("ws_label") or ABSTAIN), float(row.get("ws_confidence") or 0.0)


def stage1_label_from_density(label: str) -> str:
    return "explicit_density" if label in EXPLICIT_DENSITY_LABELS else "unclear_or_no_evidence"


def confidence_for_stage1(
    density_label: str,
    source_confidence: float,
    negative_confidence: float,
) -> float:
    if density_label in EXPLICIT_DENSITY_LABELS:
        return max(float(source_confidence), 1e-6)
    return max(float(source_confidence), float(negative_confidence))


def passes_gate(row: dict[str, Any], gate: str, stage1_label: str, source_mode: str) -> bool:
    if stage1_label == "unclear_or_no_evidence":
        return True
    if source_mode == "single_lf":
        return True
    return gate.upper() in row.get("passed_gates", [])


def build_train_rows(
    ws_rows: list[dict[str, Any]],
    *,
    source_mode: str,
    single_lf_name: str,
    negative_confidence: float,
) -> dict[str, list[dict[str, Any]]]:
    stage1_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    section_cache: dict[str, dict[str, Any]] = {}
    for row in ws_rows:
        density_label, source_confidence = source_density_label(row, source_mode, single_lf_name)
        stage1_label = stage1_label_from_density(density_label)
        base = base_training_record(row, section_cache)
        base["source_density_label"] = density_label
        base["density_stage1_label"] = stage1_label
        base["ws_confidence"] = round(
            confidence_for_stage1(density_label, source_confidence, negative_confidence),
            4,
        )
        stage1_rows.append(base)
        if density_label in EXPLICIT_DENSITY_LABELS:
            stage2 = dict(base)
            stage2["density_stage2_label"] = density_label
            stage2["ws_confidence"] = round(max(float(source_confidence), 1e-6), 4)
            stage2_rows.append(stage2)
    return {"density_stage1": stage1_rows, "density_stage2": stage2_rows}


def build_eval_rows(phase5_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stage1_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    section_cache: dict[str, dict[str, Any]] = {}
    for row in phase5_rows:
        density_label = str(row.get("density_label") or "unclear")
        base = base_training_record(row, section_cache)
        base["source_density_label"] = density_label
        base["ws_confidence"] = 1.0
        base["density_stage1_label"] = stage1_label_from_density(density_label)
        stage1_rows.append(base)
        if density_label in EXPLICIT_DENSITY_LABELS:
            stage2 = dict(base)
            stage2["density_stage2_label"] = density_label
            stage2_rows.append(stage2)
    return {"density_stage1": stage1_rows, "density_stage2": stage2_rows}


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field)) for row in rows).most_common())


def maybe_limit(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    return rows if n is None else rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plan B two-stage density datasets")
    parser.add_argument("--ws-source-dir", default="outputs/phaseA1/density")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB")
    parser.add_argument("--source-mode", choices=["weighted", "uniform", "single_lf"], default="weighted")
    parser.add_argument("--single-lf-name", default="LF-D1")
    parser.add_argument("--negative-confidence", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log", default="logs/build_planb_density_two_stage.log")
    args = parser.parse_args()

    ws_source_dir = PROJECT_ROOT / args.ws_source_dir
    phase5_data_dir = PROJECT_ROOT / args.phase5_data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    log_path = PROJECT_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_mode": args.source_mode,
        "single_lf_name": args.single_lf_name if args.source_mode == "single_lf" else None,
        "negative_confidence": args.negative_confidence,
        "splits": {},
    }

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        log("[Start] build_planb_density_two_stage", log_fp)
        log(f"[Config] ws_source_dir={ws_source_dir}", log_fp)
        log(f"[Config] phase5_data_dir={phase5_data_dir}", log_fp)
        log(f"[Config] output_dir={output_dir}", log_fp)
        log(f"[Config] source_mode={args.source_mode} negative_confidence={args.negative_confidence}", log_fp)

        train_ws_rows = maybe_limit(load_jsonl(ws_source_dir / "ws_train.jsonl"), args.max_rows)
        train_payload = build_train_rows(
            train_ws_rows,
            source_mode=args.source_mode,
            single_lf_name=args.single_lf_name,
            negative_confidence=args.negative_confidence,
        )

        for task_name, rows in train_payload.items():
            task_dir = output_dir / task_name
            write_jsonl(task_dir / f"{task_name}_train_ws.jsonl", rows)
            label_field = f"{task_name}_label"
            for gate in GATE_ORDER:
                gated = [row for row in rows if passes_gate(row, gate, row["density_stage1_label"], args.source_mode)]
                write_jsonl(task_dir / f"{task_name}_train_ws_{gate.lower()}.jsonl", gated)
                summary["splits"][f"{task_name}_train_{gate.lower()}"] = {
                    "n": len(gated),
                    "label_distribution": distribution(gated, label_field),
                }
            summary["splits"][f"{task_name}_train_all"] = {
                "n": len(rows),
                "label_distribution": distribution(rows, label_field),
            }
            log(f"[Train] {task_name} all={len(rows)}", log_fp)

        for split in ("val", "test"):
            phase5_rows = maybe_limit(load_jsonl(phase5_data_dir / f"density_{split}.jsonl"), args.max_rows)
            eval_payload = build_eval_rows(phase5_rows)
            for task_name, rows in eval_payload.items():
                task_dir = output_dir / task_name
                label_field = f"{task_name}_label"
                write_jsonl(task_dir / f"{task_name}_{split}_ws.jsonl", rows)
                write_jsonl(task_dir / f"{task_name}_{split}.jsonl", rows)
                summary["splits"][f"{task_name}_{split}"] = {
                    "n": len(rows),
                    "label_distribution": distribution(rows, label_field),
                }
                log(f"[{split}] {task_name} n={len(rows)}", log_fp)

        summary_path = output_dir / "density_two_stage_summary.json"
        write_json(summary_path, summary)
        log(f"[Saved] {summary_path}", log_fp)
        log("[Done] build_planb_density_two_stage", log_fp)


if __name__ == "__main__":
    main()
