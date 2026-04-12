#!/usr/bin/env python3
"""MWS-CFE Gold sanity-check 评测。

在 N=62 gold 标注集上评测 MWS-CFE 模型，与 Phase 5.1 已有结果对比。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5_1.evaluation.gold_metrics import (
    evaluate_density_gold,
    evaluate_has_size_gold,
    evaluate_location_gold,
)

DENSITY_LABELS = ["solid", "part_solid", "ground_glass", "calcified", "unclear"]
LOCATION_LABELS_8 = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear"]
LOCATION_LABELS_9 = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral", "unclear", "no_location"]
SIZE_LABELS = ["no_size", "has_size"]


def log(msg: str, fp=None) -> None:
    print(msg, flush=True)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def predict_with_model(
    model_dir: Path,
    texts: list[str],
    label_names: list[str],
    max_length: int = 128,
    batch_size: int = 32,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        predictions.extend([label_names[pid] for pid in pred_ids])

    return predictions


def normalize_location(label: str | None) -> str:
    if label in LOCATION_LABELS_9:
        return label
    if label is None:
        return "no_location"
    return "unclear"


def main() -> None:
    parser = argparse.ArgumentParser(description="MWS-CFE Gold evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs/phaseA2")
    parser.add_argument("--model-base-dir", type=str, default=None)
    parser.add_argument("--gate", type=str, default="g2",
                        help="Quality gate used during training (for default model dir naming)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag. Model dirs = {task}_mws_cfe_{tag}. "
                             "If not set, falls back to --gate value.")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_base = Path(args.model_base_dir) if args.model_base_dir else output_dir / "models"
    tag = args.tag if args.tag else args.gate

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_dir / f"gold_eval_mws_{tag}.log", "w", encoding="utf-8", buffering=1)

    start = time.perf_counter()
    log(f"[Start] Gold evaluation for MWS-CFE tag={tag}", log_fp)
    log(f"[Config] model_base={model_base} tag={tag} gate={args.gate} max_length={args.max_length}", log_fp)

    manifest_path = PROJECT_ROOT / "outputs" / "phase5_1" / "gold_eval_manifest.jsonl"
    manifest_rows = load_jsonl(manifest_path)
    log(f"[Data] Gold manifest: {len(manifest_rows)} samples from {manifest_path}", log_fp)

    texts = [row["text_window"] for row in manifest_rows]
    all_results: dict[str, Any] = {}

    density_model_dir = model_base / f"density_mws_cfe_{tag}"
    log(f"[Density] expected model dir: {density_model_dir} (exists={density_model_dir.exists()})", log_fp)
    if density_model_dir.exists():
        log(f"[Density] Loading model from {density_model_dir}", log_fp)
        density_preds = predict_with_model(density_model_dir, texts, DENSITY_LABELS, args.max_length)
        gold_density = [row["gold_density_category"] for row in manifest_rows]
        density_metrics = evaluate_density_gold(density_preds, gold_density, DENSITY_LABELS)
        all_results["density"] = density_metrics
        log(f"[Density] accuracy={density_metrics['accuracy']:.4f} macro_f1={density_metrics['macro_f1']:.4f}", log_fp)
    else:
        log(f"[Density] Model not found at {density_model_dir}, skipping", log_fp)

    size_model_dir = model_base / f"size_mws_cfe_{tag}"
    log(f"[Size] expected model dir: {size_model_dir} (exists={size_model_dir.exists()})", log_fp)
    if size_model_dir.exists():
        log(f"[Size] Loading model from {size_model_dir}", log_fp)
        size_preds_raw = predict_with_model(size_model_dir, texts, SIZE_LABELS, args.max_length)
        size_preds = [1 if p == "has_size" else 0 for p in size_preds_raw]
        gold_has_size = [1 if row["gold_has_size"] else 0 for row in manifest_rows]
        size_metrics = evaluate_has_size_gold(size_preds, gold_has_size)
        all_results["has_size"] = size_metrics
        log(f"[Size] accuracy={size_metrics['accuracy']:.4f} f1={size_metrics['f1']:.4f}", log_fp)
    else:
        log(f"[Size] Model not found at {size_model_dir}, skipping", log_fp)

    location_model_dir = model_base / f"location_mws_cfe_{tag}"
    log(f"[Location] expected model dir: {location_model_dir} (exists={location_model_dir.exists()})", log_fp)
    if location_model_dir.exists():
        log(f"[Location] Loading model from {location_model_dir}", log_fp)
        loc_preds_8 = predict_with_model(location_model_dir, texts, LOCATION_LABELS_8, args.max_length)
        gold_location = [normalize_location(row.get("gold_location_lobe")) for row in manifest_rows]
        location_metrics = evaluate_location_gold(loc_preds_8, gold_location, LOCATION_LABELS_9)
        all_results["location"] = location_metrics
        log(f"[Location] accuracy={location_metrics['accuracy']:.4f} macro_f1={location_metrics['macro_f1']:.4f}", log_fp)
    else:
        log(f"[Location] Model not found at {location_model_dir}, skipping", log_fp)

    result_path = results_dir / f"gold_eval_mws_{tag}.json"
    result_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"[Saved] {result_path}", log_fp)

    elapsed = time.perf_counter() - start
    log(f"[Done] {elapsed:.1f}s", log_fp)
    log_fp.close()


if __name__ == "__main__":
    main()
