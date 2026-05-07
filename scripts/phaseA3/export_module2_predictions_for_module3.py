#!/usr/bin/env python3
"""Export Module 2 mention-level predictions for Module 3 case-bundle recovery.

Default mode is conservative and local-safe:
- reads existing per-sample probability files when available;
- exports constructed Phase5 facts with explicit provenance when final model
  predictions are unavailable;
- does not load Transformer models unless --allow-model-inference is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VALID_DENSITIES = ["solid", "part_solid", "ground_glass", "calcified"]
VALID_LOCATIONS = ["RUL", "RML", "RLL", "LUL", "LLL", "lingula", "bilateral"]
LOCATION_LABELS = VALID_LOCATIONS + ["unclear"]
STAGE1_LABELS = ["explicit_density", "unclear_or_no_evidence"]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase5_rows(phase5_data_dir: Path, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        path = phase5_data_dir / f"{task}_{split}.jsonl"
        for row in _load_jsonl(path):
            out = dict(row)
            out["source_split"] = split
            out["source_path"] = str(path)
            rows.append(out)
    return rows


def _note_to_case(case_bundle_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bundle in _load_jsonl(case_bundle_path):
        case_id = str(bundle.get("case_id"))
        for fact in bundle.get("radiology_facts", []) or []:
            note_id = fact.get("note_id")
            if note_id:
                mapping[str(note_id)] = case_id
    return mapping


def _size_threshold(result: dict[str, Any]) -> float:
    for key_path in [
        ("chosen_threshold",),
        ("threshold_tuning", "selected_threshold"),
    ]:
        current: Any = result
        for key in key_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)):
            return float(current)
    return 0.5


def _stage1_threshold(result: dict[str, Any]) -> float:
    threshold = result.get("threshold_tuning", {}).get("selected_threshold")
    return float(threshold) if isinstance(threshold, (int, float)) else 0.5


def _model_tag(result: dict[str, Any], fallback: str) -> str:
    return str(result.get("tag") or fallback)


def _model_path_from_result(result: dict[str, Any], fallback: str | None = None) -> Path:
    value = result.get("model_dir") or result.get("model_path") or result.get("training_args", {}).get("output_dir") or fallback
    if not value:
        raise FileNotFoundError("model path not found in result JSON")
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_density(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "partsolid": "part_solid",
        "groundglass": "ground_glass",
        "ggo": "ground_glass",
    }
    return aliases.get(text, text)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _base_record(
    row: dict[str, Any],
    *,
    task: str,
    model_tag: str,
    note_to_case: dict[str, str],
) -> dict[str, Any]:
    note_id = row.get("note_id")
    sample_id = row.get("sample_id")
    return {
        "case_id": note_to_case.get(str(note_id)),
        "report_id": note_id,
        "note_id": note_id,
        "subject_id": row.get("subject_id"),
        "mention_id": row.get("mention_id") or sample_id,
        "sample_id": sample_id,
        "mention_text": row.get("mention_text"),
        "task": task,
        "predicted_label": None,
        "probability": None,
        "confidence": None,
        "model_tag": model_tag,
        "source_split": row.get("source_split"),
        "source_path": row.get("source_path"),
        "gold_or_constructed_label": None,
        "label_quality": row.get("label_quality"),
        "prediction_source_type": None,
        "failure_reason": None,
        "model_export_required": False,
    }


def _load_size_probabilities(probability_dir: Path, tag: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not probability_dir.exists():
        return out
    for path in sorted(probability_dir.glob(f"{tag}_*_probs.jsonl")):
        split = path.name.replace(f"{tag}_", "").replace("_probs.jsonl", "")
        for row in _load_jsonl(path):
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            prob = _as_float(row.get("prob_has_size"))
            current = out.get(str(sample_id))
            if current is None or (prob is not None and prob > (current.get("prob_has_size") or -1)):
                out[str(sample_id)] = {
                    "prob_has_size": prob,
                    "probability_split": split,
                    "probability_path": str(path),
                    "row": row,
                }
    return out


def _export_density_stage1_constructed(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = _stage1_threshold(result)
    records: list[dict[str, Any]] = []
    for row in rows:
        density = _normalize_density(row.get("density_label"))
        constructed = "explicit_density" if density in VALID_DENSITIES else "unclear_or_no_evidence"
        rec = _base_record(row, task="density_stage1", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": constructed,
                "gold_or_constructed_label": constructed,
                "prediction_source_type": "constructed_fact_not_final_model",
                "failure_reason": "final_density_stage1_per_sample_predictions_missing_or_model_dir_absent_locally",
                "model_export_required": True,
                "stage1_threshold": threshold,
                "density_label_source": row.get("density_label"),
            }
        )
        records.append(rec)
    return records


def _export_density_stage2_constructed(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        density = _normalize_density(row.get("density_label"))
        rec = _base_record(row, task="density_stage2", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": density if density in VALID_DENSITIES else None,
                "gold_or_constructed_label": density,
                "prediction_source_type": "constructed_fact_not_final_model",
                "failure_reason": "final_density_stage2_per_sample_predictions_missing_or_model_dir_absent_locally",
                "model_export_required": True,
                "density_label_source": row.get("density_label"),
            }
        )
        records.append(rec)
    return records


def _export_location_constructed(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("location_label")
        predicted = label if label in VALID_LOCATIONS else None
        rec = _base_record(row, task="location", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": predicted,
                "gold_or_constructed_label": label,
                "prediction_source_type": "constructed_fact_not_final_model",
                "failure_reason": "final_location_per_sample_predictions_missing_or_model_dir_absent_locally",
                "model_export_required": True,
                "has_location": row.get("has_location"),
            }
        )
        records.append(rec)
    return records


def _export_size_from_existing_sources(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
    threshold: float,
    probabilities: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        prob_row = probabilities.get(sample_id)
        prob = _as_float(prob_row.get("prob_has_size")) if prob_row else None
        if prob is None:
            predicted = "has_size" if bool(row.get("has_size")) else "no_size"
            source_type = "constructed_fact_only_no_probability"
            failure = "has_size_probability_missing_for_this_split"
        else:
            predicted = "has_size" if prob >= threshold else "no_size"
            source_type = "final_model_probability_file"
            failure = None
        rec = _base_record(row, task="size", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": predicted,
                "probability": prob,
                "confidence": prob,
                "gold_or_constructed_label": "has_size" if bool(row.get("has_size")) else "no_size",
                "prediction_source_type": source_type,
                "failure_reason": failure,
                "model_export_required": False if prob is not None else True,
                "has_size": row.get("has_size"),
                "size_mm": _as_float(row.get("size_label")) if row.get("has_size") else None,
                "size_text": row.get("size_text"),
                "size_threshold": threshold,
                "probability_split": prob_row.get("probability_split") if prob_row else None,
                "probability_path": prob_row.get("probability_path") if prob_row else None,
            }
        )
        records.append(rec)
    return records


def _prepare_density_rows_for_inference(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.phaseA2.build_planb_density_two_stage import add_input_strategy_fields

    section_cache: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        prepared = add_input_strategy_fields(dict(row), section_cache)
        density = _normalize_density(row.get("density_label"))
        prepared["density_stage1_label"] = "explicit_density" if density in VALID_DENSITIES else "unclear_or_no_evidence"
        if density in VALID_DENSITIES:
            prepared["density_stage2_label"] = density
        out.append(prepared)
    return out


def _prepare_location_rows_for_inference(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.phaseA2.feature_augmentation import add_location_cue_augmented_text

    return [add_location_cue_augmented_text(dict(row)) for row in rows]


def _run_hf_inference(
    *,
    rows: list[dict[str, Any]],
    model_dir: Path,
    label_names: list[str],
    input_field: str,
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(label_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [str(row.get(input_field) or row.get("mention_text") or "") for row in batch]
        encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        pred_ids = np.argmax(probs, axis=-1)
        for row, pred_id, prob_vec in zip(batch, pred_ids.tolist(), probs.tolist(), strict=False):
            outputs.append(
                {
                    "row": row,
                    "predicted_label": label_names[int(pred_id)],
                    "probability": float(prob_vec[int(pred_id)]),
                    "probabilities": {label: float(prob_vec[idx]) for idx, label in enumerate(label_names)},
                }
            )
    return outputs


def _export_density_stage1_model_predictions(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
    result: dict[str, Any],
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    prepared = _prepare_density_rows_for_inference(rows)
    model_dir = _model_path_from_result(result)
    input_field = str(result.get("input_field") or "section_aware_text")
    threshold = _stage1_threshold(result)
    predictions = _run_hf_inference(
        rows=prepared,
        model_dir=model_dir,
        label_names=STAGE1_LABELS,
        input_field=input_field,
        max_length=max_length,
        batch_size=batch_size,
    )
    records: list[dict[str, Any]] = []
    for pred in predictions:
        row = pred["row"]
        positive_prob = pred["probabilities"].get("explicit_density")
        label = "explicit_density" if positive_prob is not None and positive_prob >= threshold else "unclear_or_no_evidence"
        rec = _base_record(row, task="density_stage1", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": label,
                "probability": positive_prob,
                "confidence": positive_prob if label == "explicit_density" else pred["probabilities"].get("unclear_or_no_evidence"),
                "all_probabilities": pred["probabilities"],
                "gold_or_constructed_label": row.get("density_stage1_label"),
                "prediction_source_type": "final_model_inference",
                "failure_reason": None,
                "model_export_required": False,
                "stage1_threshold": threshold,
                "density_label_source": row.get("density_label"),
            }
        )
        records.append(rec)
    return records


def _export_density_stage2_model_predictions(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
    result: dict[str, Any],
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    prepared = _prepare_density_rows_for_inference(rows)
    model_dir = _model_path_from_result(result)
    input_field = str(result.get("input_field") or "section_aware_text")
    predictions = _run_hf_inference(
        rows=prepared,
        model_dir=model_dir,
        label_names=VALID_DENSITIES,
        input_field=input_field,
        max_length=max_length,
        batch_size=batch_size,
    )
    records: list[dict[str, Any]] = []
    for pred in predictions:
        row = pred["row"]
        rec = _base_record(row, task="density_stage2", model_tag=model_tag, note_to_case=note_to_case)
        rec.update(
            {
                "predicted_label": pred["predicted_label"],
                "probability": pred["probability"],
                "confidence": pred["probability"],
                "all_probabilities": pred["probabilities"],
                "gold_or_constructed_label": row.get("density_stage2_label"),
                "prediction_source_type": "final_model_inference",
                "failure_reason": None,
                "model_export_required": False,
                "density_label_source": row.get("density_label"),
            }
        )
        records.append(rec)
    return records


def _export_location_model_predictions(
    *,
    rows: list[dict[str, Any]],
    note_to_case: dict[str, str],
    model_tag: str,
    result: dict[str, Any],
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    prepared = _prepare_location_rows_for_inference(rows)
    model_dir = _model_path_from_result(result)
    input_field = str(result.get("input_field") or "cue_augmented_text")
    model_rows = [row for row in prepared if row.get("location_label") != "no_location"]
    model_predictions_by_sample: dict[str, dict[str, Any]] = {}
    if model_rows:
        for pred in _run_hf_inference(
            rows=model_rows,
            model_dir=model_dir,
            label_names=LOCATION_LABELS,
            input_field=input_field,
            max_length=max_length,
            batch_size=batch_size,
        ):
            model_predictions_by_sample[str(pred["row"].get("sample_id"))] = pred

    records: list[dict[str, Any]] = []
    for row in prepared:
        rec = _base_record(row, task="location", model_tag=model_tag, note_to_case=note_to_case)
        if row.get("location_label") == "no_location":
            rec.update(
                {
                    "predicted_label": None,
                    "probability": 1.0,
                    "confidence": 1.0,
                    "all_probabilities": {"no_location": 1.0},
                    "gold_or_constructed_label": "no_location",
                    "prediction_source_type": "final_model_inference_with_no_location_fallback",
                    "failure_reason": None,
                    "model_export_required": False,
                    "has_location": row.get("has_location"),
                }
            )
        else:
            pred = model_predictions_by_sample.get(str(row.get("sample_id")))
            predicted_label = pred["predicted_label"] if pred else None
            rec.update(
                {
                    "predicted_label": predicted_label if predicted_label in VALID_LOCATIONS else None,
                    "probability": pred["probability"] if pred else None,
                    "confidence": pred["probability"] if pred else None,
                    "all_probabilities": pred["probabilities"] if pred else None,
                    "gold_or_constructed_label": row.get("location_label"),
                    "prediction_source_type": "final_model_inference",
                    "failure_reason": None if pred else "model_prediction_missing_after_inference",
                    "model_export_required": False,
                    "has_location": row.get("has_location"),
                }
            )
        records.append(rec)
    return records


def _summary_rows(task: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(str(row.get("prediction_source_type")) for row in records)
    failure_counter = Counter(str(row.get("failure_reason")) for row in records if row.get("failure_reason"))
    source_split_counter = Counter(str(row.get("source_split")) for row in records)
    rows = [
        {"task": task, "metric": "rows_exported", "value": len(records)},
        {"task": task, "metric": "case_aligned_rows", "value": sum(1 for row in records if row.get("case_id"))},
        {"task": task, "metric": "case_alignment_rate", "value": f"{sum(1 for row in records if row.get('case_id')) / len(records):.6f}" if records else "0.000000"},
        {"task": task, "metric": "model_export_required_rows", "value": sum(1 for row in records if row.get("model_export_required"))},
    ]
    for key, value in sorted(counter.items()):
        rows.append({"task": task, "metric": f"prediction_source_type.{key}", "value": value})
    for key, value in sorted(failure_counter.items()):
        rows.append({"task": task, "metric": f"failure_reason.{key}", "value": value})
    for key, value in sorted(source_split_counter.items()):
        rows.append({"task": task, "metric": f"source_split.{key}", "value": value})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Module2 predictions for Module3.")
    parser.add_argument("--tasks", default="density_stage1,density_stage2,size,location")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--case-bundles", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/phaseA3/module2_predictions")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/module2_prediction_export_summary.csv")
    parser.add_argument("--density-stage1-result", default="outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_density_final_g3_len128_seed42.json")
    parser.add_argument("--density-stage2-result", default="outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed42.json")
    parser.add_argument("--size-result", default="outputs/phaseA2_planB/results/mws_cfe_size_results_size_wave5_lexical_alone_seed42.json")
    parser.add_argument("--location-result", default="outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed42.json")
    parser.add_argument("--size-probability-dir", default="outputs/phaseA2_planB/size_wave5/probabilities")
    parser.add_argument("--size-probability-tag", default="size_wave5_lexical_alone_seed42")
    parser.add_argument("--allow-model-inference", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--include-all-phase5",
        action="store_true",
        help="Export every Phase5 mention. Default exports only mentions aligned to Phase4 case bundles.",
    )
    args = parser.parse_args()

    phase5_data_dir = PROJECT_ROOT / args.phase5_data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    note_to_case = _note_to_case(PROJECT_ROOT / args.case_bundles)

    density_rows_all = _phase5_rows(phase5_data_dir, "density")
    size_rows_all = _phase5_rows(phase5_data_dir, "size")
    location_rows_all = _phase5_rows(phase5_data_dir, "location")
    if args.include_all_phase5:
        density_rows = density_rows_all
        size_rows = size_rows_all
        location_rows = location_rows_all
    else:
        density_rows = [row for row in density_rows_all if str(row.get("note_id")) in note_to_case]
        size_rows = [row for row in size_rows_all if str(row.get("note_id")) in note_to_case]
        location_rows = [row for row in location_rows_all if str(row.get("note_id")) in note_to_case]

    density_stage1_result = _load_json(PROJECT_ROOT / args.density_stage1_result)
    density_stage2_result = _load_json(PROJECT_ROOT / args.density_stage2_result)
    size_result = _load_json(PROJECT_ROOT / args.size_result)
    location_result = _load_json(PROJECT_ROOT / args.location_result)

    requested = {task.strip() for task in args.tasks.split(",") if task.strip()}
    all_summary: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    if "density_stage1" in requested:
        if args.allow_model_inference:
            records = _export_density_stage1_model_predictions(
                rows=density_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(density_stage1_result, "density_stage1_final"),
                result=density_stage1_result,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
        else:
            records = _export_density_stage1_constructed(
                rows=density_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(density_stage1_result, "density_stage1_final"),
                result=density_stage1_result,
            )
        path = output_dir / "module2_density_stage1_predictions.jsonl"
        _write_jsonl(path, records)
        all_summary.extend(_summary_rows("density_stage1", records))
        all_summary.append({"task": "density_stage1", "metric": "phase5_rows_total_before_phase4_filter", "value": len(density_rows_all)})
        outputs["density_stage1"] = str(path.relative_to(PROJECT_ROOT))

    if "density_stage2" in requested:
        if args.allow_model_inference:
            records = _export_density_stage2_model_predictions(
                rows=density_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(density_stage2_result, "density_stage2_final"),
                result=density_stage2_result,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
        else:
            records = _export_density_stage2_constructed(
                rows=density_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(density_stage2_result, "density_stage2_final"),
            )
        path = output_dir / "module2_density_stage2_predictions.jsonl"
        _write_jsonl(path, records)
        all_summary.extend(_summary_rows("density_stage2", records))
        all_summary.append({"task": "density_stage2", "metric": "phase5_rows_total_before_phase4_filter", "value": len(density_rows_all)})
        outputs["density_stage2"] = str(path.relative_to(PROJECT_ROOT))

    if "size" in requested:
        probabilities = _load_size_probabilities(PROJECT_ROOT / args.size_probability_dir, args.size_probability_tag)
        threshold = _size_threshold(size_result)
        records = _export_size_from_existing_sources(
            rows=size_rows,
            note_to_case=note_to_case,
            model_tag=f"{_model_tag(size_result, args.size_probability_tag)}_probabilities",
            threshold=threshold,
            probabilities=probabilities,
        )
        path = output_dir / "module2_size_predictions.jsonl"
        _write_jsonl(path, records)
        all_summary.extend(_summary_rows("size", records))
        all_summary.append({"task": "size", "metric": "phase5_rows_total_before_phase4_filter", "value": len(size_rows_all)})
        all_summary.append({"task": "size", "metric": "size_probability_threshold", "value": f"{threshold:.6f}"})
        all_summary.append({"task": "size", "metric": "size_probability_rows_loaded", "value": len(probabilities)})
        outputs["size"] = str(path.relative_to(PROJECT_ROOT))

    if "location" in requested:
        if args.allow_model_inference:
            records = _export_location_model_predictions(
                rows=location_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(location_result, "location_final"),
                result=location_result,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
        else:
            records = _export_location_constructed(
                rows=location_rows,
                note_to_case=note_to_case,
                model_tag=_model_tag(location_result, "location_final"),
            )
        path = output_dir / "module2_location_predictions.jsonl"
        _write_jsonl(path, records)
        all_summary.extend(_summary_rows("location", records))
        all_summary.append({"task": "location", "metric": "phase5_rows_total_before_phase4_filter", "value": len(location_rows_all)})
        outputs["location"] = str(path.relative_to(PROJECT_ROOT))

    _write_csv(PROJECT_ROOT / args.summary, all_summary)
    print(json.dumps({"outputs": outputs, "summary": args.summary}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
