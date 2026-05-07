#!/usr/bin/env python3
"""Audit Module 2 per-sample prediction availability for Module 3.

This script is read-only with respect to model execution. It inspects result
JSONs, local model artifacts, existing prediction/probability files, and
case-alignable fields in the Phase5 mention datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DENSITY_STAGE1_RESULT = PROJECT_ROOT / "outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_density_final_g3_len128_seed42.json"
DENSITY_STAGE2_RESULT = PROJECT_ROOT / "outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed42.json"
SIZE_RESULT = PROJECT_ROOT / "outputs/phaseA2_planB/results/mws_cfe_size_results_size_wave5_lexical_alone_seed42.json"
LOCATION_RESULT = PROJECT_ROOT / "outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed42.json"

TASK_CONFIGS = [
    {
        "task": "density_stage1",
        "result_path": DENSITY_STAGE1_RESULT,
        "dataset_task": "density",
        "dataset_glob": "density_*.jsonl",
        "label_fields": ["density_label"],
        "required_fields": ["sample_id", "note_id", "subject_id", "mention_text", "density_label"],
        "expected_prediction": "stage1_label/probability",
    },
    {
        "task": "density_stage2",
        "result_path": DENSITY_STAGE2_RESULT,
        "dataset_task": "density",
        "dataset_glob": "density_*.jsonl",
        "label_fields": ["density_label"],
        "required_fields": ["sample_id", "note_id", "subject_id", "mention_text", "density_label"],
        "expected_prediction": "stage2_subtype/probability",
    },
    {
        "task": "size",
        "result_path": SIZE_RESULT,
        "dataset_task": "size",
        "dataset_glob": "size_*.jsonl",
        "label_fields": ["has_size", "size_label", "size_text"],
        "required_fields": ["sample_id", "note_id", "subject_id", "mention_text", "has_size", "size_label", "size_text"],
        "expected_prediction": "has_size_probability plus constructed size_mm fact",
    },
    {
        "task": "location",
        "result_path": LOCATION_RESULT,
        "dataset_task": "location",
        "dataset_glob": "location_*.jsonl",
        "label_fields": ["location_label", "has_location"],
        "required_fields": ["sample_id", "note_id", "subject_id", "mention_text", "location_label"],
        "expected_prediction": "location_label/probability",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _jsonl_stats(path: Path) -> tuple[int, list[str]]:
    count = 0
    fields: set[str] = set()
    if not path.exists():
        return 0, []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            count += 1
            if count <= 50:
                fields.update(json.loads(line).keys())
    return count, sorted(fields)


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _find_first_key(data: Any, target_keys: set[str]) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in target_keys:
                return value
            found = _find_first_key(value, target_keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_first_key(value, target_keys)
            if found is not None:
                return found
    return None


def _path_from_result_value(value: Any) -> tuple[str, Path | None]:
    if not value:
        return "", None
    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        if str(path).startswith("/data/hcf/ljx/neuro-symbolic-nodule-followup/"):
            local = PROJECT_ROOT / path.relative_to("/data/hcf/ljx/neuro-symbolic-nodule-followup")
            return raw, local
        return raw, path
    return raw, PROJECT_ROOT / path


def _model_reference(task: str, result: dict[str, Any]) -> tuple[str, Path | None]:
    if task == "size":
        return _path_from_result_value(result.get("model_path"))
    direct = result.get("model_dir")
    if direct:
        return _path_from_result_value(direct)
    output_dir = _nested_get(result, "training_args", "output_dir")
    if output_dir:
        return _path_from_result_value(output_dir)
    found = _find_first_key(result, {"output_dir", "model_dir"})
    return _path_from_result_value(found)


def _result_is_aggregate_only(result: dict[str, Any]) -> bool:
    if not result:
        return True
    prediction_keys = {"predictions", "per_sample_predictions", "prediction_rows", "logits", "probabilities"}
    return _find_first_key(result, prediction_keys) is None


def _existing_prediction_files(task: str, prediction_dir: Path) -> list[Path]:
    patterns = [
        f"module2_{task}_predictions.jsonl",
        f"*{task}*prediction*.jsonl",
        f"*{task}*prediction*.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(prediction_dir.glob(pattern))
    return sorted(set(files))


def _size_probability_files(probability_dir: Path, tag: str) -> list[Path]:
    if not probability_dir.exists():
        return []
    return sorted(probability_dir.glob(f"{tag}_*_probs.jsonl"))


def _count_jsonl_files(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        count, _ = _jsonl_stats(path)
        total += count
    return total


def _prediction_source_stats(paths: list[Path]) -> tuple[str, int]:
    source_counts: dict[str, int] = {}
    final_rows = 0
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source_type = str(row.get("prediction_source_type") or "unknown")
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
                if source_type.startswith("final_model"):
                    final_rows += 1
    return "|".join(f"{key}:{value}" for key, value in sorted(source_counts.items())), final_rows


def _hcf_command(task: str) -> str:
    return (
        "conda run -n follow-up python scripts/phaseA3/export_module2_predictions_for_module3.py "
        "--allow-model-inference "
        f"--tasks {task} "
        "--output-dir outputs/phaseA3/module2_predictions"
    )


def _audit_task(
    config: dict[str, Any],
    *,
    phase5_data_dir: Path,
    prediction_dir: Path,
    size_probability_dir: Path,
    size_probability_tag: str,
) -> dict[str, Any]:
    task = config["task"]
    result_path = Path(config["result_path"])
    result = _read_json(result_path)
    model_ref, local_model_path = _model_reference(task, result)

    dataset_paths = sorted(phase5_data_dir.glob(config["dataset_glob"]))
    dataset_rows = 0
    fields_seen: set[str] = set()
    for path in dataset_paths:
        count, fields = _jsonl_stats(path)
        dataset_rows += count
        fields_seen.update(fields)
    alignable = [field for field in ["case_id", "report_id", "note_id", "subject_id", "sample_id", "mention_id", "mention_text"] if field in fields_seen]

    prediction_files = _existing_prediction_files(task, prediction_dir)
    probability_files: list[Path] = []
    if task == "size":
        probability_files = _size_probability_files(size_probability_dir, size_probability_tag)

    local_model_exists = bool(local_model_path and local_model_path.exists())
    final_model_files = []
    if local_model_path and local_model_path.exists():
        if local_model_path.is_file():
            final_model_files = [local_model_path.name]
        else:
            final_model_files = sorted(path.name for path in local_model_path.iterdir() if path.is_file())

    prediction_row_count = _count_jsonl_files(prediction_files)
    prediction_source_types, final_model_prediction_rows = _prediction_source_stats(prediction_files)
    probability_row_count = _count_jsonl_files(probability_files)
    per_sample_available = bool(prediction_files) or bool(probability_files)
    can_reexport = bool(result_path.exists() and dataset_paths and local_model_exists)
    if task == "size" and probability_files:
        can_reexport = True

    if task == "size" and probability_files:
        blocker = "none_for_has_size_probability; size_mm remains constructed_fact_only"
    elif can_reexport:
        blocker = "none"
    elif not result_path.exists():
        blocker = "missing_final_result_json"
    elif not local_model_exists:
        blocker = "final_model_dir_absent_locally"
    elif not dataset_paths:
        blocker = "dataset_split_absent"
    else:
        blocker = "unknown"

    return {
        "task": task,
        "expected_prediction": config["expected_prediction"],
        "final_result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "final_result_exists": result_path.exists(),
        "result_tag": result.get("tag"),
        "final_model_reference": model_ref,
        "final_model_local_path": str(local_model_path.relative_to(PROJECT_ROOT)) if local_model_path and str(local_model_path).startswith(str(PROJECT_ROOT)) else str(local_model_path or ""),
        "final_model_local_exists": local_model_exists,
        "final_model_files": "|".join(final_model_files[:12]),
        "aggregate_result_only": _result_is_aggregate_only(result),
        "existing_prediction_files": "|".join(str(path.relative_to(PROJECT_ROOT)) for path in prediction_files),
        "existing_prediction_rows": prediction_row_count,
        "existing_prediction_source_types": prediction_source_types,
        "final_model_prediction_rows": final_model_prediction_rows,
        "size_probability_files": "|".join(str(path.relative_to(PROJECT_ROOT)) for path in probability_files),
        "size_probability_rows": probability_row_count,
        "per_sample_prediction_available": per_sample_available,
        "can_reexport_from_model_and_dataset": can_reexport,
        "export_blocker": blocker,
        "dataset_paths": "|".join(str(path.relative_to(PROJECT_ROOT)) for path in dataset_paths),
        "dataset_rows": dataset_rows,
        "case_alignable_fields": "|".join(alignable),
        "label_fields": "|".join(config["label_fields"]),
        "hcf_export_command": _hcf_command(task),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "expected_prediction",
        "final_result_path",
        "final_result_exists",
        "result_tag",
        "final_model_reference",
        "final_model_local_path",
        "final_model_local_exists",
        "final_model_files",
        "aggregate_result_only",
        "existing_prediction_files",
        "existing_prediction_rows",
        "existing_prediction_source_types",
        "final_model_prediction_rows",
        "size_probability_files",
        "size_probability_rows",
        "per_sample_prediction_available",
        "can_reexport_from_model_and_dataset",
        "export_blocker",
        "dataset_paths",
        "dataset_rows",
        "case_alignable_fields",
        "label_fields",
        "hcf_export_command",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Module 2 Per-Sample Prediction Availability Audit",
        "",
        "本报告只审计本地文件和可重导出路径，未训练、未加载 GPU、未运行模型推理。",
        "",
        "## 结论",
        "",
    ]
    for row in rows:
        task = row["task"]
        if task == "size":
            status = "可用：已有 Wave5 has-size 逐样本概率；size_mm 仍只能来自已有 fact/mention 字段。"
        elif row["can_reexport_from_model_and_dataset"]:
            status = "可本地重导出：final model 与 dataset split 均存在。"
        else:
            status = f"不可本地重导出：{row['export_blocker']}。"
        lines.append(f"- `{task}`: {status}")
    lines.extend(["", "## 明细", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['task']}",
                "",
                f"- final result: `{row['final_result_path']}` exists={row['final_result_exists']}",
                f"- final model reference: `{row['final_model_reference']}`",
                f"- local model exists: {row['final_model_local_exists']}",
                f"- aggregate result only: {row['aggregate_result_only']}",
                f"- existing prediction rows: {row['existing_prediction_rows']}",
                f"- existing prediction source types: `{row['existing_prediction_source_types']}`",
                f"- final model prediction rows: {row['final_model_prediction_rows']}",
                f"- size probability rows: {row['size_probability_rows']}",
                f"- dataset rows: {row['dataset_rows']}",
                f"- alignable fields: `{row['case_alignable_fields']}`",
                f"- blocker: `{row['export_blocker']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## HCF 导出命令",
            "",
            "如需生成真正的 density/location final model 逐样本预测，应在 HCF 环境运行以下命令；本地未执行：",
            "",
            "```bash",
        ]
    )
    for row in rows:
        if row["task"] != "size" and not row["can_reexport_from_model_and_dataset"]:
            lines.append(row["hcf_export_command"])
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Module2 prediction availability for Module3.")
    parser.add_argument("--phase5-data-dir", default="outputs/phase5/datasets")
    parser.add_argument("--prediction-dir", default="outputs/phaseA3/module2_predictions")
    parser.add_argument("--size-probability-dir", default="outputs/phaseA2_planB/size_wave5/probabilities")
    parser.add_argument("--size-probability-tag", default="size_wave5_lexical_alone_seed42")
    parser.add_argument("--output", default="outputs/phaseA3/tables/module2_prediction_availability.csv")
    parser.add_argument("--report", default="reports/module3_module2_prediction_availability_report.md")
    args = parser.parse_args()

    rows = [
        _audit_task(
            config,
            phase5_data_dir=PROJECT_ROOT / args.phase5_data_dir,
            prediction_dir=PROJECT_ROOT / args.prediction_dir,
            size_probability_dir=PROJECT_ROOT / args.size_probability_dir,
            size_probability_tag=args.size_probability_tag,
        )
        for config in TASK_CONFIGS
    ]

    _write_csv(PROJECT_ROOT / args.output, rows)
    _write_report(PROJECT_ROOT / args.report, rows)
    print(json.dumps({"tasks": len(rows), "output": args.output, "report": args.report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
