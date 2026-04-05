from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5.analysis_common import (
    Logger,
    ensure_dir,
    exam_group_key,
    format_seconds,
    get_gold_labels,
    load_task_splits,
    prepare_hf_environment,
    primary_metric_name,
    primary_metric_value,
    run_svm_experiment,
    save_json,
)


def compact_metrics(task: str, result: dict) -> dict:
    metrics = result["metrics"]
    payload = {
        "primary_metric_name": primary_metric_name(task),
        "primary_metric": primary_metric_value(task, metrics),
        "accuracy": float(metrics["accuracy"]),
        "fit_seconds": result["fit_seconds"],
        "predict_seconds": result["predict_seconds"],
        "config": result["config"],
    }
    if task == "size":
        payload["f1"] = float(metrics["f1"])
        payload["macro_f1"] = float(metrics["macro_f1"])
        payload["precision"] = float(metrics["precision"])
        payload["recall"] = float(metrics["recall"])
    else:
        payload["macro_f1"] = float(metrics["macro_f1"])
    return payload


def build_exam_group_analysis(task: str, test_rows, predictions: list[str], top_k: int = 5) -> dict:
    exam_counter = Counter(str(row.get("exam_name") or "") for row in test_rows)
    top_exam_names = {name for name, _ in exam_counter.most_common(top_k)}
    groups: dict[str, list[int]] = defaultdict(list)
    y_true = get_gold_labels(task, test_rows)
    for idx, row in enumerate(test_rows):
        group = exam_group_key(str(row.get("exam_name") or ""), top_exam_names)
        groups[group].append(idx)

    analysis = {}
    for group, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        gold = [y_true[idx] for idx in indices]
        pred = [predictions[idx] for idx in indices]
        if task == "size":
            from scripts.phase5.analysis_common import evaluate_task

            metrics = evaluate_task(task, gold, pred)
            analysis[group] = {
                "count": len(indices),
                "f1": float(metrics["f1"]),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        else:
            from scripts.phase5.analysis_common import evaluate_task

            metrics = evaluate_task(task, gold, pred)
            analysis[group] = {
                "count": len(indices),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
    return analysis


def main() -> None:
    dataset_dir = PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    output_dir = PROJECT_ROOT / "outputs" / "phase5" / "results" / "ablation"
    log_path = PROJECT_ROOT / "logs" / "run_ablation.log"
    ensure_dir(output_dir)

    with Logger(log_path) as logger:
        total_start = time.perf_counter()
        logger.log("[Start] run_ablation")
        logger.log(f"[Paths] dataset_dir={dataset_dir} output_dir={output_dir}")
        prepare_hf_environment(logger)
        task_splits = {task: load_task_splits(task, dataset_dir) for task in ("density", "size", "location")}

        logger.log("[Step 1/4] A1 section-aware mention_text vs full_text")
        ablation_a1 = {}
        for task in ("density", "size", "location"):
            start = time.perf_counter()
            mention_result = run_svm_experiment(task, task_splits[task]["train"], task_splits[task]["test"], text_field="mention_text")
            full_text_result = run_svm_experiment(task, task_splits[task]["train"], task_splits[task]["test"], text_field="full_text")
            ablation_a1[task] = {
                "mention_text": compact_metrics(task, mention_result),
                "full_text": compact_metrics(task, full_text_result),
                "delta_primary_metric": round(
                    compact_metrics(task, mention_result)["primary_metric"] - compact_metrics(task, full_text_result)["primary_metric"],
                    6,
                ),
            }
            logger.log(f"[A1] task={task} done | elapsed={format_seconds(time.perf_counter() - start)}")
        save_json(output_dir / "ablation_a1_window.json", ablation_a1)

        logger.log("[Step 2/4] A2 exam_name 特征增强 vs 默认 mention_text，并按 exam_name 分组评测")
        ablation_a2 = {}
        for task in ("density", "size", "location"):
            start = time.perf_counter()
            plain_result = run_svm_experiment(task, task_splits[task]["train"], task_splits[task]["test"], text_field="mention_text")
            exam_augmented_result = run_svm_experiment(
                task,
                task_splits[task]["train"],
                task_splits[task]["test"],
                text_field="mention_text",
                prepend_exam=True,
            )
            ablation_a2[task] = {
                "plain_mention_text": compact_metrics(task, plain_result),
                "mention_text_plus_exam_name": compact_metrics(task, exam_augmented_result),
                "delta_primary_metric": round(
                    compact_metrics(task, exam_augmented_result)["primary_metric"] - compact_metrics(task, plain_result)["primary_metric"],
                    6,
                ),
                "exam_group_analysis": build_exam_group_analysis(task, task_splits[task]["test"], plain_result["predictions"]),
            }
            logger.log(f"[A2] task={task} done | elapsed={format_seconds(time.perf_counter() - start)}")
        save_json(output_dir / "ablation_a2_exam_filter.json", ablation_a2)

        logger.log("[Step 3/4] A3 explicit-only 训练子集 vs all silver labels")
        ablation_a3 = {}
        for task in ("density", "size", "location"):
            start = time.perf_counter()
            all_train_rows = task_splits[task]["train"]
            if task == "size":
                explicit_rows = [
                    row
                    for row in all_train_rows
                    if row.get("label_quality") == "explicit" or not bool(row.get("has_size"))
                ]
                subset_note = "size 任务中 explicit 样本几乎全为正类；为保证二分类可训练，保留全部 no_size 负类并仅收缩正类到 explicit 子集。"
            else:
                explicit_rows = [row for row in all_train_rows if row.get("label_quality") == "explicit"]
                subset_note = "仅使用 label_quality == explicit 的训练样本。"
            all_result = run_svm_experiment(task, all_train_rows, task_splits[task]["test"])
            explicit_result = run_svm_experiment(task, explicit_rows, task_splits[task]["test"])
            ablation_a3[task] = {
                "train_counts": {"all": len(all_train_rows), "explicit_only": len(explicit_rows)},
                "subset_note": subset_note,
                "all_silver_labels": compact_metrics(task, all_result),
                "explicit_only": compact_metrics(task, explicit_result),
                "delta_primary_metric": round(
                    compact_metrics(task, explicit_result)["primary_metric"] - compact_metrics(task, all_result)["primary_metric"],
                    6,
                ),
            }
            logger.log(f"[A3] task={task} done | explicit_train={len(explicit_rows)} | elapsed={format_seconds(time.perf_counter() - start)}")
        save_json(output_dir / "ablation_a3_label_quality.json", ablation_a3)

        logger.log("[Step 4/4] 写出 ablation_summary.json")
        summary = {
            "a1_window": {
                task: {
                    "mention_text_primary_metric": ablation_a1[task]["mention_text"]["primary_metric"],
                    "full_text_primary_metric": ablation_a1[task]["full_text"]["primary_metric"],
                    "delta_primary_metric": ablation_a1[task]["delta_primary_metric"],
                }
                for task in ("density", "size", "location")
            },
            "a2_exam_feature": {
                task: {
                    "plain_primary_metric": ablation_a2[task]["plain_mention_text"]["primary_metric"],
                    "plus_exam_primary_metric": ablation_a2[task]["mention_text_plus_exam_name"]["primary_metric"],
                    "delta_primary_metric": ablation_a2[task]["delta_primary_metric"],
                }
                for task in ("density", "size", "location")
            },
            "a3_label_quality": {
                task: {
                    "all_primary_metric": ablation_a3[task]["all_silver_labels"]["primary_metric"],
                    "explicit_only_primary_metric": ablation_a3[task]["explicit_only"]["primary_metric"],
                    "delta_primary_metric": ablation_a3[task]["delta_primary_metric"],
                }
                for task in ("density", "size", "location")
            },
            "takeaways": [
                "A1 用 full_text 替代 mention_text，用于验证 section-aware 窗口是否更高效且更准。",
                "A2 同时给出 exam_name 特征增强与 exam_name 分组表现，作为 chest CT 过滤设计的简化讨论证据。",
                "A3 检验只用 explicit 样本训练是否会牺牲覆盖率，从而反证保留全部 silver labels 的合理性。",
            ],
        }
        save_json(output_dir / "ablation_summary.json", summary)
        logger.log(f"[Done] run_ablation completed in {format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
