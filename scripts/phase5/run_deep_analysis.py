from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase5.analysis_common import (
    DEFAULT_RANDOM_SEED,
    DENSITY_LABELS,
    LOCATION_LABELS,
    LocalBertClassifier,
    Logger,
    compute_distribution,
    ensure_dir,
    format_seconds,
    get_gold_labels,
    load_task_splits,
    predict_regex,
    prepare_hf_environment,
    run_svm_experiment,
    safe_round,
    sample_rows,
    save_json,
    summarize_confidence,
    support_assessment,
)


def build_failure_region_payload(task: str, rows, predictions, selected_indices, note: str) -> dict:
    sampled_cases = []
    for idx in selected_indices:
        row = rows[idx]
        prediction = predictions[idx]
        sampled_cases.append(
            {
                "sample_id": row.get("sample_id"),
                "note_id": row.get("note_id"),
                "exam_name": row.get("exam_name"),
                "section": row.get("section"),
                "mention_text": row.get("mention_text"),
                "predicted_label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probability_by_label,
                "text_support": support_assessment(task, prediction.label, str(row.get("mention_text") or "")),
            }
        )

    support_counter = Counter(case["text_support"]["assessment"] for case in sampled_cases)
    return {
        "task": task,
        "subset_size": len(rows),
        "prediction_distribution": compute_distribution(pred.label for pred in predictions),
        "non_default_prediction_count": len(selected_indices),
        "sampled_case_count": len(sampled_cases),
        "sampled_text_support_distribution": {key: int(value) for key, value in sorted(support_counter.items())},
        "note": note,
        "sampled_cases": sampled_cases,
    }


def build_agreement_payload(task: str, rows, regex_predictions, svm_predictions, bert_predictions) -> dict:
    total = len(rows)
    triplets = []
    disagreement_examples = []
    all_same = 0
    bert_only = 0
    regex_only = 0
    svm_only = 0
    all_diff = 0
    pairwise = {"regex_vs_svm": 0, "regex_vs_bert": 0, "svm_vs_bert": 0}
    disagreement_counter = Counter()

    for row, regex_pred, svm_pred, bert_pred in zip(rows, regex_predictions, svm_predictions, bert_predictions, strict=False):
        bert_label = bert_pred.label
        pairwise["regex_vs_svm"] += int(regex_pred == svm_pred)
        pairwise["regex_vs_bert"] += int(regex_pred == bert_label)
        pairwise["svm_vs_bert"] += int(svm_pred == bert_label)

        if regex_pred == svm_pred == bert_label:
            all_same += 1
        elif regex_pred == svm_pred != bert_label:
            bert_only += 1
        elif regex_pred == bert_label != svm_pred:
            svm_only += 1
        elif svm_pred == bert_label != regex_pred:
            regex_only += 1
        else:
            all_diff += 1

        triplet_key = f"regex={regex_pred} | svm={svm_pred} | bert={bert_label}"
        disagreement_counter[triplet_key] += 1
        if len({regex_pred, svm_pred, bert_label}) > 1 and len(disagreement_examples) < 50:
            disagreement_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "mention_text": row.get("mention_text"),
                    "gold_label": get_gold_labels(task, [row])[0],
                    "regex_prediction": regex_pred,
                    "svm_prediction": svm_pred,
                    "bert_prediction": bert_label,
                    "bert_confidence": bert_pred.confidence,
                }
            )

    return {
        "task": task,
        "sample_count": total,
        "pairwise_agreement_rate": {key: safe_round(value / total if total else 0.0) for key, value in pairwise.items()},
        "three_way_agreement": {
            "all_same_count": all_same,
            "all_same_rate": safe_round(all_same / total if total else 0.0),
            "bert_only_disagree_count": bert_only,
            "regex_only_disagree_count": regex_only,
            "svm_only_disagree_count": svm_only,
            "all_different_count": all_diff,
        },
        "top_prediction_patterns": [
            {"pattern": pattern, "count": int(count)}
            for pattern, count in disagreement_counter.most_common(20)
        ],
        "sample_disagreements": disagreement_examples,
    }


def main() -> None:
    dataset_dir = PROJECT_ROOT / "outputs" / "phase5" / "datasets"
    model_dir = PROJECT_ROOT / "outputs" / "phase5" / "models"
    output_dir = PROJECT_ROOT / "outputs" / "phase5" / "results" / "deep_analysis"
    log_path = PROJECT_ROOT / "logs" / "run_deep_analysis.log"
    ensure_dir(output_dir)
    rng = random.Random(DEFAULT_RANDOM_SEED)

    with Logger(log_path) as logger:
        total_start = time.perf_counter()
        logger.log("[Start] run_deep_analysis")
        logger.log(f"[Paths] dataset_dir={dataset_dir} model_dir={model_dir} output_dir={output_dir}")
        prepare_hf_environment(logger)

        task_splits = {task: load_task_splits(task, dataset_dir) for task in ("density", "size", "location")}
        for task, splits in task_splits.items():
            logger.log(f"[Data] task={task} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        logger.log("[Step 1/5] 加载本地 PubMedBERT checkpoint")
        bert_models = {
            task: LocalBertClassifier(task, model_dir / f"{task}_pubmedbert")
            for task in ("density", "size", "location")
        }

        logger.log("[Step 2/5] 分析 regex 失败区域")
        density_unclear_rows = [row for row in task_splits["density"]["test"] if row.get("density_label") == "unclear"]
        density_predictions = bert_models["density"].predict_rows(density_unclear_rows)
        density_candidate_indices = [idx for idx, pred in enumerate(density_predictions) if pred.label != "unclear"]
        density_selected = sample_rows(density_unclear_rows, density_candidate_indices, 20, rng)
        density_payload = build_failure_region_payload(
            "density",
            density_unclear_rows,
            density_predictions,
            density_selected,
            "关注 regex 标为 unclear 的样本中，BERT 是否能给出更具体密度判断。",
        )
        save_json(output_dir / "density_unclear_predictions.json", density_payload)
        save_json(output_dir / "sampled_predictions_density.json", density_payload["sampled_cases"])
        logger.log(
            f"[Density] unclear_subset={len(density_unclear_rows)} non_unclear_predictions={len(density_candidate_indices)}"
        )

        size_null_rows = [row for row in task_splits["size"]["test"] if not bool(row.get("has_size"))]
        size_predictions = bert_models["size"].predict_rows(size_null_rows)
        size_candidate_indices = [idx for idx, pred in enumerate(size_predictions) if pred.label == "has_size"]
        size_selected = sample_rows(size_null_rows, size_candidate_indices, 20, rng)
        size_payload = build_failure_region_payload(
            "size",
            size_null_rows,
            size_predictions,
            size_selected,
            "关注 regex 未识别到尺寸的样本中，BERT 是否能补出有意义的尺寸信号。",
        )
        save_json(output_dir / "size_null_predictions.json", size_payload)
        save_json(output_dir / "sampled_predictions_size.json", size_payload["sampled_cases"])
        logger.log(f"[Size] no_size_subset={len(size_null_rows)} predicted_has_size={len(size_candidate_indices)}")

        location_null_rows = [row for row in task_splits["location"]["test"] if row.get("location_label") == "no_location"]
        location_predictions = bert_models["location"].predict_rows(location_null_rows)
        location_candidate_indices = [
            idx for idx, pred in enumerate(location_predictions) if pred.label not in {"no_location", "unclear"}
        ]
        location_selected = sample_rows(location_null_rows, location_candidate_indices, 20, rng)
        location_payload = build_failure_region_payload(
            "location",
            location_null_rows,
            location_predictions,
            location_selected,
            "关注 regex 标为 no_location 的样本中，BERT 是否能补出具体肺叶位置。",
        )
        save_json(output_dir / "location_null_predictions.json", location_payload)
        save_json(output_dir / "sampled_predictions_location.json", location_payload["sampled_cases"])
        logger.log(
            f"[Location] no_location_subset={len(location_null_rows)} predicted_specific_location={len(location_candidate_indices)}"
        )

        logger.log("[Step 3/5] 训练 ML-SVM 并做三方法一致性分析")
        agreement_payload = {}
        confidence_payload = {}
        summary: dict[str, Any] = {"failure_region_analysis": {}, "agreement_analysis": {}, "confidence_analysis": {}}

        for task in ("density", "size", "location"):
            task_start = time.perf_counter()
            train_rows = task_splits[task]["train"]
            test_rows = task_splits[task]["test"]
            logger.log(f"[Agreement] task={task} | 训练 ML-SVM")
            svm_run = run_svm_experiment(task, train_rows, test_rows)
            regex_predictions = predict_regex(task, test_rows)
            bert_predictions = bert_models[task].predict_rows(test_rows)
            agreement_payload[task] = build_agreement_payload(
                task,
                test_rows,
                regex_predictions,
                svm_run["predictions"],
                bert_predictions,
            )
            confidence_payload[task] = summarize_confidence(task, get_gold_labels(task, test_rows), bert_predictions, test_rows)
            summary["agreement_analysis"][task] = agreement_payload[task]["three_way_agreement"]
            summary["confidence_analysis"][task] = confidence_payload[task]["bucket_summary"]
            logger.log(
                f"[Agreement] task={task} done | three_way={agreement_payload[task]['three_way_agreement']['all_same_rate']:.4f} | elapsed={format_seconds(time.perf_counter() - task_start)}"
            )

        save_json(output_dir / "model_agreement_analysis.json", agreement_payload)
        save_json(output_dir / "confidence_analysis.json", confidence_payload)

        logger.log("[Step 4/5] 汇总 failure region 结果")
        summary["failure_region_analysis"] = {
            "density": {
                "subset_size": density_payload["subset_size"],
                "distribution": density_payload["prediction_distribution"],
                "sampled_support_distribution": density_payload["sampled_text_support_distribution"],
            },
            "size": {
                "subset_size": size_payload["subset_size"],
                "distribution": size_payload["prediction_distribution"],
                "sampled_support_distribution": size_payload["sampled_text_support_distribution"],
            },
            "location": {
                "subset_size": location_payload["subset_size"],
                "distribution": location_payload["prediction_distribution"],
                "sampled_support_distribution": location_payload["sampled_text_support_distribution"],
            },
        }
        summary["takeaways"] = [
            "silver label 场景下整体分数几乎饱和，因此重点转移到 regex 失败区域的补充预测能力。",
            "sampled_text_support_distribution 用于近似人工检查：若 supported 占比更高，说明 BERT 在 regex 空白区可能携带增量价值。",
            "confidence_analysis 的准确率仍基于 silver label，应仅作为置信度稳定性讨论，而非最终人工真实性能。",
        ]

        logger.log("[Step 5/5] 写出 deep_analysis_summary.json")
        save_json(output_dir / "deep_analysis_summary.json", summary)
        logger.log(f"[Done] run_deep_analysis completed in {format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
