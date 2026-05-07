import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.schema_validator import validate_instance
from src.rules.cdsg_executor import CDSGExecutor
from src.rules.lung_rads_engine import generate_recommendation as generate_flat_recommendation


def _log(msg: str, log_fp) -> None:
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _recommendation_tuple(rec: dict) -> tuple:
    return (
        rec.get("recommendation_level"),
        rec.get("recommendation_action"),
        rec.get("followup_interval"),
        rec.get("followup_modality"),
    )


def _classify_mismatch(flat: dict, cdsg: dict) -> str:
    cdsg_missing = set(cdsg.get("missing_information") or [])
    flat_missing = set(flat.get("missing_information") or [])
    cdsg_abstains = bool(cdsg.get("abstention_reason"))
    flat_actionable = flat.get("recommendation_level") != "insufficient_data"

    if cdsg_abstains and flat_actionable and "density_category" in cdsg_missing:
        return "case_bundle_density_missing_or_unclear__cdsg_abstains__flat_solid_fallback"
    if cdsg_abstains and flat_actionable and "nodule_size" in cdsg_missing:
        return "case_bundle_size_missing__cdsg_abstains"
    if cdsg_abstains and not flat_actionable:
        return "both_insufficient_but_abstention_metadata_differs"
    if flat.get("lung_rads_category") != cdsg.get("lung_rads_category"):
        if "density_category" in flat_missing and "density_category" not in cdsg_missing:
            return "dominant_selection_delta_from_flat_density_fallback"
        return "category_delta_from_graph_or_modifier"
    if _recommendation_tuple(flat) != _recommendation_tuple(cdsg):
        return "terminal_template_delta"
    if cdsg_missing != flat_missing:
        return "missing_info_policy_delta"
    return "metadata_or_schema_delta"


def _write_metric_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_mismatches(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "mismatch_reason",
        "flat_recommendation_level",
        "cdsg_recommendation_level",
        "flat_lung_rads_category",
        "cdsg_lung_rads_category",
        "flat_followup_interval",
        "cdsg_followup_interval",
        "flat_followup_modality",
        "cdsg_followup_modality",
        "cdsg_abstention_reason",
        "flat_missing_information",
        "cdsg_missing_information",
        "cdsg_terminal_node_id",
        "cdsg_visited_nodes",
        "cdsg_matched_edges",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Module 3 CDSG output with flat Lung-RADS baseline.")
    parser.add_argument("--input", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument("--graph", default="outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json")
    parser.add_argument("--comparison", default="outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_comparison.csv")
    parser.add_argument("--mismatches", default="outputs/phaseA3/tables/cdsg_vs_flat_lung_rads_mismatches.csv")
    args = parser.parse_args()

    log_path = Path("logs/phaseA3_compare_cdsg_with_flat_lung_rads.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] compare_cdsg_with_flat_lung_rads", log_fp)
        bundles = _load_jsonl(Path(args.input))
        executor = CDSGExecutor.from_path(args.graph)

        total = len(bundles)
        exact_match_count = 0
        recommendation_match_count = 0
        category_match_count = 0
        cdsg_schema_valid_count = 0
        cdsg_abstention_count = 0
        flat_insufficient_count = 0
        hard_evaluable_count = 0
        hard_evaluable_recommendation_match_count = 0
        hard_evaluable_category_match_count = 0
        strict_hard_comparable_count = 0
        strict_hard_comparable_recommendation_match_count = 0
        strict_hard_comparable_category_match_count = 0
        flat_density_fallback_count = 0
        mismatch_rows: list[dict[str, str]] = []
        mismatch_reason_counter = Counter()
        cdsg_abstention_counter = Counter()

        for idx, bundle in enumerate(bundles, start=1):
            flat = generate_flat_recommendation(bundle)
            cdsg = executor.execute(bundle)
            schema_errors = validate_instance(cdsg, "module3_recommendation_schema.json")
            if not schema_errors:
                cdsg_schema_valid_count += 1

            recommendation_match = _recommendation_tuple(flat) == _recommendation_tuple(cdsg)
            category_match = flat.get("lung_rads_category") == cdsg.get("lung_rads_category")
            exact_match = recommendation_match and category_match
            flat_missing = set(flat.get("missing_information") or [])

            if recommendation_match:
                recommendation_match_count += 1
            if category_match:
                category_match_count += 1
            if exact_match:
                exact_match_count += 1

            if flat.get("recommendation_level") == "insufficient_data":
                flat_insufficient_count += 1
            if cdsg.get("abstention_reason"):
                cdsg_abstention_count += 1
                cdsg_abstention_counter[str(cdsg.get("abstention_reason"))] += 1
            else:
                hard_evaluable_count += 1
                if recommendation_match:
                    hard_evaluable_recommendation_match_count += 1
                if category_match:
                    hard_evaluable_category_match_count += 1

            if "density_category" in flat_missing:
                flat_density_fallback_count += 1

            if not cdsg.get("abstention_reason") and "density_category" not in flat_missing:
                strict_hard_comparable_count += 1
                if recommendation_match:
                    strict_hard_comparable_recommendation_match_count += 1
                if category_match:
                    strict_hard_comparable_category_match_count += 1

            if not exact_match:
                reason = _classify_mismatch(flat, cdsg)
                mismatch_reason_counter[reason] += 1
                mismatch_rows.append(
                    {
                        "case_id": str(bundle.get("case_id")),
                        "mismatch_reason": reason,
                        "flat_recommendation_level": str(flat.get("recommendation_level")),
                        "cdsg_recommendation_level": str(cdsg.get("recommendation_level")),
                        "flat_lung_rads_category": str(flat.get("lung_rads_category")),
                        "cdsg_lung_rads_category": str(cdsg.get("lung_rads_category")),
                        "flat_followup_interval": str(flat.get("followup_interval")),
                        "cdsg_followup_interval": str(cdsg.get("followup_interval")),
                        "flat_followup_modality": str(flat.get("followup_modality")),
                        "cdsg_followup_modality": str(cdsg.get("followup_modality")),
                        "cdsg_abstention_reason": str(cdsg.get("abstention_reason")),
                        "flat_missing_information": "|".join(flat.get("missing_information") or []),
                        "cdsg_missing_information": "|".join(cdsg.get("missing_information") or []),
                        "cdsg_terminal_node_id": str(cdsg.get("generation_metadata", {}).get("terminal_node_id")),
                        "cdsg_visited_nodes": "|".join(cdsg.get("visited_nodes") or []),
                        "cdsg_matched_edges": "|".join(cdsg.get("matched_edges") or []),
                    }
                )

            if idx % 50 == 0:
                _log(f"[Progress] compared={idx}/{total}", log_fp)

        metric_rows: list[dict[str, str]] = [
            {"metric": "total_cases", "value": str(total)},
            {"metric": "cdsg_schema_valid_count", "value": str(cdsg_schema_valid_count)},
            {"metric": "cdsg_schema_valid_rate", "value": f"{_rate(cdsg_schema_valid_count, total):.6f}"},
            {"metric": "exact_match_count", "value": str(exact_match_count)},
            {"metric": "exact_match_rate", "value": f"{_rate(exact_match_count, total):.6f}"},
            {"metric": "recommendation_match_count", "value": str(recommendation_match_count)},
            {"metric": "recommendation_match_rate", "value": f"{_rate(recommendation_match_count, total):.6f}"},
            {"metric": "category_match_count", "value": str(category_match_count)},
            {"metric": "category_match_rate", "value": f"{_rate(category_match_count, total):.6f}"},
            {"metric": "hard_evaluable_count", "value": str(hard_evaluable_count)},
            {
                "metric": "hard_evaluable_recommendation_match_rate",
                "value": f"{_rate(hard_evaluable_recommendation_match_count, hard_evaluable_count):.6f}",
            },
            {
                "metric": "hard_evaluable_category_match_rate",
                "value": f"{_rate(hard_evaluable_category_match_count, hard_evaluable_count):.6f}",
            },
            {"metric": "strict_hard_comparable_count", "value": str(strict_hard_comparable_count)},
            {
                "metric": "strict_hard_comparable_recommendation_match_rate",
                "value": f"{_rate(strict_hard_comparable_recommendation_match_count, strict_hard_comparable_count):.6f}",
            },
            {
                "metric": "strict_hard_comparable_category_match_rate",
                "value": f"{_rate(strict_hard_comparable_category_match_count, strict_hard_comparable_count):.6f}",
            },
            {"metric": "flat_density_fallback_count", "value": str(flat_density_fallback_count)},
            {"metric": "flat_insufficient_count", "value": str(flat_insufficient_count)},
            {"metric": "cdsg_abstention_count", "value": str(cdsg_abstention_count)},
            {"metric": "cdsg_abstention_rate", "value": f"{_rate(cdsg_abstention_count, total):.6f}"},
            {"metric": "mismatch_count", "value": str(len(mismatch_rows))},
        ]
        for key, value in sorted(cdsg_abstention_counter.items()):
            metric_rows.append({"metric": f"cdsg_abstention_reason.{key}", "value": str(value)})
        for key, value in sorted(mismatch_reason_counter.items()):
            metric_rows.append({"metric": f"mismatch_reason.{key}", "value": str(value)})

        _write_metric_rows(Path(args.comparison), metric_rows)
        _write_mismatches(Path(args.mismatches), mismatch_rows)
        _log(f"[Summary] total={total}", log_fp)
        _log(f"[Summary] recommendation_match_rate={_rate(recommendation_match_count, total):.6f}", log_fp)
        _log(f"[Summary] category_match_rate={_rate(category_match_count, total):.6f}", log_fp)
        _log(f"[Summary] hard_evaluable_recommendation_match_rate={_rate(hard_evaluable_recommendation_match_count, hard_evaluable_count):.6f}", log_fp)
        _log(f"[Summary] cdsg_abstention_count={cdsg_abstention_count}", log_fp)
        _log(f"[Done] comparison={args.comparison} mismatches={args.mismatches}", log_fp)


if __name__ == "__main__":
    main()
