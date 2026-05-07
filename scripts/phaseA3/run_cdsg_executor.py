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


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 3 CDSG executor on Phase4 case bundles.")
    parser.add_argument("--input", default="outputs/phase4/cache/case_bundles_eval.jsonl")
    parser.add_argument("--graph", default="outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json")
    parser.add_argument("--output", default="outputs/phaseA3/recommendations/cdsg_phase4_recommendations.jsonl")
    parser.add_argument("--summary", default="outputs/phaseA3/tables/cdsg_phase4_summary.csv")
    args = parser.parse_args()

    log_path = Path("logs/phaseA3_run_cdsg_executor.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_cdsg_executor", log_fp)
        _log(f"[Config] input={args.input}", log_fp)
        _log(f"[Config] graph={args.graph}", log_fp)
        _log(f"[Config] output={args.output}", log_fp)

        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"case bundle input not found: {input_path}")

        bundles = _load_jsonl(input_path)
        executor = CDSGExecutor.from_path(args.graph)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        schema_valid_count = 0
        error_count = 0
        recommendations = []
        level_counter = Counter()
        category_counter = Counter()
        abstention_counter = Counter()
        missing_counter = Counter()

        with output_path.open("w", encoding="utf-8", buffering=1) as out_fp:
            for idx, bundle in enumerate(bundles, start=1):
                try:
                    recommendation = executor.execute(bundle)
                    errors = validate_instance(recommendation, "module3_recommendation_schema.json")
                    recommendation["_schema_errors_for_run"] = errors
                    if not errors:
                        schema_valid_count += 1
                except Exception as exc:
                    error_count += 1
                    _log(f"[Error] idx={idx} case_id={bundle.get('case_id')} error={repr(exc)}", log_fp)
                    continue

                stored = dict(recommendation)
                stored.pop("_schema_errors_for_run", None)
                out_fp.write(json.dumps(stored, ensure_ascii=False) + "\n")
                out_fp.flush()
                recommendations.append(recommendation)

                level_counter[str(recommendation.get("recommendation_level"))] += 1
                category_counter[str(recommendation.get("lung_rads_category"))] += 1
                abstention_counter[str(recommendation.get("abstention_reason"))] += 1
                missing_counter.update(recommendation.get("missing_information") or [])

                if idx % 50 == 0:
                    _log(f"[Progress] processed={idx}/{len(bundles)} errors={error_count}", log_fp)

        total = len(bundles)
        emitted = len(recommendations)
        abstention_count = sum(1 for item in recommendations if item.get("abstention_reason"))
        anchor_count = sum(1 for item in recommendations if item.get("guideline_anchor"))
        reasoning_count = sum(1 for item in recommendations if item.get("reasoning_path"))

        summary_rows: list[dict[str, str]] = [
            {"metric": "total_input_cases", "value": str(total)},
            {"metric": "emitted_recommendations", "value": str(emitted)},
            {"metric": "error_count", "value": str(error_count)},
            {"metric": "schema_valid_count", "value": str(schema_valid_count)},
            {"metric": "schema_valid_rate", "value": f"{_rate(schema_valid_count, emitted):.6f}"},
            {"metric": "guideline_anchor_nonempty_count", "value": str(anchor_count)},
            {"metric": "guideline_anchor_nonempty_rate", "value": f"{_rate(anchor_count, emitted):.6f}"},
            {"metric": "reasoning_path_nonempty_count", "value": str(reasoning_count)},
            {"metric": "reasoning_path_nonempty_rate", "value": f"{_rate(reasoning_count, emitted):.6f}"},
            {"metric": "abstention_count", "value": str(abstention_count)},
            {"metric": "abstention_rate", "value": f"{_rate(abstention_count, emitted):.6f}"},
        ]
        for key, value in sorted(level_counter.items()):
            summary_rows.append({"metric": f"recommendation_level.{key}", "value": str(value)})
        for key, value in sorted(category_counter.items()):
            summary_rows.append({"metric": f"lung_rads_category.{key}", "value": str(value)})
        for key, value in sorted(abstention_counter.items()):
            summary_rows.append({"metric": f"abstention_reason.{key}", "value": str(value)})
        for key, value in sorted(missing_counter.items()):
            summary_rows.append({"metric": f"missing_information.{key}", "value": str(value)})

        _write_summary(Path(args.summary), summary_rows)
        _log(f"[Summary] emitted={emitted} errors={error_count}", log_fp)
        _log(f"[Summary] schema_valid_rate={_rate(schema_valid_count, emitted):.6f}", log_fp)
        _log(f"[Summary] abstention_count={abstention_count}", log_fp)
        _log(f"[Done] recommendations={output_path} summary={args.summary}", log_fp)


if __name__ == "__main__":
    main()
