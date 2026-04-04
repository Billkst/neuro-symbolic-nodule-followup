import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.lung_rads_engine import generate_recommendation


def _log(msg: str, log_fp) -> None:
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/case_bundles.jsonl")
    parser.add_argument("--output", default="outputs/recommendations.jsonl")
    args = parser.parse_args()

    log_path = Path("logs/run_recommendation_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_recommendation_baseline", log_fp)
        _log(f"[Config] input={args.input} output={args.output}", log_fp)

        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"case bundles 文件不存在: {input_path}")

        bundles = _load_jsonl(input_path)
        _log(f"[Data] total_bundles={len(bundles)}", log_fp)

        level_counter = Counter()
        category_counter = Counter()
        success_count = 0
        error_count = 0

        with output_path.open("w", encoding="utf-8", buffering=1) as out_fp:
            for idx, bundle in enumerate(bundles, start=1):
                try:
                    recommendation = generate_recommendation(bundle)
                except Exception as exc:
                    error_count += 1
                    case_id = bundle.get("case_id") if isinstance(bundle, dict) else None
                    _log(f"[Error] idx={idx} case_id={case_id} error={repr(exc)}", log_fp)
                    continue

                out_fp.write(json.dumps(recommendation, ensure_ascii=False) + "\n")
                out_fp.flush()
                success_count += 1

                level_counter[str(recommendation.get("recommendation_level") or "null")] += 1
                category_counter[str(recommendation.get("lung_rads_category") or "null")] += 1

                if idx % 50 == 0:
                    _log(f"[Progress] processed={idx} success={success_count} errors={error_count}", log_fp)

        _log(f"[Summary] success={success_count} errors={error_count}", log_fp)
        _log("[Summary] recommendation_level_distribution", log_fp)
        for k, v in level_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Summary] lung_rads_category_distribution", log_fp)
        for k, v in category_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Done] run_recommendation_baseline completed", log_fp)


if __name__ == "__main__":
    main()
