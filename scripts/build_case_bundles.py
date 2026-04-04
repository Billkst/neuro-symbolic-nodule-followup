import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemblers.case_bundle_assembler import assemble_case_bundles


def _log(msg: str, log_fp) -> None:
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiology-facts", default="outputs/radiology_facts.jsonl")
    parser.add_argument("--smoking-results", default="outputs/smoking_results.jsonl", help="可选")
    parser.add_argument("--output", default="outputs/case_bundles.jsonl")
    args = parser.parse_args()

    log_path = Path("logs/build_case_bundles.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] build_case_bundles", log_fp)
        _log(
            f"[Config] radiology_facts={args.radiology_facts} smoking_results={args.smoking_results} output={args.output}",
            log_fp,
        )

        radiology_path = Path(args.radiology_facts)
        if not radiology_path.exists():
            raise FileNotFoundError(f"radiology facts 文件不存在: {radiology_path}")
        radiology_facts = _load_jsonl(radiology_path)
        _log(f"[Data] radiology_facts_loaded={len(radiology_facts)}", log_fp)

        smoking_path = Path(args.smoking_results)
        smoking_dict = None
        smoking_loaded = 0
        if smoking_path.exists():
            smoking_rows = _load_jsonl(smoking_path)
            smoking_loaded = len(smoking_rows)
            smoking_dict = {}
            for row in smoking_rows:
                sid = _to_int(row.get("subject_id"), 0)
                if sid <= 0:
                    continue
                if sid not in smoking_dict:
                    smoking_dict[sid] = row
            _log(f"[Data] smoking_results_loaded={smoking_loaded} unique_subjects={len(smoking_dict)}", log_fp)
        else:
            _log(f"[Data] smoking_results_not_found={smoking_path} -> 使用 None", log_fp)

        bundles = assemble_case_bundles(
            radiology_facts=radiology_facts,
            smoking_results=smoking_dict,
            demographics=None,
        )

        with output_path.open("w", encoding="utf-8", buffering=1) as out_fp:
            for bundle in bundles:
                out_fp.write(json.dumps(bundle, ensure_ascii=False) + "\n")
                out_fp.flush()

        quality_counter = Counter(bundle.get("label_quality") for bundle in bundles)

        _log(f"[Summary] bundles_assembled={len(bundles)}", log_fp)
        _log("[Summary] label_quality_distribution", log_fp)
        for k, v in quality_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Done] build_case_bundles completed", log_fp)


if __name__ == "__main__":
    main()
