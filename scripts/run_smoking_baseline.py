import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.extractors.smoking_extractor import extract_smoking_eligibility


def _log(msg: str, log_fp) -> None:
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/mimic_note_extracted/note/discharge.csv.gz",
        help="输入 discharge.csv.gz 路径",
    )
    parser.add_argument("--nrows", type=int, default=None, help="仅处理前 n 行")
    parser.add_argument(
        "--output",
        default="outputs/smoking_results.jsonl",
        help="输出 JSONL 路径",
    )
    args = parser.parse_args()

    log_path = Path("logs/run_smoking_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_smoking_baseline", log_fp)
        _log(f"[Config] input={args.input} nrows={args.nrows} output={args.output}", log_fp)

        df = pd.read_csv(args.input, nrows=args.nrows)
        _log(f"[Data] loaded_rows={len(df)}", log_fp)

        text_col = "text" if "text" in df.columns else ("note_text" if "note_text" in df.columns else None)
        if text_col is None:
            raise ValueError("未找到文本列，期望存在 text 或 note_text")

        status_counter = Counter()
        quality_counter = Counter()

        with Path(args.output).open("w", encoding="utf-8", buffering=1) as out_fp:
            for row_idx, (_, row) in enumerate(df.iterrows()):
                subject_id = int(row.get("subject_id", 0) or 0)
                note_id = row.get("note_id")
                if not isinstance(note_id, str) or not note_id:
                    note_id = f"{subject_id}-DS-{row_idx}"

                text = row.get(text_col)
                result = extract_smoking_eligibility(subject_id=subject_id, note_id=note_id, text=str(text or ""))

                out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fp.flush()

                status_counter[result["smoking_status_norm"]] += 1
                quality_counter[result["evidence_quality"]] += 1

                if (row_idx + 1) % 1000 == 0:
                    _log(f"[Progress] processed={row_idx + 1}", log_fp)

        _log("[Summary] smoking_status_norm", log_fp)
        for k, v in status_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Summary] evidence_quality", log_fp)
        for k, v in quality_counter.most_common():
            _log(f"  - {k}: {v}", log_fp)

        _log("[Done] run_smoking_baseline completed", log_fp)


if __name__ == "__main__":
    main()
