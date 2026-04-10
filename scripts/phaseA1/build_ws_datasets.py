from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weak_supervision.aggregation import weighted_majority_vote
from src.weak_supervision.base import ABSTAIN, LFOutput, MentionRecord
from src.weak_supervision.labeling_functions import ALL_LFS
from src.weak_supervision.quality_gate import GATE_ORDER, evaluate_gate


DEFAULT_WEIGHTS = {
    "density": {
        "LF-D1": 1.0, "LF-D2": 0.75, "LF-D3": 0.85, "LF-D4": 0.9, "LF-D5": 0.85,
    },
    "size": {
        "LF-S1": 1.0, "LF-S2": 0.9, "LF-S3": 0.8, "LF-S4": 0.7, "LF-S5": 0.9,
    },
    "location": {
        "LF-L1": 1.0, "LF-L2": 0.85, "LF-L3": 0.9, "LF-L4": 0.75, "LF-L5": 0.7,
    },
}

TASKS = ["density", "size", "location"]
SPLITS = ["train", "val", "test"]


def _log(msg: str, fp=None) -> None:
    print(msg, flush=True)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_lfs_for_task(
    record: MentionRecord,
    task: str,
    lf_list: list,
) -> list[LFOutput]:
    outputs = []
    for lf_fn in lf_list:
        try:
            result = lf_fn(record)
            outputs.append(result)
        except Exception:
            outputs.append(LFOutput(lf_name=lf_fn.__name__, label=ABSTAIN))
    return outputs


def process_record(
    record: MentionRecord,
    task: str,
    lf_list: list,
    weights: dict[str, float],
) -> dict:
    lf_outputs = run_lfs_for_task(record, task, lf_list)
    agg = weighted_majority_vote(lf_outputs, weights=weights)
    gate = evaluate_gate(agg)

    lf_details = []
    for o in lf_outputs:
        lf_details.append({
            "lf_name": o.lf_name,
            "label": o.label,
            "confidence": o.confidence,
            "evidence_span": o.evidence_span,
        })

    return {
        "sample_id": record["sample_id"],
        "note_id": record["note_id"],
        "subject_id": record["subject_id"],
        "exam_name": record.get("exam_name", ""),
        "section": record.get("section", ""),
        "mention_text": record["mention_text"],
        "full_text": record.get("full_text", ""),

        "ws_label": agg.label,
        "ws_confidence": round(agg.confidence, 4),
        "lf_coverage": agg.lf_coverage,
        "lf_agreement": round(agg.lf_agreement, 4),
        "supporting_lfs": agg.supporting_lfs,
        "all_votes": {k: round(v, 4) for k, v in agg.all_votes.items()},
        "evidence_spans": agg.evidence_spans,

        "gate_level": gate.gate_level,
        "passed_gates": gate.passed_gates,

        "lf_details": lf_details,

        "original_label": _get_original_label(record, task),
        "label_quality": record.get("label_quality", ""),
    }


def _get_original_label(record: MentionRecord, task: str) -> str:
    if task == "density":
        return record.get("density_label", "unclear")
    elif task == "size":
        return str(record.get("has_size", False)).lower()
    elif task == "location":
        loc = record.get("location_label")
        return loc if loc is not None else "no_location"
    return ""


def build_training_record(ws_record: dict, task: str) -> dict:
    rec = {
        "sample_id": ws_record["sample_id"],
        "note_id": ws_record["note_id"],
        "subject_id": ws_record["subject_id"],
        "exam_name": ws_record["exam_name"],
        "section": ws_record["section"],
        "mention_text": ws_record["mention_text"],
        "full_text": ws_record["full_text"],
        "label_quality": ws_record["label_quality"],
        "ws_confidence": ws_record["ws_confidence"],
        "lf_coverage": ws_record["lf_coverage"],
        "gate_level": ws_record["gate_level"],
        "passed_gates": ws_record["passed_gates"],
    }

    if task == "density":
        rec["density_label"] = ws_record["ws_label"]
    elif task == "size":
        rec["has_size"] = ws_record["ws_label"] == "true"
        rec["size_label"] = None
        rec["size_text"] = None
    elif task == "location":
        rec["location_label"] = ws_record["ws_label"]
        rec["has_location"] = ws_record["ws_label"] not in ("no_location", ABSTAIN)

    return rec


def compute_lf_stats(ws_records: list[dict], task: str) -> dict:
    lf_names = sorted({d["lf_name"] for r in ws_records for d in r["lf_details"]})
    total = len(ws_records)

    coverage = {}
    label_dist_per_lf = {}
    for lf_name in lf_names:
        non_abstain = 0
        labels = Counter()
        for r in ws_records:
            for d in r["lf_details"]:
                if d["lf_name"] == lf_name:
                    if d["label"] != ABSTAIN:
                        non_abstain += 1
                        labels[d["label"]] += 1
        coverage[lf_name] = {
            "count": non_abstain,
            "rate": round(non_abstain / total, 4) if total > 0 else 0,
        }
        label_dist_per_lf[lf_name] = dict(labels.most_common())

    conflict_count = 0
    for r in ws_records:
        active = [d["label"] for d in r["lf_details"] if d["label"] != ABSTAIN]
        if len(set(active)) > 1:
            conflict_count += 1
    conflict_rate = round(conflict_count / total, 4) if total > 0 else 0

    pairwise_overlap = {}
    for i, lf_a in enumerate(lf_names):
        for lf_b in lf_names[i + 1:]:
            both_active = 0
            both_agree = 0
            for r in ws_records:
                details = {d["lf_name"]: d["label"] for d in r["lf_details"]}
                a_label = details.get(lf_a, ABSTAIN)
                b_label = details.get(lf_b, ABSTAIN)
                if a_label != ABSTAIN and b_label != ABSTAIN:
                    both_active += 1
                    if a_label == b_label:
                        both_agree += 1
            pair_key = f"{lf_a}+{lf_b}"
            pairwise_overlap[pair_key] = {
                "overlap_count": both_active,
                "overlap_rate": round(both_active / total, 4) if total > 0 else 0,
                "agreement_rate": round(both_agree / both_active, 4) if both_active > 0 else 0,
            }

    agg_label_dist = Counter(r["ws_label"] for r in ws_records)
    gate_dist = Counter(r["gate_level"] for r in ws_records)

    return {
        "task": task,
        "total_samples": total,
        "lf_coverage": coverage,
        "lf_label_distributions": label_dist_per_lf,
        "conflict_rate": conflict_rate,
        "conflict_count": conflict_count,
        "pairwise_overlap": pairwise_overlap,
        "aggregated_label_distribution": dict(agg_label_dist.most_common()),
        "gate_distribution": {g: gate_dist.get(g, 0) for g in GATE_ORDER + ["REJECTED"]},
        "gate_retention": {
            g: sum(1 for r in ws_records if g in r["passed_gates"])
            for g in GATE_ORDER
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/phase5/datasets")
    parser.add_argument("--output-dir", default="outputs/phaseA1")
    parser.add_argument("--log", default="logs/build_ws_datasets.log")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log(f"[Start] build_ws_datasets", log_fp)
        _log(f"[Config] input_dir={input_dir} output_dir={output_dir} nrows={args.nrows} tasks={args.tasks}", log_fp)

        all_stats = {}

        for task in args.tasks:
            _log(f"\n[Task: {task}]", log_fp)
            lf_list = ALL_LFS[task]
            weights = DEFAULT_WEIGHTS[task]
            _log(f"  LFs: {[fn.__name__ for fn in lf_list]}", log_fp)
            _log(f"  Weights: {weights}", log_fp)

            task_output_dir = output_dir / task
            task_output_dir.mkdir(parents=True, exist_ok=True)

            for split in SPLITS:
                src_file = input_dir / f"{task}_{split}.jsonl"
                if not src_file.exists():
                    _log(f"  [WARN] {src_file} not found, skipping", log_fp)
                    continue

                _log(f"  [Split: {split}] loading {src_file}", log_fp)
                records = load_jsonl(src_file)
                if args.nrows:
                    records = records[:args.nrows]
                _log(f"    loaded {len(records)} records", log_fp)

                t0 = time.time()
                ws_records = []
                for idx, record in enumerate(records):
                    ws_rec = process_record(record, task, lf_list, weights)
                    ws_records.append(ws_rec)
                    if (idx + 1) % 50000 == 0:
                        _log(f"    processed {idx + 1}/{len(records)} ({time.time() - t0:.1f}s)", log_fp)

                elapsed = time.time() - t0
                _log(f"    processed {len(ws_records)} records in {elapsed:.1f}s", log_fp)

                ws_path = task_output_dir / f"ws_{split}.jsonl"
                write_jsonl(ws_path, ws_records)
                _log(f"    wrote {ws_path}", log_fp)

                non_abstain = [r for r in ws_records if r["ws_label"] != ABSTAIN]
                training_records = [build_training_record(r, task) for r in non_abstain]
                train_path = task_output_dir / f"{task}_{split}_ws.jsonl"
                write_jsonl(train_path, training_records)
                _log(f"    wrote {train_path} ({len(training_records)} non-ABSTAIN records)", log_fp)

                if split == "train":
                    _log(f"  [Stats: {task}] computing LF statistics on train split", log_fp)
                    stats = compute_lf_stats(ws_records, task)
                    all_stats[task] = stats
                    stats_path = task_output_dir / "lf_stats.json"
                    write_json(stats_path, stats)
                    _log(f"    wrote {stats_path}", log_fp)

                    _log(f"    coverage: { {k: v['rate'] for k, v in stats['lf_coverage'].items()} }", log_fp)
                    _log(f"    conflict_rate: {stats['conflict_rate']}", log_fp)
                    _log(f"    agg_label_dist: {stats['aggregated_label_distribution']}", log_fp)
                    _log(f"    gate_retention: {stats['gate_retention']}", log_fp)

                    for gate in GATE_ORDER:
                        gated = [r for r in training_records if gate in r.get("passed_gates", [])]
                        gated_path = task_output_dir / f"{task}_train_ws_{gate.lower()}.jsonl"
                        write_jsonl(gated_path, gated)
                        _log(f"    wrote {gated_path} ({len(gated)} records)", log_fp)

        summary_path = output_dir / "ws_summary.json"
        write_json(summary_path, all_stats)
        _log(f"\n[Summary] wrote {summary_path}", log_fp)
        _log("[Done] build_ws_datasets completed", log_fp)


if __name__ == "__main__":
    main()
