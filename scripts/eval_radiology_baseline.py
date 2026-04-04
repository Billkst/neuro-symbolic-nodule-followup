import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.eval.manifest_builder import load_manifest, _load_jsonl
from src.extractors.radiology_extractor import extract_radiology_facts
from src.parsers.section_parser import parse_sections


def _log(msg, log_fp):
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def extract_full_text_variant(note_id, subject_id, exam_name, report_text):
    from src.extractors.nodule_extractor import (
        extract_change_status,
        extract_density,
        extract_location,
        extract_morphology,
        extract_recommendation_cue,
        extract_size,
        segment_nodule_mentions,
    )
    from src.extractors.modality_classifier import classify_body_site, classify_modality
    from datetime import datetime, timezone

    modality = classify_modality(exam_name)
    body_site = classify_body_site(exam_name)

    nodule_mentions = segment_nodule_mentions(report_text)
    recommendation_cue = extract_recommendation_cue(report_text)

    nodules = []
    for idx, mention in enumerate(nodule_mentions, start=1):
        size_mm, size_text = extract_size(mention)
        density_category, density_text = extract_density(mention)
        location_lobe, location_text = extract_location(mention)
        change_status, change_text = extract_change_status(mention)
        morphology = extract_morphology(mention)

        score = sum([
            size_mm is not None,
            density_category not in (None, "unclear"),
            location_lobe not in (None, "unclear"),
        ])
        confidence = "high" if score == 3 else ("medium" if score == 2 else "low")

        nodule_obj = {
            "nodule_id_in_report": idx,
            "size_mm": size_mm, "size_text": size_text,
            "density_category": density_category, "density_text": density_text,
            "location_lobe": location_lobe, "location_text": location_text,
            "count_type": "single" if len(nodule_mentions) == 1 else ("multiple" if len(nodule_mentions) > 1 else "unclear"),
            "change_status": change_status, "change_text": change_text,
            "calcification": morphology["calcification"],
            "spiculation": morphology["spiculation"],
            "lobulation": morphology["lobulation"],
            "cavitation": morphology["cavitation"],
            "perifissural": morphology["perifissural"],
            "lung_rads_category": None,
            "recommendation_cue": recommendation_cue,
            "evidence_span": mention,
            "confidence": confidence,
            "missing_flags": [],
        }
        missing = [k for k, v in nodule_obj.items() if k != "missing_flags" and v is None]
        nodule_obj["missing_flags"] = missing
        nodules.append(nodule_obj)

    return {
        "note_id": note_id,
        "subject_id": subject_id,
        "exam_name": exam_name,
        "modality": modality,
        "body_site": body_site,
        "report_text": report_text,
        "sections": {"indication": None, "technique": None, "comparison": None, "findings": None, "impression": None},
        "nodule_count": len(nodules),
        "nodules": nodules,
        "extraction_metadata": {
            "extractor_version": "full_text_regex_0.1",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_name": "full_text_regex",
        },
    }


def run_variant(variant_name, samples, log_fp):
    results = []
    for idx, sample in enumerate(samples, start=1):
        text = sample.get("text", "")
        note_id = sample["note_id"]
        subject_id = sample["subject_id"]
        exam_name = sample.get("exam_name", "")

        if variant_name == "section_aware_regex":
            sections = parse_sections(text)
            fact = extract_radiology_facts(note_id, subject_id, exam_name, text, sections)
        elif variant_name == "full_text_regex":
            fact = extract_full_text_variant(note_id, subject_id, exam_name, text)
        else:
            sections = parse_sections(text)
            fact = extract_radiology_facts(note_id, subject_id, exam_name, text, sections)

        results.append(fact)
        if idx % 100 == 0:
            _log(f"  [{variant_name}] processed={idx}/{len(samples)}", log_fp)

    return results


def compute_metrics(facts):
    from src.eval.radiology_metrics import evaluate_radiology
    return evaluate_radiology(facts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_config.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    manifests_dir = Path(config["output"]["manifests_dir"])
    results_dir = Path(config["output"]["eval_results_dir"])
    comparisons_dir = Path(config["output"]["comparisons_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else manifests_dir / "radiology_explicit_eval.json"
    manifest = load_manifest(manifest_path)
    samples = manifest["samples"]

    log_path = Path("logs/eval_radiology_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] eval_radiology_baseline", log_fp)
        _log(f"[Config] manifest={manifest_path} samples={len(samples)}", log_fp)

        variants = config["baselines"]["radiology_variants"]
        all_results = {}

        for variant in variants:
            _log(f"[Variant] Running {variant}...", log_fp)
            facts = run_variant(variant, samples, log_fp)
            metrics = compute_metrics(facts)
            all_results[variant] = {"facts": facts, "metrics": metrics}

            facts_path = results_dir / f"radiology_facts_{variant}.jsonl"
            with facts_path.open("w", encoding="utf-8") as fp:
                for f in facts:
                    fp.write(json.dumps(f, ensure_ascii=False) + "\n")

            metrics_path = results_dir / f"radiology_metrics_{variant}.json"
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, ensure_ascii=False, indent=2)

            _log(f"  nodule_detect_rate={metrics.get('nodule_detect_rate', 0):.4f}", log_fp)
            _log(f"  size_mm_extract_rate={metrics.get('size_mm_extract_rate', 0):.4f}", log_fp)
            _log(f"  density_category_extract_rate={metrics.get('density_category_extract_rate', 0):.4f}", log_fp)
            _log(f"  location_lobe_extract_rate={metrics.get('location_lobe_extract_rate', 0):.4f}", log_fp)

        if len(all_results) >= 2:
            comparison = {"variants": {}}
            for name, data in all_results.items():
                comparison["variants"][name] = data["metrics"]

            keys_to_compare = [
                "nodule_detect_rate", "size_mm_extract_rate",
                "density_category_extract_rate", "location_lobe_extract_rate",
                "change_status_extract_rate", "recommendation_cue_extract_rate",
                "avg_nodules_per_note", "total_nodules",
            ]
            variant_names = list(all_results.keys())
            if len(variant_names) >= 2:
                deltas = {}
                m0 = all_results[variant_names[0]]["metrics"]
                m1 = all_results[variant_names[1]]["metrics"]
                for key in keys_to_compare:
                    v0 = m0.get(key, 0)
                    v1 = m1.get(key, 0)
                    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                        deltas[key] = {"baseline": v0, "variant": v1, "delta": v1 - v0}
                comparison["delta"] = {
                    "baseline": variant_names[0],
                    "variant": variant_names[1],
                    "deltas": deltas,
                }

            comp_path = comparisons_dir / "radiology_comparison.json"
            with comp_path.open("w", encoding="utf-8") as fp:
                json.dump(comparison, fp, ensure_ascii=False, indent=2)
            _log(f"[Comparison] saved to {comp_path}", log_fp)

        _log("[Done] eval_radiology_baseline completed", log_fp)


if __name__ == "__main__":
    main()
