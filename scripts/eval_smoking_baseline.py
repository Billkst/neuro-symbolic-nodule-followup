import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.eval.manifest_builder import load_manifest
from src.extractors.smoking_extractor import (
    extract_smoking_eligibility,
    extract_smoking_status,
    find_social_history_section,
)


def _log(msg, log_fp):
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def extract_social_history_only(subject_id, note_id, text):
    section_name, section_text = find_social_history_section(text)
    if not section_text:
        from src.extractors.smoking_extractor import _is_deidentified
        from datetime import datetime, timezone
        return {
            "subject_id": subject_id,
            "note_id": note_id,
            "source_section": None,
            "smoking_status_raw": None,
            "smoking_status_norm": "unknown",
            "pack_year_value": None, "pack_year_text": None,
            "ppd_value": None, "ppd_text": None,
            "years_smoked_value": None, "years_smoked_text": None,
            "quit_years_value": None, "quit_years_text": None,
            "evidence_span": None,
            "ever_smoker_flag": None,
            "eligible_for_high_risk_screening": None,
            "eligibility_criteria_applied": None,
            "eligibility_reason": None,
            "evidence_quality": "none",
            "extraction_metadata": {
                "extractor_version": "social_history_only_v1.0",
                "extraction_timestamp": datetime.now().isoformat(),
                "model_name": "social_history_only",
            },
            "missing_flags": [
                "source_section", "smoking_status_raw", "pack_year_value",
                "ppd_value", "years_smoked_value", "quit_years_value",
                "evidence_span", "ever_smoker_flag",
            ],
            "data_quality_notes": "Social History section not found or de-identified. No fallback applied.",
        }

    result = extract_smoking_eligibility(subject_id, note_id, section_text)
    result["extraction_metadata"]["model_name"] = "social_history_only"
    result["extraction_metadata"]["extractor_version"] = "social_history_only_v1.0"
    return result


def run_variant(variant_name, samples, log_fp):
    results = []
    for idx, sample in enumerate(samples, start=1):
        text = sample.get("text", "")
        subject_id = sample["subject_id"]
        note_id = sample["note_id"]

        if variant_name == "social_history_only":
            result = extract_social_history_only(subject_id, note_id, text)
        elif variant_name == "social_history_plus_fallback":
            result = extract_smoking_eligibility(subject_id, note_id, text)
        else:
            result = extract_smoking_eligibility(subject_id, note_id, text)

        results.append(result)
        if idx % 100 == 0:
            _log(f"  [{variant_name}] processed={idx}/{len(samples)}", log_fp)

    return results


def compute_metrics(results):
    from src.eval.smoking_metrics import evaluate_smoking
    return evaluate_smoking(results)


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

    manifest_path = Path(args.manifest) if args.manifest else manifests_dir / "smoking_explicit_eval.json"
    manifest = load_manifest(manifest_path)
    samples = manifest["samples"]

    log_path = Path("logs/eval_smoking_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] eval_smoking_baseline", log_fp)
        _log(f"[Config] manifest={manifest_path} samples={len(samples)}", log_fp)

        variants = config["baselines"]["smoking_variants"]
        all_results = {}

        for variant in variants:
            _log(f"[Variant] Running {variant}...", log_fp)
            results = run_variant(variant, samples, log_fp)
            metrics = compute_metrics(results)
            all_results[variant] = {"results": results, "metrics": metrics}

            results_path = results_dir / f"smoking_results_{variant}.jsonl"
            with results_path.open("w", encoding="utf-8") as fp:
                for r in results:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")

            metrics_path = results_dir / f"smoking_metrics_{variant}.json"
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, ensure_ascii=False, indent=2)

            _log(f"  non_unknown_rate={metrics.get('non_unknown_rate', 0):.4f}", log_fp)
            _log(f"  pack_year_parse_rate={metrics.get('pack_year_parse_rate', 0):.4f}", log_fp)
            _log(f"  evidence_quality_distribution={metrics.get('evidence_quality_distribution', {})}", log_fp)

        if len(all_results) >= 2:
            comparison = {"variants": {}}
            for name, data in all_results.items():
                comparison["variants"][name] = data["metrics"]

            keys_to_compare = [
                "non_unknown_rate", "ever_smoker_rate", "eligible_rate",
                "fallback_trigger_rate", "pack_year_parse_rate",
                "ppd_parse_rate", "years_smoked_parse_rate",
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

            comp_path = comparisons_dir / "smoking_comparison.json"
            with comp_path.open("w", encoding="utf-8") as fp:
                json.dump(comparison, fp, ensure_ascii=False, indent=2)
            _log(f"[Comparison] saved to {comp_path}", log_fp)

        _log("[Done] eval_smoking_baseline completed", log_fp)


if __name__ == "__main__":
    main()
