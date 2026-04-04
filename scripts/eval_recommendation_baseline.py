import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.assemblers.case_bundle_assembler import assemble_case_bundles
from src.eval.manifest_builder import load_manifest, _load_jsonl
from src.rules.lung_rads_engine import generate_recommendation


def _log(msg, log_fp):
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def generate_cue_only_recommendation(case_bundle):
    from datetime import datetime, timezone

    cue = None
    for fact in case_bundle.get("radiology_facts", []):
        for nodule in fact.get("nodules", []):
            if nodule.get("recommendation_cue"):
                cue = nodule["recommendation_cue"]
                break
        if cue:
            break

    if not cue:
        return {
            "case_id": case_bundle.get("case_id", ""),
            "recommendation_level": "insufficient_data",
            "recommendation_action": None,
            "followup_interval": None,
            "followup_modality": None,
            "lung_rads_category": None,
            "guideline_source": "cue_extraction_only",
            "guideline_anchor": None,
            "reasoning_path": ["no_recommendation_cue_found"],
            "triggered_rules": [],
            "input_facts_used": {},
            "missing_information": ["recommendation_cue"],
            "uncertainty_note": "No recommendation cue found in report text.",
            "output_type": "cue_based",
            "generation_metadata": {
                "engine_version": "cue_only_v0.1",
                "generation_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "rules_version": "none",
            },
        }

    import re
    interval = None
    modality = None
    interval_match = re.search(r"(\d+)\s*(month|year|week)", cue, re.IGNORECASE)
    if interval_match:
        num = interval_match.group(1)
        unit = interval_match.group(2).lower()
        interval = f"{num} {unit}s" if int(num) > 1 else f"{num} {unit}"

    if re.search(r"\b(CT|LDCT|low.dose)\b", cue, re.IGNORECASE):
        modality = "CT"
    elif re.search(r"\b(PET|PET.CT)\b", cue, re.IGNORECASE):
        modality = "PET-CT"

    return {
        "case_id": case_bundle.get("case_id", ""),
        "recommendation_level": "actionable",
        "recommendation_action": cue,
        "followup_interval": interval,
        "followup_modality": modality,
        "lung_rads_category": None,
        "guideline_source": "cue_extraction_only",
        "guideline_anchor": None,
        "reasoning_path": ["extracted_recommendation_cue", f"cue_text={cue[:100]}"],
        "triggered_rules": ["cue_extraction"],
        "input_facts_used": {"recommendation_cue": cue},
        "missing_information": [],
        "uncertainty_note": "Recommendation derived from report cue text only, no structured rule applied.",
        "output_type": "cue_based",
        "generation_metadata": {
            "engine_version": "cue_only_v0.1",
            "generation_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rules_version": "none",
        },
    }


def run_variant(variant_name, bundles, log_fp):
    results = []
    for idx, bundle in enumerate(bundles, start=1):
        try:
            if variant_name == "cue_only":
                rec = generate_cue_only_recommendation(bundle)
            elif variant_name == "structured_rule":
                rec = generate_recommendation(bundle)
            else:
                rec = generate_recommendation(bundle)
        except Exception as exc:
            _log(f"  [Error] idx={idx} case_id={bundle.get('case_id')} error={repr(exc)}", log_fp)
            continue

        results.append(rec)
        if idx % 50 == 0:
            _log(f"  [{variant_name}] processed={idx}/{len(bundles)}", log_fp)

    return results


def compute_metrics(recommendations):
    from src.eval.recommendation_metrics import evaluate_recommendations
    return evaluate_recommendations(recommendations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--radiology-facts", default=None)
    parser.add_argument("--smoking-results", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    manifests_dir = Path(config["output"]["manifests_dir"])
    results_dir = Path(config["output"]["eval_results_dir"])
    comparisons_dir = Path(config["output"]["comparisons_dir"])
    cache_dir = Path(config["output"]["cache_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs/eval_recommendation_baseline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] eval_recommendation_baseline", log_fp)

        rad_facts_path = Path(args.radiology_facts) if args.radiology_facts else cache_dir / "radiology_facts_eval.jsonl"
        if not rad_facts_path.exists():
            rad_facts_path = Path("outputs/radiology_facts.jsonl")
        if not rad_facts_path.exists():
            _log("[Error] No radiology facts found. Run build_phase4_eval_sets.py first.", log_fp)
            return

        radiology_facts = _load_jsonl(rad_facts_path)
        _log(f"[Data] radiology_facts={len(radiology_facts)}", log_fp)

        smoking_dict = None
        smk_path = Path(args.smoking_results) if args.smoking_results else cache_dir / "smoking_results_eval.jsonl"
        if not smk_path.exists():
            smk_path = Path("outputs/smoking_results.jsonl")
        if smk_path.exists():
            smoking_rows = _load_jsonl(smk_path)
            smoking_dict = {}
            for row in smoking_rows:
                sid = _to_int(row.get("subject_id"), 0)
                if sid > 0 and sid not in smoking_dict:
                    smoking_dict[sid] = row
            _log(f"[Data] smoking_results={len(smoking_rows)} unique_subjects={len(smoking_dict)}", log_fp)

        bundles = assemble_case_bundles(
            radiology_facts=radiology_facts,
            smoking_results=smoking_dict,
            demographics=None,
        )
        _log(f"[Data] case_bundles={len(bundles)}", log_fp)

        bundles_path = cache_dir / "case_bundles_eval.jsonl"
        bundles_path.parent.mkdir(parents=True, exist_ok=True)
        with bundles_path.open("w", encoding="utf-8") as fp:
            for b in bundles:
                fp.write(json.dumps(b, ensure_ascii=False) + "\n")

        variants = config["baselines"]["recommendation_variants"]
        all_results = {}

        for variant in variants:
            _log(f"[Variant] Running {variant}...", log_fp)
            recs = run_variant(variant, bundles, log_fp)
            metrics = compute_metrics(recs)
            all_results[variant] = {"recommendations": recs, "metrics": metrics}

            recs_path = results_dir / f"recommendations_{variant}.jsonl"
            with recs_path.open("w", encoding="utf-8") as fp:
                for r in recs:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")

            metrics_path = results_dir / f"recommendation_metrics_{variant}.json"
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, ensure_ascii=False, indent=2)

            _log(f"  actionable_rate={metrics.get('actionable_rate', 0):.4f}", log_fp)
            _log(f"  insufficient_data_rate={metrics.get('insufficient_data_rate', 0):.4f}", log_fp)
            _log(f"  guideline_anchor_presence_rate={metrics.get('guideline_anchor_presence_rate', 0):.4f}", log_fp)

        if len(all_results) >= 2:
            comparison = {"variants": {}}
            for name, data in all_results.items():
                comparison["variants"][name] = data["metrics"]

            keys_to_compare = [
                "actionable_rate", "monitoring_rate", "insufficient_data_rate",
                "guideline_anchor_presence_rate", "reasoning_path_nonempty_rate",
                "triggered_rules_nonempty_rate",
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

            comp_path = comparisons_dir / "recommendation_comparison.json"
            with comp_path.open("w", encoding="utf-8") as fp:
                json.dump(comparison, fp, ensure_ascii=False, indent=2)
            _log(f"[Comparison] saved to {comp_path}", log_fp)

        _log("[Done] eval_recommendation_baseline completed", log_fp)


if __name__ == "__main__":
    main()
