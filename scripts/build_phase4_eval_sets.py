import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.eval.manifest_builder import (
    build_case_study_set,
    build_radiology_explicit_eval,
    build_recommendation_eval,
    build_smoking_explicit_eval,
    save_manifest,
)


def _log(msg, log_fp):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_config.yaml")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--radiology-facts", default=None)
    parser.add_argument("--smoking-results", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    seed = config["pipeline"]["random_seed"]
    manifests_dir = Path(config["output"]["manifests_dir"])
    cache_dir = Path(config["output"]["cache_dir"])
    nrows = args.nrows or config["pipeline"].get("nrows_limit")

    log_path = Path("logs/build_phase4_eval_sets.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] build_phase4_eval_sets", log_fp)
        _log(f"[Config] seed={seed} nrows={nrows}", log_fp)

        _log("[Step 1/4] Building radiology_explicit_eval manifest...", log_fp)
        rad_cfg = config["eval_sets"]["radiology_explicit_eval"]
        rad_manifest = build_radiology_explicit_eval(
            nrows=nrows,
            target_size=rad_cfg["target_size"],
            seed=seed,
            min_explicit_fields=rad_cfg.get("min_explicit_fields", 1),
            cache_dir=cache_dir,
        )
        save_manifest(rad_manifest, manifests_dir / "radiology_explicit_eval.json")
        _log(f"  selected={rad_manifest['selected_count']} from {rad_manifest['total_candidates']} candidates", log_fp)

        _log("[Step 2/4] Building smoking_explicit_eval manifest...", log_fp)
        smk_cfg = config["eval_sets"]["smoking_explicit_eval"]
        smk_manifest = build_smoking_explicit_eval(
            nrows=nrows,
            target_size=smk_cfg["target_size"],
            seed=seed,
            cache_dir=cache_dir,
        )
        save_manifest(smk_manifest, manifests_dir / "smoking_explicit_eval.json")
        _log(f"  selected={smk_manifest['selected_count']} from {smk_manifest['total_candidates']} candidates", log_fp)

        radiology_facts = []
        smoking_results = []

        rad_facts_path = Path(args.radiology_facts) if args.radiology_facts else None
        if rad_facts_path and rad_facts_path.exists():
            radiology_facts = _load_jsonl(rad_facts_path)
            _log(f"  loaded {len(radiology_facts)} radiology facts from {rad_facts_path}", log_fp)
        else:
            _log("  [Info] No pre-computed radiology facts provided. Running extraction on manifest samples...", log_fp)
            from src.extractors.radiology_extractor import extract_radiology_facts
            from src.parsers.section_parser import parse_sections

            for sample in rad_manifest["samples"]:
                text = sample.get("text", "")
                sections = parse_sections(text)
                fact = extract_radiology_facts(
                    note_id=sample["note_id"],
                    subject_id=sample["subject_id"],
                    exam_name=sample.get("exam_name", ""),
                    report_text=text,
                    sections=sections,
                )
                radiology_facts.append(fact)

            facts_cache = cache_dir / "radiology_facts_eval.jsonl"
            facts_cache.parent.mkdir(parents=True, exist_ok=True)
            with facts_cache.open("w", encoding="utf-8") as fp:
                for f in radiology_facts:
                    fp.write(json.dumps(f, ensure_ascii=False) + "\n")
            _log(f"  extracted and cached {len(radiology_facts)} radiology facts", log_fp)

        smk_results_path = Path(args.smoking_results) if args.smoking_results else None
        if smk_results_path and smk_results_path.exists():
            smoking_results = _load_jsonl(smk_results_path)
            _log(f"  loaded {len(smoking_results)} smoking results from {smk_results_path}", log_fp)
        else:
            _log("  [Info] No pre-computed smoking results provided. Running extraction on manifest samples...", log_fp)
            from src.extractors.smoking_extractor import extract_smoking_eligibility

            for sample in smk_manifest["samples"]:
                result = extract_smoking_eligibility(
                    subject_id=sample["subject_id"],
                    note_id=sample["note_id"],
                    text=sample.get("text", ""),
                )
                smoking_results.append(result)

            smk_cache = cache_dir / "smoking_results_eval.jsonl"
            smk_cache.parent.mkdir(parents=True, exist_ok=True)
            with smk_cache.open("w", encoding="utf-8") as fp:
                for r in smoking_results:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            _log(f"  extracted and cached {len(smoking_results)} smoking results", log_fp)

        _log("[Step 3/4] Building recommendation_eval manifest...", log_fp)
        rec_cfg = config["eval_sets"]["recommendation_eval"]
        rec_manifest = build_recommendation_eval(
            radiology_facts=radiology_facts,
            target_size=rec_cfg["target_size"],
            seed=seed,
        )
        save_manifest(rec_manifest, manifests_dir / "recommendation_eval.json")
        _log(f"  selected={rec_manifest['selected_count']}", log_fp)

        _log("[Step 4/4] Building case_study_set manifest...", log_fp)
        cs_cfg = config["eval_sets"]["case_study_set"]
        cs_manifest = build_case_study_set(
            radiology_facts=radiology_facts,
            smoking_results=smoking_results,
            target_size=cs_cfg["target_size"],
            seed=seed,
        )
        save_manifest(cs_manifest, manifests_dir / "case_study_set.json")
        _log(f"  selected={cs_manifest['selected_count']} coverage_met={cs_manifest['coverage_targets_met']}", log_fp)

        _log("[Summary]", log_fp)
        _log(f"  radiology_explicit_eval: {rad_manifest['selected_count']} samples", log_fp)
        _log(f"  smoking_explicit_eval: {smk_manifest['selected_count']} samples", log_fp)
        _log(f"  recommendation_eval: {rec_manifest['selected_count']} samples", log_fp)
        _log(f"  case_study_set: {cs_manifest['selected_count']} samples", log_fp)
        _log(f"  manifests saved to: {manifests_dir}", log_fp)
        _log("[Done] build_phase4_eval_sets completed", log_fp)


if __name__ == "__main__":
    main()
