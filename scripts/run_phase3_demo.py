import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemblers.case_bundle_assembler import assemble_case_bundles
from src.data.filters import filter_chest_ct, filter_nodule_reports
from src.data.loader import load_discharge, load_radiology, load_radiology_detail
from src.extractors.radiology_extractor import extract_radiology_facts
from src.extractors.smoking_extractor import extract_smoking_eligibility
from src.parsers.section_parser import parse_sections
from src.pipeline.schema_validator import validate_instance
from src.rules.lung_rads_engine import generate_recommendation


DEMO_SAMPLE_SIZE = 50
RR_NOTE_ID_PATTERN = re.compile(r"^\d+-RR-\d+$")
DS_NOTE_ID_PATTERN = re.compile(r"^\d+-DS-\d+$")


def _log(msg: str, log_fp) -> None:
    print(msg, flush=True)
    log_fp.write(msg + "\n")
    log_fp.flush()


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _radiology_note_id(subject_id: int, note_seq, note_id_raw) -> str:
    raw = str(note_id_raw or "").strip()
    if RR_NOTE_ID_PATTERN.match(raw):
        return raw
    seq = _to_int(note_seq, 0)
    if seq > 0:
        return f"{subject_id}-RR-{seq}"
    fallback = _to_int(note_id_raw, 0)
    return f"{subject_id}-RR-{fallback if fallback > 0 else 0}"


def _discharge_note_id(subject_id: int, note_id_raw, fallback_index: int) -> str:
    raw = str(note_id_raw or "").strip()
    if DS_NOTE_ID_PATTERN.match(raw):
        return raw
    return f"{subject_id}-DS-{fallback_index}"


def _validate_and_collect(instance: dict, schema_name: str, error_bucket: list, object_id: str, stage: str) -> bool:
    errors = validate_instance(instance, schema_name)
    if not errors:
        return True
    error_bucket.append(
        {
            "stage": stage,
            "id": object_id,
            "schema": schema_name,
            "errors": errors,
        }
    )
    return False


def main():
    log_path = Path("logs/run_phase3_demo.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path("outputs/phase3_demo_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        _log("[Start] run_phase3_demo", log_fp)
        _log(f"[Config] DEMO_SAMPLE_SIZE={DEMO_SAMPLE_SIZE}", log_fp)

        timings = {}
        validation_errors = []

        t0 = time.perf_counter()
        radiology_df = load_radiology(nrows=100000)
        detail_df = load_radiology_detail(nrows=200000)
        discharge_df = load_discharge(nrows=50000)
        timings["step1_load_data_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step1] radiology={len(radiology_df)} radiology_detail={len(detail_df)} discharge={len(discharge_df)} time={timings['step1_load_data_sec']}s",
            log_fp,
        )

        t0 = time.perf_counter()
        chest_ct_df = filter_chest_ct(radiology_df, detail_df)
        nodule_df = filter_nodule_reports(chest_ct_df)
        candidates_df = nodule_df.head(DEMO_SAMPLE_SIZE).copy()
        timings["step2_filter_candidates_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step2] chest_ct={len(chest_ct_df)} nodule_candidates={len(nodule_df)} sampled={len(candidates_df)} time={timings['step2_filter_candidates_sec']}s",
            log_fp,
        )

        t0 = time.perf_counter()
        radiology_facts = []
        radiology_schema_valid = 0
        radiology_schema_invalid = 0
        for idx, (_, row) in enumerate(candidates_df.iterrows(), start=1):
            subject_id = _to_int(row.get("subject_id"), 0)
            note_id = _radiology_note_id(subject_id, row.get("note_seq"), row.get("note_id"))
            exam_name = str(row.get("exam_name") or "")
            text = str(row.get("text") or "")
            sections = parse_sections(text)

            try:
                fact = extract_radiology_facts(
                    note_id=note_id,
                    subject_id=subject_id,
                    exam_name=exam_name,
                    report_text=text,
                    sections=sections,
                )
                radiology_facts.append(fact)
                ok = _validate_and_collect(
                    fact,
                    "radiology_fact_schema",
                    validation_errors,
                    object_id=note_id,
                    stage="radiology_extraction",
                )
                if ok:
                    radiology_schema_valid += 1
                else:
                    radiology_schema_invalid += 1
            except Exception as exc:
                radiology_schema_invalid += 1
                validation_errors.append(
                    {
                        "stage": "radiology_extraction",
                        "id": note_id,
                        "schema": "radiology_fact_schema",
                        "errors": [f"runtime_error: {repr(exc)}"],
                    }
                )

            if idx % 10 == 0:
                _log(f"[Step3-Progress] processed={idx}", log_fp)

        timings["step3_radiology_extract_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step3] extracted={len(radiology_facts)} schema_valid={radiology_schema_valid} schema_invalid={radiology_schema_invalid} time={timings['step3_radiology_extract_sec']}s",
            log_fp,
        )

        t0 = time.perf_counter()
        subject_ids = {int(item.get("subject_id")) for item in radiology_facts if item.get("subject_id") is not None}
        text_col = "text" if "text" in discharge_df.columns else ("note_text" if "note_text" in discharge_df.columns else None)
        if text_col is None:
            raise ValueError("discharge 数据中未找到 text/note_text 列")

        discharge_match_df = discharge_df[discharge_df["subject_id"].isin(list(subject_ids))].copy()
        smoking_results = []
        smoking_dict = {}
        smoking_schema_valid = 0
        smoking_schema_invalid = 0

        for idx, (_, row) in enumerate(discharge_match_df.iterrows(), start=1):
            subject_id = _to_int(row.get("subject_id"), 0)
            note_id = _discharge_note_id(subject_id, row.get("note_id"), idx)
            text = str(row.get(text_col) or "")

            try:
                result = extract_smoking_eligibility(subject_id=subject_id, note_id=note_id, text=text)
                smoking_results.append(result)
                if subject_id not in smoking_dict:
                    smoking_dict[subject_id] = result
                ok = _validate_and_collect(
                    result,
                    "smoking_eligibility_schema",
                    validation_errors,
                    object_id=note_id,
                    stage="smoking_extraction",
                )
                if ok:
                    smoking_schema_valid += 1
                else:
                    smoking_schema_invalid += 1
            except Exception as exc:
                smoking_schema_invalid += 1
                validation_errors.append(
                    {
                        "stage": "smoking_extraction",
                        "id": note_id,
                        "schema": "smoking_eligibility_schema",
                        "errors": [f"runtime_error: {repr(exc)}"],
                    }
                )

        timings["step4_smoking_extract_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step4] matched_discharge={len(discharge_match_df)} smoking_extracted={len(smoking_results)} smoking_subjects={len(smoking_dict)} schema_valid={smoking_schema_valid} schema_invalid={smoking_schema_invalid} time={timings['step4_smoking_extract_sec']}s",
            log_fp,
        )

        t0 = time.perf_counter()
        case_bundles = assemble_case_bundles(
            radiology_facts=radiology_facts,
            smoking_results=smoking_dict,
            demographics=None,
        )
        bundle_schema_valid = 0
        bundle_schema_invalid = 0
        for bundle in case_bundles:
            case_id = str(bundle.get("case_id") or "unknown_case")
            ok = _validate_and_collect(
                bundle,
                "case_bundle_schema",
                validation_errors,
                object_id=case_id,
                stage="bundle_assembly",
            )
            if ok:
                bundle_schema_valid += 1
            else:
                bundle_schema_invalid += 1
        timings["step5_bundle_assembly_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step5] bundles={len(case_bundles)} schema_valid={bundle_schema_valid} schema_invalid={bundle_schema_invalid} time={timings['step5_bundle_assembly_sec']}s",
            log_fp,
        )

        t0 = time.perf_counter()
        recommendations = []
        recommendation_schema_valid = 0
        recommendation_schema_invalid = 0
        for bundle in case_bundles:
            case_id = str(bundle.get("case_id") or "unknown_case")
            try:
                rec = generate_recommendation(bundle)
                recommendations.append(rec)
                ok = _validate_and_collect(
                    rec,
                    "recommendation_schema",
                    validation_errors,
                    object_id=case_id,
                    stage="recommendation_generation",
                )
                if ok:
                    recommendation_schema_valid += 1
                else:
                    recommendation_schema_invalid += 1
            except Exception as exc:
                recommendation_schema_invalid += 1
                validation_errors.append(
                    {
                        "stage": "recommendation_generation",
                        "id": case_id,
                        "schema": "recommendation_schema",
                        "errors": [f"runtime_error: {repr(exc)}"],
                    }
                )

        timings["step6_recommendation_sec"] = round(time.perf_counter() - t0, 3)
        _log(
            f"[Step6] recommendations={len(recommendations)} schema_valid={recommendation_schema_valid} schema_invalid={recommendation_schema_invalid} time={timings['step6_recommendation_sec']}s",
            log_fp,
        )

        output = {
            "pipeline_version": "0.1.0",
            "demo_sample_size": DEMO_SAMPLE_SIZE,
            "timing_seconds": timings,
            "summary": {
                "radiology_candidates": len(candidates_df),
                "radiology_facts_extracted": len(radiology_facts),
                "radiology_schema_valid": radiology_schema_valid,
                "radiology_schema_invalid": radiology_schema_invalid,
                "smoking_extracted": len(smoking_results),
                "smoking_schema_valid": smoking_schema_valid,
                "smoking_schema_invalid": smoking_schema_invalid,
                "case_bundles_assembled": len(case_bundles),
                "bundle_schema_valid": bundle_schema_valid,
                "bundle_schema_invalid": bundle_schema_invalid,
                "recommendations_generated": len(recommendations),
                "recommendation_schema_valid": recommendation_schema_valid,
                "recommendation_schema_invalid": recommendation_schema_invalid,
            },
            "radiology_facts_sample": radiology_facts[:10],
            "smoking_results_sample": smoking_results[:5],
            "case_bundles_sample": case_bundles[:5],
            "recommendations_sample": recommendations[:5],
            "validation_errors": validation_errors,
        }

        with output_path.open("w", encoding="utf-8", buffering=1) as out_fp:
            json.dump(output, out_fp, ensure_ascii=False, indent=2)
            out_fp.flush()

        _log(f"[Step7] wrote_output={output_path}", log_fp)
        _log(f"[Summary] {json.dumps(output['summary'], ensure_ascii=False)}", log_fp)
        _log(f"[Summary] validation_error_count={len(validation_errors)}", log_fp)
        _log("[Done] run_phase3_demo completed", log_fp)


if __name__ == "__main__":
    main()
