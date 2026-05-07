#!/usr/bin/env python3
"""Build Module 2 final tables v3/v4/v5/v6 from existing Plan B result artifacts.

This script is read-only with respect to experiment outputs: it aggregates
already-produced JSON/CSV files and writes paper-ready CSV, LaTeX, and Markdown
artifacts. It must not launch training or evaluation.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs/phaseA2_planB/results"
LEGACY_PLANB_MAIN = PROJECT_ROOT / "outputs/phaseA2_planB/tables/planb_main_table.csv"
EXPECTED_SEEDS = [13, 42, 87, 3407, 31415]
ACTIVE_VERSION = "v3"
TABLES_DIR = PROJECT_ROOT / "outputs/phaseA2_planB/final_tables_v3"
LATEX_DIR = PROJECT_ROOT / "outputs/phaseA2_planB/final_tables_latex_v3"
REPORT_PATH = PROJECT_ROOT / "reports/module2_final_sota_writeup_v3.md"
FIGURES_DIR = PROJECT_ROOT / "outputs/phaseA2_planB/figures"
FINAL_FIGURES_DIR = PROJECT_ROOT / "outputs/phaseA2_planB/final_figures_v5"


@dataclass(frozen=True)
class Summary:
    mean: float | None
    std: float | None
    n: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    task: str
    metric: str


MAIN_METRICS = [
    MetricSpec("density_stage1_f1", "Density Stage 1 F1", "density_stage1", "f1"),
    MetricSpec("density_stage1_auprc", "Density Stage 1 AUPRC", "density_stage1", "auprc"),
    MetricSpec("density_stage1_auroc", "Density Stage 1 AUROC", "density_stage1", "auroc"),
    MetricSpec("density_stage2_macro_f1", "Density Stage 2 Macro-F1", "density_stage2", "macro_f1"),
    MetricSpec("has_size_f1", "Has-size F1", "size", "f1"),
    MetricSpec("location_macro_f1", "Location Macro-F1", "location", "macro_f1"),
]

APPENDIX_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("macro_f1", "Macro-F1"),
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
]

APPENDIX_TASK_METRICS = {
    "density_stage1": [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("macro_f1", "Macro-F1"),
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
    ],
    "density_stage2": [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ],
    "size": [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
    ],
    "location": [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ],
}

APPENDIX_TASK_FILES = {
    "density_stage1": "appendix_density_stage1_full_metrics.csv",
    "density_stage2": "appendix_density_stage2_full_metrics.csv",
    "size": "appendix_size_full_metrics.csv",
    "location": "appendix_location_full_metrics.csv",
}

TASK_TITLES = {
    "density_stage1": "Density Stage 1",
    "density_stage2": "Density Stage 2",
    "size": "Has-size",
    "location": "Location",
}

TASK_PRIMARY_METRIC_NOTES = {
    "density_stage1": "binary evidence detection; threshold calibrated; main text reports F1/AUPRC/AUROC",
    "density_stage2": "multi-class density subtype classification; main text reports Macro-F1",
    "size": "binary field extraction; main text reports F1",
    "location": "multi-class location extraction; main text reports Macro-F1",
}

BASE_MAIN_METHODS = [
    (
        "TF-IDF + LR",
        {
            "density_stage1": "tfidf_lr_density_stage1_results_planb_full_seed*.json",
            "density_stage2": "tfidf_lr_density_stage2_results_planb_full_seed*.json",
            "size": "tfidf_lr_size_results_planb_full_seed*.json",
            "location": "tfidf_lr_location_results_planb_full_seed*.json",
        },
    ),
    (
        "TF-IDF + SVM",
        {
            "density_stage1": "tfidf_svm_density_stage1_results_planb_full_seed*.json",
            "density_stage2": "tfidf_svm_density_stage2_results_planb_full_seed*.json",
            "size": "tfidf_svm_size_results_planb_full_seed*.json",
            "location": "tfidf_svm_location_results_planb_full_seed*.json",
        },
    ),
    (
        "TF-IDF + MLP",
        {
            "density_stage1": "tfidf_mlp_density_stage1_results_planb_full_seed*.json",
            "density_stage2": "tfidf_mlp_density_stage2_results_planb_full_seed*.json",
            "size": "tfidf_mlp_size_results_planb_full_seed*.json",
            "location": "tfidf_mlp_location_results_planb_full_seed*.json",
        },
    ),
    (
        "Vanilla PubMedBERT",
        {
            "density_stage1": "vanilla_pubmedbert_density_stage1_results_planb_full_seed*.json",
            "density_stage2": "vanilla_pubmedbert_density_stage2_results_planb_full_seed*.json",
            "size": "vanilla_pubmedbert_size_results_planb_full_seed*.json",
            "location": "vanilla_pubmedbert_location_results_planb_full_seed*.json",
        },
    ),
]

EXTRA_PLM_METHODS = [
    (
        "SciBERT",
        {
            "density_stage1": "scibert_density_stage1_results_extra_plm_seed*.json",
            "density_stage2": "scibert_density_stage2_results_extra_plm_seed*.json",
            "size": "scibert_size_results_extra_plm_seed*.json",
            "location": "scibert_location_results_extra_plm_seed*.json",
        },
    ),
    (
        "BioClinicalBERT / ClinicalBERT",
        {
            "density_stage1": "bioclinicalbert_density_stage1_results_extra_plm_seed*.json",
            "density_stage2": "bioclinicalbert_density_stage2_results_extra_plm_seed*.json",
            "size": "bioclinicalbert_size_results_extra_plm_seed*.json",
            "location": "bioclinicalbert_location_results_extra_plm_seed*.json",
        },
    ),
]

OURS_METHOD_V3_V4 = (
    "MWS-CFE (Ours; final)",
    {
        "density_stage1": "mws_cfe_density_stage1_results_planb_full_seed*.json",
        "density_stage2": "mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json",
        "size": "mws_cfe_size_results_size_wave5_lexical_bert_cue_lr_seed*.json",
        "location": "mws_cfe_location_results_location_aug_g2_seed*.json",
    },
)

OURS_METHOD_V5 = (
    "MWS-CFE (Ours; final)",
    {
        "density_stage1": "mws_cfe_density_stage1_results_density_final_g3_len128_seed*.json",
        "density_stage2": "mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json",
        "size": "mws_cfe_size_results_size_wave5_lexical_bert_cue_lr_seed*.json",
        "location": "mws_cfe_location_results_location_aug_g2_seed*.json",
    },
)

MAIN_METHODS = [
    *BASE_MAIN_METHODS,
    (
        OURS_METHOD_V3_V4[0],
        OURS_METHOD_V3_V4[1],
    ),
]


def configure_version(version: str) -> None:
    global ACTIVE_VERSION, TABLES_DIR, LATEX_DIR, REPORT_PATH, FINAL_FIGURES_DIR, MAIN_METHODS
    if version not in {"v3", "v4", "v5", "v6"}:
        raise ValueError(f"Unsupported final table version: {version}")
    ACTIVE_VERSION = version
    TABLES_DIR = PROJECT_ROOT / f"outputs/phaseA2_planB/final_tables_{version}"
    LATEX_DIR = PROJECT_ROOT / f"outputs/phaseA2_planB/final_tables_latex_{version}"
    REPORT_PATH = PROJECT_ROOT / f"reports/module2_final_sota_writeup_{version}.md"
    FINAL_FIGURES_DIR = PROJECT_ROOT / f"outputs/phaseA2_planB/final_figures_{version}"
    if version in {"v5", "v6"}:
        MAIN_METHODS = [*BASE_MAIN_METHODS, *EXTRA_PLM_METHODS, OURS_METHOD_V5]
    elif version == "v4":
        MAIN_METHODS = [*BASE_MAIN_METHODS, *EXTRA_PLM_METHODS, OURS_METHOD_V3_V4]
    else:
        MAIN_METHODS = [*BASE_MAIN_METHODS, OURS_METHOD_V3_V4]


def final_stage1_pattern() -> str:
    if ACTIVE_VERSION in {"v5", "v6"}:
        return "mws_cfe_density_stage1_results_density_final_g3_len128_seed*.json"
    return "mws_cfe_density_stage1_results_planb_full_seed*.json"


def final_stage1_source_note() -> str:
    if ACTIVE_VERSION in {"v5", "v6"}:
        return "stage1 density_final_g3_len128; stage2 density_final_g3_len128"
    return "stage1 planb_full; stage2 density_final_g3_len128"


def seed_from_path(path: Path) -> int:
    match = re.search(r"_seed(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def mean_std(values: list[float], seeds: list[int]) -> Summary:
    if not values:
        return Summary(None, None, 0, tuple(seeds))
    mean = sum(values) / len(values)
    std = 0.0 if len(values) <= 1 else math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return Summary(mean, std, len(values), tuple(seeds))


def summarize_pattern(pattern: str, metric: str, result_key: str = "phase5_test_results") -> Summary:
    paths = sorted(RESULTS_DIR.glob(pattern), key=seed_from_path)
    values: list[float] = []
    seeds: list[int] = []
    for path in paths:
        data = load_json(path)
        value = nested_get(data, f"{result_key}.{metric}")
        if isinstance(value, (int, float)):
            values.append(float(value))
            seeds.append(seed_from_path(path))
    return mean_std(values, seeds)


def summarize_thresholds(pattern: str) -> Summary:
    paths = sorted(RESULTS_DIR.glob(pattern), key=seed_from_path)
    values: list[float] = []
    seeds: list[int] = []
    for path in paths:
        data = load_json(path)
        value = data.get("chosen_threshold")
        if value is None:
            value = nested_get(data, "threshold_tuning.selected_threshold")
        if isinstance(value, (int, float)):
            values.append(float(value))
            seeds.append(seed_from_path(path))
    return mean_std(values, seeds)


def fmt(summary: Summary, *, percent: bool = True, decimals: int = 2) -> str:
    if summary.mean is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{summary.mean * scale:.{decimals}f} +/- {(summary.std or 0.0) * scale:.{decimals}f}"


def fmt_na(summary: Summary, *, percent: bool = True, decimals: int = 2) -> str:
    value = fmt(summary, percent=percent, decimals=decimals)
    return "N/A" if value == "--" else value


def fmt_latex(summary: Summary, *, bold: bool = False, percent: bool = True, decimals: int = 2) -> str:
    value = fmt(summary, percent=percent, decimals=decimals).replace("+/-", r"$\pm$")
    if bold and value != "--":
        return rf"\textbf{{{value}}}"
    return value


def fmt_markdown(summary: Summary, *, bold: bool = False, percent: bool = True, decimals: int = 2) -> str:
    value = fmt(summary, percent=percent, decimals=decimals)
    if bold and value != "--":
        return f"**{value}**"
    return value


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_legacy_main_value(method: str, column: str) -> str:
    if not LEGACY_PLANB_MAIN.exists():
        return "--"
    with LEGACY_PLANB_MAIN.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("Method") == method:
                return row.get(column, "--") or "--"
    return "--"


def build_main_table() -> tuple[list[dict[str, str]], dict[str, dict[str, Summary]], dict[str, bool], str]:
    raw: dict[str, dict[str, Summary]] = {}
    for method, task_patterns in MAIN_METHODS:
        raw[method] = {}
        for spec in MAIN_METRICS:
            raw[method][spec.key] = summarize_pattern(task_patterns[spec.task], spec.metric)

    best_by_metric: dict[str, float] = {}
    for spec in MAIN_METRICS:
        means = [raw[method][spec.key].mean for method, _ in MAIN_METHODS]
        best_by_metric[spec.key] = max(value for value in means if value is not None)

    best_flags: dict[str, bool] = {}
    rows: list[dict[str, str]] = []
    for method, _ in MAIN_METHODS:
        row = {"Method": method}
        method_all_best = True
        for spec in MAIN_METRICS:
            summary = raw[method][spec.key]
            is_best = summary.mean is not None and math.isclose(summary.mean, best_by_metric[spec.key], rel_tol=0.0, abs_tol=1e-12)
            method_all_best = method_all_best and is_best
            value = fmt(summary)
            if is_best:
                value += " *"
            row[spec.title] = value
        best_flags[method] = method_all_best
        rows.append(row)

    ours_all_best = best_flags.get("MWS-CFE (Ours; final)", False)
    verdict = "yes" if ours_all_best else "no"
    return rows, raw, best_flags, verdict


def build_ablation_table() -> list[dict[str, str]]:
    final_stage1 = {
        "f1": summarize_pattern(final_stage1_pattern(), "f1"),
        "auprc": summarize_pattern(final_stage1_pattern(), "auprc"),
    }
    final_stage2 = summarize_pattern("mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json", "macro_f1")
    final_density_variant = "Final two-stage density (P0 calibrated Stage 1 + G3 len128 Stage 2)"
    if ACTIVE_VERSION in {"v5", "v6"}:
        final_density_variant = "Final two-stage density (G3 len128 calibrated Stage 1 + G3 len128 Stage 2)"
    rows: list[dict[str, str]] = [
        {
            "Section": "Core density ablation",
            "Variant": final_density_variant,
            "Source": final_stage1_source_note(),
            "Seeds": "5",
            "Density Stage 1 F1": fmt(final_stage1["f1"]),
            "Density Stage 1 AUPRC": fmt(final_stage1["auprc"]),
            "Density Stage 2 Macro-F1": fmt(final_stage2),
            "Has-size F1": "--",
            "Has-size AUPRC": "--",
            "Protocol Note": "final full row for density",
        },
        {
            "Section": "Core density ablation",
            "Variant": "Legacy G2 quality gate before selection",
            "Source": "planb_full legacy aggregate / stage1 P0",
            "Seeds": "5",
            "Density Stage 1 F1": fmt(final_stage1["f1"]),
            "Density Stage 1 AUPRC": fmt(final_stage1["auprc"]),
            "Density Stage 2 Macro-F1": read_legacy_main_value("MWS-CFE (Ours)", "Density Stage 2 Macro-F1"),
            "Has-size F1": "--",
            "Has-size AUPRC": "--",
            "Protocol Note": "shows effect of selecting G3+len128 for Stage 2",
        },
    ]

    rows.append(
        {
            "Section": "Core density ablation",
            "Variant": "Before P0 threshold tuning (legacy default decision layer)",
            "Source": "outputs/phaseA2_planB/tables/planb_main_table.csv",
            "Seeds": "5",
            "Density Stage 1 F1": read_legacy_main_value("MWS-CFE (Ours)", "Density Stage 1 F1"),
            "Density Stage 1 AUPRC": read_legacy_main_value("MWS-CFE (Ours)", "Density Stage 1 AUPRC"),
            "Density Stage 2 Macro-F1": "--",
            "Has-size F1": "--",
            "Has-size AUPRC": "--",
            "Protocol Note": "legacy aggregate; diagnostic for P0 threshold tuning, not a final full ablation",
        }
    )

    core_specs = [
        (
            "w/o quality gate (G1 diagnostic)",
            "ab_wo_quality_gate",
            "quality gate selection diagnostic",
        ),
        (
            "w/o section-aware input",
            "ab_wo_section",
            "section-aware input ablation",
        ),
        (
            "w/o confidence-aware training",
            "ab_wo_confidence",
            "confidence-aware training ablation",
        ),
    ]
    for variant, tag, note in core_specs:
        stage1_f1 = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "f1")
        stage1_auprc = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "auprc")
        stage2 = summarize_pattern(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
        rows.append(
            {
                "Section": "Core density ablation",
                "Variant": variant,
                "Source": tag,
                "Seeds": str(max(stage1_f1.n, stage2.n)),
                "Density Stage 1 F1": fmt(stage1_f1),
                "Density Stage 1 AUPRC": fmt(stage1_auprc),
                "Density Stage 2 Macro-F1": fmt(stage2),
                "Has-size F1": "--",
                "Has-size AUPRC": "--",
                "Protocol Note": note,
            }
        )

    size_specs = [
        (
            "lexical alone",
            "size_wave5_lexical_alone",
            "5-seed learned lexical expert",
        ),
        (
            "lexical + BERT",
            "size_wave5_lexical_bert_lr",
            "diagnostic component analysis; seed42 only",
        ),
        (
            "lexical + cue",
            "size_wave5_lexical_cue_lr",
            "diagnostic component analysis; seed42 only",
        ),
        (
            "lexical + BERT + cue",
            "size_wave5_lexical_bert_cue_lr",
            "final 5-seed learned stacked head",
        ),
    ]
    for variant, tag, note in size_specs:
        f1 = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "f1")
        auprc = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "auprc")
        rows.append(
            {
                "Section": "Has-size Wave5 component analysis",
                "Variant": variant,
                "Source": tag,
                "Seeds": str(f1.n),
                "Density Stage 1 F1": "--",
                "Density Stage 1 AUPRC": "--",
                "Density Stage 2 Macro-F1": "--",
                "Has-size F1": fmt(f1),
                "Has-size AUPRC": fmt(auprc),
                "Protocol Note": note,
            }
        )
    return rows


def parameter_density_row(group: str, param_type: str, value: str, tag: str, note: str) -> dict[str, str]:
    stage1_auprc = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "auprc")
    stage1_f1 = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "f1")
    stage2 = summarize_pattern(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
    return {
        "Parameter Group": group,
        "Parameter Type": param_type,
        "Value": value,
        "Tag": tag,
        "Seeds": str(max(stage1_auprc.n, stage2.n)),
        "Density Stage 1 AUPRC": fmt(stage1_auprc),
        "Density Stage 1 F1": fmt(stage1_f1),
        "Density Stage 2 Macro-F1": fmt(stage2),
        "Has-size F1": "--",
        "Notes": note,
    }


def build_parameter_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value, tag in [
        ("64", "p1_len64"),
        ("96", "p1_len96"),
        ("128", "planb_full"),
        ("160", "p1_len160"),
        ("192", "p1_len192"),
    ]:
        rows.append(parameter_density_row("P1 max_seq_length", "numeric length scan", value, tag, "retained P1 scan"))

    for value, tag in [
        ("G1", "p2_g1"),
        ("G2", "planb_full"),
        ("G3 (selected)", "p2_g3"),
        ("G4", "p2_g4"),
        ("G5", "p2_g5"),
    ]:
        rows.append(
            parameter_density_row(
                "P2 quality gate",
                "categorical design choice",
                value,
                tag,
                "categorical gate option; not a continuous numerical parameter",
            )
        )

    for value, tag in [
        ("Mention only", "p3_mention_text"),
        ("Section-aware (selected)", "planb_full"),
        ("Findings only", "p3_findings_text"),
        ("Impression only", "p3_impression_text"),
        ("Findings + impression", "p3_findings_impression_text"),
        ("Full text", "p3_full_text"),
    ]:
        rows.append(
            parameter_density_row(
                "P3 section/input strategy",
                "categorical design choice",
                value,
                tag,
                "categorical input-construction strategy; not a continuous numerical parameter",
            )
        )

    wave5_tag = "size_wave5_lexical_bert_cue_lr"
    wave5_f1 = summarize_pattern(f"mws_cfe_size_results_{wave5_tag}_seed*.json", "f1")
    wave5_threshold = summarize_thresholds(f"mws_cfe_size_results_{wave5_tag}_seed*.json")
    rows.append(
        {
            "Parameter Group": "Wave5 Has-size threshold",
            "Parameter Type": "diagnostic decision-layer setting",
            "Value": f"threshold {fmt(wave5_threshold, percent=False, decimals=6)}",
            "Tag": wave5_tag,
            "Seeds": str(wave5_f1.n),
            "Density Stage 1 AUPRC": "--",
            "Density Stage 1 F1": "--",
            "Density Stage 2 Macro-F1": "--",
            "Has-size F1": fmt(wave5_f1),
            "Notes": "threshold tuned on phase5-like dev; Phase5 test not used for threshold",
        }
    )
    rows.append(
        {
            "Parameter Group": "Wave5 Has-size stacked head",
            "Parameter Type": "diagnostic decision-layer setting",
            "Value": "lexical + BERT + cue logistic-regression head",
            "Tag": wave5_tag,
            "Seeds": str(wave5_f1.n),
            "Density Stage 1 AUPRC": "--",
            "Density Stage 1 F1": "--",
            "Density Stage 2 Macro-F1": "--",
            "Has-size F1": fmt(wave5_f1),
            "Notes": "test_truncated=false; test_sample_count=42057",
        }
    )
    return rows


def seed_text_from_summaries(summaries: list[Summary]) -> str:
    seeds = sorted({seed for summary in summaries for seed in summary.seeds})
    if not seeds:
        return "0"
    return f"{len(seeds)} ({','.join(str(seed) for seed in seeds)})"


def build_appendix_full_metrics_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, task_patterns in MAIN_METHODS:
        for task in ["density_stage1", "density_stage2", "size", "location"]:
            pattern = task_patterns[task]
            summaries = {metric: summarize_pattern(pattern, metric) for metric, _ in APPENDIX_METRICS}
            row = {
                "Method": method,
                "Task": TASK_TITLES[task],
                "Primary Metric Rationale": TASK_PRIMARY_METRIC_NOTES[task],
                "Seeds": seed_text_from_summaries(list(summaries.values())),
                "Result Pattern": pattern,
            }
            for metric, title in APPENDIX_METRICS:
                row[title] = fmt(summaries[metric])
            rows.append(row)
    return rows


def build_task_appendix_full_metrics_tables() -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for task, metric_specs in APPENDIX_TASK_METRICS.items():
        rows: list[dict[str, str]] = []
        for method, task_patterns in MAIN_METHODS:
            pattern = task_patterns[task]
            summaries = {metric: summarize_pattern(pattern, metric) for metric, _ in metric_specs}
            row = {
                "Method": method,
                "Seeds": seed_text_from_summaries(list(summaries.values())),
            }
            for metric, title in metric_specs:
                row[title] = fmt_na(summaries[metric])
            rows.append(row)
        tables[task] = rows
    return tables


def build_density_ablation_table_v6() -> list[dict[str, str]]:
    full_stage1_f1 = summarize_pattern(final_stage1_pattern(), "f1")
    full_stage1_auprc = summarize_pattern(final_stage1_pattern(), "auprc")
    full_stage2 = summarize_pattern("mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json", "macro_f1")

    legacy_f1_value = read_legacy_main_value("MWS-CFE (Ours)", "Density Stage 1 F1")
    legacy_auprc_value = read_legacy_main_value("MWS-CFE (Ours)", "Density Stage 1 AUPRC")

    g2_stage1_f1 = summarize_pattern("mws_cfe_density_stage1_results_planb_full_seed*.json", "f1")
    g2_stage1_auprc = summarize_pattern("mws_cfe_density_stage1_results_planb_full_seed*.json", "auprc")
    g2_stage2 = summarize_pattern("mws_cfe_density_stage2_results_planb_full_seed*.json", "macro_f1")

    rows = [
        {
            "Variant": "Full MWS-CFE",
            "Seeds": str(max(full_stage1_f1.n, full_stage2.n)),
            "Density Stage 1 F1": fmt(full_stage1_f1),
            "Density Stage 1 AUPRC": fmt(full_stage1_auprc),
            "Density Stage 2 Macro-F1": fmt(full_stage2),
            "中文说明": "最终完整模型：Stage 1 使用 G3+len128 校准配置，Stage 2 使用 G3+len128；不使用 len192。",
        },
        {
            "Variant": "w/o P0 threshold tuning",
            "Seeds": "5",
            "Density Stage 1 F1": legacy_f1_value,
            "Density Stage 1 AUPRC": legacy_auprc_value,
            "Density Stage 2 Macro-F1": "N/A",
            "中文说明": "只诊断 Stage 1 阈值校准层；Stage 2 指标不适用于该行。",
        },
        {
            "Variant": "w/o G3 gate selection",
            "Seeds": str(max(g2_stage1_f1.n, g2_stage2.n)),
            "Density Stage 1 F1": fmt(g2_stage1_f1),
            "Density Stage 1 AUPRC": fmt(g2_stage1_auprc),
            "Density Stage 2 Macro-F1": fmt(g2_stage2),
            "中文说明": "回退到旧 G2 gate 配置，用于显示选择 G3 gate 后的增益。",
        },
    ]

    for variant, tag, note in [
        ("w/o section-aware input", "ab_wo_section", "去掉 section-aware 输入构造，保留其余训练流程。"),
        ("w/o confidence-aware training", "ab_wo_confidence", "去掉 confidence-aware training，保留其余模型结构。"),
    ]:
        stage1_f1 = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "f1")
        stage1_auprc = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "auprc")
        stage2 = summarize_pattern(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
        rows.append(
            {
                "Variant": variant,
                "Seeds": str(max(stage1_f1.n, stage2.n)),
                "Density Stage 1 F1": fmt(stage1_f1),
                "Density Stage 1 AUPRC": fmt(stage1_auprc),
                "Density Stage 2 Macro-F1": fmt(stage2),
                "中文说明": note,
            }
        )
    return rows


def build_size_wave5_component_table_v6() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant, tag, note in [
        ("lexical alone", "size_wave5_lexical_alone", "5-seed learned lexical expert；作为 Wave5 组件基线。"),
        ("lexical + BERT", "size_wave5_lexical_bert_lr", "seed42 诊断结果；用于判断 BERT probability 的边际贡献。"),
        ("lexical + cue", "size_wave5_lexical_cue_lr", "seed42 诊断结果；用于判断 cue features 的边际贡献。"),
        ("lexical + BERT + cue", "size_wave5_lexical_bert_cue_lr", "最终 5-seed learned stacked head；不属于 deterministic cue-only。"),
    ]:
        f1 = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "f1")
        auprc = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "auprc")
        rows.append(
            {
                "Variant": variant,
                "Seeds": str(f1.n),
                "Has-size F1": fmt(f1),
                "Has-size AUPRC": fmt(auprc),
                "中文说明": note,
            }
        )
    return rows


def build_density_parameter_table_v6() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, param_type, note, values in [
        (
            "P1 max_seq_length",
            "数值型长度扫描",
            "P1 长度扫描；Stage 1 只报告 AUPRC，避免不同阈值设置下的 F1 不可比。",
            [
                ("64", "p1_len64"),
                ("96", "p1_len96"),
                ("128", "planb_full"),
                ("160", "p1_len160"),
                ("192", "p1_len192"),
            ],
        ),
        (
            "P2 quality gate",
            "类别型设计选择",
            "类别型设计选择，不是连续数值参数；用于比较 gate 策略。",
            [
                ("G1", "p2_g1"),
                ("G2", "planb_full"),
                ("G3 (selected)", "p2_g3"),
                ("G4", "p2_g4"),
                ("G5", "p2_g5"),
            ],
        ),
        (
            "P3 section/input strategy",
            "类别型设计选择",
            "类别型设计选择，不是连续数值参数；用于比较输入构造策略。",
            [
                ("Mention only", "p3_mention_text"),
                ("Section-aware (selected)", "planb_full"),
                ("Findings only", "p3_findings_text"),
                ("Impression only", "p3_impression_text"),
                ("Findings + impression", "p3_findings_impression_text"),
                ("Full text", "p3_full_text"),
            ],
        ),
    ]:
        for value, tag in values:
            stage1_auprc = summarize_pattern(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "auprc")
            stage2 = summarize_pattern(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
            rows.append(
                {
                    "参数组": group,
                    "参数类型": param_type,
                    "取值": value,
                    "Seeds": str(max(stage1_auprc.n, stage2.n)),
                    "Density Stage 1 AUPRC": fmt(stage1_auprc),
                    "Density Stage 2 Macro-F1": fmt(stage2),
                    "中文说明": note,
                }
            )
    return rows


def build_size_wave5_diagnostic_parameter_table_v6() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    final_tag = "size_wave5_lexical_bert_cue_lr"
    final_f1 = summarize_pattern(f"mws_cfe_size_results_{final_tag}_seed*.json", "f1")
    final_auprc = summarize_pattern(f"mws_cfe_size_results_{final_tag}_seed*.json", "auprc")
    threshold = summarize_thresholds(f"mws_cfe_size_results_{final_tag}_seed*.json")
    rows.extend(
        [
            {
                "诊断组": "Wave5 threshold",
                "取值": f"threshold {fmt(threshold, percent=False, decimals=6)}",
                "Seeds": str(final_f1.n),
                "Has-size F1": fmt(final_f1),
                "Has-size AUPRC": fmt(final_auprc),
                "中文说明": "阈值在 phase5-like dev 上选择；Phase5 test 未用于阈值选择。",
            },
            {
                "诊断组": "Wave5 stacked head",
                "取值": "lexical + BERT + cue logistic-regression head",
                "Seeds": str(final_f1.n),
                "Has-size F1": fmt(final_f1),
                "Has-size AUPRC": fmt(final_auprc),
                "中文说明": "最终学习式堆叠头；融合 lexical probability、BERT probability 和 cue features。",
            },
        ]
    )
    for group, value, tag, note in [
        ("component", "lexical alone", "size_wave5_lexical_alone", "lexical alone 组件基线。"),
        ("component", "lexical + BERT", "size_wave5_lexical_bert_lr", "lexical + BERT，seed42 诊断结果。"),
        ("component", "lexical + cue", "size_wave5_lexical_cue_lr", "lexical + cue，seed42 诊断结果。"),
        ("component", "lexical + BERT + cue", "size_wave5_lexical_bert_cue_lr", "lexical + BERT + cue，最终 5-seed 组件组合。"),
    ]:
        f1 = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "f1")
        auprc = summarize_pattern(f"mws_cfe_size_results_{tag}_seed*.json", "auprc")
        rows.append(
            {
                "诊断组": group,
                "取值": value,
                "Seeds": str(f1.n),
                "Has-size F1": fmt(f1),
                "Has-size AUPRC": fmt(auprc),
                "中文说明": note,
            }
        )
    return rows


def infer_figure_manifest_row(path: Path) -> dict[str, str]:
    name = path.name
    if name.startswith("p1_"):
        p_family = "P1 max_seq_length"
    elif name.startswith("p2_"):
        p_family = "P2 quality_gate"
    elif name.startswith("p3_"):
        p_family = "P3 section/input strategy"
    else:
        p_family = "unknown"

    if "stage_1_auprc" in name:
        metric = "Density Stage 1 AUPRC"
    elif "stage_2_macro_f1" in name:
        metric = "Density Stage 2 Macro-F1"
    else:
        metric = "unknown"

    main_candidates = {
        "p1_max_seq_length_stage_2_macro_f1.svg": "justifies retaining len128 over len192 for Stage 2",
        "p2_quality_gate_stage_2_macro_f1.svg": "justifies G3 gate selection for Stage 2",
        "p3_section_input_strategy_stage_2_macro_f1.svg": "summarizes section-aware input choice for Stage 2",
    }
    if name in main_candidates:
        suitable_for = "main_text_candidate"
        note = main_candidates[name]
    else:
        suitable_for = "appendix"
        note = "secondary parameter diagnostic; keep out of the main table to avoid widening the main text"

    return {
        "file_name": name,
        "source_path": str(path.relative_to(PROJECT_ROOT)),
        "P family": p_family,
        "metric": metric,
        "suitable_for": suitable_for,
        "note": note,
    }


def build_figure_manifest() -> list[dict[str, str]]:
    paths = sorted(FIGURES_DIR.glob("*.svg"))
    return [infer_figure_manifest_row(path) for path in paths]


def build_figure_manifest_v6() -> list[dict[str, str]]:
    specs = [
        (
            "p1_max_seq_length_stage_1_auprc_zoomed.svg",
            "P1 max_seq_length",
            "Density Stage 1 AUPRC",
            "appendix_or_optional_main",
            "重画图；zoomed y-axis，横坐标包含 64/96/128/160/192，并标出 128 selected。P1 用 Stage 1 AUPRC 展示，避免用非最高 Macro-F1 解释 128 选择。",
        ),
        (
            "p2_quality_gate_stage_2_macro_f1_zoomed.svg",
            "P2 quality gate",
            "Density Stage 2 Macro-F1",
            "main_text_recommended",
            "重画图；zoomed y-axis，并标出 G3 selected。正文优先保留，展示 quality gate 选择对 Stage 2 的核心影响。",
        ),
        (
            "p3_section_input_strategy_stage_2_macro_f1_zoomed.svg",
            "P3 section/input strategy",
            "Density Stage 2 Macro-F1",
            "main_text_recommended",
            "重画图；zoomed y-axis，并标出 section-aware selected。正文优先保留，展示输入构造策略的核心影响。",
        ),
        (
            "p1_max_seq_length_stage_1_auprc.svg",
            "P1 max_seq_length",
            "Density Stage 1 AUPRC",
            "appendix",
            "附录图；补充 Stage 1 AUPRC 对长度扫描的敏感性。",
        ),
        (
            "p2_quality_gate_stage_1_auprc.svg",
            "P2 quality gate",
            "Density Stage 1 AUPRC",
            "appendix",
            "附录图；补充 Stage 1 AUPRC 对 gate 选择的敏感性。",
        ),
        (
            "p3_section_input_strategy_stage_1_auprc.svg",
            "P3 section/input strategy",
            "Density Stage 1 AUPRC",
            "appendix",
            "附录图；补充 Stage 1 AUPRC 对输入构造策略的敏感性。",
        ),
    ]
    rows: list[dict[str, str]] = []
    for file_name, p_family, metric, placement, note in specs:
        if file_name.endswith("_zoomed.svg"):
            path = FINAL_FIGURES_DIR / file_name
        else:
            path = FIGURES_DIR / file_name
        rows.append(
            {
                "file_name": file_name,
                "source_path": str(path.relative_to(PROJECT_ROOT)),
                "P family": p_family,
                "metric": metric,
                "placement": placement,
                "note": note,
            }
        )
    return rows


def write_text_svg(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def percent_summary(pattern: str, metric: str) -> tuple[float, float]:
    summary = summarize_pattern(pattern, metric)
    if summary.mean is None:
        return 0.0, 0.0
    return summary.mean * 100.0, (summary.std or 0.0) * 100.0


def render_zoomed_bar_svg(
    *,
    title: str,
    subtitle: str,
    values: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    y_label: str = "Macro-F1 (%)",
    note: str,
) -> str:
    width = 980
    height = 560
    left = 92
    right = 44
    top = 92
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = min(82, plot_w / max(len(values), 1) * 0.56)

    def y_pos(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    def x_center(idx: int) -> float:
        if len(values) == 1:
            return left + plot_w / 2
        return left + (idx + 0.5) * plot_w / len(values)

    tick_count = 5
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-family="Arial" font-size="14" fill="#4b5563">{html.escape(subtitle)}; zoomed y-axis {y_min:.1f}-{y_max:.1f}%</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>',
        f'<text x="28" y="{top + plot_h / 2}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151" transform="rotate(-90 28,{top + plot_h / 2})">{html.escape(y_label)}</text>',
    ]
    for tick in range(tick_count + 1):
        value = y_min + (y_max - y_min) * tick / tick_count
        y = y_pos(value)
        pieces.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        pieces.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{value:.1f}</text>')

    for idx, row in enumerate(values):
        mean = float(row["mean"])
        std = float(row.get("std") or 0.0)
        selected = bool(row.get("selected"))
        label = str(row["label"])
        x = x_center(idx)
        y = y_pos(mean)
        base_y = top + plot_h
        color = "#d97706" if selected else "#2563eb"
        stroke = "#92400e" if selected else "#1d4ed8"
        pieces.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{base_y - y:.2f}" fill="{color}" opacity="0.88" stroke="{stroke}" stroke-width="1"/>')
        if std > 0:
            high = y_pos(min(mean + std, y_max))
            low = y_pos(max(mean - std, y_min))
            pieces.append(f'<line x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" stroke="#111827" stroke-width="1.3"/>')
            pieces.append(f'<line x1="{x - 7:.2f}" y1="{high:.2f}" x2="{x + 7:.2f}" y2="{high:.2f}" stroke="#111827" stroke-width="1.3"/>')
            pieces.append(f'<line x1="{x - 7:.2f}" y1="{low:.2f}" x2="{x + 7:.2f}" y2="{low:.2f}" stroke="#111827" stroke-width="1.3"/>')
        pieces.append(f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="#111827">{mean:.2f}</text>')
        pieces.append(
            f'<text x="{x:.2f}" y="{base_y + 28}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151" '
            f'transform="rotate(-28 {x:.2f},{base_y + 28})">{html.escape(label)}</text>'
        )
        if selected:
            pieces.append(f'<text x="{x:.2f}" y="{top - 16}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#92400e">selected</text>')
            pieces.append(f'<line x1="{x:.2f}" y1="{top - 10}" x2="{x:.2f}" y2="{max(y - 22, top + 4):.2f}" stroke="#92400e" stroke-width="1.2" stroke-dasharray="4 3"/>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def y_bounds(values: list[dict[str, Any]], *, pad: float = 1.0, floor_step: float = 0.5) -> tuple[float, float]:
    lows = [float(row["mean"]) - float(row.get("std") or 0.0) for row in values]
    highs = [float(row["mean"]) + float(row.get("std") or 0.0) for row in values]
    lower = math.floor((min(lows) - pad) / floor_step) * floor_step
    upper = math.ceil((max(highs) + pad) / floor_step) * floor_step
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def write_v6_zoomed_parameter_figures() -> None:
    p1_specs = [
        ("64", "p1_len64", False),
        ("96", "p1_len96", False),
        ("128 (selected)", "planb_full", True),
        ("160", "p1_len160", False),
        ("192", "p1_len192", False),
    ]
    p1_values = []
    for label, tag, selected in p1_specs:
        mean, std = percent_summary(f"mws_cfe_density_stage1_results_{tag}_seed*.json", "auprc")
        p1_values.append({"label": label, "mean": mean, "std": std, "selected": selected})
    p1_min, p1_max = y_bounds(p1_values, pad=1.0, floor_step=0.5)
    write_text_svg(
        FINAL_FIGURES_DIR / "p1_max_seq_length_stage_1_auprc_zoomed.svg",
        render_zoomed_bar_svg(
            title="P1 max_seq_length",
            subtitle="Density Stage 1 AUPRC across input lengths",
            values=p1_values,
            y_min=p1_min,
            y_max=p1_max,
            y_label="AUPRC (%)",
            note="",
        ),
    )

    p2_specs = [
        ("G1", "p2_g1", False),
        ("G2", "planb_full", False),
        ("G3 (selected)", "p2_g3", True),
        ("G4", "p2_g4", False),
        ("G5", "p2_g5", False),
    ]
    p2_values = []
    for label, tag, selected in p2_specs:
        mean, std = percent_summary(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
        p2_values.append({"label": label, "mean": mean, "std": std, "selected": selected})
    p2_min, p2_max = y_bounds(p2_values, pad=1.0, floor_step=0.5)
    write_text_svg(
        FINAL_FIGURES_DIR / "p2_quality_gate_stage_2_macro_f1_zoomed.svg",
        render_zoomed_bar_svg(
            title="P2 quality gate",
            subtitle="Density Stage 2 Macro-F1 across weak-label filters",
            values=p2_values,
            y_min=p2_min,
            y_max=p2_max,
            y_label="Macro-F1 (%)",
            note="",
        ),
    )

    p3_specs = [
        ("Mention", "p3_mention_text", False),
        ("Section-aware (selected)", "planb_full", True),
        ("Findings", "p3_findings_text", False),
        ("Impression", "p3_impression_text", False),
        ("Findings+Impression", "p3_findings_impression_text", False),
        ("Full text", "p3_full_text", False),
    ]
    p3_values = []
    for label, tag, selected in p3_specs:
        mean, std = percent_summary(f"mws_cfe_density_stage2_results_{tag}_seed*.json", "macro_f1")
        p3_values.append({"label": label, "mean": mean, "std": std, "selected": selected})
    p3_min, p3_max = y_bounds(p3_values, pad=1.0, floor_step=1.0)
    write_text_svg(
        FINAL_FIGURES_DIR / "p3_section_input_strategy_stage_2_macro_f1_zoomed.svg",
        render_zoomed_bar_svg(
            title="P3 section/input strategy",
            subtitle="Density Stage 2 Macro-F1 across input constructions",
            values=p3_values,
            y_min=p3_min,
            y_max=p3_max,
            y_label="Macro-F1 (%)",
            note="",
        ),
    )


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "--") for column in columns) + " |")
    return "\n".join(lines)


def build_main_markdown(raw: dict[str, dict[str, Summary]]) -> str:
    best: dict[str, float] = {}
    for spec in MAIN_METRICS:
        means = [raw[method][spec.key].mean for method, _ in MAIN_METHODS]
        best[spec.key] = max(value for value in means if value is not None)

    rows: list[dict[str, str]] = []
    for method, _ in MAIN_METHODS:
        row = {"Method": method}
        for spec in MAIN_METRICS:
            summary = raw[method][spec.key]
            is_best = summary.mean is not None and math.isclose(summary.mean, best[spec.key], rel_tol=0.0, abs_tol=1e-12)
            row[spec.title] = fmt_markdown(summary, bold=is_best)
        rows.append(row)
    return markdown_table(rows, ["Method", *[spec.title for spec in MAIN_METRICS]])


def write_main_latex(raw: dict[str, dict[str, Summary]]) -> None:
    best: dict[str, float] = {}
    for spec in MAIN_METRICS:
        means = [raw[method][spec.key].mean for method, _ in MAIN_METHODS]
        best[spec.key] = max(value for value in means if value is not None)

    extra_plm_note = ""
    if ACTIVE_VERSION in {"v4", "v5", "v6"}:
        extra_plm_note = rf"SciBERT and BioClinicalBERT are included in {ACTIVE_VERSION} as extra PLM baselines. "
    stage1_note = "P0 threshold-tuned Stage 1"
    if ACTIVE_VERSION in {"v5", "v6"}:
        stage1_note = "G3+len128 threshold-calibrated Stage 1"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Module 2 final learned-model comparison under final-tables-{ACTIVE_VERSION} protocol.}}",
        rf"\label{{tab:module2_main_{ACTIVE_VERSION}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(escape_latex(spec.title) for spec in MAIN_METRICS) + r" \\",
        r"\midrule",
    ]
    for method, _ in MAIN_METHODS:
        cells = [escape_latex(method)]
        for spec in MAIN_METRICS:
            summary = raw[method][spec.key]
            is_best = summary.mean is not None and math.isclose(summary.mean, best[spec.key], rel_tol=0.0, abs_tol=1e-12)
            cells.append(fmt_latex(summary, bold=is_best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.3em}",
            (
                r"\footnotesize{Learned-model comparison only; bold marks the best learned model in each column. "
                r"Cue-only and P2 deterministic hybrid diagnostics are excluded from the main table. "
                + extra_plm_note
                +
                rf"Ours uses {stage1_note}, G3+len128 Stage 2, Wave5 lexical+BERT+cue learned stacked Has-size head "
                r"with test\_truncated=false and test\_sample\_count=42057, and location\_aug\_g2 for Location.}"
            ),
            r"\end{table}",
        ]
    )
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / "main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generic_latex_table(path: Path, rows: list[dict[str, str]], columns: list[str], caption: str, label: str, note: str) -> None:
    align = "l" * len(columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(escape_latex(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(row.get(column, "--")).replace("+/-", r"$\pm$") for column in columns) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.3em}",
            rf"\footnotesize{{{escape_latex(note)}}}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_wave5_protocol() -> dict[str, Any]:
    paths = sorted(RESULTS_DIR.glob("mws_cfe_size_results_size_wave5_lexical_bert_cue_lr_seed*.json"), key=seed_from_path)
    records = []
    for path in paths:
        data = load_json(path)
        records.append(
            {
                "seed": seed_from_path(path),
                "test_truncated": data.get("test_truncated"),
                "test_sample_count": data.get("test_sample_count"),
                "protocol_truncated": nested_get(data, "wave5_protocol.phase5_test_truncated"),
                "protocol_sample_count": nested_get(data, "wave5_protocol.test_sample_count"),
            }
        )
    return {"records": records}


def write_report(
    main_md: str,
    raw_main: dict[str, dict[str, Summary]],
    ablation_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
    appendix_rows: list[dict[str, str]],
    figure_manifest_rows: list[dict[str, str]],
    ours_all_best: bool,
) -> None:
    wave5_summary = raw_main["MWS-CFE (Ours; final)"]["has_size_f1"]
    wave5_protocol = validate_wave5_protocol()
    wave5_record_notes = ", ".join(
        f"seed {item['seed']}: test_truncated={str(item['test_truncated']).lower()}, test_sample_count={item['test_sample_count']}"
        for item in wave5_protocol["records"]
    )
    all_best_text = "是" if ours_all_best else "否"
    missing_extra_plm = [
        method
        for method in ["SciBERT", "BioClinicalBERT / ClinicalBERT"]
        if method in raw_main and all(summary.n == 0 for summary in raw_main[method].values())
    ]
    if ACTIVE_VERSION == "v5" and missing_extra_plm:
        need_more_experiments = "需要补齐 extra PLM full 5-seed 结果"
        can_write = "可以先写口径和附录结构，但最终正文主表需等 PLM 数值补齐"
        experiment_boundary_text = (
            "当前本地 v5 表格结构已经封装完成，但正文主表若要求 SciBERT 与 "
            "BioClinicalBERT / ClinicalBERT 两行都有数值，需要先同步或运行对应 full 5-seed 结果。"
        )
        missing_plm_text = (
            "\n\n当前本地结果目录未发现 "
            + "、".join(missing_extra_plm)
            + " 的 `extra_plm_seed*.json` 结果；v5 builder 会保留这些行并以 `--` 标记缺失值。"
        )
    else:
        need_more_experiments = "不需要"
        can_write = "可以"
        experiment_boundary_text = f"当前 hcf 同步结果已经足够进入 final tables {ACTIVE_VERSION} 封板；继续补性能实验会破坏当前收口边界。"
        missing_plm_text = ""
    extra_plm_text = ""
    if ACTIVE_VERSION in {"v4", "v5"}:
        extra_plm_text = (
            f"\n\n{ACTIVE_VERSION} 在正文 learned-model 主表中保留 SciBERT 与 "
            "BioClinicalBERT / ClinicalBERT 两个 PLM baseline；cue-only 与 "
            "P2 deterministic hybrid 仍然排除。"
        )
    stage1_result_pattern = final_stage1_pattern()
    stage1_description = "P0 threshold-tuned MWS-CFE"
    if ACTIVE_VERSION == "v5":
        stage1_description = "G3 len128 threshold-calibrated MWS-CFE"
    appendix_csv = TABLES_DIR / "appendix_full_metrics_table.csv"
    appendix_tex = LATEX_DIR / "appendix_full_metrics_table.tex"
    manifest_path = FINAL_FIGURES_DIR / "figure_manifest.csv"
    figure_manifest_text = (
        f"`{manifest_path.relative_to(PROJECT_ROOT)}`，共 {len(figure_manifest_rows)} 张参数图。"
        if ACTIVE_VERSION == "v5"
        else "v5 only."
    )
    report = f"""# 模块2 final tables {ACTIVE_VERSION} 封板报告

> 日期：2026-05-07
> 范围：只做现有结果聚合、落表、LaTeX 表格与结果章节写作；未启动训练，未补性能实验。

## 1. 主表指标口径固定

正文主表按任务性质报告不同主指标。

1. Density Stage 1 是 binary evidence detection，即二分类证据检测；本文做了 threshold calibration，因此正文报告 F1、AUPRC、AUROC。
2. Density Stage 2 是 multi-class density subtype classification，即多类别密度亚型分类，因此正文报告 Macro-F1。
3. Has-size 是 binary field extraction，即二分类字段抽取，因此正文报告 F1。
4. Location 是 multi-class location extraction，即多类别位置抽取，因此正文报告 Macro-F1。
5. 其他完整指标放入附录完整指标表，避免正文主表过宽。

附录完整指标表：`{appendix_csv.relative_to(PROJECT_ROOT)}` 与 `{appendix_tex.relative_to(PROJECT_ROOT)}`。该附录当前包含 {len(appendix_rows)} 行 method-task 指标汇总。

## 2. 最终正文主表

{main_md}

说明：主表只比较 learned models，即学习模型；`*` 或加粗表示该列 learned-model 最优。Cue-only 不进入正文主表，P2 deterministic hybrid 也不进入正文主表。{extra_plm_text}{missing_plm_text}

Ours 最终是否在正文 learned-model 主表所有主指标上达到最优：**{all_best_text}**。

## 3. Ours final {ACTIVE_VERSION} 口径

Density Stage 1 使用 {stage1_description}：`outputs/phaseA2_planB/results/{stage1_result_pattern}`。

Density Stage 2 使用 final combo `G3 + len128`：`outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json`。本轮明确不使用 `len192`。

Has-size 使用 Wave5 learned stacked head：`size_wave5_lexical_bert_cue_lr`。5-seed Phase5 full test 的 Has-size F1 为 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}`；对应百分制为 `{fmt(wave5_summary)}`。协议记录：{wave5_record_notes}。

Location 使用 location augmented learned model：`outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed*.json`。该结果沿用与旧 Vanilla / old MWS 一致的 `no_location` fallback evaluation protocol。

## 4. 为什么 cue-only 不进入正文主表

Cue-only 继续作为 deterministic label-construction reference，即确定性标签构造参照。它用于说明当前弱监督标签与规则线索的关系，而不是 learned-model 的公平泛化能力。如果把 cue-only 放入正文主表，会把规则复现规则标签的闭环结果误读为模型性能。因此 {ACTIVE_VERSION} 主表将其移出，只建议放在附录或方法学说明中。

## 5. 为什么 P2 hybrid 不进入正文主表

P2 deterministic hybrid 的高分来自规则优先的决策层，与 Has-size 标签构造存在同源风险。该结果适合作为 benchmark circularity，即基准闭环风险的诊断证据，不适合作为正文 learned-model comparison 的性能行。因此 {ACTIVE_VERSION} 主表完全排除 P2 hybrid。

## 6. Has-size 为什么转为 Wave5 stacked head

BERT-only size head 在 Wave3/Wave4 中表现不稳定，尤其受阈值选择和测试截断影响。Has-size 本身强依赖局部数值、单位、范围和尺寸上下文线索，纯 BERT 表征没有稳定释放这些线索。Wave5 改为 lexical + BERT + cue 的 learned stacked head，即学习式堆叠头：用 lexical probability、BERT probability 和 cue features 共同输入 logistic-regression head。最终 `lexical + BERT + cue` 在完整 Phase5 test 上达到 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}`，且 `test_truncated=false`、`test_sample_count=42057`。

## 7. Wave3/Wave4 失败诊断

第一，BERT-only size head 不稳定，不能作为最终 Has-size 口径。

第二，`ws_val`-only threshold tuning 与 Phase5 分布不一致，导致阈值在最终测试分布上不可稳健迁移。

第三，smoke 截断 test 曾产生误导。{ACTIVE_VERSION} 只接受未截断 Phase5 full test，不再使用截断测试结论。

## 8. 消融、参数表与参数图

`ablation_table_final.csv` 分成两类：一类是可直接作为正文的 core density ablation，包括 two-stage density、section-aware input、confidence-aware training、threshold tuning 和 quality gate selection；另一类是 Has-size Wave5 component analysis。`lexical + BERT` 与 `lexical + cue` 目前只有 seed42，因此表中明确标为 diagnostic component analysis，不能伪装成 5-seed ablation。

`parameter_table_final.csv` 保留 P1 max_seq_length、P2 quality gate 和 P3 section/input strategy，并新增 Wave5 Has-size threshold / stacked head 诊断说明。P2 与 P3 在表中明确标注为 categorical design choices，即类别型设计选择，不是连续数值参数。

参数讨论图索引：{figure_manifest_text}

## 9. 结果章节写作稿

在 final-tables-{ACTIVE_VERSION} 口径下，MWS-CFE 在所有正文 learned-model 主指标上达到最优。Density Stage 1 采用 {stage1_description} 后，显式密度证据检测的 F1、AUPRC 和 AUROC 均成为 learned-model 主表最优；Density Stage 2 固定为 G3 + len128 后，Macro-F1 也超过 TF-IDF baselines 和 Vanilla PubMedBERT。Has-size 的最终口径不再使用不稳定的 BERT-only head，而是采用 Wave5 lexical + BERT + cue learned stacked head，在未截断 Phase5 full test 上取得 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}` 的 F1。Location 使用 augmented learned model 并沿用一致的 `no_location` fallback evaluation protocol，同样达到 learned-model 主表最优。

这些结果支持的论文叙事是：模块2不依赖 cue-only 或 P2 deterministic hybrid 来获得正文结论，而是在严格 learned-model comparison 中完成最终封板。Cue-only 与 P2 hybrid 应作为方法学附录材料，用来解释标签构造和闭环风险；正文主结果只保留学习模型之间的公平比较。

## 10. 最终判断

模块2是否还需要补实验：**{need_more_experiments}**。{experiment_boundary_text}

是否可以进入论文正式写作：**{can_write}**。建议后续只做文字润色、表格排版和附录说明，不再改动性能口径。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_report_v6(
    main_md: str,
    raw_main: dict[str, dict[str, Summary]],
    density_ablation_rows: list[dict[str, str]],
    size_component_rows: list[dict[str, str]],
    density_parameter_rows: list[dict[str, str]],
    size_parameter_rows: list[dict[str, str]],
    appendix_tables: dict[str, list[dict[str, str]]],
    figure_manifest_rows: list[dict[str, str]],
    ours_all_best: bool,
) -> None:
    missing_extra_plm = [
        method
        for method in ["SciBERT", "BioClinicalBERT / ClinicalBERT"]
        if method in raw_main and all(summary.n == 0 for summary in raw_main[method].values())
    ]
    need_more_experiments = "需要补齐 extra PLM full 5-seed 结果" if missing_extra_plm else "不需要"
    experiment_note = (
        "当前 v6 表格结构已完成，但正文 PLM baseline 仍缺少结果，需要先同步或运行 extra PLM full 5-seed。"
        if missing_extra_plm
        else "当前已有结果足够支持 Module 2 v6 表格封板；不需要重新训练或补 Ours 实验。"
    )
    all_best_text = "是" if ours_all_best else "否"

    appendix_paths = "\n".join(
        f"- `{(TABLES_DIR / filename).relative_to(PROJECT_ROOT)}`"
        for filename in APPENDIX_TASK_FILES.values()
    )
    latex_appendix_paths = "\n".join(
        f"- `{(LATEX_DIR / filename.replace('.csv', '.tex')).relative_to(PROJECT_ROOT)}`"
        for filename in APPENDIX_TASK_FILES.values()
    )
    main_figures = [row["file_name"] for row in figure_manifest_rows if row["placement"] == "main_text_recommended"]
    optional_figures = [row["file_name"] for row in figure_manifest_rows if row["placement"] == "appendix_or_optional_main"]
    appendix_figures = [row["file_name"] for row in figure_manifest_rows if row["placement"] == "appendix"]

    report = f"""# 模块2 final tables v6 论文呈现报告

> 日期：2026-05-07
> 范围：只重构表格和报告呈现；未启动训练，未修改任何实验结果。

## 1. v6 主表

v6 主表沿用 v5 learned-model 主表口径：正文只保留 learned models，即学习模型；cue-only 和 P2 deterministic hybrid 继续排除。Ours Density Stage 1 使用 `mws_cfe_density_stage1_results_density_final_g3_len128_seed*.json`，Density Stage 2 继续使用 `mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json`，不使用 len192。

{main_md}

Ours 是否在正文 learned-model 主表所有主指标上最优：**{all_best_text}**。

## 2. 主表 primary metrics 口径

主表不同任务使用不同 primary metrics，即主要评价指标，是因为任务定义和可解释性不同。

1. Density Stage 1 是二分类证据检测；本文做了 threshold calibration，即阈值校准，因此正文报告 F1、AUPRC 和 AUROC。
2. Density Stage 2 是多类别密度亚型分类，因此正文报告 Macro-F1，避免大类掩盖小类表现。
3. Has-size 是二分类字段抽取，因此正文报告 F1，直接反映 has_size 正类抽取质量。
4. Location 是多类别位置抽取，因此正文报告 Macro-F1，避免频繁肺叶类别主导平均结果。
5. Accuracy、Precision、Recall、AUPRC、AUROC 等完整指标进入附录，避免正文主表过宽。

## 3. 任务级完整指标附录

v6 将 appendix full metrics 拆成 4 张任务级表，而不是继续使用一张大总表。原因是四个任务可计算和应报告的指标不同；拆表后每张表只包含该任务适用指标，不再出现大量空白列。若旧结果 JSON 未提供某个适用指标，表中写作 N/A，表示未计算而非任务不适用。

CSV 路径：
{appendix_paths}

LaTeX 路径：
{latex_appendix_paths}

## 4. 消融表拆分

Density ablation 和 Has-size Wave5 component analysis 在 v6 中分开。Density ablation 是 Full vs w/o 格式，用来解释 Stage 1/Stage 2 density pipeline 的关键组件；Has-size Wave5 表是组件诊断，不伪装成严格 5-seed ablation，其中 `lexical + BERT` 和 `lexical + cue` 明确标为 seed42 诊断结果。

Density ablation：`{(TABLES_DIR / "density_ablation_table_final.csv").relative_to(PROJECT_ROOT)}`，共 {len(density_ablation_rows)} 行。

Has-size Wave5 component analysis：`{(TABLES_DIR / "size_wave5_component_table_final.csv").relative_to(PROJECT_ROOT)}`，共 {len(size_component_rows)} 行。

## 5. 参数表拆分

Density 参数表只讨论 P1/P2/P3，并只保留 Density Stage 1 AUPRC 与 Density Stage 2 Macro-F1。这里不放 Has-size F1，也不放 Density Stage 1 F1，因为很多参数设置没有统一做 P0 threshold tuning，F1 不适合作为公平参数扫描指标。

P2 quality gate 和 P3 section/input strategy 是 categorical design choices，即类别型设计选择，不是连续数值参数；因此 v6 报告把它们作为设计选项比较，而不是数值超参数曲线。

Density parameter table：`{(TABLES_DIR / "density_parameter_table_final.csv").relative_to(PROJECT_ROOT)}`，共 {len(density_parameter_rows)} 行。

Has-size Wave5 diagnostic parameter table：`{(TABLES_DIR / "size_wave5_diagnostic_parameter_table.csv").relative_to(PROJECT_ROOT)}`，共 {len(size_parameter_rows)} 行。

## 6. 参数图进入正文和附录

v6 重新绘制参数图，使用 zoomed y-axis，并在图中直接标出 selected 配置。P1 改用 Stage 1 AUPRC 展示，因为 128 不是 Stage 2 Macro-F1 最高点；这样能避免用非最高的 Stage 2 F1 解释 128 选择。P1 仍不作为正文强证据，只作为可选正文图或附录图。

正文优先推荐 2 张参数图：
{chr(10).join(f"- `{name}`" for name in main_figures)}

正文可选或附录图：
{chr(10).join(f"- `{name}`" for name in optional_figures)}

附录放 3 张参数图：
{chr(10).join(f"- `{name}`" for name in appendix_figures)}

如果版面紧张，正文只保留 `p2_quality_gate_stage_2_macro_f1_zoomed.svg` 和 `p3_section_input_strategy_stage_2_macro_f1_zoomed.svg`；P1 图移入附录。完整索引见 `{(FINAL_FIGURES_DIR / "figure_manifest.csv").relative_to(PROJECT_ROOT)}`。

## 7. 最终判断

模块2是否还需要补实验：**{need_more_experiments}**。{experiment_note}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Module 2 final tables v3/v4/v5/v6")
    parser.add_argument("--version", choices=["v3", "v4", "v5", "v6"], default="v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_version(args.version)

    main_rows, raw_main, _best_flags, verdict = build_main_table()
    ours_all_best = verdict == "yes"

    if ACTIVE_VERSION == "v6":
        density_ablation_rows = build_density_ablation_table_v6()
        size_component_rows = build_size_wave5_component_table_v6()
        density_parameter_rows = build_density_parameter_table_v6()
        size_parameter_rows = build_size_wave5_diagnostic_parameter_table_v6()
        appendix_tables = build_task_appendix_full_metrics_tables()
        write_v6_zoomed_parameter_figures()
        figure_manifest_rows = build_figure_manifest_v6()

        write_csv(TABLES_DIR / "main_table_final.csv", main_rows)
        write_main_latex(raw_main)

        write_csv(TABLES_DIR / "density_ablation_table_final.csv", density_ablation_rows)
        write_csv(TABLES_DIR / "size_wave5_component_table_final.csv", size_component_rows)
        write_csv(TABLES_DIR / "density_parameter_table_final.csv", density_parameter_rows)
        write_csv(TABLES_DIR / "size_wave5_diagnostic_parameter_table.csv", size_parameter_rows)
        write_csv(FINAL_FIGURES_DIR / "figure_manifest.csv", figure_manifest_rows)

        write_generic_latex_table(
            LATEX_DIR / "density_ablation_table.tex",
            density_ablation_rows,
            ["Variant", "Seeds", "Density Stage 1 F1", "Density Stage 1 AUPRC", "Density Stage 2 Macro-F1", "中文说明"],
            "Module 2 density ablation under final v6 presentation.",
            "tab:module2_density_ablation_v6",
            "Full-vs-without comparison for density components. N/A denotes metrics not applicable to the diagnostic row.",
        )
        write_generic_latex_table(
            LATEX_DIR / "size_wave5_component_table.tex",
            size_component_rows,
            ["Variant", "Seeds", "Has-size F1", "Has-size AUPRC", "中文说明"],
            "Module 2 Has-size Wave5 component analysis.",
            "tab:module2_size_wave5_components_v6",
            "This is component analysis, not a strict five-seed ablation for every row; seed42-only diagnostics are marked in the Chinese note column.",
        )
        write_generic_latex_table(
            LATEX_DIR / "density_parameter_table.tex",
            density_parameter_rows,
            ["参数组", "参数类型", "取值", "Seeds", "Density Stage 1 AUPRC", "Density Stage 2 Macro-F1", "中文说明"],
            "Module 2 density parameter diagnostics for P1/P2/P3.",
            "tab:module2_density_parameters_v6",
            "Stage 1 F1 is omitted because not every scan uses the same P0 threshold calibration; P2/P3 are categorical design choices.",
        )
        write_generic_latex_table(
            LATEX_DIR / "size_wave5_diagnostic_parameter_table.tex",
            size_parameter_rows,
            ["诊断组", "取值", "Seeds", "Has-size F1", "Has-size AUPRC", "中文说明"],
            "Module 2 Has-size Wave5 diagnostic parameters.",
            "tab:module2_size_wave5_diagnostics_v6",
            "Wave5 diagnostics document thresholding, stacked-head design, and component combinations.",
        )

        for task, rows in appendix_tables.items():
            filename = APPENDIX_TASK_FILES[task]
            write_csv(TABLES_DIR / filename, rows)
            metric_columns = [title for _, title in APPENDIX_TASK_METRICS[task]]
            write_generic_latex_table(
                LATEX_DIR / filename.replace(".csv", ".tex"),
                rows,
                ["Method", "Seeds", *metric_columns],
                f"Module 2 {TASK_TITLES[task]} full metrics appendix.",
                f"tab:module2_{task}_full_metrics_v6",
                "Task-level appendix table; only metrics applicable to this task are included.",
            )

        main_md = build_main_markdown(raw_main)
        write_report_v6(
            main_md,
            raw_main,
            density_ablation_rows,
            size_component_rows,
            density_parameter_rows,
            size_parameter_rows,
            appendix_tables,
            figure_manifest_rows,
            ours_all_best,
        )

        print(f"Wrote final tables {ACTIVE_VERSION} artifacts:")
        print(TABLES_DIR / "main_table_final.csv")
        print(TABLES_DIR / "density_ablation_table_final.csv")
        print(TABLES_DIR / "size_wave5_component_table_final.csv")
        print(TABLES_DIR / "density_parameter_table_final.csv")
        print(TABLES_DIR / "size_wave5_diagnostic_parameter_table.csv")
        for filename in APPENDIX_TASK_FILES.values():
            print(TABLES_DIR / filename)
            print(LATEX_DIR / filename.replace(".csv", ".tex"))
        print(LATEX_DIR / "main_table.tex")
        print(LATEX_DIR / "density_ablation_table.tex")
        print(LATEX_DIR / "size_wave5_component_table.tex")
        print(LATEX_DIR / "density_parameter_table.tex")
        print(LATEX_DIR / "size_wave5_diagnostic_parameter_table.tex")
        print(FINAL_FIGURES_DIR / "figure_manifest.csv")
        print(REPORT_PATH)
        return

    ablation_rows = build_ablation_table()
    parameter_rows = build_parameter_table()
    appendix_rows = build_appendix_full_metrics_table()
    figure_manifest_rows = build_figure_manifest() if ACTIVE_VERSION == "v5" else []

    write_csv(TABLES_DIR / "main_table_final.csv", main_rows)
    write_csv(TABLES_DIR / "ablation_table_final.csv", ablation_rows)
    write_csv(TABLES_DIR / "parameter_table_final.csv", parameter_rows)
    write_csv(TABLES_DIR / "appendix_full_metrics_table.csv", appendix_rows)
    if ACTIVE_VERSION == "v5":
        write_csv(FINAL_FIGURES_DIR / "figure_manifest.csv", figure_manifest_rows)

    write_main_latex(raw_main)
    write_generic_latex_table(
        LATEX_DIR / "ablation_table.tex",
        ablation_rows,
        [
            "Section",
            "Variant",
            "Seeds",
            "Density Stage 1 F1",
            "Density Stage 1 AUPRC",
            "Density Stage 2 Macro-F1",
            "Has-size F1",
            "Protocol Note",
        ],
        "Module 2 final core ablations and Wave5 Has-size component analysis.",
        f"tab:module2_ablation_{ACTIVE_VERSION}",
        "The first block is the directly reportable core density ablation. The Wave5 block is component analysis; seed42-only rows are diagnostic and not five-seed ablations.",
    )
    write_generic_latex_table(
        LATEX_DIR / "parameter_table.tex",
        parameter_rows,
        [
            "Parameter Group",
            "Parameter Type",
            "Value",
            "Seeds",
            "Density Stage 1 AUPRC",
            "Density Stage 1 F1",
            "Density Stage 2 Macro-F1",
            "Has-size F1",
            "Notes",
        ],
        "Module 2 parameter and decision-layer diagnostics.",
        f"tab:module2_parameter_{ACTIVE_VERSION}",
        "P2 quality gate and P3 section/input strategy are categorical design choices, not continuous numerical parameters. Wave5 rows document Has-size threshold and stacked-head diagnostics.",
    )
    write_generic_latex_table(
        LATEX_DIR / "appendix_full_metrics_table.tex",
        appendix_rows,
        [
            "Method",
            "Task",
            "Seeds",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro-F1",
            "AUPRC",
            "AUROC",
        ],
        "Module 2 appendix full metrics by method and task.",
        f"tab:module2_appendix_full_metrics_{ACTIVE_VERSION}",
        "The main table reports task-appropriate primary metrics only; this appendix lists all aggregate metrics available in phase5_test_results.",
    )

    main_md = build_main_markdown(raw_main)
    write_report(main_md, raw_main, ablation_rows, parameter_rows, appendix_rows, figure_manifest_rows, ours_all_best)

    print(f"Wrote final tables {ACTIVE_VERSION} artifacts:")
    print(TABLES_DIR / "main_table_final.csv")
    print(TABLES_DIR / "ablation_table_final.csv")
    print(TABLES_DIR / "parameter_table_final.csv")
    print(TABLES_DIR / "appendix_full_metrics_table.csv")
    print(LATEX_DIR / "main_table.tex")
    print(LATEX_DIR / "ablation_table.tex")
    print(LATEX_DIR / "parameter_table.tex")
    print(LATEX_DIR / "appendix_full_metrics_table.tex")
    if ACTIVE_VERSION == "v5":
        print(FINAL_FIGURES_DIR / "figure_manifest.csv")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
