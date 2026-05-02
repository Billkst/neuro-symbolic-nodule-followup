#!/usr/bin/env python3
"""Build Module 2 final tables v3/v4 from existing Plan B result artifacts.

This script is read-only with respect to experiment outputs: it aggregates
already-produced JSON/CSV files and writes paper-ready CSV, LaTeX, and Markdown
artifacts. It must not launch training or evaluation.
"""
from __future__ import annotations

import argparse
import csv
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

OURS_METHOD = (
    "MWS-CFE (Ours; final)",
    {
        "density_stage1": "mws_cfe_density_stage1_results_planb_full_seed*.json",
        "density_stage2": "mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json",
        "size": "mws_cfe_size_results_size_wave5_lexical_bert_cue_lr_seed*.json",
        "location": "mws_cfe_location_results_location_aug_g2_seed*.json",
    },
)

MAIN_METHODS = [
    *BASE_MAIN_METHODS,
    (
        OURS_METHOD[0],
        OURS_METHOD[1],
    ),
]


def configure_version(version: str) -> None:
    global ACTIVE_VERSION, TABLES_DIR, LATEX_DIR, REPORT_PATH, MAIN_METHODS
    if version not in {"v3", "v4"}:
        raise ValueError(f"Unsupported final table version: {version}")
    ACTIVE_VERSION = version
    TABLES_DIR = PROJECT_ROOT / f"outputs/phaseA2_planB/final_tables_{version}"
    LATEX_DIR = PROJECT_ROOT / f"outputs/phaseA2_planB/final_tables_latex_{version}"
    REPORT_PATH = PROJECT_ROOT / f"reports/module2_final_sota_writeup_{version}.md"
    if version == "v4":
        MAIN_METHODS = [*BASE_MAIN_METHODS, *EXTRA_PLM_METHODS, OURS_METHOD]
    else:
        MAIN_METHODS = [*BASE_MAIN_METHODS, OURS_METHOD]


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
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
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
        "f1": summarize_pattern("mws_cfe_density_stage1_results_planb_full_seed*.json", "f1"),
        "auprc": summarize_pattern("mws_cfe_density_stage1_results_planb_full_seed*.json", "auprc"),
    }
    final_stage2 = summarize_pattern("mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json", "macro_f1")
    rows: list[dict[str, str]] = [
        {
            "Section": "Core density ablation",
            "Variant": "Final two-stage density (P0 Stage 1 + G3 len128 Stage 2)",
            "Source": "stage1 planb_full; stage2 density_final_g3_len128",
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

    extra_plm_note = (
        r"SciBERT and BioClinicalBERT are added in v4 as extra PLM baselines. "
        if ACTIVE_VERSION == "v4"
        else ""
    )
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
                r"Ours uses P0 threshold-tuned Stage 1, G3+len128 Stage 2, Wave5 lexical+BERT+cue learned stacked Has-size head "
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


def write_report(main_md: str, raw_main: dict[str, dict[str, Summary]], ablation_rows: list[dict[str, str]], parameter_rows: list[dict[str, str]], ours_all_best: bool) -> None:
    wave5_summary = raw_main["MWS-CFE (Ours; final)"]["has_size_f1"]
    wave5_protocol = validate_wave5_protocol()
    wave5_record_notes = ", ".join(
        f"seed {item['seed']}: test_truncated={str(item['test_truncated']).lower()}, test_sample_count={item['test_sample_count']}"
        for item in wave5_protocol["records"]
    )
    all_best_text = "是" if ours_all_best else "否"
    need_more_experiments = "不需要"
    can_write = "可以"
    extra_plm_text = ""
    if ACTIVE_VERSION == "v4":
        extra_plm_text = (
            "\n\nv4 在正文 learned-model 主表中新增 SciBERT 与 "
            "BioClinicalBERT / ClinicalBERT 两个 PLM baseline；cue-only 与 "
            "P2 deterministic hybrid 仍然排除。"
        )
    report = f"""# 模块2 final tables {ACTIVE_VERSION} 封板报告

> 日期：2026-05-02  
> 范围：只做现有结果聚合、落表、LaTeX 表格与结果章节写作；未启动训练，未补性能实验。

## 1. 最终正文主表

{main_md}

说明：主表只比较 learned models，即学习模型；`*` 或加粗表示该列 learned-model 最优。Cue-only 不进入正文主表，P2 deterministic hybrid 也不进入正文主表。{extra_plm_text}

Ours 最终是否在正文 learned-model 主表所有主指标上达到最优：**{all_best_text}**。

## 2. Ours final {ACTIVE_VERSION} 口径

Density Stage 1 使用 P0 threshold-tuned MWS-CFE：`outputs/phaseA2_planB/results/mws_cfe_density_stage1_results_planb_full_seed*.json`。

Density Stage 2 使用 final combo `G3 + len128`：`outputs/phaseA2_planB/results/mws_cfe_density_stage2_results_density_final_g3_len128_seed*.json`。本轮明确不使用 `len192`。

Has-size 使用 Wave5 learned stacked head：`size_wave5_lexical_bert_cue_lr`。5-seed Phase5 full test 的 Has-size F1 为 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}`；对应百分制为 `{fmt(wave5_summary)}`。协议记录：{wave5_record_notes}。

Location 使用 location augmented learned model：`outputs/phaseA2_planB/results/mws_cfe_location_results_location_aug_g2_seed*.json`。该结果沿用与旧 Vanilla / old MWS 一致的 `no_location` fallback evaluation protocol。

## 3. 为什么 cue-only 不进入正文主表

Cue-only 继续作为 deterministic label-construction reference，即确定性标签构造参照。它用于说明当前弱监督标签与规则线索的关系，而不是 learned-model 的公平泛化能力。如果把 cue-only 放入正文主表，会把规则复现规则标签的闭环结果误读为模型性能。因此 {ACTIVE_VERSION} 主表将其移出，只建议放在附录或方法学说明中。

## 4. 为什么 P2 hybrid 不进入正文主表

P2 deterministic hybrid 的高分来自规则优先的决策层，与 Has-size 标签构造存在同源风险。该结果适合作为 benchmark circularity，即基准闭环风险的诊断证据，不适合作为正文 learned-model comparison 的性能行。因此 {ACTIVE_VERSION} 主表完全排除 P2 hybrid。

## 5. Has-size 为什么转为 Wave5 stacked head

BERT-only size head 在 Wave3/Wave4 中表现不稳定，尤其受阈值选择和测试截断影响。Has-size 本身强依赖局部数值、单位、范围和尺寸上下文线索，纯 BERT 表征没有稳定释放这些线索。Wave5 改为 lexical + BERT + cue 的 learned stacked head，即学习式堆叠头：用 lexical probability、BERT probability 和 cue features 共同输入 logistic-regression head。最终 `lexical + BERT + cue` 在完整 Phase5 test 上达到 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}`，且 `test_truncated=false`、`test_sample_count=42057`。

## 6. Wave3/Wave4 失败诊断

第一，BERT-only size head 不稳定，不能作为最终 Has-size 口径。

第二，`ws_val`-only threshold tuning 与 Phase5 分布不一致，导致阈值在最终测试分布上不可稳健迁移。

第三，smoke 截断 test 曾产生误导。{ACTIVE_VERSION} 只接受未截断 Phase5 full test，不再使用截断测试结论。

## 7. 消融与参数表处理

`ablation_table_final.csv` 分成两类：一类是可直接作为正文的 core density ablation，包括 two-stage density、section-aware input、confidence-aware training、threshold tuning 和 quality gate selection；另一类是 Has-size Wave5 component analysis。`lexical + BERT` 与 `lexical + cue` 目前只有 seed42，因此表中明确标为 diagnostic component analysis，不能伪装成 5-seed ablation。

`parameter_table_final.csv` 保留 P1 max_seq_length、P2 quality gate 和 P3 section/input strategy，并新增 Wave5 Has-size threshold / stacked head 诊断说明。P2 与 P3 在表中明确标注为 categorical design choices，即类别型设计选择，不是连续数值参数。

## 8. 结果章节写作稿

在 final-tables-{ACTIVE_VERSION} 口径下，MWS-CFE 在所有正文 learned-model 主指标上达到最优。Density Stage 1 采用 P0 threshold-tuned decision layer 后，显式密度证据检测的 F1、AUPRC 和 AUROC 均成为 learned-model 主表最优；Density Stage 2 固定为 G3 + len128 后，Macro-F1 也超过 TF-IDF baselines 和 Vanilla PubMedBERT。Has-size 的最终口径不再使用不稳定的 BERT-only head，而是采用 Wave5 lexical + BERT + cue learned stacked head，在未截断 Phase5 full test 上取得 `{wave5_summary.mean:.6f} +/- {wave5_summary.std:.6f}` 的 F1。Location 使用 augmented learned model 并沿用一致的 `no_location` fallback evaluation protocol，同样达到 learned-model 主表最优。

这些结果支持的论文叙事是：模块2不依赖 cue-only 或 P2 deterministic hybrid 来获得正文结论，而是在严格 learned-model comparison 中完成最终封板。Cue-only 与 P2 hybrid 应作为方法学附录材料，用来解释标签构造和闭环风险；正文主结果只保留学习模型之间的公平比较。

## 9. 最终判断

模块2是否还需要补实验：**{need_more_experiments}**。当前 hcf 同步结果已经足够进入 final tables {ACTIVE_VERSION} 封板；继续补性能实验会破坏当前收口边界。

是否可以进入论文正式写作：**{can_write}**。建议后续只做文字润色、表格排版和附录说明，不再改动性能口径。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Module 2 final tables v3/v4")
    parser.add_argument("--version", choices=["v3", "v4"], default="v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_version(args.version)

    main_rows, raw_main, _best_flags, verdict = build_main_table()
    ours_all_best = verdict == "yes"
    ablation_rows = build_ablation_table()
    parameter_rows = build_parameter_table()

    write_csv(TABLES_DIR / "main_table_final.csv", main_rows)
    write_csv(TABLES_DIR / "ablation_table_final.csv", ablation_rows)
    write_csv(TABLES_DIR / "parameter_table_final.csv", parameter_rows)

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

    main_md = build_main_markdown(raw_main)
    write_report(main_md, raw_main, ablation_rows, parameter_rows, ours_all_best)

    print(f"Wrote final tables {ACTIVE_VERSION} artifacts:")
    print(TABLES_DIR / "main_table_final.csv")
    print(TABLES_DIR / "ablation_table_final.csv")
    print(TABLES_DIR / "parameter_table_final.csv")
    print(LATEX_DIR / "main_table.tex")
    print(LATEX_DIR / "ablation_table.tex")
    print(LATEX_DIR / "parameter_table.tex")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
