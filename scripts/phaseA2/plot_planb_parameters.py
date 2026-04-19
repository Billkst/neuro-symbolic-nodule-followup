#!/usr/bin/env python3
"""Render Plan B parameter discussion figures as dependency-free SVG files."""
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VALUE_RE = re.compile(r"(?P<mean>-?\d+(?:\.\d+)?)\s*\+/-\s*(?P<std>\d+(?:\.\d+)?)")
METRICS = [
    ("Density Stage 1 auprc", "Stage 1 AUPRC"),
    ("Density Stage 2 macro_f1", "Stage 2 Macro-F1"),
]


def parse_value(value: str) -> tuple[float | None, float | None]:
    match = VALUE_RE.search(value or "")
    if not match:
        return None, None
    return float(match.group("mean")), float(match.group("std"))


def read_parameter_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
    )


def render_svg(title: str, values: list[tuple[str, float, float | None]]) -> str:
    width = 920
    height = 520
    left = 90
    right = 30
    top = 70
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    means = [value for _, value, _ in values]
    max_y = max(max(means) * 1.12, 1.0) if means else 1.0
    if max_y <= 1.2:
        max_y = 100.0

    def x_pos(idx: int) -> float:
        if len(values) <= 1:
            return left + plot_w / 2
        return left + idx * (plot_w / (len(values) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - (value / max_y) * plot_h

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" fill="#111">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.4"/>',
    ]
    for tick in range(0, 6):
        value = max_y * tick / 5
        y = y_pos(value)
        pieces.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        pieces.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#444">{value:.0f}</text>')
    points = []
    for idx, (label, mean, std) in enumerate(values):
        x = x_pos(idx)
        y = y_pos(mean)
        points.append(f"{x:.2f},{y:.2f}")
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#1f77b4"/>')
        pieces.append(f'<text x="{x:.2f}" y="{y - 12:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111">{mean:.2f}</text>')
        if std is not None:
            y_low = y_pos(max(mean - std, 0.0))
            y_high = y_pos(mean + std)
            pieces.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="#1f77b4" stroke-width="1.2"/>')
            pieces.append(f'<line x1="{x - 5:.2f}" y1="{y_high:.2f}" x2="{x + 5:.2f}" y2="{y_high:.2f}" stroke="#1f77b4" stroke-width="1.2"/>')
            pieces.append(f'<line x1="{x - 5:.2f}" y1="{y_low:.2f}" x2="{x + 5:.2f}" y2="{y_low:.2f}" stroke="#1f77b4" stroke-width="1.2"/>')
        pieces.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#333" transform="rotate(-35 {x:.2f},{top + plot_h + 28})">{html.escape(label)}</text>'
        )
    if len(points) >= 2:
        pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>')
    pieces.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#444">Values are mean +/- std over completed seeds; y-axis is percent.</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Plan B parameter figures")
    parser.add_argument("--summary-csv", default="outputs/phaseA2_planB/tables/planb_parameter_summary.csv")
    parser.add_argument("--output-dir", default="outputs/phaseA2_planB/figures")
    args = parser.parse_args()

    summary_path = PROJECT_ROOT / args.summary_csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_parameter_rows(summary_path)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["Parameter"]].append(row)

    written: list[Path] = []
    for parameter, param_rows in grouped.items():
        for metric_key, metric_display in METRICS:
            values = []
            for row in param_rows:
                mean, std = parse_value(row.get(metric_key, ""))
                if mean is None:
                    continue
                values.append((row["Value"], mean, std))
            if not values:
                continue
            title = f"{parameter}: {metric_display}"
            svg = render_svg(title, values)
            out_path = output_dir / f"{safe_name(parameter)}_{safe_name(metric_display)}.svg"
            out_path.write_text(svg, encoding="utf-8")
            written.append(out_path)
    for path in written:
        print(f"[Saved] {path}", flush=True)
    if not written:
        print("[WARN] no figures generated; check completed parameter result files", flush=True)


if __name__ == "__main__":
    main()
