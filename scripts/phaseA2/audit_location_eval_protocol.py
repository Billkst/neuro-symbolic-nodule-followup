#!/usr/bin/env python3
"""Print Location evaluation protocol fields from result JSON files."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_protocol_block(payload: dict[str, Any], split: str) -> dict[str, Any]:
    value = payload.get(split)
    return value if isinstance(value, dict) else {}


def row_for(path: Path, split: str) -> dict[str, Any]:
    payload = load_json(path)
    block = pick_protocol_block(payload, split)
    seed = payload.get("seed")
    if seed is None:
        match = re.search(r"_seed(\d+)", str(payload.get("tag") or path.name))
        seed = int(match.group(1)) if match else None
    return {
        "path": str(path),
        "method": payload.get("method"),
        "task": payload.get("task"),
        "tag": payload.get("tag"),
        "seed": seed,
        "split": split,
        "macro_f1": block.get("macro_f1"),
        "accuracy": block.get("accuracy"),
        "note": block.get("note"),
        "no_location_count": block.get("no_location_count"),
        "has_location_count": block.get("has_location_count"),
    }


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "path",
        "method",
        "task",
        "tag",
        "seed",
        "split",
        "macro_f1",
        "accuracy",
        "note",
        "no_location_count",
        "has_location_count",
    ]
    print("\t".join(columns), flush=True)
    for row in rows:
        print("\t".join("" if row.get(col) is None else str(row.get(col)) for col in columns), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Location result JSON evaluation protocol fields")
    parser.add_argument("paths", nargs="+", help="Location result JSON paths or glob patterns")
    parser.add_argument(
        "--split",
        choices=["ws_val_results", "ws_test_results", "phase5_test_results"],
        default="phase5_test_results",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of TSV")
    args = parser.parse_args()

    rows = [row_for(path, args.split) for path in expand_paths(args.paths)]
    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
