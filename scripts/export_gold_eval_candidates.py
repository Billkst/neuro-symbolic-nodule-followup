"""Export 80-row mention-level gold evaluation candidate CSV from Phase 5 test split.

Stratified sampling across 4 strata:
  A (30): Clear positive — explicit quality, non-unclear density, has_size, has_location
  B (25): Missing/absent info — at least one gap in density/size/location
  C (15): Boundary/unusual — multi-nodule or atypical mention patterns
  D (10): Rare classes — part_solid density or lingula location

Input:  outputs/phase5/datasets/density_test.jsonl
Output: data/gold_eval_candidates_v1.csv
Seed:   42
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
TOTAL_SAMPLES = 80
WINDOW_RADIUS = 200

REAL_LOBES = {"RUL", "RML", "RLL", "LUL", "LLL", "bilateral", "lingula"}
BOUNDARY_PATTERN = re.compile(
    r"(multiple|several|numerous|bilateral|scattered)", re.IGNORECASE
)

OUTPUT_COLUMNS = [
    "sample_id",
    "subject_id",
    "note_id",
    "mention_text",
    "text_window",
    "silver_density_category",
    "silver_has_size",
    "silver_size_mm",
    "silver_location_lobe",
    "split",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_text_window(full_text: str, mention_text: str, radius: int = WINDOW_RADIUS) -> str:
    ft_norm = re.sub(r"\s+", " ", full_text)
    mt_norm = re.sub(r"\s+", " ", mention_text)
    idx = ft_norm.find(mt_norm)
    if idx < 0:
        return mention_text
    start = max(0, idx - radius)
    end = min(len(ft_norm), idx + len(mt_norm) + radius)
    return ft_norm[start:end]


def _location(row: dict) -> str:
    loc = row.get("location_label")
    if loc is None:
        return "no_location"
    return loc



def stratify(rows: list[dict], seed: int = SEED) -> list[dict]:
    import random
    rng = random.Random(seed)
    random.seed(seed)

    used_ids: set[str] = set()

    def take(pool: list[dict], n: int) -> list[dict]:
        available = [r for r in pool if r["sample_id"] not in used_ids]
        rng.shuffle(available)
        picked = available[:n]
        for r in picked:
            used_ids.add(r["sample_id"])
        return picked

    # --- Stratum D: Rare classes (10) ---
    pool_part_solid = [r for r in rows if r["density_label"] == "part_solid"]
    pool_part_solid_explicit = [r for r in pool_part_solid if r["label_quality"] == "explicit"]
    pool_part_solid_rest = [r for r in pool_part_solid if r["label_quality"] != "explicit"]
    d_part_solid = take(pool_part_solid_explicit, 6)
    if len(d_part_solid) < 6:
        d_part_solid += take(pool_part_solid_rest, 6 - len(d_part_solid))

    pool_lingula = [r for r in rows if _location(r) == "lingula"]
    pool_lingula_good = [r for r in pool_lingula if r["density_label"] != "unclear"]
    pool_lingula_rest = [r for r in pool_lingula if r["density_label"] == "unclear"]
    d_lingula = take(pool_lingula_good, 4)
    if len(d_lingula) < 4:
        d_lingula += take(pool_lingula_rest, 4 - len(d_lingula))

    stratum_d = d_part_solid + d_lingula

    # --- Stratum C: Boundary/unusual (15) ---
    pool_boundary = [
        r for r in rows
        if BOUNDARY_PATTERN.search(r["mention_text"])
        and r["density_label"] != "unclear"
    ]
    stratum_c = take(pool_boundary, 15)

    # --- Stratum A: Clear positive (30) ---
    pool_a = [
        r for r in rows
        if r["label_quality"] == "explicit"
        and r["density_label"] != "unclear"
        and r["has_size"] is True
        and _location(r) in REAL_LOBES
    ]

    density_pools_a: dict[str, list[dict]] = {}
    for r in pool_a:
        density_pools_a.setdefault(r["density_label"], []).append(r)

    density_targets = {
        "ground_glass": 14,
        "solid": 10,
        "calcified": 4,
        "part_solid": 2,
    }
    stratum_a: list[dict] = []
    for density, target in density_targets.items():
        pool = density_pools_a.get(density, [])
        stratum_a += take(pool, target)

    if len(stratum_a) < 30:
        backfill_pool = [
            r for r in pool_a
            if r["sample_id"] not in used_ids
        ]
        stratum_a += take(backfill_pool, 30 - len(stratum_a))

    # --- Stratum B: Missing/absent info (25) ---
    # B1: unclear density but has something else (8)
    pool_b1 = [
        r for r in rows
        if r["density_label"] == "unclear"
        and (r["has_size"] is True or _location(r) in REAL_LOBES)
    ]
    stratum_b1 = take(pool_b1, 8)

    # B2: no size, non-unclear density (8)
    pool_b2 = [
        r for r in rows
        if r["has_size"] is False
        and r["density_label"] != "unclear"
    ]
    stratum_b2 = take(pool_b2, 8)

    # B3: no location, non-unclear density, has size (9)
    pool_b3 = [
        r for r in rows
        if _location(r) in {"no_location", "unclear"}
        and r["density_label"] != "unclear"
        and r["has_size"] is True
    ]
    stratum_b3 = take(pool_b3, 9)

    stratum_b = stratum_b1 + stratum_b2 + stratum_b3

    all_selected = stratum_d + stratum_c + stratum_a + stratum_b
    assert len(all_selected) == TOTAL_SAMPLES, (
        f"Expected {TOTAL_SAMPLES}, got {len(all_selected)} "
        f"(D={len(stratum_d)}, C={len(stratum_c)}, A={len(stratum_a)}, B={len(stratum_b)})"
    )
    assert len({r["sample_id"] for r in all_selected}) == TOTAL_SAMPLES, "Duplicate sample_id"

    # Tag stratum for traceability
    for r in stratum_a:
        r["_stratum"] = "A_clear_positive"
    for r in stratum_b:
        r["_stratum"] = "B_missing_info"
    for r in stratum_c:
        r["_stratum"] = "C_boundary"
    for r in stratum_d:
        r["_stratum"] = "D_rare_class"

    return all_selected


def to_output_row(row: dict) -> dict:
    text_window = build_text_window(row["full_text"], row["mention_text"])
    loc = row.get("location_label")
    return {
        "sample_id": row["sample_id"],
        "subject_id": row["subject_id"],
        "note_id": row["note_id"],
        "mention_text": row["mention_text"],
        "text_window": text_window,
        "silver_density_category": row["density_label"],
        "silver_has_size": row["has_size"],
        "silver_size_mm": row["size_label"] if row["size_label"] is not None else "",
        "silver_location_lobe": loc if loc is not None else "no_location",
        "split": "test",
    }


def validate(output_rows: list[dict]) -> None:
    assert len(output_rows) == TOTAL_SAMPLES, f"Row count {len(output_rows)} != {TOTAL_SAMPLES}"
    ids = [r["sample_id"] for r in output_rows]
    assert len(set(ids)) == TOTAL_SAMPLES, "Duplicate sample_id in output"
    for r in output_rows:
        assert r["text_window"], f"Empty text_window for {r['sample_id']}"
        assert r["silver_density_category"], f"Empty density for {r['sample_id']}"
        assert r["silver_location_lobe"], f"Empty location for {r['sample_id']}"
        assert r["split"] == "test", f"Non-test split for {r['sample_id']}"

    densities = [r["silver_density_category"] for r in output_rows]
    for d in ["ground_glass", "solid", "calcified", "part_solid"]:
        assert densities.count(d) >= 1, f"Missing density class: {d}"

    has_size_count = sum(1 for r in output_rows if r["silver_has_size"])
    no_size_count = sum(1 for r in output_rows if not r["silver_has_size"])
    assert has_size_count >= 20, f"Too few has_size=True: {has_size_count}"
    assert no_size_count >= 8, f"Too few has_size=False: {no_size_count}"

    locations = [r["silver_location_lobe"] for r in output_rows]
    assert locations.count("lingula") >= 2, f"Too few lingula: {locations.count('lingula')}"

    print(f"  [OK] All validations passed ({TOTAL_SAMPLES} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "outputs" / "phase5" / "datasets" / "density_test.jsonl"),
        help="Path to density_test.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "gold_eval_candidates_v1.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[Config] seed={SEED} total={TOTAL_SAMPLES} window_radius={WINDOW_RADIUS}", flush=True)
    print(f"[Input]  {input_path}", flush=True)
    print(f"[Output] {output_path}", flush=True)

    print("[Step 1/4] Loading test split ...", flush=True)
    rows = load_jsonl(input_path)
    print(f"  loaded {len(rows)} mention-level samples", flush=True)

    print("[Step 2/4] Stratified sampling ...", flush=True)
    selected = stratify(rows, seed=SEED)
    from collections import Counter
    stratum_counts = Counter(r["_stratum"] for r in selected)
    for s, c in sorted(stratum_counts.items()):
        print(f"  {s}: {c}", flush=True)

    print("[Step 3/4] Building output rows ...", flush=True)
    output_rows = [to_output_row(r) for r in selected]

    print("[Step 4/4] Validating & writing ...", flush=True)
    validate(output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[Done] Wrote {len(output_rows)} rows to {output_path}", flush=True)

    print(f"\n[Summary]", flush=True)
    density_dist = Counter(r["silver_density_category"] for r in output_rows)
    print(f"  density: {dict(sorted(density_dist.items()))}", flush=True)
    loc_dist = Counter(r["silver_location_lobe"] for r in output_rows)
    print(f"  location: {dict(sorted(loc_dist.items()))}", flush=True)
    size_count = sum(1 for r in output_rows if r["silver_has_size"])
    print(f"  has_size: True={size_count} False={TOTAL_SAMPLES - size_count}", flush=True)


if __name__ == "__main__":
    main()
