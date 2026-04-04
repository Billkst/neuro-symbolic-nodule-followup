from pathlib import Path
import zipfile

import pandas as pd


def _find_member(zip_file: zipfile.ZipFile, filename: str) -> str:
    members = zip_file.namelist()
    candidates = [name for name in members if name.lower().endswith(f"/{filename}") or name.lower() == filename]
    if not candidates:
        raise FileNotFoundError(f"{filename} not found inside zip archive")
    candidates.sort(key=len)
    return candidates[0]


def load_patients_from_zip(zip_path="data/mimic-iv-3.1.zip"):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = _find_member(zf, "patients.csv.gz")
        with zf.open(member) as f:
            return pd.read_csv(
                f,
                compression="gzip",
                usecols=[
                    "subject_id",
                    "gender",
                    "anchor_age",
                    "anchor_year",
                    "anchor_year_group",
                    "dod",
                ],
            )


def load_admissions_from_zip(zip_path="data/mimic-iv-3.1.zip"):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = _find_member(zf, "admissions.csv.gz")
        with zf.open(member) as f:
            return pd.read_csv(f, compression="gzip")
