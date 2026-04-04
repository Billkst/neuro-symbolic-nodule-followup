from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/mimic_note_extracted/note")


def load_radiology(data_dir=DATA_DIR, nrows=None):
    return pd.read_csv(data_dir / "radiology.csv.gz", nrows=nrows)


def load_radiology_detail(data_dir=DATA_DIR, nrows=None):
    return pd.read_csv(data_dir / "radiology_detail.csv.gz", nrows=nrows)


def load_discharge(data_dir=DATA_DIR, nrows=None):
    return pd.read_csv(data_dir / "discharge.csv.gz", nrows=nrows)
