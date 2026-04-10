"""Labeling functions for density, size, and location tasks."""

from src.weak_supervision.labeling_functions.density_lfs import DENSITY_LFS
from src.weak_supervision.labeling_functions.size_lfs import SIZE_LFS
from src.weak_supervision.labeling_functions.location_lfs import LOCATION_LFS

ALL_LFS = {
    "density": DENSITY_LFS,
    "size": SIZE_LFS,
    "location": LOCATION_LFS,
}

__all__ = ["DENSITY_LFS", "SIZE_LFS", "LOCATION_LFS", "ALL_LFS"]
