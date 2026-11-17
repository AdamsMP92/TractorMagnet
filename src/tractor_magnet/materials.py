"""Room-temperature permanent-magnet grades used by the browser app.

The remanence intervals are taken from the physical magnet data published by
supermagnete.de: https://www.supermagnete.de/physical-magnet-data
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MagnetGrade:
    """Named permanent-magnet grade with a specified remanence interval."""

    key: str
    material: str
    grade: str
    remanence_min: float
    remanence_max: float

    @property
    def nominal_remanence(self) -> float:
        """Return the interval midpoint rounded to the nearest millitesla."""

        midpoint = 0.5 * (self.remanence_min + self.remanence_max)
        return math.floor(1000.0 * midpoint + 0.5) / 1000.0

    @property
    def label(self) -> str:
        return (
            f"{self.material} {self.grade} "
            f"({self.remanence_min:.3f}–{self.remanence_max:.3f} T)"
        )

    def contains(self, remanence: float) -> bool:
        return self.remanence_min <= remanence <= self.remanence_max


MAGNET_GRADES = (
    MagnetGrade("ferrite_hf24_23", "Ferrite", "HF 24/23", 0.350, 0.365),
    MagnetGrade("ferrite_y30bh", "Ferrite", "Y30BH", 0.380, 0.390),
    MagnetGrade("ferrite_y35", "Ferrite", "Y35", 0.400, 0.410),
    MagnetGrade("ndfeb_n35", "NdFeB", "N35", 1.170, 1.210),
    MagnetGrade("ndfeb_n42", "NdFeB", "N42", 1.290, 1.320),
    MagnetGrade("ndfeb_n45", "NdFeB", "N45", 1.320, 1.370),
    MagnetGrade("ndfeb_n48", "NdFeB", "N48", 1.370, 1.420),
    MagnetGrade("ndfeb_n52", "NdFeB", "N52", 1.420, 1.470),
)

MAGNET_GRADES_BY_KEY = {grade.key: grade for grade in MAGNET_GRADES}


def magnet_grade(key: str) -> MagnetGrade:
    """Return a supported grade or raise a user-facing validation error."""

    try:
        return MAGNET_GRADES_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unsupported magnet grade: {key!r}") from error
