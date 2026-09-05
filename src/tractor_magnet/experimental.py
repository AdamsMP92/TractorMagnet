"""Experimental Figure-4 geometries derived from the archived STL samples."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from tractor_magnet.constants import MILLIMETRE

_STL_NAME = re.compile(
    r"(?P<index>\d+)_Tractor_(?P<a_integer>\d+)_(?P<a_decimal>\d+)__"
    r"(?P<b_integer>\d+)_(?P<b_decimal>\d+)\.stl"
)


@dataclass(frozen=True, slots=True)
class ExperimentalPoint:
    """One tested geometry and its outcome in the original Figure-4 data."""

    a: float
    b: float
    outcome: str
    source: str
    sample_set: str | None = None
    stl_filename: str | None = None


def geometry_from_stl_name(filename: str) -> tuple[float, float]:
    """Decode ``a`` and ``b`` in metres from a sample STL filename."""

    match = _STL_NAME.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"unsupported tractor-magnet STL filename: {filename!r}")

    def coordinate(integer: str, decimal: str) -> float:
        return float(f"{integer}.{decimal}") * MILLIMETRE

    return (
        coordinate(match["a_integer"], match["a_decimal"]),
        coordinate(match["b_integer"], match["b_decimal"]),
    )


def load_figure_4_experimental_points(
    table_path: Path,
    stl_root: Path,
) -> tuple[ExperimentalPoint, ...]:
    """Load outcome labels and verify every STL-backed coordinate from its filename."""

    points: list[ExperimentalPoint] = []
    with table_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source = row["source"]
            a = float(row["a_mm"]) * MILLIMETRE
            b = float(row["b_mm"]) * MILLIMETRE
            sample_set = row["sample_set"] or None
            stl_filename = row["stl_filename"] or None

            if source == "stl":
                if sample_set is None or stl_filename is None:
                    raise ValueError("STL rows require sample_set and stl_filename")
                stl_path = stl_root / sample_set / stl_filename
                if not stl_path.is_file():
                    raise FileNotFoundError(stl_path)
                stl_a, stl_b = geometry_from_stl_name(stl_filename)
                if abs(stl_a - a) > 1.0e-12 or abs(stl_b - b) > 1.0e-12:
                    raise ValueError(f"table coordinates disagree with {stl_path}")
            elif source != "supplementary_table":
                raise ValueError(f"unsupported experimental point source: {source!r}")

            points.append(
                ExperimentalPoint(
                    a=a,
                    b=b,
                    outcome=row["outcome"],
                    source=source,
                    sample_set=sample_set,
                    stl_filename=stl_filename,
                )
            )

    if len({(point.a, point.b) for point in points}) != len(points):
        raise ValueError("experimental Figure-4 coordinates must be unique")
    return tuple(points)
