from pathlib import Path

import pytest

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.experimental import (
    geometry_from_stl_name,
    load_figure_4_experimental_points,
)


def test_geometry_from_stl_name_decodes_decimal_coordinates() -> None:
    a, b = geometry_from_stl_name("17_Tractor_13_0__10_2.stl")

    assert a / MILLIMETRE == pytest.approx(13.0)
    assert b / MILLIMETRE == pytest.approx(10.2)


def test_geometry_from_stl_name_rejects_unrelated_names() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        geometry_from_stl_name("sample.stl")


def test_figure_4_table_is_verified_against_79_stl_files() -> None:
    repository = Path(__file__).parents[1]
    points = load_figure_4_experimental_points(
        repository / "data/figure_4_experimental_points.csv",
        repository / "STL_Samples",
    )

    assert len(points) == 82
    assert sum(point.source == "stl" for point in points) == 79
    assert sum(point.source == "supplementary_table" for point in points) == 3
    assert {point.outcome for point in points} == {
        "possible",
        "partial",
        "not_possible",
        "transverse_unstable",
    }
