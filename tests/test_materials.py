import pytest

from tractor_magnet.materials import MAGNET_GRADES, magnet_grade


def test_supported_grades_cover_ferrite_and_ndfeb() -> None:
    assert {grade.material for grade in MAGNET_GRADES} == {"Ferrite", "NdFeB"}
    assert len(MAGNET_GRADES) == 8


@pytest.mark.parametrize(
    ("key", "nominal"),
    (
        ("ferrite_hf24_23", 0.358),
        ("ferrite_y30bh", 0.385),
        ("ferrite_y35", 0.405),
        ("ndfeb_n35", 1.190),
        ("ndfeb_n42", 1.305),
        ("ndfeb_n45", 1.345),
        ("ndfeb_n48", 1.395),
        ("ndfeb_n52", 1.445),
    ),
)
def test_nominal_remanence_is_midpoint_rounded_to_one_millitesla(
    key: str,
    nominal: float,
) -> None:
    grade = magnet_grade(key)

    assert grade.nominal_remanence == pytest.approx(nominal)
    assert grade.contains(nominal)
