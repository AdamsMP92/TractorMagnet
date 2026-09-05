import math

import pytest

from tractor_magnet.models import CylinderMagnet
from tractor_magnet.moments import cylinder_geometric_moment, magnetization_moment_z


def test_cylinder_zero_order_moment_is_its_volume() -> None:
    magnet = CylinderMagnet(radius=0.0075, height=0.01, remanence=1.3)

    assert cylinder_geometric_moment(magnet, (0, 0, 0)) == pytest.approx(magnet.volume)
    assert magnetization_moment_z(magnet, (0, 0, 0)) == pytest.approx(magnet.dipole_moment_z)


def test_centred_cylinder_odd_moments_vanish() -> None:
    magnet = CylinderMagnet(radius=0.005, height=0.01, remanence=0.39)

    assert cylinder_geometric_moment(magnet, (1, 0, 0)) == 0.0
    assert cylinder_geometric_moment(magnet, (0, 3, 0)) == 0.0
    assert cylinder_geometric_moment(magnet, (0, 0, 5)) == 0.0


def test_second_moments_match_elementary_cylinder_integrals() -> None:
    magnet = CylinderMagnet(radius=0.005, height=0.01, remanence=0.39)

    transverse = math.pi * magnet.radius**4 * magnet.height / 4.0
    axial = magnet.volume * magnet.height**2 / 12.0

    assert cylinder_geometric_moment(magnet, (2, 0, 0)) == pytest.approx(transverse)
    assert cylinder_geometric_moment(magnet, (0, 2, 0)) == pytest.approx(transverse)
    assert cylinder_geometric_moment(magnet, (0, 0, 2)) == pytest.approx(axial)


@pytest.mark.parametrize(
    ("field", "value"),
    [("radius", 0.0), ("height", -1.0), ("remanence", float("nan"))],
)
def test_invalid_cylinder_parameters_are_rejected(field: str, value: float) -> None:
    parameters = {"radius": 0.005, "height": 0.01, "remanence": 0.39}
    parameters[field] = value

    with pytest.raises(ValueError):
        CylinderMagnet(**parameters)
