import numpy as np
import pytest

from tractor_magnet.configurations import article_minimal_configuration
from tractor_magnet.multipole import MultipoleExpansion
from tractor_magnet.point_dipole import PointDipoleModel


@pytest.mark.parametrize("point", [(0.018, 0.0), (0.025, 0.003), (0.04, -0.002)])
def test_mpex0_is_the_point_dipole_model(point: tuple[float, float]) -> None:
    configuration = article_minimal_configuration()
    dipoles = PointDipoleModel(configuration)
    multipoles = MultipoleExpansion(configuration, order=0)

    assert multipoles.energy(point) == pytest.approx(dipoles.energy(point), rel=2.0e-13)
    assert multipoles.force(point) == pytest.approx(dipoles.force(point), rel=2.0e-13)
    assert multipoles.hessian(point) == pytest.approx(dipoles.hessian(point), rel=2.0e-13)


def test_mpex2_respects_mirror_symmetry() -> None:
    model = MultipoleExpansion(article_minimal_configuration(), order=2)
    x_coordinate = 0.025
    y_coordinate = 0.002

    assert model.energy((x_coordinate, y_coordinate)) == pytest.approx(
        model.energy((x_coordinate, -y_coordinate)), rel=1.0e-13
    )
    assert model.force((x_coordinate, 0.0))[1] == pytest.approx(0.0, abs=1.0e-15)
    assert model.hessian((x_coordinate, 0.0))[0, 1] == pytest.approx(0.0, abs=1.0e-12)


def test_higher_order_changes_short_range_force() -> None:
    configuration = article_minimal_configuration()
    point = (0.018, 0.0)
    mpex0_force = MultipoleExpansion(configuration, order=0).force(point)
    mpex2_force = MultipoleExpansion(configuration, order=2).force(point)

    assert not np.isclose(mpex0_force[0], mpex2_force[0], rtol=1.0e-3, atol=0.0)


def test_fourth_order_has_converged_towards_sixth_order() -> None:
    configuration = article_minimal_configuration()
    point = (0.017, 0.0)
    forces = {
        order: MultipoleExpansion(configuration, order=order).force(point)[0] for order in (2, 4, 6)
    }

    assert abs(forces[4] - forces[6]) < abs(forces[2] - forces[6])
