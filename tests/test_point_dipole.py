import numpy as np
import pytest
from scipy.optimize import brentq

from tractor_magnet.configurations import article_minimal_configuration
from tractor_magnet.equilibrium import find_axial_equilibria
from tractor_magnet.point_dipole import (
    PointDipoleModel,
    equilibrium_moment_ratio,
    reduced_geometry_bounds,
    reduced_moment_ratio,
)


def test_article_configuration_has_moment_ratio_two_fifteenths() -> None:
    configuration = article_minimal_configuration()
    attractor, repulsor, _ = configuration.sources

    ratio = abs(repulsor.magnet.dipole_moment_z / attractor.magnet.dipole_moment_z)

    assert ratio == pytest.approx(2.0 / 15.0)
    assert ratio < 0.5


def test_appendix_a_equilibrium_relation_matches_force_root() -> None:
    a = 0.014
    b = 0.012
    required_ratio = 2.0 / 15.0
    expected_x = brentq(
        lambda x: equilibrium_moment_ratio(x, a, b) - required_ratio,
        0.010,
        0.015,
    )
    model = PointDipoleModel(article_minimal_configuration(a, b))

    equilibria = find_axial_equilibria(model, (0.005, 0.05))

    assert any(item.point[0] == pytest.approx(expected_x, abs=1.0e-10) for item in equilibria)


def test_reduced_bounds_map_back_to_moment_ratio_bounds() -> None:
    xi = 2.0
    beta_low, beta_mid, beta_up = reduced_geometry_bounds(xi)

    ratios = [reduced_moment_ratio(xi, beta) for beta in (beta_low, beta_mid, beta_up)]

    assert beta_low < beta_mid < beta_up
    assert np.all(np.asarray(ratios) > 0.0)
    assert np.all(np.asarray(ratios) < 0.5)
