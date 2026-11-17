import math

import numpy as np
import pytest

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.stability_map import (
    compute_stability_map,
    evaluate_stability_point,
    geometry_is_forbidden,
)


def test_geometry_exclusion_reproduces_equation_6() -> None:
    assert geometry_is_forbidden(9.0 * MILLIMETRE, 5.0 * MILLIMETRE)
    assert not geometry_is_forbidden(12.0 * MILLIMETRE, 5.0 * MILLIMETRE)
    assert geometry_is_forbidden(15.0 * MILLIMETRE, 4.9 * MILLIMETRE)


def test_mpex0_reference_geometry_has_expected_figure_4_quantities() -> None:
    point = evaluate_stability_point(
        14.0 * MILLIMETRE,
        12.0 * MILLIMETRE,
        order=0,
    )

    assert point.has_local_minimum
    assert point.passes_friction_threshold
    assert point.equilibrium_x / MILLIMETRE == pytest.approx(21.83870, abs=1.0e-4)
    assert point.positive_x_barrier * 1.0e3 == pytest.approx(125.16, rel=2.0e-3)
    assert point.negative_x_barrier * 1.0e3 == pytest.approx(40.43, rel=2.0e-3)
    assert point.transverse_barrier * 1.0e3 == pytest.approx(102.23, rel=3.0e-2)
    assert point.transverse_half_span / MILLIMETRE == pytest.approx(17.0)
    assert 0.0 < point.transverse_barrier_position < point.transverse_half_span
    assert point.transverse_friction_crossing / MILLIMETRE == pytest.approx(0.4681, rel=2e-3)
    assert point.centering_tolerance / MILLIMETRE == pytest.approx(1.25)
    assert point.passes_centering_tolerance
    assert point.chi == pytest.approx(math.log10(abs(point.curvature_y - point.curvature_x)))


def test_axial_barrier_scan_stops_at_first_follower_contact() -> None:
    point = evaluate_stability_point(
        12.0 * MILLIMETRE,
        8.0 * MILLIMETRE,
        order=0,
    )

    expected_contact_x = math.sqrt(12.5**2 - 8.0**2)
    assert point.positive_x_scan_limit / MILLIMETRE == pytest.approx(expected_contact_x)
    assert point.positive_x_barrier_position >= point.positive_x_scan_limit
    assert point.positive_x_barrier_position <= point.equilibrium_x
    assert point.negative_x_scan_limit > point.equilibrium_x


def test_soft_transverse_trap_fails_practical_centering_tolerance() -> None:
    point = evaluate_stability_point(
        12.0 * MILLIMETRE,
        8.0 * MILLIMETRE,
        order=4,
    )

    assert point.passes_friction_threshold
    assert point.transverse_friction_crossing / MILLIMETRE == pytest.approx(2.375, rel=2e-3)
    assert not point.passes_centering_tolerance


def test_evaluation_mask_skips_unselected_locations() -> None:
    a_values = np.asarray((12.0, 14.0)) * MILLIMETRE
    b_values = np.asarray((8.0, 12.0)) * MILLIMETRE
    selected = np.asarray(((False, False), (False, True)))

    result = compute_stability_map(
        a_values,
        b_values,
        order=0,
        axial_samples=121,
        barrier_samples=101,
        evaluation_mask=selected,
    )

    assert np.isnan(result.equilibrium_x[~selected]).all()
    assert np.isfinite(result.equilibrium_x[selected]).all()


def test_evaluation_mask_must_match_map_shape() -> None:
    with pytest.raises(ValueError, match="evaluation_mask"):
        compute_stability_map(
            np.asarray((12.0, 14.0)) * MILLIMETRE,
            np.asarray((8.0, 12.0)) * MILLIMETRE,
            order=0,
            evaluation_mask=np.ones((1, 2), dtype=bool),
        )
