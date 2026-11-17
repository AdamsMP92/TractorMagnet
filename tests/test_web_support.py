import math

import numpy as np
import pytest

from tractor_magnet.configurations import minimal_configuration
from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.stability_map import geometry_is_forbidden
from web_support import WebInputs, calculate_web


def _inputs(**overrides) -> WebInputs:
    values = {
        "follower_radius_mm": 7.5,
        "follower_height_mm": 10.0,
        "follower_remanence_t": 1.3,
        "attractor_radius_mm": 7.5,
        "attractor_height_mm": 10.0,
        "attractor_remanence_t": 1.3,
        "repulsor_radius_mm": 5.0,
        "repulsor_height_mm": 10.0,
        "repulsor_remanence_t": 0.39,
        "a_max_mm": 19.0,
        "b_max_mm": 14.0,
        "grid_points": 11,
        "refinement_order": 0,
        "friction_threshold_mn": 10.0,
        "centering_tolerance_mm": 1.25,
    }
    values.update(overrides)
    return WebInputs.from_millimetres(**values)


def test_web_inputs_reproduce_article_moment_ratio() -> None:
    inputs = _inputs()

    assert inputs.moment_ratio == pytest.approx(2.0 / 15.0)
    assert inputs.passes_moment_condition
    assert inputs.follower.polarity == 1
    assert inputs.attractor.polarity == -1
    assert inputs.repulsor.polarity == 1
    assert inputs.a_min_m / MILLIMETRE == pytest.approx(0.0)
    assert inputs.b_min_m / MILLIMETRE == pytest.approx(4.0)


def test_map_minima_follow_geometry_and_rounding_rules() -> None:
    inputs = _inputs(attractor_radius_mm=6.2, repulsor_radius_mm=5.9)

    assert inputs.a_min_m / MILLIMETRE == pytest.approx(0.0)
    assert inputs.b_min_m / MILLIMETRE == pytest.approx(4.0)
    assert math.hypot(inputs.a_min_m, inputs.b_min_m) < (
        inputs.attractor_radius_m + inputs.repulsor_radius_m
    )


def test_custom_minimal_configuration_and_overlap_rule() -> None:
    inputs = _inputs(attractor_radius_mm=4.0, repulsor_radius_mm=3.0)
    configuration = minimal_configuration(
        8.0 * MILLIMETRE,
        5.0 * MILLIMETRE,
        follower=inputs.follower,
        attractor=inputs.attractor,
        repulsor=inputs.repulsor,
    )

    assert configuration.sources[0].position == (-8.0 * MILLIMETRE, 0.0)
    assert not geometry_is_forbidden(
        8.0 * MILLIMETRE,
        5.0 * MILLIMETRE,
        attractor_radius=4.0 * MILLIMETRE,
        repulsor_radius=3.0 * MILLIMETRE,
    )
    assert geometry_is_forbidden(
        8.0 * MILLIMETRE,
        2.5 * MILLIMETRE,
        attractor_radius=4.0 * MILLIMETRE,
        repulsor_radius=3.0 * MILLIMETRE,
    )


def test_small_web_calculation_reuses_dipole_result_for_order_zero() -> None:
    calculation = calculate_web(_inputs())

    assert calculation.refined_map is calculation.dipole_map
    assert calculation.dipole_map.chi.shape == (11, 11)
    assert math.isfinite(calculation.elapsed_s)
    assert np.array_equal(
        calculation.refined_map.geometry_forbidden,
        calculation.dipole_map.geometry_forbidden,
    )
    assert calculation.dipole_map.geometry_forbidden[:, 0].any()
    assert calculation.dipole_map.geometry_forbidden[0, :].all()
    assert np.isnan(calculation.dipole_map.chi[calculation.dipole_map.geometry_forbidden]).all()


def test_web_validation_rejects_unsupported_order_and_invalid_map_limit() -> None:
    with pytest.raises(ValueError, match="multipole order"):
        _inputs(refinement_order=6)
    with pytest.raises(ValueError, match="minimum"):
        _inputs(b_max_mm=4.0)
    with pytest.raises(ValueError, match="between 11 and 101"):
        _inputs(grid_points=102)
