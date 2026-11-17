import math
from pathlib import Path

import numpy as np
import pytest

from tractor_magnet.configurations import minimal_configuration
from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.stability_map import geometry_is_forbidden
from web_support import WebInputs, calculate_web


def _inputs(**overrides) -> WebInputs:
    values = {
        "height_mm": 10.0,
        "follower_radius_mm": 7.5,
        "follower_grade": "ndfeb_n42",
        "follower_remanence_t": 1.3,
        "attractor_radius_mm": 7.5,
        "attractor_grade": "ndfeb_n42",
        "attractor_remanence_t": 1.3,
        "repulsor_radius_mm": 5.0,
        "repulsor_grade": "ferrite_y30bh",
        "repulsor_remanence_t": 0.39,
        "resolution_mm": 1.0,
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
    assert inputs.follower.height == inputs.attractor.height == inputs.repulsor.height
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


def test_web_calculation_returns_an_adaptive_mpex4_map() -> None:
    calculation = calculate_web(_inputs())

    assert calculation.refined_map.order == 4
    assert calculation.refined_map.stable.any()
    assert calculation.line_fit is not None
    assert calculation.line_fit.a_min < calculation.line_fit.a_max
    assert calculation.line_fit.points >= 2
    assert math.isfinite(calculation.elapsed_s)
    assert calculation.evaluated_points < calculation.full_grid_points
    assert calculation.search_extent_m > calculation.refined_map.a_values[-1]
    assert calculation.refined_map.geometry_forbidden[:, 0].any()
    assert np.isnan(
        calculation.refined_map.chi[calculation.refined_map.geometry_forbidden]
    ).all()

    reference = np.load(
        Path(__file__).parents[1]
        / "results/figure_4_stability_maps/figure_4_theory_mpex4.npz"
    )
    reference_stable = (
        reference["has_local_minimum"]
        & reference["passes_friction_threshold"]
        & reference["passes_centering_tolerance"]
        & ~reference["geometry_forbidden"]
    )[::10, ::10]
    a_indices = np.rint(reference["a_values"][::10] / MILLIMETRE).astype(int)
    b_indices = np.rint(reference["b_values"][::10] / MILLIMETRE - 4.0).astype(int)
    adaptive_stable = calculation.refined_map.stable[np.ix_(b_indices, a_indices)]

    assert np.array_equal(adaptive_stable, reference_stable)


def test_web_validation_rejects_excessive_or_nonpositive_resolution() -> None:
    with pytest.raises(ValueError, match="between 0.1 mm and 2.0 mm"):
        _inputs(resolution_mm=0.05)
    with pytest.raises(ValueError, match="positive"):
        _inputs(resolution_mm=0.0)


def test_web_validation_rejects_remanence_outside_selected_grade() -> None:
    with pytest.raises(ValueError, match="Follower remanence"):
        _inputs(follower_grade="ndfeb_n35", follower_remanence_t=1.3)
    with pytest.raises(ValueError, match="Unsupported magnet grade"):
        _inputs(repulsor_grade="unknown")
