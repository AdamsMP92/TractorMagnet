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


def test_web_calculation_returns_a_direct_validated_mpex4_curve() -> None:
    calculation = calculate_web(_inputs())

    curve = calculation.design_curve
    assert curve.order == 4
    assert curve.a_values.size > 2
    assert curve.a_values.shape == curve.b_values.shape == curve.equilibrium_x.shape
    assert math.isfinite(calculation.elapsed_s)
    assert curve.search_extent > curve.a_values[-1]
    assert curve.mpex4_curvature_evaluations > curve.a_values.size
    assert curve.full_validity_evaluations < curve.mpex4_curvature_evaluations
    assert all(
        not geometry_is_forbidden(
            float(a),
            float(b),
            attractor_radius=calculation.inputs.attractor.radius,
            repulsor_radius=calculation.inputs.repulsor.radius,
        )
        for a, b in curve.points
    )

    reference = np.load(
        Path(__file__).parents[1]
        / "results/figure_4_stability_maps/figure_4_theory_mpex4.npz"
    )
    stable = (
        reference["has_local_minimum"]
        & reference["passes_friction_threshold"]
        & reference["passes_centering_tolerance"]
        & ~reference["geometry_forbidden"]
    )
    differences = reference["curvature_difference"]
    reference_a = reference["a_values"]
    reference_b = reference["b_values"]
    reference_roots: dict[float, float] = {}
    for a in curve.a_values:
        column = int(np.argmin(abs(reference_a - a)))
        roots: list[float] = []
        for row in range(reference_b.size - 1):
            if not stable[row, column] or not stable[row + 1, column]:
                continue
            lower = differences[row, column]
            upper = differences[row + 1, column]
            if lower == 0.0:
                roots.append(float(reference_b[row]))
            elif np.signbit(lower) != np.signbit(upper):
                fraction = abs(lower) / (abs(lower) + abs(upper))
                roots.append(
                    float(reference_b[row] + fraction * (reference_b[row + 1] - reference_b[row]))
                )
        if roots:
            reference_roots[round(float(a), 12)] = max(roots)

    common = [
        abs(float(b) - reference_roots[round(float(a), 12)])
        for a, b in curve.points
        if round(float(a), 12) in reference_roots
    ]
    assert len(common) >= 3
    assert max(common) < 0.1 * MILLIMETRE


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
