"""Direct search for the practically valid MPEX4 tractor-magnet design curve."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.models import CylinderMagnet
from tractor_magnet.stability_map import (
    StabilityPoint,
    evaluate_curvature_point,
    evaluate_stability_point,
    geometry_is_forbidden,
)


@dataclass(frozen=True, slots=True)
class DesignCurveResult:
    """Validated ``(a, b)`` points satisfying the planar curvature balance."""

    order: int
    a_values: NDArray[np.float64]
    b_values: NDArray[np.float64]
    equilibrium_x: NDArray[np.float64]
    resolution: float
    search_extent: float
    scout_curvature_evaluations: int
    mpex4_curvature_evaluations: int
    full_validity_evaluations: int

    @property
    def points(self) -> NDArray[np.float64]:
        """Return the valid curve as an ``(N, 2)`` array."""

        return np.column_stack((self.a_values, self.b_values))


def _axis(lower: float, upper: float, step: float) -> NDArray[np.float64]:
    start = math.ceil(lower / step - 1.0e-12) * step
    stop = math.ceil(upper / step - 1.0e-12) * step
    count = int(round((stop - start) / step))
    return start + np.arange(count + 1, dtype=float) * step


def _ceil_millimetres(value: float) -> float:
    return math.ceil(value / MILLIMETRE - 1.0e-12) * MILLIMETRE


def _interpolated_roots(
    coordinates: NDArray[np.float64],
    values: list[float],
) -> list[float]:
    roots: list[float] = []
    for left, right, value_left, value_right in zip(
        coordinates[:-1], coordinates[1:], values[:-1], values[1:], strict=True
    ):
        if not math.isfinite(value_left) or not math.isfinite(value_right):
            continue
        if value_left == 0.0:
            roots.append(float(left))
        elif np.signbit(value_left) != np.signbit(value_right):
            fraction = abs(value_left) / (abs(value_left) + abs(value_right))
            roots.append(float(left + fraction * (right - left)))
    if values and values[-1] == 0.0:
        roots.append(float(coordinates[-1]))
    return roots


def _point_is_practical(point: StabilityPoint) -> bool:
    return (
        point.has_local_minimum
        and point.passes_friction_threshold
        and point.passes_centering_tolerance
    )


def compute_design_curve(
    resolution: float,
    *,
    friction_threshold: float,
    centering_tolerance: float,
    follower: CylinderMagnet,
    attractor: CylinderMagnet,
    repulsor: CylinderMagnet,
    max_expansions: int = 6,
) -> DesignCurveResult:
    """Trace and validate ``Hyy - Hxx = 0`` without constructing a 2D MPEX4 map.

    An automatically expanding, curvature-only MPEX0 scout locates a generous
    candidate interval. MPEX4 roots are then solved directly in ``b`` at each
    requested ``a``. The expensive force-barrier, centering, and collision
    checks run only at those roots.
    """

    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a positive finite number")
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")

    b_min = max(0.0, math.floor(repulsor.radius / MILLIMETRE) * MILLIMETRE - MILLIMETRE)
    radius_sum = attractor.radius + repulsor.radius
    coarse_step = max(resolution, MILLIMETRE)
    extent = max(20.0 * MILLIMETRE, _ceil_millimetres(1.6 * radius_sum))
    guard = max(3.0 * MILLIMETRE, 2.0 * coarse_step)
    scout_curvature_evaluations = 0
    full_validity_evaluations = 0
    scout_roots: list[tuple[float, float]] = []
    scout_valid: list[tuple[float, float]] = []

    for _ in range(max_expansions):
        scout_roots = []
        scout_valid = []
        a_values = _axis(0.0, extent, coarse_step)
        b_values = _axis(b_min, extent, coarse_step)
        for a in a_values:
            differences: list[float] = []
            for b in b_values:
                if geometry_is_forbidden(
                    float(a),
                    float(b),
                    attractor_radius=attractor.radius,
                    repulsor_radius=repulsor.radius,
                ):
                    differences.append(float("nan"))
                    continue
                point = evaluate_curvature_point(
                    float(a),
                    float(b),
                    order=0,
                    axial_samples=121,
                    follower=follower,
                    attractor=attractor,
                    repulsor=repulsor,
                )
                scout_curvature_evaluations += 1
                differences.append(point.curvature_difference)
            roots = _interpolated_roots(b_values, differences)
            if not roots:
                continue
            selected_root = max(roots)
            scout_roots.append((float(a), selected_root))
            full_validity_evaluations += 1
            stability = evaluate_stability_point(
                float(a),
                selected_root,
                order=0,
                friction_threshold=friction_threshold,
                centering_tolerance=centering_tolerance,
                axial_samples=121,
                barrier_samples=101,
                follower=follower,
                attractor=attractor,
                repulsor=repulsor,
            )
            if _point_is_practical(stability):
                scout_valid.append((float(a), selected_root))
        if scout_valid:
            outermost = max(max(a, b) for a, b in scout_valid)
            if outermost <= extent - guard:
                break
        extent = _ceil_millimetres(1.4 * extent)
    else:
        raise ValueError("The automatic MPEX0 scout did not find a closed design curve.")

    if not scout_valid:
        raise ValueError("The MPEX0 scout found no practically valid curvature balance.")

    roots_array = np.asarray(scout_roots, dtype=float)
    valid_array = np.asarray(scout_valid, dtype=float)
    padding = max(2.0 * MILLIMETRE, 4.0 * resolution)
    fine_a = _axis(
        max(0.0, float(np.min(valid_array[:, 0])) - padding),
        min(extent, float(np.max(valid_array[:, 0])) + padding),
        resolution,
    )
    local_half_width = max(2.0 * MILLIMETRE, 4.0 * resolution)
    local_step = min(
        MILLIMETRE,
        max(0.5 * MILLIMETRE, 2.0 * resolution),
    )
    curvature_cache: dict[tuple[float, float], float] = {}
    mpex4_curvature_evaluations = 0

    def difference(a: float, b: float) -> float:
        nonlocal mpex4_curvature_evaluations
        key = (round(a, 12), round(b, 12))
        if key not in curvature_cache:
            if geometry_is_forbidden(
                a,
                b,
                attractor_radius=attractor.radius,
                repulsor_radius=repulsor.radius,
            ):
                curvature_cache[key] = float("nan")
            else:
                point = evaluate_curvature_point(
                    a,
                    b,
                    order=4,
                    follower=follower,
                    attractor=attractor,
                    repulsor=repulsor,
                )
                curvature_cache[key] = point.curvature_difference
                mpex4_curvature_evaluations += 1
        return curvature_cache[key]

    def exact_roots(a: float, samples: NDArray[np.float64]) -> list[float]:
        values = [difference(a, float(b)) for b in samples]
        roots: list[float] = []
        for left, right, value_left, value_right in zip(
            samples[:-1], samples[1:], values[:-1], values[1:], strict=True
        ):
            if not math.isfinite(value_left) or not math.isfinite(value_right):
                continue
            if value_left == 0.0:
                roots.append(float(left))
            elif np.signbit(value_left) != np.signbit(value_right):
                roots.append(
                    float(
                        brentq(
                            lambda b: difference(a, b),
                            float(left),
                            float(right),
                            xtol=1.0e-9,
                            rtol=4.0 * np.finfo(float).eps,
                        )
                    )
                )
        if values[-1] == 0.0:
            roots.append(float(samples[-1]))
        return sorted(set(roots), reverse=True)

    def validated_root(a: float) -> tuple[float, float] | None:
        nonlocal full_validity_evaluations
        predicted_b = float(np.interp(a, roots_array[:, 0], roots_array[:, 1]))
        local_samples = _axis(
            max(b_min, predicted_b - local_half_width),
            min(extent, predicted_b + local_half_width),
            local_step,
        )
        roots = exact_roots(a, local_samples)
        if not roots:
            roots = exact_roots(a, _axis(b_min, extent, coarse_step))
        for root in roots:
            full_validity_evaluations += 1
            stability = evaluate_stability_point(
                a,
                root,
                order=4,
                friction_threshold=friction_threshold,
                centering_tolerance=centering_tolerance,
                follower=follower,
                attractor=attractor,
                repulsor=repulsor,
            )
            if _point_is_practical(stability):
                return root, stability.equilibrium_x
        return None

    curve: dict[float, tuple[float, float]] = {}
    evaluated_a: set[float] = set()
    lower = float(fine_a[0])
    upper = float(fine_a[-1])
    for _ in range(max_expansions):
        for a_value in _axis(lower, upper, resolution):
            key = round(float(a_value), 12)
            if key in evaluated_a:
                continue
            evaluated_a.add(key)
            point = validated_root(float(a_value))
            if point is not None:
                curve[key] = point
        if not curve:
            break
        valid_a = sorted(curve)
        touches_lower = valid_a[0] <= lower + 0.5 * resolution
        touches_upper = valid_a[-1] >= upper - 0.5 * resolution
        if not touches_lower and not touches_upper:
            break
        new_lower = max(0.0, lower - padding) if touches_lower else lower
        new_upper = min(extent, upper + padding) if touches_upper else upper
        if new_lower == lower and new_upper == upper:
            break
        lower, upper = new_lower, new_upper

    if not curve:
        raise ValueError("MPEX4 found no practically valid design curve.")

    ordered_a = np.asarray(sorted(curve), dtype=float)
    ordered_b = np.asarray([curve[float(a)][0] for a in ordered_a], dtype=float)
    ordered_equilibrium = np.asarray([curve[float(a)][1] for a in ordered_a], dtype=float)

    return DesignCurveResult(
        order=4,
        a_values=ordered_a,
        b_values=ordered_b,
        equilibrium_x=ordered_equilibrium,
        resolution=resolution,
        search_extent=extent,
        scout_curvature_evaluations=scout_curvature_evaluations,
        mpex4_curvature_evaluations=mpex4_curvature_evaluations,
        full_validity_evaluations=full_validity_evaluations,
    )
