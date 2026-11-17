"""Adaptive search for the narrow practical tractor-magnet stability region."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.models import CylinderMagnet
from tractor_magnet.stability_map import StabilityMap, compute_stability_map


@dataclass(frozen=True, slots=True)
class AdaptiveMapResult:
    """Final MPEX4 map and diagnostics from the adaptive search."""

    stability_map: StabilityMap
    search_extent: float
    resolution: float
    evaluated_points: int
    full_grid_points: int
    scout_points: int


def _axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(math.ceil((upper - lower) / step))
    return np.linspace(lower, lower + count * step, count + 1)


def _ceil_millimetres(value: float) -> float:
    return math.ceil(value / MILLIMETRE - 1.0e-12) * MILLIMETRE


def _scout_seed(
    stability_map: StabilityMap,
    *,
    centering_tolerance: float,
    centering_margin: float,
) -> np.ndarray:
    return (
        stability_map.force_stable
        & np.isfinite(stability_map.transverse_friction_crossing)
        & (
            stability_map.transverse_friction_crossing
            <= centering_tolerance + centering_margin
        )
    )


def _distance_mask(
    seed: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    *,
    padding: float,
) -> np.ndarray:
    rows, columns = np.where(seed)
    if not rows.size:
        return np.zeros((target_b.size, target_a.size), dtype=bool)
    tree = cKDTree(np.column_stack((seed_a[columns], seed_b[rows])))
    a_grid, b_grid = np.meshgrid(target_a, target_b)
    points = np.column_stack((a_grid.ravel(), b_grid.ravel()))
    distance, _ = tree.query(points)
    return distance.reshape(a_grid.shape) <= padding


def _crop_to_stable_region(
    stability_map: StabilityMap,
    *,
    margin: float,
) -> StabilityMap:
    rows, columns = np.where(stability_map.stable)
    if not rows.size:
        raise ValueError("MPEX4 found no practical stability region.")
    upper = max(
        float(np.max(stability_map.a_values[columns])),
        float(np.max(stability_map.b_values[rows])),
    ) + margin
    column_stop = min(
        stability_map.a_values.size,
        int(np.searchsorted(stability_map.a_values, upper, side="right")) + 1,
    )
    row_stop = min(
        stability_map.b_values.size,
        int(np.searchsorted(stability_map.b_values, upper, side="right")) + 1,
    )
    slices = (slice(0, row_stop), slice(0, column_stop))
    values: dict[str, object] = {}
    for field in fields(StabilityMap):
        value = getattr(stability_map, field.name)
        if field.name == "order":
            values[field.name] = value
        elif field.name == "a_values":
            values[field.name] = value[:column_stop]
        elif field.name == "b_values":
            values[field.name] = value[:row_stop]
        else:
            values[field.name] = value[slices]
    return StabilityMap(**values)


def compute_adaptive_stability_map(
    resolution: float,
    *,
    friction_threshold: float,
    centering_tolerance: float,
    follower: CylinderMagnet,
    attractor: CylinderMagnet,
    repulsor: CylinderMagnet,
    max_expansions: int = 6,
) -> AdaptiveMapResult:
    """Find and refine the practical region without scanning a large MPEX4 rectangle.

    A cheap, automatically expanding MPEX0 scout identifies a relaxed candidate
    region.  A fine MPEX0 pass tightens it, then only that band is evaluated at
    MPEX4.  If a refined stable point touches the candidate boundary, the band
    expands automatically before the final crop.
    """

    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a positive finite number")
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")

    b_min = max(0.0, math.floor(repulsor.radius / MILLIMETRE) * MILLIMETRE - MILLIMETRE)
    radius_sum = attractor.radius + repulsor.radius
    coarse_step = max(resolution, MILLIMETRE)
    initial_extent = max(
        20.0 * MILLIMETRE,
        _ceil_millimetres(1.6 * radius_sum),
    )
    extent = initial_extent
    guard = max(3.0 * MILLIMETRE, 2.0 * coarse_step)
    centering_margin = max(MILLIMETRE, 2.0 * resolution)
    scout: StabilityMap | None = None
    seed: np.ndarray | None = None

    for _ in range(max_expansions):
        scout = compute_stability_map(
            _axis(0.0, extent, coarse_step),
            _axis(b_min, extent, coarse_step),
            order=0,
            friction_threshold=friction_threshold,
            centering_tolerance=centering_tolerance,
            axial_samples=121,
            barrier_samples=101,
            follower=follower,
            attractor=attractor,
            repulsor=repulsor,
        )
        seed = _scout_seed(
            scout,
            centering_tolerance=centering_tolerance,
            centering_margin=centering_margin,
        )
        rows, columns = np.where(seed)
        if rows.size:
            outermost = max(
                float(np.max(scout.a_values[columns])),
                float(np.max(scout.b_values[rows])),
            )
            if outermost <= extent - guard:
                break
        extent = _ceil_millimetres(1.4 * extent)
    else:
        raise ValueError("The automatic MPEX0 search did not find a closed candidate region.")

    assert scout is not None and seed is not None
    a_values = _axis(0.0, extent, resolution)
    b_values = _axis(b_min, extent, resolution)
    broad_padding = max(3.0 * MILLIMETRE, 2.0 * coarse_step)
    fine_scout_mask = _distance_mask(
        seed,
        scout.a_values,
        scout.b_values,
        a_values,
        b_values,
        padding=broad_padding,
    )
    fine_scout = compute_stability_map(
        a_values,
        b_values,
        order=0,
        friction_threshold=friction_threshold,
        centering_tolerance=centering_tolerance,
        axial_samples=121,
        barrier_samples=101,
        follower=follower,
        attractor=attractor,
        repulsor=repulsor,
        evaluation_mask=fine_scout_mask,
    )
    fine_seed = _scout_seed(
        fine_scout,
        centering_tolerance=centering_tolerance,
        centering_margin=centering_margin,
    )
    refinement_padding = max(MILLIMETRE, 2.0 * resolution)
    candidate = _distance_mask(
        fine_seed,
        a_values,
        b_values,
        a_values,
        b_values,
        padding=refinement_padding,
    )
    if not np.any(candidate):
        raise ValueError("The MPEX0 scout found no refinement candidates.")

    refined: StabilityMap | None = None
    dilation_steps = max(1, int(math.ceil(refinement_padding / resolution)))
    for _ in range(3):
        refined = compute_stability_map(
            a_values,
            b_values,
            order=4,
            friction_threshold=friction_threshold,
            centering_tolerance=centering_tolerance,
            follower=follower,
            attractor=attractor,
            repulsor=repulsor,
            evaluation_mask=candidate,
        )
        candidate_boundary = candidate & ~binary_erosion(candidate, border_value=0)
        if not np.any(refined.stable & candidate_boundary):
            break
        candidate = binary_dilation(candidate, iterations=dilation_steps)
    else:
        raise ValueError("The adaptive MPEX4 region did not close after refinement.")

    assert refined is not None
    cropped = _crop_to_stable_region(
        refined,
        margin=max(2.0 * MILLIMETRE, 4.0 * resolution),
    )
    evaluated = candidate & ~refined.geometry_forbidden
    scout_evaluated = int(np.count_nonzero(~scout.geometry_forbidden)) + int(
        np.count_nonzero(fine_scout_mask & ~fine_scout.geometry_forbidden)
    )
    return AdaptiveMapResult(
        stability_map=cropped,
        search_extent=extent,
        resolution=resolution,
        evaluated_points=int(np.count_nonzero(evaluated)),
        full_grid_points=int(candidate.size),
        scout_points=scout_evaluated,
    )
