"""Stability-map calculation for the minimal tractor-magnet geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from tractor_magnet.configurations import article_minimal_configuration, minimal_configuration
from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.models import CylinderMagnet, PlanarConfiguration, Point2D
from tractor_magnet.multipole import MultipoleExpansion

_DX = (1, 0, 0)
_DY = (0, 1, 0)


@dataclass(frozen=True, slots=True)
class StabilityPoint:
    """Calculated quantities at one geometry point ``(a, b)``.

    Force barriers are positive magnitudes. Curvatures and their difference
    use SI units (N/m), as required by the definition of ``chi`` in Fig. 4(f).
    Missing quantities are represented by NaN.
    """

    equilibrium_x: float
    binding_energy: float
    negative_x_barrier: float
    negative_x_barrier_position: float
    positive_x_barrier: float
    positive_x_barrier_position: float
    transverse_barrier: float
    transverse_barrier_position: float
    transverse_friction_crossing: float
    transverse_half_span: float
    centering_tolerance: float
    negative_x_scan_limit: float
    positive_x_scan_limit: float
    transverse_scan_limit: float
    curvature_x: float
    curvature_y: float
    curvature_difference: float
    chi: float
    has_local_minimum: bool
    passes_friction_threshold: bool
    passes_centering_tolerance: bool


@dataclass(frozen=True, slots=True)
class CurvaturePoint:
    """Outer equilibrium and planar curvatures without force-barrier scans."""

    equilibrium_x: float
    curvature_x: float
    curvature_y: float
    curvature_difference: float
    has_local_minimum: bool


@dataclass(frozen=True, slots=True)
class StabilityMap:
    """Figure-4 quantities evaluated on a rectangular ``(a, b)`` grid."""

    order: int
    a_values: NDArray[np.float64]
    b_values: NDArray[np.float64]
    equilibrium_x: NDArray[np.float64]
    binding_energy: NDArray[np.float64]
    negative_x_barrier: NDArray[np.float64]
    negative_x_barrier_position: NDArray[np.float64]
    positive_x_barrier: NDArray[np.float64]
    positive_x_barrier_position: NDArray[np.float64]
    transverse_barrier: NDArray[np.float64]
    transverse_barrier_position: NDArray[np.float64]
    transverse_friction_crossing: NDArray[np.float64]
    transverse_half_span: NDArray[np.float64]
    centering_tolerance: NDArray[np.float64]
    negative_x_scan_limit: NDArray[np.float64]
    positive_x_scan_limit: NDArray[np.float64]
    transverse_scan_limit: NDArray[np.float64]
    curvature_x: NDArray[np.float64]
    curvature_y: NDArray[np.float64]
    curvature_difference: NDArray[np.float64]
    chi: NDArray[np.float64]
    has_local_minimum: NDArray[np.bool_]
    passes_friction_threshold: NDArray[np.bool_]
    passes_centering_tolerance: NDArray[np.bool_]
    geometry_forbidden: NDArray[np.bool_]

    @property
    def force_stable(self) -> NDArray[np.bool_]:
        """Return the region selected by force amplitudes alone."""

        return self.passes_friction_threshold & ~self.geometry_forbidden

    @property
    def stable(self) -> NDArray[np.bool_]:
        """Return the force-stable region with a sufficiently narrow friction dead zone."""

        return self.force_stable & self.passes_centering_tolerance


@dataclass(frozen=True, slots=True)
class _ForceBarriers:
    """Force extrema, their positions, and accessible scan limits."""

    negative_x: float
    negative_x_position: float
    positive_x: float
    positive_x_position: float
    transverse: float
    transverse_position: float
    transverse_friction_crossing: float
    negative_x_limit: float
    positive_x_limit: float
    transverse_limit: float


@dataclass(frozen=True, slots=True)
class SymmetryLineFit:
    """Linear fit ``b = slope * a + intercept`` to ``Hyy - Hxx = 0``."""

    order: int
    slope: float
    intercept: float
    rms_residual: float
    points: int
    a_min: float
    a_max: float


def geometry_is_forbidden(
    a: float,
    b: float,
    *,
    attractor_radius: float = 7.5 * MILLIMETRE,
    repulsor_radius: float = 5.0 * MILLIMETRE,
) -> bool:
    """Return whether the three fixed magnets overlap.

    The defaults reduce to Eq. (6) of the article. The first condition checks
    the two symmetric repulsors; the second checks either repulsor against the
    attractor.
    """

    return b < repulsor_radius or math.hypot(a, b) < attractor_radius + repulsor_radius


def _sign_change_roots(
    model: MultipoleExpansion,
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
) -> list[float]:
    roots: list[float] = []

    def axial_force(x_coordinate: float) -> float:
        return float(model.force((x_coordinate, 0.0))[0])

    for left, right, value_left, value_right in zip(
        coordinates[:-1], coordinates[1:], values[:-1], values[1:], strict=True
    ):
        if not np.isfinite(value_left) or not np.isfinite(value_right):
            continue
        if value_left == 0.0:
            roots.append(float(left))
        elif np.signbit(value_left) != np.signbit(value_right):
            roots.append(
                float(
                    brentq(
                        axial_force,
                        float(left),
                        float(right),
                        xtol=1.0e-12,
                        rtol=4.0 * np.finfo(float).eps,
                    )
                )
            )
    if values[-1] == 0.0:
        roots.append(float(coordinates[-1]))

    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1.0e-10:
            unique.append(root)
    return unique


def _minimal_model(
    a: float,
    b: float,
    *,
    order: int,
    follower: CylinderMagnet | None,
    attractor: CylinderMagnet | None,
    repulsor: CylinderMagnet | None,
) -> MultipoleExpansion:
    custom_magnets = (follower, attractor, repulsor)
    if any(magnet is not None for magnet in custom_magnets):
        if not all(magnet is not None for magnet in custom_magnets):
            raise ValueError("follower, attractor, and repulsor must be supplied together")
        configuration = minimal_configuration(
            a,
            b,
            follower=follower,
            attractor=attractor,
            repulsor=repulsor,
        )
    else:
        configuration = article_minimal_configuration(a, b)
    return MultipoleExpansion(configuration, order=order)


def _outer_minimum(
    model: MultipoleExpansion,
    axial_interval: tuple[float, float],
    axial_samples: int,
) -> tuple[list[float], tuple[float, NDArray[np.float64], float] | None]:
    x_coordinates = np.linspace(*axial_interval, axial_samples)
    axial_force = -model.derivative_array(x_coordinates, 0.0, _DX)
    roots = _sign_change_roots(model, x_coordinates, axial_force)
    minima: list[tuple[float, NDArray[np.float64], float]] = []
    for root in roots:
        hessian = model.hessian((root, 0.0))
        if np.all(np.linalg.eigvalsh(hessian) > 0.0):
            minima.append((root, hessian, model.energy((root, 0.0))))
    return roots, max(minima, key=lambda item: item[0]) if minima else None


def evaluate_curvature_point(
    a: float,
    b: float,
    *,
    order: int,
    axial_interval: tuple[float, float] = (0.1e-3, 80.0e-3),
    axial_samples: int = 241,
    follower: CylinderMagnet | None = None,
    attractor: CylinderMagnet | None = None,
    repulsor: CylinderMagnet | None = None,
) -> CurvaturePoint:
    """Evaluate only the outer equilibrium and its planar Hessian.

    This is the inexpensive primitive used to locate ``Hyy - Hxx = 0`` before
    the complete force-barrier and centering checks are applied.
    """

    if a < 0.0 or b <= 0.0:
        raise ValueError("a must be non-negative and b must be positive")
    if axial_samples < 3:
        raise ValueError("axial_samples must be at least three")
    model = _minimal_model(
        a,
        b,
        order=order,
        follower=follower,
        attractor=attractor,
        repulsor=repulsor,
    )
    _, minimum = _outer_minimum(model, axial_interval, axial_samples)
    if minimum is None:
        nan = float("nan")
        return CurvaturePoint(nan, nan, nan, nan, False)
    equilibrium_x, hessian, _ = minimum
    curvature_x = float(hessian[0, 0])
    curvature_y = float(hessian[1, 1])
    return CurvaturePoint(
        equilibrium_x=equilibrium_x,
        curvature_x=curvature_x,
        curvature_y=curvature_y,
        curvature_difference=curvature_y - curvature_x,
        has_local_minimum=True,
    )


def _force_barriers(
    model: MultipoleExpansion,
    equilibrium_x: float,
    preceding_root: float,
    *,
    x_max: float,
    transverse_half_span: float,
    friction_threshold: float,
    samples: int,
) -> _ForceBarriers:
    configuration = model.configuration
    origin = (equilibrium_x, 0.0)
    left_distance = _collision_free_ray_distance(
        configuration,
        origin,
        (-1.0, 0.0),
        equilibrium_x - preceding_root,
    )
    right_distance = _collision_free_ray_distance(
        configuration,
        origin,
        (1.0, 0.0),
        x_max - equilibrium_x,
    )
    transverse_distance = _collision_free_ray_distance(
        configuration,
        origin,
        (0.0, 1.0),
        transverse_half_span,
    )

    # Parameterize every path outwards from the equilibrium. This makes the
    # first extremum the one belonging to the local basin on both x sides.
    left_coordinates = equilibrium_x - np.linspace(0.0, left_distance, samples)
    right_coordinates = equilibrium_x + np.linspace(0.0, right_distance, samples)
    transverse_coordinates = np.linspace(0.0, transverse_distance, samples)

    left_force = -model.derivative_array(left_coordinates, 0.0, _DX)
    right_force = -model.derivative_array(right_coordinates, 0.0, _DX)
    transverse_force = -model.derivative_array(equilibrium_x, transverse_coordinates, _DY)

    def first_local_extreme(values: NDArray[np.float64], *, maximum: bool) -> tuple[int, float]:
        signed = -values if maximum else values
        for index in range(1, signed.size - 1):
            if signed[index] <= signed[index - 1] and signed[index] <= signed[index + 1]:
                return index, float(values[index])
        index = int(np.nanargmax(values) if maximum else np.nanargmin(values))
        return index, float(values[index])

    # The barriers belong to the local basin around the equilibrium. Farther
    # extrema can occur after the follower has already escaped that basin and
    # must not be mistaken for a stabilizing force barrier.
    positive_x_index, positive_x_force = first_local_extreme(left_force, maximum=True)
    negative_x_index, negative_x_force = first_local_extreme(right_force, maximum=False)
    transverse_index, transverse_force_minimum = first_local_extreme(
        transverse_force, maximum=False
    )
    positive_x = max(0.0, positive_x_force)
    negative_x = max(0.0, -negative_x_force)
    transverse = max(0.0, -transverse_force_minimum)
    transverse_position = float(transverse_coordinates[transverse_index])
    restoring_force = -transverse_force
    threshold_indices = np.flatnonzero(restoring_force >= friction_threshold)
    friction_crossing = float("nan")
    if threshold_indices.size:
        index = int(threshold_indices[0])
        if index == 0:
            friction_crossing = float(transverse_coordinates[0])
        else:
            force_left = restoring_force[index - 1]
            force_right = restoring_force[index]
            fraction = (friction_threshold - force_left) / (force_right - force_left)
            friction_crossing = float(
                transverse_coordinates[index - 1]
                + fraction * (transverse_coordinates[index] - transverse_coordinates[index - 1])
            )

    return _ForceBarriers(
        negative_x=negative_x,
        negative_x_position=float(right_coordinates[negative_x_index]),
        positive_x=positive_x,
        positive_x_position=float(left_coordinates[positive_x_index]),
        transverse=transverse,
        transverse_position=transverse_position,
        transverse_friction_crossing=friction_crossing,
        negative_x_limit=float(right_coordinates[-1]),
        positive_x_limit=float(left_coordinates[-1]),
        transverse_limit=float(transverse_coordinates[-1]),
    )


def _collision_free_ray_distance(
    configuration: PlanarConfiguration,
    origin: Point2D,
    direction: Point2D,
    requested_distance: float,
) -> float:
    """Return the accessible ray length before the follower first touches a source."""

    if requested_distance < 0.0:
        raise ValueError("requested_distance must be non-negative")
    direction_array = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction_array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and non-zero")
    direction_array /= norm
    origin_array = np.asarray(origin, dtype=float)
    accessible = float(requested_distance)

    for source in configuration.sources:
        displacement = origin_array - np.asarray(source.position, dtype=float)
        contact_distance = configuration.follower.radius + source.magnet.radius
        radial_clearance = float(np.dot(displacement, displacement) - contact_distance**2)
        if radial_clearance < 0.0:
            return 0.0

        projection = float(np.dot(displacement, direction_array))
        discriminant = projection**2 - radial_clearance
        if discriminant < 0.0:
            continue
        entry = -projection - math.sqrt(max(0.0, discriminant))
        if 0.0 <= entry < accessible:
            accessible = entry

    return accessible


def evaluate_stability_point(
    a: float,
    b: float,
    *,
    order: int,
    friction_threshold: float = 10.0e-3,
    axial_interval: tuple[float, float] = (0.1e-3, 80.0e-3),
    transverse_half_span: float | None = None,
    centering_tolerance: float | None = None,
    axial_samples: int = 241,
    barrier_samples: int = 201,
    follower: CylinderMagnet | None = None,
    attractor: CylinderMagnet | None = None,
    repulsor: CylinderMagnet | None = None,
) -> StabilityPoint:
    """Evaluate all six Fig. 4 quantities for one geometry.

    The rightmost axial stationary minimum is the exterior follower trap. The
    force barriers are the extrema surrounding that minimum, following the
    recipe in Sec. II of the article.
    """

    if a < 0.0 or b <= 0.0:
        raise ValueError("a must be non-negative and b must be positive")
    if axial_samples < 3 or barrier_samples < 3:
        raise ValueError("sample counts must be at least three")

    model = _minimal_model(
        a,
        b,
        order=order,
        follower=follower,
        attractor=attractor,
        repulsor=repulsor,
    )
    configuration = model.configuration
    repulsor_radius = configuration.sources[1].magnet.radius
    selected_half_span = (
        b + repulsor_radius if transverse_half_span is None else transverse_half_span
    )
    selected_centering_tolerance = (
        repulsor_radius / 4.0 if centering_tolerance is None else centering_tolerance
    )
    if selected_half_span <= 0.0:
        raise ValueError("transverse_half_span must be positive")
    if selected_centering_tolerance <= 0.0:
        raise ValueError("centering_tolerance must be positive")
    roots, minimum = _outer_minimum(model, axial_interval, axial_samples)
    if minimum is None:
        nan = float("nan")
        return StabilityPoint(
            equilibrium_x=nan,
            binding_energy=nan,
            negative_x_barrier=nan,
            negative_x_barrier_position=nan,
            positive_x_barrier=nan,
            positive_x_barrier_position=nan,
            transverse_barrier=nan,
            transverse_barrier_position=nan,
            transverse_friction_crossing=nan,
            transverse_half_span=selected_half_span,
            centering_tolerance=selected_centering_tolerance,
            negative_x_scan_limit=nan,
            positive_x_scan_limit=nan,
            transverse_scan_limit=nan,
            curvature_x=nan,
            curvature_y=nan,
            curvature_difference=nan,
            chi=nan,
            has_local_minimum=False,
            passes_friction_threshold=False,
            passes_centering_tolerance=False,
        )

    equilibrium_x, hessian, energy = minimum
    roots_to_left = [root for root in roots if root < equilibrium_x - 1.0e-10]
    preceding_root = max(roots_to_left, default=axial_interval[0])
    barriers = _force_barriers(
        model,
        equilibrium_x,
        preceding_root,
        x_max=axial_interval[1],
        transverse_half_span=selected_half_span,
        friction_threshold=friction_threshold,
        samples=barrier_samples,
    )
    curvature_x = float(hessian[0, 0])
    curvature_y = float(hessian[1, 1])
    difference = curvature_y - curvature_x
    chi = -math.inf if difference == 0.0 else math.log10(abs(difference))
    barrier_values = (barriers.negative_x, barriers.positive_x, barriers.transverse)
    passes_threshold = all(barrier >= friction_threshold for barrier in barrier_values)
    passes_centering = (
        math.isfinite(barriers.transverse_friction_crossing)
        and barriers.transverse_friction_crossing <= selected_centering_tolerance
    )

    return StabilityPoint(
        equilibrium_x=equilibrium_x,
        binding_energy=abs(float(energy)),
        negative_x_barrier=barriers.negative_x,
        negative_x_barrier_position=barriers.negative_x_position,
        positive_x_barrier=barriers.positive_x,
        positive_x_barrier_position=barriers.positive_x_position,
        transverse_barrier=barriers.transverse,
        transverse_barrier_position=barriers.transverse_position,
        transverse_friction_crossing=barriers.transverse_friction_crossing,
        transverse_half_span=selected_half_span,
        centering_tolerance=selected_centering_tolerance,
        negative_x_scan_limit=barriers.negative_x_limit,
        positive_x_scan_limit=barriers.positive_x_limit,
        transverse_scan_limit=barriers.transverse_limit,
        curvature_x=curvature_x,
        curvature_y=curvature_y,
        curvature_difference=difference,
        chi=chi,
        has_local_minimum=True,
        passes_friction_threshold=passes_threshold,
        passes_centering_tolerance=passes_centering,
    )


def compute_stability_map(
    a_values: ArrayLike,
    b_values: ArrayLike,
    *,
    order: int,
    friction_threshold: float = 10.0e-3,
    centering_tolerance: float | None = None,
    axial_samples: int = 241,
    barrier_samples: int = 201,
    follower: CylinderMagnet | None = None,
    attractor: CylinderMagnet | None = None,
    repulsor: CylinderMagnet | None = None,
    evaluation_mask: ArrayLike | None = None,
    progress: bool = False,
) -> StabilityMap:
    """Evaluate a Figure-4 stability map sequentially and deterministically."""

    a_array = np.asarray(a_values, dtype=float)
    b_array = np.asarray(b_values, dtype=float)
    if a_array.ndim != 1 or b_array.ndim != 1 or not a_array.size or not b_array.size:
        raise ValueError("a_values and b_values must be non-empty one-dimensional arrays")
    if np.any(a_array < 0.0) or np.any(b_array < 0.0):
        raise ValueError("geometry coordinates must be non-negative")

    shape = (b_array.size, a_array.size)
    selected = np.ones(shape, dtype=bool)
    if evaluation_mask is not None:
        selected = np.asarray(evaluation_mask, dtype=bool)
        if selected.shape != shape:
            raise ValueError("evaluation_mask must match the (b, a) grid shape")
    names = (
        "equilibrium_x",
        "binding_energy",
        "negative_x_barrier",
        "negative_x_barrier_position",
        "positive_x_barrier",
        "positive_x_barrier_position",
        "transverse_barrier",
        "transverse_barrier_position",
        "transverse_friction_crossing",
        "transverse_half_span",
        "centering_tolerance",
        "negative_x_scan_limit",
        "positive_x_scan_limit",
        "transverse_scan_limit",
        "curvature_x",
        "curvature_y",
        "curvature_difference",
        "chi",
    )
    fields = {name: np.full(shape, np.nan, dtype=float) for name in names}
    has_minimum = np.zeros(shape, dtype=bool)
    passes_threshold = np.zeros(shape, dtype=bool)
    passes_centering = np.zeros(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    attractor_radius = attractor.radius if attractor is not None else 7.5 * MILLIMETRE
    repulsor_radius = repulsor.radius if repulsor is not None else 5.0 * MILLIMETRE

    for row, b in enumerate(b_array):
        if progress:
            print(f"MPEX{order}: row {row + 1}/{b_array.size} (b={b / MILLIMETRE:.2f} mm)")
        for column, a in enumerate(a_array):
            forbidden[row, column] = geometry_is_forbidden(
                float(a),
                float(b),
                attractor_radius=attractor_radius,
                repulsor_radius=repulsor_radius,
            )
            if forbidden[row, column]:
                continue
            if not selected[row, column]:
                continue
            point = evaluate_stability_point(
                float(a),
                float(b),
                order=order,
                friction_threshold=friction_threshold,
                centering_tolerance=centering_tolerance,
                axial_samples=axial_samples,
                barrier_samples=barrier_samples,
                follower=follower,
                attractor=attractor,
                repulsor=repulsor,
            )
            for name in names:
                fields[name][row, column] = getattr(point, name)
            has_minimum[row, column] = point.has_local_minimum
            passes_threshold[row, column] = point.passes_friction_threshold
            passes_centering[row, column] = point.passes_centering_tolerance

    return StabilityMap(
        order=order,
        a_values=a_array,
        b_values=b_array,
        has_local_minimum=has_minimum,
        passes_friction_threshold=passes_threshold,
        passes_centering_tolerance=passes_centering,
        geometry_forbidden=forbidden,
        **fields,
    )


def symmetry_line_points(
    stability_map: StabilityMap,
    *,
    practical_only: bool = False,
) -> NDArray[np.float64]:
    """Interpolate sign changes of ``Hyy - Hxx`` along each b-column.

    With ``practical_only=True``, both samples bracketing a crossing must lie
    inside the complete practical-stability mask.
    """

    points: list[tuple[float, float]] = []
    eligible = (
        stability_map.stable
        if practical_only
        else stability_map.has_local_minimum & ~stability_map.geometry_forbidden
    )
    differences = stability_map.curvature_difference

    for column, a in enumerate(stability_map.a_values):
        candidates: list[float] = []
        for row in range(stability_map.b_values.size - 1):
            if not eligible[row, column] or not eligible[row + 1, column]:
                continue
            lower = differences[row, column]
            upper = differences[row + 1, column]
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            b_lower = stability_map.b_values[row]
            b_upper = stability_map.b_values[row + 1]
            if lower == 0.0:
                candidates.append(float(b_lower))
            elif np.signbit(lower) != np.signbit(upper):
                fraction = abs(lower) / (abs(lower) + abs(upper))
                candidates.append(float(b_lower + fraction * (b_upper - b_lower)))
        if candidates:
            points.append((float(a), max(candidates)))

    return np.asarray(points, dtype=float).reshape((-1, 2))


def fit_symmetry_line(
    stability_map: StabilityMap,
    *,
    a_interval: tuple[float, float] | None = (10.0e-3, 14.5e-3),
    practical_only: bool = False,
) -> SymmetryLineFit:
    """Fit the symmetry contour over a requested or automatically found interval."""

    points = symmetry_line_points(stability_map, practical_only=practical_only)
    if a_interval is None:
        selected = points
    else:
        selected = points[(points[:, 0] >= a_interval[0]) & (points[:, 0] <= a_interval[1])]
    if selected.shape[0] < 2:
        raise ValueError("fewer than two symmetry-line points in the fit interval")
    slope, intercept = np.polyfit(selected[:, 0], selected[:, 1], deg=1)
    residuals = selected[:, 1] - (slope * selected[:, 0] + intercept)
    return SymmetryLineFit(
        order=stability_map.order,
        slope=float(slope),
        intercept=float(intercept),
        rms_residual=float(np.sqrt(np.mean(residuals**2))),
        points=int(selected.shape[0]),
        a_min=float(np.min(selected[:, 0])),
        a_max=float(np.max(selected[:, 0])),
    )
