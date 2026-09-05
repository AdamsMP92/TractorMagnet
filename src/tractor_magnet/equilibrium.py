"""One-dimensional root search and two-dimensional stability classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, root

from tractor_magnet.base import EnergyModel


@dataclass(frozen=True, slots=True)
class Equilibrium:
    """A stationary point and its local linear stability information."""

    point: tuple[float, float]
    energy: float
    force: tuple[float, float]
    hessian: tuple[tuple[float, float], tuple[float, float]]
    eigenvalues: tuple[float, float]
    stable: bool


def classify_equilibrium(
    model: EnergyModel,
    point: tuple[float, float],
    *,
    curvature_tolerance: float = 0.0,
) -> Equilibrium:
    """Classify a stationary point using the eigenvalues of the Hessian."""

    force = model.force(point)
    hessian = model.hessian(point)
    eigenvalues = np.linalg.eigvalsh(hessian)
    return Equilibrium(
        point=point,
        energy=model.energy(point),
        force=(float(force[0]), float(force[1])),
        hessian=(
            (float(hessian[0, 0]), float(hessian[0, 1])),
            (float(hessian[1, 0]), float(hessian[1, 1])),
        ),
        eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        stable=bool(np.all(eigenvalues > curvature_tolerance)),
    )


def _unique_sorted(values: list[float], tolerance: float) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def find_axial_equilibria(
    model: EnergyModel,
    x_interval: tuple[float, float],
    *,
    samples: int = 400,
    root_tolerance: float = 1.0e-12,
) -> list[Equilibrium]:
    """Find stationary points on the mirror-symmetry axis y=0.

    The interval is sampled for sign changes of F_x and each bracket is
    refined with Brent's method. Stability is then checked in both planar
    directions.
    """

    x_min, x_max = x_interval
    if not x_min < x_max:
        raise ValueError("x_interval must be strictly increasing")
    if samples < 2:
        raise ValueError("samples must be at least two")

    grid = np.linspace(x_min, x_max, samples)

    def axial_force(x_coordinate: float) -> float:
        return float(model.force((float(x_coordinate), 0.0))[0])

    force_values: NDArray[np.float64] = np.asarray(
        [axial_force(float(x_coordinate)) for x_coordinate in grid]
    )
    roots: list[float] = []

    for left, right, force_left, force_right in zip(
        grid[:-1], grid[1:], force_values[:-1], force_values[1:], strict=True
    ):
        if not np.isfinite(force_left) or not np.isfinite(force_right):
            continue
        if force_left == 0.0:
            roots.append(float(left))
        elif np.signbit(force_left) != np.signbit(force_right):
            roots.append(
                float(
                    brentq(
                        axial_force,
                        float(left),
                        float(right),
                        xtol=root_tolerance,
                        rtol=4.0 * np.finfo(float).eps,
                    )
                )
            )
    if force_values[-1] == 0.0:
        roots.append(float(grid[-1]))

    return [
        classify_equilibrium(model, (root, 0.0))
        for root in _unique_sorted(roots, 10.0 * root_tolerance)
    ]


def find_planar_equilibria(
    model: EnergyModel,
    x_interval: tuple[float, float],
    y_interval: tuple[float, float],
    *,
    seeds_per_axis: int = 7,
    position_tolerance: float = 1.0e-8,
    force_tolerance: float = 1.0e-9,
) -> list[Equilibrium]:
    """Find distinct stationary points from a regular grid of initial seeds."""

    x_min, x_max = x_interval
    y_min, y_max = y_interval
    if not x_min < x_max or not y_min < y_max:
        raise ValueError("search intervals must be strictly increasing")
    if seeds_per_axis < 2:
        raise ValueError("seeds_per_axis must be at least two")

    roots: list[NDArray[np.float64]] = []
    x_seeds = np.linspace(x_min, x_max, seeds_per_axis)
    y_seeds = np.linspace(y_min, y_max, seeds_per_axis)

    def force(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return model.force((float(coordinates[0]), float(coordinates[1])))

    def force_jacobian(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return -model.hessian((float(coordinates[0]), float(coordinates[1])))

    for x_seed in x_seeds:
        for y_seed in y_seeds:
            solution = root(
                force,
                np.asarray((x_seed, y_seed)),
                jac=force_jacobian,
            )
            coordinates = np.asarray(solution.x, dtype=float)
            residual = np.linalg.norm(force(coordinates))
            inside_domain = (
                x_min - position_tolerance <= coordinates[0] <= x_max + position_tolerance
                and y_min - position_tolerance <= coordinates[1] <= y_max + position_tolerance
            )
            is_new = all(
                np.linalg.norm(coordinates - existing) > position_tolerance for existing in roots
            )
            if solution.success and residual <= force_tolerance and inside_domain and is_new:
                roots.append(coordinates)

    return [
        classify_equilibrium(model, (float(coordinates[0]), float(coordinates[1])))
        for coordinates in sorted(roots, key=lambda value: (value[0], value[1]))
    ]
