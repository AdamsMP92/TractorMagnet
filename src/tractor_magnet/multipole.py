"""Cartesian multipole expansion of the magnetostatic interaction energy."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tractor_magnet.constants import MU0_OVER_4PI
from tractor_magnet.kernels import (
    inverse_distance_derivative,
    inverse_distance_derivative_function,
)
from tractor_magnet.models import PlacedMagnet, PlanarConfiguration, Point2D
from tractor_magnet.moments import magnetization_moment_z
from tractor_magnet.multi_index import (
    MultiIndex,
    add_indices,
    cartesian_indices,
    multi_factorial,
    total_order,
)

_ZERO: MultiIndex = (0, 0, 0)
_DX: MultiIndex = (1, 0, 0)
_DY: MultiIndex = (0, 1, 0)
_DXX: MultiIndex = (2, 0, 0)
_DXY: MultiIndex = (1, 1, 0)
_DYY: MultiIndex = (0, 2, 0)


class MultipoleExpansion:
    """MPEX-N model with moments of each cylinder retained through order N.

    The source Hertz vector is expanded using Eq. (B2), and the follower
    correlation integral is expanded using Eq. (B13). Cylindrical symmetry
    removes every moment with an odd Cartesian component.
    """

    def __init__(self, configuration: PlanarConfiguration, order: int = 4) -> None:
        if not isinstance(order, int) or order < 0 or order % 2:
            raise ValueError("order must be a non-negative even integer")
        self.configuration = configuration
        self.order = order
        self._interaction_terms = tuple(
            (source, self._build_terms(source)) for source in configuration.sources
        )

    def _build_terms(self, source: PlacedMagnet) -> dict[MultiIndex, float]:
        terms: defaultdict[MultiIndex, float] = defaultdict(float)
        indices = tuple(cartesian_indices(self.order, even_only=True))
        follower = self.configuration.follower

        for follower_index in indices:
            follower_moment = magnetization_moment_z(follower, follower_index)
            follower_weight = follower_moment / multi_factorial(follower_index)
            for source_index in indices:
                source_moment = magnetization_moment_z(source.magnet, source_index)
                source_weight = (
                    (-1.0) ** total_order(source_index)
                    * source_moment
                    / multi_factorial(source_index)
                )
                base_index = add_indices(follower_index, source_index)
                coefficient = MU0_OVER_4PI * follower_weight * source_weight
                terms[add_indices(base_index, _DXX)] += coefficient
                terms[add_indices(base_index, _DYY)] += coefficient

        return {index: coefficient for index, coefficient in terms.items() if coefficient != 0.0}

    def derivative(self, point: Point2D, derivative: MultiIndex = _ZERO) -> float:
        """Evaluate a Cartesian derivative of the potential energy."""

        if min(derivative) < 0:
            raise ValueError("derivative orders must be non-negative")

        total = 0.0
        for source, terms in self._interaction_terms:
            displacement = (
                point[0] - source.position[0],
                point[1] - source.position[1],
                0.0,
            )
            for kernel_index, coefficient in terms.items():
                differentiated_index = add_indices(kernel_index, derivative)
                total += coefficient * inverse_distance_derivative(
                    differentiated_index, displacement
                )
        return total

    def energy(self, point: Point2D) -> float:
        """Return potential energy in joules."""

        return self.derivative(point)

    def derivative_array(
        self,
        x_coordinates: ArrayLike,
        y_coordinates: ArrayLike,
        derivative: MultiIndex = _ZERO,
    ) -> NDArray[np.float64]:
        """Evaluate an energy derivative on broadcast-compatible coordinates."""

        if min(derivative) < 0:
            raise ValueError("derivative orders must be non-negative")
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x_coordinates, dtype=float), np.asarray(y_coordinates, dtype=float)
        )
        total = np.zeros(x_values.shape, dtype=float)

        for source, terms in self._interaction_terms:
            dx = x_values - source.position[0]
            dy = y_values - source.position[1]
            if np.any((dx == 0.0) & (dy == 0.0)):
                raise ValueError("the multipole kernel is singular at a source centre")
            for kernel_index, coefficient in terms.items():
                differentiated_index = add_indices(kernel_index, derivative)
                kernel = inverse_distance_derivative_function(differentiated_index)
                total += coefficient * np.asarray(kernel(dx, dy, 0.0), dtype=float)
        return total

    def energy_grid(
        self, x_coordinates: ArrayLike, y_coordinates: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return mesh coordinates and potential energy on a rectilinear grid."""

        x_values = np.asarray(x_coordinates, dtype=float)
        y_values = np.asarray(y_coordinates, dtype=float)
        if x_values.ndim != 1 or y_values.ndim != 1 or not x_values.size or not y_values.size:
            raise ValueError("grid coordinates must be non-empty one-dimensional arrays")
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        return x_grid, y_grid, self.derivative_array(x_grid, y_grid)

    def force(self, point: Point2D) -> NDArray[np.float64]:
        """Return -grad(U) in newtons."""

        return -np.asarray([self.derivative(point, _DX), self.derivative(point, _DY)], dtype=float)

    def hessian(self, point: Point2D) -> NDArray[np.float64]:
        """Return the planar Hessian of U in joules per square metre."""

        xx = self.derivative(point, _DXX)
        xy = self.derivative(point, _DXY)
        yy = self.derivative(point, _DYY)
        return np.asarray(((xx, xy), (xy, yy)), dtype=float)
