"""Numerical datasets used to reproduce figures from the article."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tractor_magnet.configurations import (
    article_minimal_configuration,
    supplementary_configurations,
)
from tractor_magnet.constants import CENTIMETRE, MILLIJOULE
from tractor_magnet.equilibrium import Equilibrium, find_planar_equilibria
from tractor_magnet.models import PlanarConfiguration
from tractor_magnet.multipole import MultipoleExpansion


@dataclass(frozen=True, slots=True)
class EnergyLandscapeCase:
    """Plot metadata for one tabulated supplementary configuration."""

    slug: str
    title: str
    configuration: PlanarConfiguration
    expected_equilibrium_x: float
    energy_limits: tuple[float, float]
    table: str


@dataclass(frozen=True, slots=True)
class EnergyLandscape:
    """Computed energy grid and equilibria for one supplementary case."""

    case: EnergyLandscapeCase
    order: int
    x_grid: NDArray[np.float64]
    y_grid: NDArray[np.float64]
    energy: NDArray[np.float64]
    equilibria: tuple[Equilibrium, ...]

    @property
    def stable_equilibrium(self) -> Equilibrium:
        """Return the stable root closest to the position reported in the figure."""

        stable = tuple(equilibrium for equilibrium in self.equilibria if equilibrium.stable)
        if not stable:
            raise RuntimeError(f"no stable equilibrium found for {self.case.slug}")
        return min(
            stable,
            key=lambda equilibrium: abs(equilibrium.point[0] - self.case.expected_equilibrium_x),
        )


def supplementary_energy_landscape_cases() -> tuple[EnergyLandscapeCase, ...]:
    """Return plot specifications read from Tables I-V and Figs. 3-7."""

    configurations = supplementary_configurations()
    specifications = (
        ("minimal_example_1", "Minimal example I", 1.89, (-1.75, -1.25), "Table I"),
        ("minimal_example_2", "Minimal example II", 2.10, (-2.50, -2.00), "Table II"),
        ("minimal_example_3", "Minimal example III", 1.88, (-3.20, -2.70), "Table III"),
        ("advanced_example_1", "Advanced example I", 2.32, (-3.10, -2.60), "Table IV"),
        ("advanced_example_2", "Advanced example II", 1.98, (-6.20, -5.70), "Table V"),
    )
    return tuple(
        EnergyLandscapeCase(
            slug=slug,
            title=title,
            configuration=configurations[slug],
            expected_equilibrium_x=x_cm * CENTIMETRE,
            energy_limits=(limits_mj[0] * MILLIJOULE, limits_mj[1] * MILLIJOULE),
            table=table,
        )
        for slug, title, x_cm, limits_mj, table in specifications
    )


def compute_supplementary_energy_landscape(
    case: EnergyLandscapeCase,
    *,
    order: int = 4,
    points: int = 201,
) -> EnergyLandscape:
    """Compute one landscape over the domain used in Figs. 3-7."""

    if points < 2:
        raise ValueError("points must be at least two")

    x_coordinates = np.linspace(1.0, 3.0, points) * CENTIMETRE
    y_coordinates = np.linspace(-1.0, 1.0, points) * CENTIMETRE
    model = MultipoleExpansion(case.configuration, order=order)
    x_grid, y_grid, energy = model.energy_grid(x_coordinates, y_coordinates)
    equilibria = tuple(
        find_planar_equilibria(
            model,
            (float(x_coordinates[0]), float(x_coordinates[-1])),
            (float(y_coordinates[0]), float(y_coordinates[-1])),
        )
    )
    return EnergyLandscape(
        case=case,
        order=order,
        x_grid=x_grid,
        y_grid=y_grid,
        energy=energy,
        equilibria=equilibria,
    )


def figure_5_force_curves(
    x_coordinates: ArrayLike,
    *,
    orders: Iterable[int] = (0, 2, 4, 6),
    configuration: PlanarConfiguration | None = None,
) -> dict[int, NDArray[np.float64]]:
    """Evaluate the axial MPEX force curves shown in Fig. 5.

    Coordinates are supplied in metres and returned forces are in newtons.
    Models are evaluated sequentially to keep resource usage predictable.
    """

    x_values = np.asarray(x_coordinates, dtype=float)
    if x_values.ndim != 1 or x_values.size == 0:
        raise ValueError("x_coordinates must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x_values)):
        raise ValueError("x_coordinates must be finite")

    selected_configuration = configuration or article_minimal_configuration()
    curves: dict[int, NDArray[np.float64]] = {}
    for order in orders:
        model = MultipoleExpansion(selected_configuration, order=order)
        curves[order] = np.asarray(
            [model.force((float(x_coordinate), 0.0))[0] for x_coordinate in x_values],
            dtype=float,
        )
    return curves
