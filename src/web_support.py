"""Validated adapter between the scientific core and the browser app.

The public package remains SI based.  This module owns only browser-facing
defaults, unit conversion, staged MPEX calculations, and compact summaries.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.models import CylinderMagnet
from tractor_magnet.stability_map import StabilityMap, compute_stability_map


@dataclass(frozen=True, slots=True)
class WebInputs:
    """Complete, hashable input state for one browser calculation."""

    follower_radius_m: float
    follower_height_m: float
    follower_remanence_t: float
    attractor_radius_m: float
    attractor_height_m: float
    attractor_remanence_t: float
    repulsor_radius_m: float
    repulsor_height_m: float
    repulsor_remanence_t: float
    a_min_m: float
    a_max_m: float
    b_min_m: float
    b_max_m: float
    grid_points: int = 21
    refinement_order: int = 4
    friction_threshold_n: float = 10.0e-3
    centering_tolerance_m: float = 1.25e-3

    @classmethod
    def from_millimetres(
        cls,
        *,
        follower_radius_mm: float,
        follower_height_mm: float,
        follower_remanence_t: float,
        attractor_radius_mm: float,
        attractor_height_mm: float,
        attractor_remanence_t: float,
        repulsor_radius_mm: float,
        repulsor_height_mm: float,
        repulsor_remanence_t: float,
        a_max_mm: float,
        b_max_mm: float,
        grid_points: int,
        refinement_order: int,
        friction_threshold_mn: float,
        centering_tolerance_mm: float,
    ) -> WebInputs:
        """Construct validated SI inputs from the units shown by the app."""

        attractor_radius_mm = float(attractor_radius_mm)
        repulsor_radius_mm = float(repulsor_radius_mm)
        # The lower a boundary is the a-coordinate of the fixed-magnet
        # collision circle at 90 degrees, measured from the positive a-axis.
        # Hence a_min = (R_att + R_rep) cos(90 degrees) = 0.
        a_min_mm = 0.0
        b_min_mm = max(1.0, float(math.floor(repulsor_radius_mm)))
        result = cls(
            follower_radius_m=float(follower_radius_mm) * MILLIMETRE,
            follower_height_m=float(follower_height_mm) * MILLIMETRE,
            follower_remanence_t=float(follower_remanence_t),
            attractor_radius_m=attractor_radius_mm * MILLIMETRE,
            attractor_height_m=float(attractor_height_mm) * MILLIMETRE,
            attractor_remanence_t=float(attractor_remanence_t),
            repulsor_radius_m=repulsor_radius_mm * MILLIMETRE,
            repulsor_height_m=float(repulsor_height_mm) * MILLIMETRE,
            repulsor_remanence_t=float(repulsor_remanence_t),
            a_min_m=float(a_min_mm) * MILLIMETRE,
            a_max_m=float(a_max_mm) * MILLIMETRE,
            b_min_m=float(b_min_mm) * MILLIMETRE,
            b_max_m=float(b_max_mm) * MILLIMETRE,
            grid_points=int(grid_points),
            refinement_order=int(refinement_order),
            friction_threshold_n=float(friction_threshold_mn) * 1.0e-3,
            centering_tolerance_m=float(centering_tolerance_mm) * MILLIMETRE,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Reject non-physical or computationally excessive app inputs."""

        values = {
            name: value
            for name, value in (
                ("follower radius", self.follower_radius_m),
                ("follower height", self.follower_height_m),
                ("follower remanence", self.follower_remanence_t),
                ("attractor radius", self.attractor_radius_m),
                ("attractor height", self.attractor_height_m),
                ("attractor remanence", self.attractor_remanence_t),
                ("repulsor radius", self.repulsor_radius_m),
                ("repulsor height", self.repulsor_height_m),
                ("repulsor remanence", self.repulsor_remanence_t),
                ("a maximum", self.a_max_m),
                ("b minimum", self.b_min_m),
                ("b maximum", self.b_max_m),
                ("friction threshold", self.friction_threshold_n),
                ("centering tolerance", self.centering_tolerance_m),
            )
        }
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Non-finite input: {', '.join(invalid)}")
        nonpositive = [name for name, value in values.items() if value <= 0.0]
        if nonpositive:
            raise ValueError(f"Inputs must be positive: {', '.join(nonpositive)}")
        if not math.isfinite(self.a_min_m) or self.a_min_m < 0.0:
            raise ValueError("The a minimum must be finite and non-negative.")
        if self.a_min_m >= self.a_max_m or self.b_min_m >= self.b_max_m:
            raise ValueError("Each map minimum must be smaller than its maximum.")
        if self.refinement_order not in (0, 2, 4):
            raise ValueError("The multipole order must be 0, 2, or 4.")
        if not 11 <= self.grid_points <= 101:
            raise ValueError("The web grid must contain between 11 and 101 points per axis.")

    @property
    def follower(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.follower_radius_m,
            self.follower_height_m,
            self.follower_remanence_t,
            polarity=1,
        )

    @property
    def attractor(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.attractor_radius_m,
            self.attractor_height_m,
            self.attractor_remanence_t,
            polarity=-1,
        )

    @property
    def repulsor(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.repulsor_radius_m,
            self.repulsor_height_m,
            self.repulsor_remanence_t,
            polarity=1,
        )

    @property
    def moment_ratio(self) -> float:
        """Return ``|m_rep/m_att|`` for one repulsor."""

        return abs(self.repulsor.dipole_moment_z / self.attractor.dipole_moment_z)

    @property
    def passes_moment_condition(self) -> bool:
        """Return the Appendix-A necessary condition for a stable dipole trap."""

        return self.moment_ratio < 0.5


@dataclass(frozen=True, slots=True)
class WebCalculation:
    """Staged dipole and multipole results retained by the browser."""

    inputs: WebInputs
    dipole_map: StabilityMap
    refined_map: StabilityMap
    elapsed_s: float


def _map(inputs: WebInputs, order: int) -> StabilityMap:
    a_values = np.linspace(inputs.a_min_m, inputs.a_max_m, inputs.grid_points)
    b_values = np.linspace(inputs.b_min_m, inputs.b_max_m, inputs.grid_points)
    return compute_stability_map(
        a_values,
        b_values,
        order=order,
        friction_threshold=inputs.friction_threshold_n,
        centering_tolerance=inputs.centering_tolerance_m,
        follower=inputs.follower,
        attractor=inputs.attractor,
        repulsor=inputs.repulsor,
    )


def calculate_web(inputs: WebInputs) -> WebCalculation:
    """Run the point-dipole baseline and requested MPEX refinement."""

    inputs.validate()
    started = time.perf_counter()
    dipole_map = _map(inputs, 0)
    if inputs.refinement_order == 0:
        refined_map = dipole_map
    else:
        refined_map = _map(inputs, inputs.refinement_order)
    return WebCalculation(
        inputs=inputs,
        dipole_map=dipole_map,
        refined_map=refined_map,
        elapsed_s=time.perf_counter() - started,
    )
