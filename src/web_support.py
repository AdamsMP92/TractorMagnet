"""Validated adapter between the scientific core and the browser app."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from tractor_magnet.adaptive_map import compute_adaptive_stability_map
from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.materials import magnet_grade
from tractor_magnet.models import CylinderMagnet
from tractor_magnet.stability_map import StabilityMap


@dataclass(frozen=True, slots=True)
class WebInputs:
    """Complete, hashable input state for one adaptive browser calculation."""

    follower_radius_m: float
    height_m: float
    follower_grade: str
    follower_remanence_t: float
    attractor_radius_m: float
    attractor_grade: str
    attractor_remanence_t: float
    repulsor_radius_m: float
    repulsor_grade: str
    repulsor_remanence_t: float
    resolution_m: float
    friction_threshold_n: float = 10.0e-3
    centering_tolerance_m: float = 1.25e-3

    @classmethod
    def from_millimetres(
        cls,
        *,
        height_mm: float,
        follower_radius_mm: float,
        follower_grade: str,
        follower_remanence_t: float,
        attractor_radius_mm: float,
        attractor_grade: str,
        attractor_remanence_t: float,
        repulsor_radius_mm: float,
        repulsor_grade: str,
        repulsor_remanence_t: float,
        resolution_mm: float,
        friction_threshold_mn: float,
        centering_tolerance_mm: float,
    ) -> WebInputs:
        """Construct validated SI inputs from the units shown by the app."""

        result = cls(
            height_m=float(height_mm) * MILLIMETRE,
            follower_radius_m=float(follower_radius_mm) * MILLIMETRE,
            follower_grade=str(follower_grade),
            follower_remanence_t=float(follower_remanence_t),
            attractor_radius_m=float(attractor_radius_mm) * MILLIMETRE,
            attractor_grade=str(attractor_grade),
            attractor_remanence_t=float(attractor_remanence_t),
            repulsor_radius_m=float(repulsor_radius_mm) * MILLIMETRE,
            repulsor_grade=str(repulsor_grade),
            repulsor_remanence_t=float(repulsor_remanence_t),
            resolution_m=float(resolution_mm) * MILLIMETRE,
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
                ("common magnet height", self.height_m),
                ("follower radius", self.follower_radius_m),
                ("follower remanence", self.follower_remanence_t),
                ("attractor radius", self.attractor_radius_m),
                ("attractor remanence", self.attractor_remanence_t),
                ("repulsor radius", self.repulsor_radius_m),
                ("repulsor remanence", self.repulsor_remanence_t),
                ("resolution", self.resolution_m),
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
        if not 0.1 * MILLIMETRE <= self.resolution_m <= 2.0 * MILLIMETRE:
            raise ValueError("Resolution must lie between 0.1 mm and 2.0 mm.")
        assignments = (
            ("follower", self.follower_grade, self.follower_remanence_t),
            ("attractor", self.attractor_grade, self.attractor_remanence_t),
            ("repulsor", self.repulsor_grade, self.repulsor_remanence_t),
        )
        for name, grade_key, remanence in assignments:
            grade = magnet_grade(grade_key)
            if not grade.contains(remanence):
                raise ValueError(
                    f"{name.capitalize()} remanence must lie between "
                    f"{grade.remanence_min:.3f} T and {grade.remanence_max:.3f} T "
                    f"for {grade.material} {grade.grade}."
                )

    @property
    def a_min_m(self) -> float:
        return 0.0

    @property
    def b_min_m(self) -> float:
        return max(
            0.0,
            math.floor(self.repulsor_radius_m / MILLIMETRE) * MILLIMETRE - MILLIMETRE,
        )

    @property
    def follower(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.follower_radius_m,
            self.height_m,
            self.follower_remanence_t,
            polarity=1,
        )

    @property
    def attractor(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.attractor_radius_m,
            self.height_m,
            self.attractor_remanence_t,
            polarity=-1,
        )

    @property
    def repulsor(self) -> CylinderMagnet:
        return CylinderMagnet(
            self.repulsor_radius_m,
            self.height_m,
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
    """Adaptive MPEX4 result retained by the browser."""

    inputs: WebInputs
    refined_map: StabilityMap
    search_extent_m: float
    evaluated_points: int
    full_grid_points: int
    scout_points: int
    elapsed_s: float


def calculate_web(inputs: WebInputs) -> WebCalculation:
    """Run the automatic MPEX0 scout and sparse MPEX4 refinement."""

    inputs.validate()
    started = time.perf_counter()
    result = compute_adaptive_stability_map(
        inputs.resolution_m,
        friction_threshold=inputs.friction_threshold_n,
        centering_tolerance=inputs.centering_tolerance_m,
        follower=inputs.follower,
        attractor=inputs.attractor,
        repulsor=inputs.repulsor,
    )
    return WebCalculation(
        inputs=inputs,
        refined_map=result.stability_map,
        search_extent_m=result.search_extent,
        evaluated_points=result.evaluated_points,
        full_grid_points=result.full_grid_points,
        scout_points=result.scout_points,
        elapsed_s=time.perf_counter() - started,
    )
