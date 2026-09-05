"""Point-dipole reference model and analytical Appendix-A relations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from tractor_magnet.constants import MU0_OVER_4PI
from tractor_magnet.models import PlanarConfiguration, Point2D


class PointDipoleModel:
    """Planar interaction model corresponding to Eqs. (A6)-(A8)."""

    def __init__(self, configuration: PlanarConfiguration) -> None:
        self.configuration = configuration

    def energy(self, point: Point2D) -> float:
        follower_moment = self.configuration.follower.dipole_moment_z
        total = 0.0
        for source in self.configuration.sources:
            dx = point[0] - source.position[0]
            dy = point[1] - source.position[1]
            radius_squared = dx * dx + dy * dy
            if radius_squared == 0.0:
                raise ValueError("point-dipole energy is singular at a source centre")
            total += source.magnet.dipole_moment_z / radius_squared**1.5
        return MU0_OVER_4PI * follower_moment * total

    def force(self, point: Point2D) -> NDArray[np.float64]:
        follower_moment = self.configuration.follower.dipole_moment_z
        force = np.zeros(2, dtype=float)
        for source in self.configuration.sources:
            displacement = np.asarray(point, dtype=float) - np.asarray(source.position)
            radius_squared = float(displacement @ displacement)
            if radius_squared == 0.0:
                raise ValueError("point-dipole force is singular at a source centre")
            coefficient = (
                3.0
                * MU0_OVER_4PI
                * follower_moment
                * source.magnet.dipole_moment_z
                / radius_squared**2.5
            )
            force += coefficient * displacement
        return force

    def hessian(self, point: Point2D) -> NDArray[np.float64]:
        follower_moment = self.configuration.follower.dipole_moment_z
        hessian = np.zeros((2, 2), dtype=float)
        identity = np.eye(2)
        for source in self.configuration.sources:
            displacement = np.asarray(point, dtype=float) - np.asarray(source.position)
            radius_squared = float(displacement @ displacement)
            if radius_squared == 0.0:
                raise ValueError("point-dipole Hessian is singular at a source centre")
            prefactor = MU0_OVER_4PI * follower_moment * source.magnet.dipole_moment_z
            hessian += prefactor * (
                -3.0 * identity / radius_squared**2.5
                + 15.0 * np.outer(displacement, displacement) / radius_squared**3.5
            )
        return hessian


def equilibrium_moment_ratio(x_equilibrium: float, a: float, b: float) -> float:
    """Return m_rep/m_att required by Eq. (A9)."""

    if min(x_equilibrium, a, b) <= 0.0:
        raise ValueError("x_equilibrium, a and b must be positive")
    return (x_equilibrium**2 + b**2) ** 2.5 / (2.0 * x_equilibrium * (x_equilibrium + a) ** 4)


def reduced_geometry_bounds(xi: float) -> tuple[float, float, float]:
    """Return (beta_low, beta_mid, beta_up) from Eqs. (A15)-(A16)."""

    if xi <= 0.0:
        raise ValueError("xi must be positive")
    beta_low = xi / math.sqrt(5.0 * xi + 4.0)
    beta_mid = xi / math.sqrt(2.0 * xi + 1.0)
    beta_up = 2.0 * xi / math.sqrt(5.0 * xi + 1.0)
    return beta_low, beta_mid, beta_up


def reduced_moment_ratio(xi: float, beta_ratio: float) -> float:
    """Return the dimensionless moment ratio in Eq. (A16)."""

    if xi <= 0.0 or beta_ratio <= 0.0:
        raise ValueError("xi and beta_ratio must be positive")
    return (xi**2 + beta_ratio**2) ** 2.5 / (2.0 * xi * (xi + 1.0) ** 4)


def reduced_binding_energy(xi: float, beta_ratio: float) -> float:
    """Return the dimensionless binding energy in Eq. (A23)."""

    if xi <= 0.0 or beta_ratio <= 0.0:
        raise ValueError("xi and beta_ratio must be positive")
    return (xi**2 + beta_ratio**2) / (xi * (xi + 1.0) ** 4) - 1.0 / (xi + 1.0) ** 3
