"""Cartesian multipole moments of uniformly magnetized cylinders."""

from __future__ import annotations

import math

from scipy.special import beta

from tractor_magnet.models import CylinderMagnet
from tractor_magnet.multi_index import MultiIndex


def cylinder_geometric_moment(magnet: CylinderMagnet, index: MultiIndex) -> float:
    """Return the volume integral of x^i y^j z^k for a centred cylinder.

    This is the product of Eqs. (B4)-(B7), without magnetization.
    """

    x_order, y_order, z_order = index
    if min(index) < 0:
        raise ValueError("multi-index components must be non-negative")
    if any(component % 2 for component in index):
        return 0.0

    radial = magnet.radius ** (x_order + y_order + 2) / (x_order + y_order + 2)
    angular = 2.0 * beta((x_order + 1) / 2.0, (y_order + 1) / 2.0)
    axial = magnet.height ** (z_order + 1) / (2.0**z_order * (z_order + 1))
    result = radial * angular * axial
    if not math.isfinite(result):
        raise ArithmeticError("cylinder moment evaluation produced a non-finite value")
    return float(result)


def magnetization_moment_z(magnet: CylinderMagnet, index: MultiIndex) -> float:
    """Return the signed z-component of the moment in Eq. (B3)."""

    return magnet.magnetization_z * cylinder_geometric_moment(magnet, index)
