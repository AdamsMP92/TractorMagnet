"""Shared protocol for energy models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tractor_magnet.models import Point2D


class EnergyModel(Protocol):
    """Potential energy and its first two planar derivatives."""

    def energy(self, point: Point2D) -> float: ...

    def force(self, point: Point2D) -> NDArray[np.float64]: ...

    def hessian(self, point: Point2D) -> NDArray[np.float64]: ...
