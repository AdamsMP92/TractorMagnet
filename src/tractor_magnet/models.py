"""Immutable descriptions of cylindrical magnets and planar configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tractor_magnet.constants import VACUUM_PERMEABILITY

Point2D = tuple[float, float]


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


@dataclass(frozen=True, slots=True)
class CylinderMagnet:
    """Uniformly magnetized cylinder whose axis is parallel to the z-axis.

    All dimensions are in metres and ``remanence`` is in tesla. ``polarity``
    gives the sign of the z-component of the magnetization.
    """

    radius: float
    height: float
    remanence: float
    polarity: int = 1

    def __post_init__(self) -> None:
        _require_positive_finite("radius", self.radius)
        _require_positive_finite("height", self.height)
        _require_positive_finite("remanence", self.remanence)
        if self.polarity not in (-1, 1):
            raise ValueError(f"polarity must be -1 or 1, got {self.polarity!r}")

    @property
    def volume(self) -> float:
        """Cylinder volume in cubic metres."""

        return math.pi * self.radius**2 * self.height

    @property
    def magnetization_z(self) -> float:
        """Signed uniform magnetization in amperes per metre."""

        return self.polarity * self.remanence / VACUUM_PERMEABILITY

    @property
    def dipole_moment_z(self) -> float:
        """Signed magnetic dipole moment in ampere square metres."""

        return self.magnetization_z * self.volume


@dataclass(frozen=True, slots=True)
class PlacedMagnet:
    """A cylindrical source magnet placed in the z=0 plane."""

    magnet: CylinderMagnet
    position: Point2D

    def __post_init__(self) -> None:
        if len(self.position) != 2 or not all(math.isfinite(value) for value in self.position):
            raise ValueError("position must contain two finite coordinates")


@dataclass(frozen=True, slots=True)
class PlanarConfiguration:
    """A movable follower and a fixed collection of source magnets."""

    follower: CylinderMagnet
    sources: tuple[PlacedMagnet, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("a configuration needs at least one source magnet")
