"""Models for the two-dimensional tractor-magnet configuration."""

from tractor_magnet.configurations import (
    article_minimal_configuration,
    minimal_configuration,
    supplementary_configurations,
)
from tractor_magnet.equilibrium import Equilibrium, find_axial_equilibria, find_planar_equilibria
from tractor_magnet.models import CylinderMagnet, PlacedMagnet, PlanarConfiguration
from tractor_magnet.multipole import MultipoleExpansion
from tractor_magnet.point_dipole import PointDipoleModel
from tractor_magnet.stability_map import (
    StabilityMap,
    StabilityPoint,
    compute_stability_map,
    evaluate_stability_point,
)

__all__ = [
    "CylinderMagnet",
    "Equilibrium",
    "MultipoleExpansion",
    "PlanarConfiguration",
    "PlacedMagnet",
    "PointDipoleModel",
    "StabilityMap",
    "StabilityPoint",
    "article_minimal_configuration",
    "compute_stability_map",
    "evaluate_stability_point",
    "find_axial_equilibria",
    "find_planar_equilibria",
    "minimal_configuration",
    "supplementary_configurations",
]
