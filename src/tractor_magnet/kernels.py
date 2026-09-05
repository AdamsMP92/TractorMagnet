"""Analytical derivatives of the inverse-distance Green function."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import sympy as sp

from tractor_magnet.multi_index import MultiIndex

_X, _Y, _Z = sp.symbols("x y z", real=True)
_INVERSE_DISTANCE = 1 / sp.sqrt(_X**2 + _Y**2 + _Z**2)


@cache
def inverse_distance_derivative_function(
    index: MultiIndex,
) -> Callable[[float, float, float], float]:
    """Compile a Cartesian derivative of 1/r to a numerical function."""

    if min(index) < 0:
        raise ValueError("multi-index components must be non-negative")
    expression = sp.diff(_INVERSE_DISTANCE, _X, index[0], _Y, index[1], _Z, index[2])
    return sp.lambdify((_X, _Y, _Z), expression, modules="numpy", cse=True)


def inverse_distance_derivative(
    index: MultiIndex, displacement: tuple[float, float, float]
) -> float:
    """Evaluate D^index(1/r) away from the singularity."""

    if displacement == (0.0, 0.0, 0.0):
        raise ValueError("the inverse-distance kernel is singular at the source centre")
    function = inverse_distance_derivative_function(index)
    return float(function(*displacement))
