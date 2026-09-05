"""Utilities for three-dimensional Cartesian multi-indices."""

from __future__ import annotations

import math
from collections.abc import Iterator

MultiIndex = tuple[int, int, int]


def cartesian_indices(max_order: int, *, even_only: bool = False) -> Iterator[MultiIndex]:
    """Yield all three-dimensional multi-indices through ``max_order``."""

    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer")

    for total in range(max_order + 1):
        for x_order in range(total, -1, -1):
            for y_order in range(total - x_order, -1, -1):
                z_order = total - x_order - y_order
                index = (x_order, y_order, z_order)
                if not even_only or all(component % 2 == 0 for component in index):
                    yield index


def add_indices(left: MultiIndex, right: MultiIndex) -> MultiIndex:
    """Return the component-wise sum of two multi-indices."""

    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def multi_factorial(index: MultiIndex) -> int:
    """Return alpha! = alpha_x! alpha_y! alpha_z!."""

    return math.prod(math.factorial(component) for component in index)


def total_order(index: MultiIndex) -> int:
    """Return |alpha|."""

    return sum(index)
