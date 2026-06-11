"""Geometric shape primitives for placing obstacles and source regions.

Each shape rasterises to an ``(n, n)`` boolean **interior** mask (no ghost
border) via :meth:`Shape.mask`. Shapes use **normalised** coordinates in
``[0, 1]`` — ``x`` increases to the right, ``y`` increases downward (matching the
screen/grid convention) — so the same scene works at any grid resolution.

These are pure geometry: numpy only, no solver, no UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def _coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cell-centre coordinate grids ``(X, Y)``, each shape ``(n, n)``."""
    centres = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(centres, centres)  # x varies along columns, y along rows
    return x, y


@runtime_checkable
class Shape(Protocol):
    """Anything that can rasterise itself to an interior boolean mask."""

    def mask(self, n: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Circle:
    """A filled disc centred at ``(cx, cy)`` with radius ``r`` (all normalised)."""

    cx: float
    cy: float
    r: float

    def mask(self, n: int) -> np.ndarray:
        x, y = _coords(n)
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2


@dataclass(frozen=True, slots=True)
class Ellipse:
    """A filled ellipse with semi-axes ``rx``, ``ry`` centred at ``(cx, cy)``."""

    cx: float
    cy: float
    rx: float
    ry: float

    def mask(self, n: int) -> np.ndarray:
        x, y = _coords(n)
        return ((x - self.cx) / self.rx) ** 2 + ((y - self.cy) / self.ry) ** 2 <= 1.0


@dataclass(frozen=True, slots=True)
class Rectangle:
    """An axis-aligned rectangle spanning ``[x0, x1] × [y0, y1]`` (normalised)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def mask(self, n: int) -> np.ndarray:
        x, y = _coords(n)
        lo_x, hi_x = min(self.x0, self.x1), max(self.x0, self.x1)
        lo_y, hi_y = min(self.y0, self.y1), max(self.y0, self.y1)
        return (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)


@dataclass(frozen=True, slots=True)
class Plate:
    """A rotated rectangular plate — handy as an angled airfoil-like obstacle.

    Centred at ``(cx, cy)`` with the given ``length`` and ``thickness``
    (normalised), rotated ``angle`` degrees clockwise from horizontal.
    """

    cx: float
    cy: float
    length: float
    thickness: float
    angle: float = 0.0

    def mask(self, n: int) -> np.ndarray:
        x, y = _coords(n)
        theta = np.radians(self.angle)
        cos, sin = np.cos(theta), np.sin(theta)
        # Coordinates in the plate's local (rotated) frame.
        dx, dy = x - self.cx, y - self.cy
        local_x = dx * cos + dy * sin
        local_y = -dx * sin + dy * cos
        return (np.abs(local_x) <= self.length / 2) & (np.abs(local_y) <= self.thickness / 2)


def rasterize(shapes: tuple[Shape, ...] | list[Shape], n: int) -> np.ndarray:
    """Union the masks of several shapes into one ``(n, n)`` boolean array."""
    mask = np.zeros((n, n), dtype=bool)
    for shape in shapes:
        mask |= shape.mask(n)
    return mask
