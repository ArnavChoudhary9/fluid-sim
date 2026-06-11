"""Grid geometry and the *ghost-cell* indexing convention.

Stam's Stable Fluids represents an ``n x n`` simulation inside arrays of shape
``(n + 2, n + 2)``. The extra ring of cells (indices ``0`` and ``n + 1`` on each
axis) are *ghost cells*: they hold boundary values and are written by
:func:`fluidsim.core.boundary.set_bnd`. The interior — the cells the physics
actually updates — is the slice ``[1 : n + 1, 1 : n + 1]``.

We use the convention ``array[y, x]`` (row index first). ``y`` increases
*downward* so that it matches image / screen coordinates; the renderer therefore
needs no vertical flip.

This module is pure geometry: it has no NumPy state of its own and no knowledge
of velocity, dye, or solvers. Keeping the convention in one tiny place means the
rest of the core can speak in terms of :attr:`Grid.interior` instead of
hand-writing ``1:n+1`` slices everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Grid:
    """Describes a padded ``(n + 2, n + 2)`` simulation grid.

    Parameters
    ----------
    n:
        Number of interior cells per axis.
    """

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"grid size must be >= 1, got {self.n}")

    @property
    def padded(self) -> int:
        """Side length including the one-cell ghost border."""
        return self.n + 2

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of a scalar field array, ``(n + 2, n + 2)``."""
        return (self.padded, self.padded)

    @property
    def interior(self) -> tuple[slice, slice]:
        """Slice selecting the interior (non-ghost) cells."""
        s = slice(1, self.n + 1)
        return (s, s)

    def clamp(self, coord: float) -> float:
        """Clamp a continuous coordinate to the valid sampling range.

        Semi-Lagrangian advection back-traces particle positions; those positions
        must stay inside ``[0.5, n + 0.5]`` so bilinear interpolation only ever
        touches real cells. This helper centralises that rule.
        """
        lo, hi = 0.5, self.n + 0.5
        if coord < lo:
            return lo
        if coord > hi:
            return hi
        return coord
