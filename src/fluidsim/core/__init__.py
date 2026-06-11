"""Physics core for the fluid simulation.

This subpackage is **pure**: it depends only on NumPy (and optionally Numba) and
knows nothing about pygame, rendering, or input. Everything outside ``core`` is a
consumer of the :class:`~fluidsim.core.solver_base.BaseSolver` interface returned
by :func:`~fluidsim.core.factory.make_solver`.

Import a solver with::

    from fluidsim.core import make_solver
    from fluidsim.config import SimConfig

    solver = make_solver(SimConfig(n=128, backend="auto"))
    solver.add_dye(64, 64, (255, 0, 0), radius=4)
    solver.step(1 / 60)
"""

from __future__ import annotations

from .factory import make_solver
from .fields import FluidState
from .grid import Grid
from .solver_base import BaseSolver

__all__ = ["make_solver", "BaseSolver", "FluidState", "Grid"]
