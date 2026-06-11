"""fluidsim — an interactive 2D Stable-Fluids simulation.

The public surface kept here is intentionally small and **NumPy-only**: importing
:mod:`fluidsim` never pulls in pygame or numba. Build a solver and drive the
physics with nothing but the core::

    import fluidsim

    solver = fluidsim.make_solver(fluidsim.SimConfig(n=128))
    solver.add_dye(64, 64, (255, 0, 0), radius=5)
    solver.step(1 / 60)
    image = solver.density_field  # read-only view

To launch the interactive window, use :func:`fluidsim.app.run` (which imports
pygame on demand) or run ``python -m fluidsim``.
"""

from __future__ import annotations

from .config import AppConfig, BrushConfig, RenderConfig, SimConfig
from .core import make_solver

__version__ = "0.1.0"

__all__ = [
    "make_solver",
    "SimConfig",
    "BrushConfig",
    "RenderConfig",
    "AppConfig",
    "__version__",
]
