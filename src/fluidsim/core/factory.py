"""Backend selection with graceful fallback.

:func:`make_solver` is the single entry point consumers use to obtain a
:class:`~fluidsim.core.solver_base.BaseSolver`. It honours
``SimConfig.backend``:

* ``"numpy"`` — always the pure-NumPy reference solver.
* ``"numba"`` — the Numba solver if available, otherwise a warning + NumPy.
* ``"auto"`` — Numba if available, else NumPy, silently.

Keeping this decision in one place means no other module ever imports a concrete
solver class, and the Numba dependency is touched only when actually requested.
"""

from __future__ import annotations

import warnings

from ..config import SimConfig
from .numpy_solver import NumpySolver
from .solver_base import BaseSolver


def make_solver(config: SimConfig) -> BaseSolver:
    """Construct the solver backend named by ``config.backend``."""
    backend = config.backend

    if backend == "numpy":
        return NumpySolver(config)

    # Importing here keeps numba off the import path for the NumPy-only case.
    from .numba_solver import NUMBA_AVAILABLE, NumbaSolver

    if backend == "numba":
        if NUMBA_AVAILABLE:
            return NumbaSolver(config)
        warnings.warn(
            "backend='numba' requested but numba is unavailable; using NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return NumpySolver(config)

    if backend == "auto":
        return NumbaSolver(config) if NUMBA_AVAILABLE else NumpySolver(config)

    # Unreachable: SimConfig validates the backend name at construction.
    raise ValueError(f"unknown backend {backend!r}")
