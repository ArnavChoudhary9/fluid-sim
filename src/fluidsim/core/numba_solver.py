"""Numba-accelerated solver backend.

:class:`NumbaSolver` subclasses :class:`~fluidsim.core.numpy_solver.NumpySolver`
and overrides only the numerically hot primitives — the linear-solve sweep, the
projection, and advection — delegating each to a compiled kernel in
:mod:`fluidsim.core._numba_kernels`. Everything else (the step orchestration,
all boundary handling, source injection, configuration) is inherited unchanged.

Because the kernels mirror the NumPy arithmetic exactly, the two backends agree
to floating-point tolerance; see ``tests/test_solver_parity.py``.

The numba import is guarded: if numba (or a compatible numba/NumPy combination)
is unavailable, ``NUMBA_AVAILABLE`` is ``False`` and the factory falls back to
the NumPy backend. Importing this module therefore never raises.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .boundary import SCALAR, apply_boundary, fluid_neighbour_count
from .numpy_solver import NumpySolver

try:
    from . import _numba_kernels as _kernels

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    _kernels = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


class NumbaSolver(NumpySolver):
    """NumPy solver with its hot loops replaced by Numba kernels."""

    def __init__(self, config: SimConfig) -> None:
        if not NUMBA_AVAILABLE:  # pragma: no cover - guarded by the factory
            raise RuntimeError("NumbaSolver requested but numba is not available")
        super().__init__(config)

    def _lin_solve(self, b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float) -> None:
        # The RGB dye field is 3-D; its diffusion is cheap and stays on the
        # well-tested NumPy path. Only 2-D velocity diffusion is JIT-accelerated.
        if x.ndim != 2:
            super()._lin_solve(b, x, x0, a, c)
            return
        inv_c = 1.0 / c
        for _ in range(self.config.iterations):
            _kernels.lin_solve_iter(x, x0, a, inv_c, self.config.n)
            self._field_bnd(b, x)

    def _project(self, u: np.ndarray, v: np.ndarray) -> None:
        n = self.config.n
        s = self.state
        p, div = s.pressure, s.divergence
        obstacle = s.obstacle
        fluid = (~obstacle).astype(np.float32)
        count = fluid_neighbour_count(obstacle).astype(np.float32)

        _kernels.divergence(u, v, div, obstacle, n)
        p[:] = 0.0
        apply_boundary(SCALAR, div, n, self.config.boundary)
        self._pressure_bnd(p)

        for _ in range(self.config.iterations):
            _kernels.project_iter(p, div, fluid, count, n)
            self._pressure_bnd(p)

        _kernels.subtract_gradient(u, v, p, n)
        self._velocity_bnd(u, v)

    def _advect(
        self,
        b: int,
        d: np.ndarray,
        d0: np.ndarray,
        vel_u: np.ndarray,
        vel_v: np.ndarray,
        dt: float,
    ) -> None:
        dt0 = dt * self.config.n
        if d.ndim == 3:
            _kernels.advect_rgb(d, d0, vel_u, vel_v, dt0, self.config.n)
        else:
            _kernels.advect_scalar(d, d0, vel_u, vel_v, dt0, self.config.n)
        self._field_bnd(b, d)
