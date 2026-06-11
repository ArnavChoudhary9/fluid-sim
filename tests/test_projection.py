"""The defining invariant of incompressible flow: projection kills divergence.

Two honest properties of Stam's collocated, central-difference projection:

* It strongly **reduces** the divergence of a smooth field (given enough
  Gauss-Seidel iterations — the Poisson solve converges slowly, so we use a small
  grid here and many sweeps).
* It is a **contraction**: re-projecting never *increases* the divergence. It is
  deliberately *not* idempotent — the central-difference divergence operator and
  the 5-point Poisson stencil differ, so a residual remains that further passes
  keep nibbling at. This is why the solver projects twice per step rather than
  expecting one pass to be exact (see docs/src/math-projection.md).
"""

from __future__ import annotations

import numpy as np

from fluidsim.config import SimConfig
from fluidsim.core import make_solver


def _central_divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = 0.5 * (
        (u[1:-1, 2:] - u[1:-1, :-2]) + (v[2:, 1:-1] - v[:-2, 1:-1])
    )
    return div


def _rms_divergence(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(_central_divergence(u, v) ** 2)))


def _smooth_velocity(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A smooth, intentionally divergent velocity field on the interior."""
    line = np.linspace(0.0, np.pi, n, dtype=np.float32)
    xs, ys = np.meshgrid(line, line)
    u = np.zeros((n + 2, n + 2), dtype=np.float32)
    v = np.zeros((n + 2, n + 2), dtype=np.float32)
    u[1:-1, 1:-1] = np.sin(2 * xs) * np.cos(ys)
    v[1:-1, 1:-1] = np.cos(xs) * np.sin(2 * ys)
    return u, v


def test_projection_removes_divergence(backend: str) -> None:
    n = 16
    solver = make_solver(SimConfig(n=n, backend=backend, iterations=200))
    u, v = _smooth_velocity(n)
    solver.state.u[:] = u
    solver.state.v[:] = v

    pre = _rms_divergence(solver.state.u, solver.state.v)
    solver._project(solver.state.u, solver.state.v)
    post = _rms_divergence(solver.state.u, solver.state.v)

    assert pre > 1e-3              # the test field really was divergent
    assert post < 0.15 * pre      # projection removed the vast majority


def test_reprojection_is_a_contraction(backend: str) -> None:
    n = 16
    solver = make_solver(SimConfig(n=n, backend=backend, iterations=200))
    u, v = _smooth_velocity(n)
    solver.state.u[:] = u
    solver.state.v[:] = v

    solver._project(solver.state.u, solver.state.v)
    after_one = _rms_divergence(solver.state.u, solver.state.v)
    solver._project(solver.state.u, solver.state.v)
    after_two = _rms_divergence(solver.state.u, solver.state.v)

    assert after_two <= after_one + 1e-6  # a second pass never adds divergence
