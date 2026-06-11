"""Advection: mass is (nearly) conserved, and obstacles do not leak."""

from __future__ import annotations

import numpy as np

from fluidsim.config import SimConfig
from fluidsim.core import make_solver
from fluidsim.core.boundary import SCALAR


def test_scalar_advection_conserves_mass(backend: str) -> None:
    """A compact blob advected by a uniform interior velocity keeps its mass."""
    n = 48
    solver = make_solver(SimConfig(n=n, backend=backend))

    # A Gaussian blob centred well away from the walls.
    line = np.arange(1, n + 1)
    xs, ys = np.meshgrid(line, line)
    blob = np.zeros((n + 2, n + 2), dtype=np.float32)
    blob[1:-1, 1:-1] = np.exp(-((xs - n / 2) ** 2 + (ys - n / 2) ** 2) / 12.0).astype(
        np.float32
    )

    u = np.full((n + 2, n + 2), 1.0, dtype=np.float32)  # uniform rightward flow
    v = np.zeros((n + 2, n + 2), dtype=np.float32)
    out = np.zeros_like(blob)

    before = float(blob[1:-1, 1:-1].sum())
    solver._advect(SCALAR, out, blob, u, v, dt=1 / 60)
    after = float(out[1:-1, 1:-1].sum())

    assert abs(after - before) / before < 0.02


def test_obstacle_does_not_leak(backend: str) -> None:
    """A full vertical wall stops dye from ever reaching the far side."""
    n = 48
    solver = make_solver(SimConfig(n=n, backend=backend, diffusion=0.0, fade=1.0))

    wall = n // 2
    solver.state.obstacle[1:-1, wall : wall + 2] = True
    solver.clear()  # evacuate anything inside the new wall

    # Push dye + flow into the wall from the left, every step.
    for _ in range(80):
        for y in range(n // 4, 3 * n // 4):
            solver.add_dye(wall - 6, y, (255.0, 255.0, 255.0), radius=2)
            solver.add_velocity(wall - 6, y, 40.0, 0.0, radius=2)
        solver.step(1 / 60)

    right_region = solver.state.density[1:-1, wall + 2 :, :]
    left_region = solver.state.density[1:-1, 1:wall, :]

    assert left_region.sum() > 1.0                       # dye built up on the left
    assert right_region.sum() < 1e-3 * left_region.sum() # essentially none leaked


def test_velocity_is_zero_inside_solids(backend: str) -> None:
    n = 32
    solver = make_solver(SimConfig(n=n, backend=backend))
    solver.state.obstacle[10:20, 10:20] = True
    solver.clear()
    for _ in range(10):
        solver.add_velocity(5, 16, 50.0, 0.0, radius=3)
        solver.step(1 / 60)

    solid = solver.state.obstacle
    assert np.allclose(solver.state.u[solid], 0.0)
    assert np.allclose(solver.state.v[solid], 0.0)
