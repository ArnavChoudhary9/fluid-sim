"""Configurable boundary conditions: inflow, outflow, periodic, moving walls."""

from __future__ import annotations

import numpy as np

from fluidsim.config import BCType, BoundaryConditions, SimConfig
from fluidsim.core import make_solver
from fluidsim.core.boundary import (
    SCALAR,
    U_FIELD,
    V_FIELD,
    apply_boundary,
    apply_pressure_boundary,
    set_bnd,
)


def _ramp(n: int) -> np.ndarray:
    field = np.zeros((n + 2, n + 2), dtype=np.float32)
    field[1:-1, 1:-1] = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    return field


def test_all_walls_matches_set_bnd() -> None:
    """The generalised path must reproduce the closed-wall default exactly."""
    n = 8
    walls = BoundaryConditions()
    for b in (SCALAR, U_FIELD, V_FIELD):
        a, c = _ramp(n), _ramp(n)
        set_bnd(b, a, n)
        apply_boundary(b, c, n, walls)
        assert np.array_equal(a, c)


def test_inflow_pins_velocity() -> None:
    n = 8
    bc = BoundaryConditions(left=BCType.INFLOW, inflow_velocity=(2.0, 0.0))
    u = np.zeros((n + 2, n + 2), dtype=np.float32)
    apply_boundary(U_FIELD, u, n, bc)
    assert np.allclose(u[1:-1, 0], 2.0)   # ghost
    assert np.allclose(u[1:-1, 1], 2.0)   # boundary cell pinned (Dirichlet)


def test_outflow_is_zero_gradient() -> None:
    n = 8
    bc = BoundaryConditions(right=BCType.OUTFLOW)
    u = _ramp(n)
    apply_boundary(U_FIELD, u, n, bc)
    assert np.array_equal(u[1:-1, n + 1], u[1:-1, n])   # copied, not negated


def test_periodic_wraps() -> None:
    n = 8
    bc = BoundaryConditions(left=BCType.PERIODIC, right=BCType.PERIODIC)
    f = _ramp(n)
    apply_boundary(SCALAR, f, n, bc)
    assert np.array_equal(f[1:-1, 0], f[1:-1, n])     # left ghost = right interior
    assert np.array_equal(f[1:-1, n + 1], f[1:-1, 1])  # right ghost = left interior


def test_moving_wall_drags_tangential_velocity() -> None:
    n = 8
    lid = 3.0
    bc = BoundaryConditions(wall_velocity=(0.0, 0.0, lid, 0.0))  # top wall moves in +x
    u = _ramp(n)
    apply_boundary(U_FIELD, u, n, bc)
    # Top ghost row enforces a wall surface velocity of `lid`: ghost = 2*lid - inner.
    assert np.allclose(u[0, 1:-1], 2.0 * lid - u[1, 1:-1])


def test_pressure_outflow_is_dirichlet_zero() -> None:
    n = 8
    bc = BoundaryConditions(right=BCType.OUTFLOW)
    p = _ramp(n)
    apply_pressure_boundary(p, n, bc)
    assert np.allclose(p[1:-1, n + 1], 0.0)


def test_wind_tunnel_flows_and_does_not_leak(backend: str) -> None:
    """Integration: inflow→outflow past a cylinder develops a downstream flow."""
    n = 64
    bc = BoundaryConditions(
        left=BCType.INFLOW, right=BCType.OUTFLOW,
        top=BCType.WALL, bottom=BCType.WALL, inflow_velocity=(1.5, 0.0),
    )
    solver = make_solver(SimConfig(n=n, backend=backend, boundary=bc))
    solver.set_velocity_field(np.full((n, n), 1.5, np.float32), np.zeros((n, n), np.float32))

    ys, xs = np.mgrid[0:n, 0:n]
    cylinder = (xs - 18) ** 2 + (ys - n // 2) ** 2 <= 36
    solver.set_obstacle_mask(cylinder)

    for _ in range(100):
        solver.step(1 / 60)

    u = solver.velocity_field[0]
    assert u[1:-1, n // 2 :].mean() > 0.3            # flow established downstream
    assert np.allclose(solver.state.u[solver.state.obstacle], 0.0)  # no leak into solid
