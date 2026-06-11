"""Boundary condition sign conventions on the ghost ring.

These are the rules from docs/src/math-boundary-conditions.md:

* scalars copy their neighbour (homogeneous Neumann),
* the velocity component normal to a wall is negated,
* corners are the average of their two edge neighbours.
"""

from __future__ import annotations

import numpy as np

from fluidsim.core.boundary import SCALAR, U_FIELD, V_FIELD, set_bnd


def _ramp(n: int) -> np.ndarray:
    """An interior field with distinct values so reflections are detectable."""
    field = np.zeros((n + 2, n + 2), dtype=np.float32)
    field[1:-1, 1:-1] = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    return field


def test_scalar_copies_neighbour() -> None:
    n = 6
    f = _ramp(n)
    set_bnd(SCALAR, f, n)
    # Left/right ghosts equal the adjacent interior column (Neumann).
    assert np.array_equal(f[1:-1, 0], f[1:-1, 1])
    assert np.array_equal(f[1:-1, n + 1], f[1:-1, n])
    # Top/bottom ghosts equal the adjacent interior row.
    assert np.array_equal(f[0, 1:-1], f[1, 1:-1])
    assert np.array_equal(f[n + 1, 1:-1], f[n, 1:-1])


def test_u_velocity_negates_across_vertical_walls() -> None:
    n = 6
    f = _ramp(n)
    set_bnd(U_FIELD, f, n)
    # x-velocity is normal to the left/right walls -> negated there.
    assert np.allclose(f[1:-1, 0], -f[1:-1, 1])
    assert np.allclose(f[1:-1, n + 1], -f[1:-1, n])
    # ...but tangential to top/bottom -> copied.
    assert np.allclose(f[0, 1:-1], f[1, 1:-1])


def test_v_velocity_negates_across_horizontal_walls() -> None:
    n = 6
    f = _ramp(n)
    set_bnd(V_FIELD, f, n)
    # y-velocity is normal to top/bottom walls -> negated there.
    assert np.allclose(f[0, 1:-1], -f[1, 1:-1])
    assert np.allclose(f[n + 1, 1:-1], -f[n, 1:-1])
    # ...tangential to left/right -> copied.
    assert np.allclose(f[1:-1, 0], f[1:-1, 1])


def test_corners_average_their_neighbours() -> None:
    n = 6
    f = _ramp(n)
    set_bnd(SCALAR, f, n)
    assert f[0, 0] == 0.5 * (f[1, 0] + f[0, 1])
    assert f[n + 1, n + 1] == 0.5 * (f[n, n + 1] + f[n + 1, n])
