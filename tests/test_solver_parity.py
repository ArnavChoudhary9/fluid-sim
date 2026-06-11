"""The NumPy and Numba backends must agree (within fast-math tolerance).

Skipped entirely when numba is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidsim.config import SimConfig
from fluidsim.core import make_solver
from fluidsim.core.numba_solver import NUMBA_AVAILABLE

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")


def _drive(backend: str, steps: int):
    solver = make_solver(SimConfig(n=32, backend=backend))
    # A fixed (no randomness) input sequence so both backends see identical input.
    for t in range(steps):
        cx = 8 + (t % 16)
        solver.add_dye(cx, 16, (200.0, 50.0, 10.0), radius=3)
        solver.add_velocity(cx, 16, 30.0, 5.0, radius=3)
        solver.step(1 / 60)
    return solver


def test_numpy_and_numba_agree() -> None:
    a = _drive("numpy", steps=25)
    b = _drive("numba", steps=25)
    assert np.allclose(a.state.u, b.state.u, atol=1e-4)
    assert np.allclose(a.state.v, b.state.v, atol=1e-4)
    assert np.allclose(a.state.density, b.state.density, atol=1e-3)
