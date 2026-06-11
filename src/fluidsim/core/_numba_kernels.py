"""Numba-JIT inner kernels for the accelerated solver.

This module is imported **only** by :mod:`fluidsim.core.numba_solver`, and only
inside a ``try/except`` there — so importing :mod:`fluidsim` never requires
``numba`` to be installed. Importing this module *does* require numba (the
top-level ``from numba import njit`` is the whole point).

Each kernel is a faithful, scalar-loop transcription of the corresponding
vectorised NumPy primitive, using the **same red-black ordering and the same
arithmetic**, so the two backends agree to floating-point tolerance. Sequential
loops are exactly where Numba beats NumPy: no per-sweep temporary arrays, and
Gauss-Seidel reads the freshest neighbour values.

Indexing convention matches the rest of the core: ``field[y, x]`` with the
interior spanning ``1..n`` on both axes.
"""

from __future__ import annotations

from numba import njit


@njit(cache=True)
def lin_solve_iter(x, x0, a, inv_c, n):
    """One red-black Gauss-Seidel sweep of ``(c·x − a·Σneighbours) = x0`` (2D)."""
    for color in range(2):  # 0 = red, then 1 = black (matches the NumPy order)
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                if (i + j) % 2 == color:
                    s = x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i]
                    x[j, i] = (x0[j, i] + a * s) * inv_c


@njit(cache=True)
def divergence(u, v, div, obstacle, n):
    """Compute ``div = -0.5·∇·u`` on the interior, zeroed inside obstacles."""
    div[:, :] = 0.0
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            if obstacle[j, i]:
                div[j, i] = 0.0
            else:
                div[j, i] = -0.5 * (
                    (u[j, i + 1] - u[j, i - 1]) + (v[j + 1, i] - v[j - 1, i])
                )


@njit(cache=True)
def project_iter(p, div, fluid, count, n):
    """One red-black sweep of the obstacle-aware pressure-Poisson solve (2D)."""
    for color in range(2):
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                if fluid[j, i] and count[j, i] > 0.0 and (i + j) % 2 == color:
                    s = (
                        p[j, i - 1] * fluid[j, i - 1]
                        + p[j, i + 1] * fluid[j, i + 1]
                        + p[j - 1, i] * fluid[j - 1, i]
                        + p[j + 1, i] * fluid[j + 1, i]
                    )
                    p[j, i] = (div[j, i] + s) / count[j, i]


@njit(cache=True)
def subtract_gradient(u, v, p, n):
    """Subtract the pressure gradient from the velocity field (2D)."""
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            u[j, i] -= 0.5 * (p[j, i + 1] - p[j, i - 1])
            v[j, i] -= 0.5 * (p[j + 1, i] - p[j - 1, i])


@njit(cache=True)
def advect_scalar(d, d0, vel_u, vel_v, dt0, n):
    """Semi-Lagrangian advection of a scalar field (2D)."""
    hi = n + 0.5
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            x = i - dt0 * vel_u[j, i]
            y = j - dt0 * vel_v[j, i]
            if x < 0.5:
                x = 0.5
            elif x > hi:
                x = hi
            if y < 0.5:
                y = 0.5
            elif y > hi:
                y = hi
            i0 = int(x)
            j0 = int(y)
            sx1 = x - i0
            sx0 = 1.0 - sx1
            sy1 = y - j0
            sy0 = 1.0 - sy1
            d[j, i] = sy0 * (sx0 * d0[j0, i0] + sx1 * d0[j0, i0 + 1]) + sy1 * (
                sx0 * d0[j0 + 1, i0] + sx1 * d0[j0 + 1, i0 + 1]
            )


@njit(cache=True)
def advect_rgb(d, d0, vel_u, vel_v, dt0, n):
    """Semi-Lagrangian advection of a 3-channel (RGB) field."""
    hi = n + 0.5
    channels = d.shape[2]
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            x = i - dt0 * vel_u[j, i]
            y = j - dt0 * vel_v[j, i]
            if x < 0.5:
                x = 0.5
            elif x > hi:
                x = hi
            if y < 0.5:
                y = 0.5
            elif y > hi:
                y = hi
            i0 = int(x)
            j0 = int(y)
            sx1 = x - i0
            sx0 = 1.0 - sx1
            sy1 = y - j0
            sy0 = 1.0 - sy1
            for c in range(channels):
                d[j, i, c] = sy0 * (
                    sx0 * d0[j0, i0, c] + sx1 * d0[j0, i0 + 1, c]
                ) + sy1 * (sx0 * d0[j0 + 1, i0, c] + sx1 * d0[j0 + 1, i0 + 1, c])
