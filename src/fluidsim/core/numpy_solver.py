"""Pure-NumPy reference implementation of the Stable-Fluids solver.

This is the canonical, dependency-light backend. It implements Jos Stam's
algorithm with three vectorised primitives — ``add source``, ``diffuse``
(implicit), ``project`` (pressure-Poisson), and ``advect`` (semi-Lagrangian) —
and weaves obstacle handling through each of them.

Two design points worth calling out:

* **Red-black Gauss-Seidel.** True sequential Gauss-Seidel cannot be vectorised
  (each cell depends on the previous one updated in the same sweep). Splitting the
  grid into a checkerboard and updating all "red" cells, then all "black" cells,
  recovers Gauss-Seidel-like convergence while staying fully vectorised. The
  Numba backend uses the *same* red-black ordering so the two agree numerically.

* **Project before *and* after advection.** Semi-Lagrangian advection traces
  velocity along a field that must already be divergence-free, and it
  re-introduces a little divergence of its own; projecting on both sides is what
  keeps the flow incompressible. Omitting either projection is a classic source
  of visible artifacts.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .boundary import (
    SCALAR,
    U_FIELD,
    V_FIELD,
    apply_boundary,
    apply_obstacle_bnd,
    apply_pressure_boundary,
    fluid_neighbour_count,
    zero_inside_obstacles,
)
from .solver_base import BaseSolver


class NumpySolver(BaseSolver):
    """Vectorised NumPy Stable-Fluids solver."""

    def __init__(self, config: SimConfig) -> None:
        super().__init__(config)
        n = config.n
        shape = self.grid.shape

        # Interior checkerboard masks for red-black Gauss-Seidel.
        yy, xx = np.indices(shape)
        interior = np.zeros(shape, dtype=bool)
        interior[1:-1, 1:-1] = True
        even = (xx + yy) % 2 == 0
        self._red = interior & even
        self._black = interior & ~even

        # Interior coordinate grids (x varies along axis 1, y along axis 0).
        line = np.arange(1, n + 1, dtype=np.float32)
        self._xs, self._ys = np.meshgrid(line, line)

    # -- Stam's two top-level steps -----------------------------------------

    def _vel_step(self, dt: float) -> None:
        s = self.state
        u, v = s.u, s.v
        u0, v0 = s.u_prev, s.v_prev
        visc = self.config.viscosity

        # 1. inject staged forces (+ optional dye-driven buoyancy)
        u += dt * s.u_src
        v += dt * s.v_src
        if self.config.buoyancy != 0.0:
            # Buoyant force proportional to the *normalised* smoke amount (mean of
            # the RGB channels mapped to ~[0, 1]) so ``buoyancy`` is independent of
            # the dye colour scale. y increases downward, so negative v is "up".
            smoke = s.density.sum(axis=2) * (1.0 / 765.0)
            v -= dt * self.config.buoyancy * smoke
        self._velocity_bnd(u, v)

        # 2. viscous diffusion (implicit solve)
        u0[:] = u
        self._diffuse(U_FIELD, u, u0, visc, dt)
        v0[:] = v
        self._diffuse(V_FIELD, v, v0, visc, dt)

        # 3. make divergence-free before advecting
        self._project(u, v)

        # 4. self-advection (back-trace along the current velocity field)
        u0[:] = u
        v0[:] = v
        self._advect(U_FIELD, u, u0, u0, v0, dt)
        self._advect(V_FIELD, v, v0, u0, v0, dt)

        # 5. project again to remove divergence advection re-introduced
        self._project(u, v)

    def _dens_step(self, dt: float) -> None:
        s = self.state
        d, d0 = s.density, s.density_prev
        diff = self.config.diffusion

        d += dt * s.density_src
        self._scalar_bnd(d)

        d0[:] = d
        self._diffuse(SCALAR, d, d0, diff, dt)

        d0[:] = d
        self._advect(SCALAR, d, d0, s.u, s.v, dt)

        # Keep dye from accumulating inside walls (re-filled by the boundary
        # copy before the next advection, so this introduces no sink).
        zero_inside_obstacles(d, s.obstacle)

    # -- Primitives ----------------------------------------------------------

    def _diffuse(self, b: int, x: np.ndarray, x0: np.ndarray, coeff: float, dt: float) -> None:
        """Implicit (backward-Euler) diffusion via red-black Gauss-Seidel."""
        if coeff <= 0.0:
            return
        a = dt * coeff * self.config.n * self.config.n
        self._lin_solve(b, x, x0, a, 1.0 + 4.0 * a)

    def _lin_solve(self, b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float) -> None:
        """Solve ``(c·x − a·Σneighbours) = x0`` with red-black sweeps."""
        inv_c = 1.0 / c
        for _ in range(self.config.iterations):
            x[self._red] = ((x0 + a * self._neighbour_sum(x)) * inv_c)[self._red]
            x[self._black] = ((x0 + a * self._neighbour_sum(x)) * inv_c)[self._black]
            self._field_bnd(b, x)

    def _project(self, u: np.ndarray, v: np.ndarray) -> None:
        """Hodge projection: subtract the gradient of pressure to enforce ∇·u = 0.

        Obstacle-aware: divergence is taken over fluid cells only, the pressure
        relaxation averages over each cell's *fluid* neighbours (Neumann at solid
        faces), and the final no-through-flow condition is re-imposed by the
        velocity boundary pass.
        """
        n = self.config.n
        s = self.state
        p, div = s.pressure, s.divergence
        obstacle = s.obstacle
        fluid = ~obstacle

        count = fluid_neighbour_count(obstacle).astype(np.float32)
        count_safe = np.where(count > 0, count, 1.0)
        solvable_red = self._red & fluid & (count > 0)
        solvable_black = self._black & fluid & (count > 0)

        div[:] = 0.0
        div[1:-1, 1:-1] = -0.5 * (
            (u[1:-1, 2:] - u[1:-1, :-2]) + (v[2:, 1:-1] - v[:-2, 1:-1])
        )
        div[obstacle] = 0.0
        p[:] = 0.0
        apply_boundary(SCALAR, div, n, self.config.boundary)
        self._pressure_bnd(p)

        for _ in range(self.config.iterations):
            new = (div + self._fluid_neighbour_sum(p, fluid)) / count_safe
            p[solvable_red] = new[solvable_red]
            new = (div + self._fluid_neighbour_sum(p, fluid)) / count_safe
            p[solvable_black] = new[solvable_black]
            self._pressure_bnd(p)

        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
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
        """Semi-Lagrangian advection: back-trace, then bilinearly sample ``d0``."""
        n = self.config.n
        dt0 = dt * n

        # Where did the stuff now at each interior cell come from one step ago?
        x = self._xs - dt0 * vel_u[1:-1, 1:-1]
        y = self._ys - dt0 * vel_v[1:-1, 1:-1]
        np.clip(x, 0.5, n + 0.5, out=x)
        np.clip(y, 0.5, n + 0.5, out=y)

        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        x1 = x0 + 1
        y1 = y0 + 1

        sx1 = x - x0
        sx0 = 1.0 - sx1
        sy1 = y - y0
        sy0 = 1.0 - sy1
        if d0.ndim == 3:  # RGB dye: add a trailing axis so weights broadcast
            sx0, sx1 = sx0[..., None], sx1[..., None]
            sy0, sy1 = sy0[..., None], sy1[..., None]

        d[1:-1, 1:-1] = sy0 * (sx0 * d0[y0, x0] + sx1 * d0[y0, x1]) + sy1 * (
            sx0 * d0[y1, x0] + sx1 * d0[y1, x1]
        )
        self._field_bnd(b, d)

    # -- Neighbour sums ------------------------------------------------------

    @staticmethod
    def _neighbour_sum(x: np.ndarray) -> np.ndarray:
        """Sum of the four orthogonal neighbours at every interior cell."""
        s = np.zeros_like(x)
        s[1:-1, 1:-1] = x[1:-1, :-2] + x[1:-1, 2:] + x[:-2, 1:-1] + x[2:, 1:-1]
        return s

    @staticmethod
    def _fluid_neighbour_sum(p: np.ndarray, fluid: np.ndarray) -> np.ndarray:
        """Sum of pressure over *fluid* neighbours only (solid faces contribute 0)."""
        s = np.zeros_like(p)
        s[1:-1, 1:-1] = (
            p[1:-1, :-2] * fluid[1:-1, :-2]
            + p[1:-1, 2:] * fluid[1:-1, 2:]
            + p[:-2, 1:-1] * fluid[:-2, 1:-1]
            + p[2:, 1:-1] * fluid[2:, 1:-1]
        )
        return s

    # -- Boundary dispatch ---------------------------------------------------

    def _field_bnd(self, b: int, x: np.ndarray) -> None:
        apply_boundary(b, x, self.config.n, self.config.boundary)
        apply_obstacle_bnd(b, x, self.state.obstacle, slip=self.config.obstacle_slip)

    def _velocity_bnd(self, u: np.ndarray, v: np.ndarray) -> None:
        self._field_bnd(U_FIELD, u)
        self._field_bnd(V_FIELD, v)
        zero_inside_obstacles(u, self.state.obstacle)
        zero_inside_obstacles(v, self.state.obstacle)

    def _scalar_bnd(self, d: np.ndarray) -> None:
        self._field_bnd(SCALAR, d)

    def _pressure_bnd(self, p: np.ndarray) -> None:
        apply_pressure_boundary(p, self.config.n, self.config.boundary)
        apply_obstacle_bnd(SCALAR, p, self.state.obstacle)
