"""The abstract solver interface — the project's least-privilege contract.

Every consumer of the physics core talks to a :class:`Solver`, never to a
concrete implementation. The interface deliberately separates three capabilities
so that each caller is handed only what it needs:

* **Source injection** — :meth:`add_dye`, :meth:`add_velocity`,
  :meth:`set_obstacle`. Called by the interaction layer. These are the *only*
  ways to mutate the simulation from outside.
* **Stepping** — :meth:`step`. Called by the main loop.
* **Reading** — :attr:`density_field`, :attr:`velocity_field`,
  :attr:`obstacle_mask`. Called by the renderer; they return read-only views.

The NumPy and Numba backends both subclass :class:`BaseSolver`, which implements
the input/output plumbing once (it is identical for both) and leaves only the
numerical kernels — :meth:`_vel_step` and :meth:`_dens_step` — abstract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import SimConfig
from .fields import DYE_CHANNELS, FluidState
from .grid import Grid


class BaseSolver(ABC):
    """Shared input/output behaviour for every solver backend.

    Subclasses implement the two numerical kernels; everything a consumer can
    *see* is defined here so the two backends stay behaviourally identical.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.grid = Grid(config.n)
        self.state = FluidState.allocate(self.grid)

    # -- Source injection (interaction layer only) --------------------------

    def add_dye(self, cx: int, cy: int, color: tuple[float, float, float], radius: int) -> None:
        """Stage dye of ``color`` in a disc of ``radius`` cells around ``(cx, cy)``.

        Coordinates are in grid space (interior cells ``1..n``). The contribution
        is added to a source buffer and applied on the next :meth:`step`, so
        several injections in one frame accumulate.
        """
        mask, region = self._disc(cx, cy, radius)
        for channel in range(DYE_CHANNELS):
            self.state.density_src[(*region, channel)][mask] += color[channel]

    def add_velocity(self, cx: int, cy: int, fx: float, fy: float, radius: int) -> None:
        """Stage a velocity impulse ``(fx, fy)`` in a disc around ``(cx, cy)``."""
        mask, region = self._disc(cx, cy, radius)
        self.state.u_src[region][mask] += fx
        self.state.v_src[region][mask] += fy

    def set_obstacle(self, cx: int, cy: int, radius: int, solid: bool) -> None:
        """Mark (or clear) a disc of cells as solid obstacle."""
        mask, region = self._disc(cx, cy, radius)
        block = self.state.obstacle[region]
        block[mask] = solid
        self.state.obstacle[region] = block
        if solid:
            # Remove any fluid quantities that were inside the new obstacle.
            self._evacuate_solids()

    def clear(self) -> None:
        """Reset velocity and dye (obstacles are preserved)."""
        self.state.reset_fluid()

    # -- Bulk setup (used by the scene/experiment layer) --------------------
    # These set whole interior fields at once. They take ``(n, n)`` interior
    # arrays (no ghost border) so callers never deal with the padding.

    def set_obstacle_mask(self, mask: np.ndarray) -> None:
        """Replace the obstacle field from an ``(n, n)`` boolean interior mask."""
        self._require_interior_shape(mask)
        self.state.obstacle[1:-1, 1:-1] = mask
        self._evacuate_solids()

    def set_velocity_field(self, u: np.ndarray, v: np.ndarray) -> None:
        """Set the interior velocity from two ``(n, n)`` arrays (initial conditions)."""
        self._require_interior_shape(u)
        self._require_interior_shape(v)
        self.state.u[1:-1, 1:-1] = u
        self.state.v[1:-1, 1:-1] = v

    def set_dye_field(self, dye: np.ndarray) -> None:
        """Set the interior dye from an ``(n, n, 3)`` RGB array (initial conditions)."""
        self._require_interior_shape(dye)
        self.state.density[1:-1, 1:-1] = dye

    def add_dye_region(self, mask: np.ndarray, color: tuple[float, float, float]) -> None:
        """Stage dye of ``color`` wherever an ``(n, n)`` boolean mask is True."""
        self._require_interior_shape(mask)
        for channel in range(DYE_CHANNELS):
            self.state.density_src[1:-1, 1:-1, channel][mask] += color[channel]

    def add_velocity_region(self, mask: np.ndarray, fx: float, fy: float) -> None:
        """Stage a velocity impulse ``(fx, fy)`` wherever an ``(n, n)`` mask is True."""
        self._require_interior_shape(mask)
        self.state.u_src[1:-1, 1:-1][mask] += fx
        self.state.v_src[1:-1, 1:-1][mask] += fy

    def set_velocity_region(self, mask: np.ndarray, fx: float, fy: float) -> None:
        """Pin (Dirichlet-set) the interior velocity wherever an ``(n, n)`` mask is True."""
        self._require_interior_shape(mask)
        ui, vi = self.state.u[1:-1, 1:-1], self.state.v[1:-1, 1:-1]
        ui[mask] = fx
        vi[mask] = fy

    # -- Stepping (main loop only) ------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""
        self._vel_step(dt)
        self._dens_step(dt)
        self._apply_fade()
        self.state.clear_sources()

    # -- Reading (renderer only) --------------------------------------------

    @property
    def density_field(self) -> np.ndarray:
        return self.state.density_view()

    @property
    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        return self.state.velocity_view()

    @property
    def obstacle_mask(self) -> np.ndarray:
        return self.state.obstacle_view()

    # -- Backend-specific numerics ------------------------------------------

    @abstractmethod
    def _vel_step(self, dt: float) -> None:
        """Advance the velocity field one step (add → diffuse → project → advect → project)."""

    @abstractmethod
    def _dens_step(self, dt: float) -> None:
        """Advance the dye field one step (add → diffuse → advect)."""

    # -- Shared helpers ------------------------------------------------------

    def _apply_fade(self) -> None:
        if self.config.fade < 1.0:
            self.state.density *= self.config.fade

    def _evacuate_solids(self) -> None:
        self.state.u[self.state.obstacle] = 0.0
        self.state.v[self.state.obstacle] = 0.0
        self.state.density[self.state.obstacle] = 0.0

    def _require_interior_shape(self, array: np.ndarray) -> None:
        expected = (self.config.n, self.config.n)
        if array.shape[:2] != expected:
            raise ValueError(
                f"expected an interior array of shape {expected}, got {array.shape}"
            )

    def _disc(self, cx: int, cy: int, radius: int) -> tuple[np.ndarray, tuple[slice, slice]]:
        """Return a boolean disc mask and the array slice it applies to.

        Clamps the disc to the interior so callers can pass any coordinate.
        """
        n = self.config.n
        x0, x1 = max(1, cx - radius), min(n, cx + radius)
        y0, y1 = max(1, cy - radius), min(n, cy + radius)
        region = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        ys, xs = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
        return mask, region
