"""The mutable simulation state and its pre-allocated scratch buffers.

:class:`FluidState` owns every array the solver touches. Splitting it out from
the solver gives us a single, inspectable container that:

* the **solver** mutates in place each step,
* the **renderer** reads (never writes), and
* tests can construct and assert against directly.

All arrays are ``float32`` for memory-bandwidth and pygame-``surfarray``
friendliness, except :attr:`obstacle`, which is a ``bool`` mask.

The dye field (:attr:`density`) has **three channels** (RGB). All three are
advected by the same velocity field, so coloured dye mixes and transports
realistically and the "cycle dye colour" control shows true colour rather than a
single-channel intensity tinted at draw time.

Scratch buffers (``*_prev``, :attr:`pressure`, :attr:`divergence`) are allocated
once and reused every step to avoid per-frame allocation churn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid

# Channel count for the dye field. Named for readability at call sites.
DYE_CHANNELS = 3


def _zeros(grid: Grid) -> np.ndarray:
    return np.zeros(grid.shape, dtype=np.float32)


def _zeros_rgb(grid: Grid) -> np.ndarray:
    return np.zeros((*grid.shape, DYE_CHANNELS), dtype=np.float32)


@dataclass(slots=True)
class FluidState:
    """All fields for one simulation, sized to ``grid``.

    Construct via :meth:`allocate`; the dataclass fields are filled in there.
    """

    grid: Grid

    # Primary fields ---------------------------------------------------------
    u: np.ndarray            # x-velocity,  shape (n+2, n+2)
    v: np.ndarray            # y-velocity,  shape (n+2, n+2)
    density: np.ndarray      # RGB dye,     shape (n+2, n+2, 3)
    obstacle: np.ndarray     # solid mask,  shape (n+2, n+2), bool

    # Per-step source buffers (filled by interaction, consumed by the solver) -
    u_src: np.ndarray
    v_src: np.ndarray
    density_src: np.ndarray

    # Reusable scratch (previous-state / linear-solver work arrays) -----------
    u_prev: np.ndarray
    v_prev: np.ndarray
    density_prev: np.ndarray
    pressure: np.ndarray
    divergence: np.ndarray

    @classmethod
    def allocate(cls, grid: Grid) -> FluidState:
        """Create a zero-initialised state for ``grid``."""
        return cls(
            grid=grid,
            u=_zeros(grid),
            v=_zeros(grid),
            density=_zeros_rgb(grid),
            obstacle=np.zeros(grid.shape, dtype=bool),
            u_src=_zeros(grid),
            v_src=_zeros(grid),
            density_src=_zeros_rgb(grid),
            u_prev=_zeros(grid),
            v_prev=_zeros(grid),
            density_prev=_zeros_rgb(grid),
            pressure=_zeros(grid),
            divergence=_zeros(grid),
        )

    # -- Source accumulation -------------------------------------------------
    # Sources accumulate between steps and are zeroed once consumed, so multiple
    # input events in a single frame add up rather than overwrite.

    def clear_sources(self) -> None:
        """Zero the per-step source buffers (called after they are consumed)."""
        self.u_src[:] = 0.0
        self.v_src[:] = 0.0
        self.density_src[:] = 0.0

    def reset_fluid(self) -> None:
        """Clear velocity and dye but keep obstacles (the ``C`` / clear action)."""
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.density[:] = 0.0
        self.clear_sources()

    # -- Read-only views for consumers --------------------------------------
    # Returning views with the writeable flag cleared lets the renderer read
    # live data with zero copies while making accidental writes raise.

    def density_view(self) -> np.ndarray:
        view = self.density.view()
        view.flags.writeable = False
        return view

    def velocity_view(self) -> tuple[np.ndarray, np.ndarray]:
        uv = self.u.view(), self.v.view()
        uv[0].flags.writeable = False
        uv[1].flags.writeable = False
        return uv

    def obstacle_view(self) -> np.ndarray:
        view = self.obstacle.view()
        view.flags.writeable = False
        return view
