"""Immutable configuration for the fluid simulation.

Configuration is split into small, frozen dataclasses so that every consumer
receives exactly the knobs it needs and *cannot* mutate shared global state
(principle of least privilege). Genuinely mutable runtime state — the current
brush size, the paused flag, the active dye colour — does **not** live here; it
lives in :mod:`fluidsim.interaction.brush` and in the main loop.

All validation happens in ``__post_init__`` so that an invalid configuration
fails loudly at construction time rather than producing silent NaNs deep inside
the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# A backend is selected by name. ``auto`` prefers Numba when it is importable
# and transparently falls back to the pure-NumPy reference solver otherwise.
Backend = Literal["numpy", "numba", "auto"]

# Tangential velocity treatment at obstacle surfaces. ``free`` (free-slip) lets
# fluid glide along walls; ``no`` (no-slip) makes it stick. See
# docs/src/math-obstacles.md.
SlipMode = Literal["free", "no"]


class BCType(Enum):
    """Condition applied at one edge of the domain.

    * ``WALL`` — a solid, closed wall: no flow through it. The default, and what
      the interactive app uses on all four sides.
    * ``INFLOW`` — fluid enters at a prescribed velocity (Dirichlet). The wind
      tunnel's upstream edge.
    * ``OUTFLOW`` — fluid leaves freely (zero-gradient / Neumann), with the
      pressure pinned so mass can exit. The wind tunnel's downstream edge.
    * ``PERIODIC`` — the edge wraps to the opposite edge. Must be set on *both*
      edges of an axis (left+right, or top+bottom).

    See docs/src/cfd-boundary-conditions.md.
    """

    WALL = "wall"
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    PERIODIC = "periodic"


@dataclass(frozen=True, slots=True)
class BoundaryConditions:
    """Per-edge boundary conditions for the domain.

    Attributes
    ----------
    left, right, top, bottom:
        The :class:`BCType` for each edge.
    inflow_velocity:
        ``(vx, vy)`` imposed at every ``INFLOW`` edge.
    wall_velocity:
        Tangential wall speeds ``(left, right, top, bottom)``. A non-zero value
        turns the corresponding ``WALL`` into a *moving* wall dragging the fluid
        along it (used by the lid-driven-cavity scene). For the left/right edges
        the value is a y-velocity; for top/bottom it is an x-velocity.
    """

    left: BCType = BCType.WALL
    right: BCType = BCType.WALL
    top: BCType = BCType.WALL
    bottom: BCType = BCType.WALL
    inflow_velocity: tuple[float, float] = (0.0, 0.0)
    wall_velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if (self.left is BCType.PERIODIC) != (self.right is BCType.PERIODIC):
            raise ValueError("PERIODIC must be set on both left and right edges")
        if (self.top is BCType.PERIODIC) != (self.bottom is BCType.PERIODIC):
            raise ValueError("PERIODIC must be set on both top and bottom edges")

    @property
    def is_static_walls(self) -> bool:
        """True when every edge is a stationary closed wall (the fast path)."""
        return (
            self.left is BCType.WALL
            and self.right is BCType.WALL
            and self.top is BCType.WALL
            and self.bottom is BCType.WALL
            and self.wall_velocity == (0.0, 0.0, 0.0, 0.0)
        )

    @property
    def all_pressure_neumann(self) -> bool:
        """True when no edge needs a special pressure condition (all WALL/INFLOW)."""
        return all(
            edge in (BCType.WALL, BCType.INFLOW)
            for edge in (self.left, self.right, self.top, self.bottom)
        )


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Physics parameters for the Stable-Fluids solver.

    Attributes
    ----------
    n:
        Logical grid resolution. The solver allocates ``(n + 2, n + 2)`` arrays
        so that a one-cell *ghost* border can hold boundary values.
    dt:
        Fixed simulation timestep, in seconds. The app loop integrates with a
        fixed-``dt`` accumulator for deterministic, machine-independent results.
    viscosity:
        Kinematic viscosity ``nu``. Higher values smear momentum faster.
    diffusion:
        Dye diffusion coefficient. Higher values blur the dye faster.
    iterations:
        Gauss-Seidel sweeps used for both the diffusion and the pressure-Poisson
        solves. More sweeps → better convergence, slower frames.
    backend:
        Which solver implementation to construct (see :data:`Backend`).
    obstacle_slip:
        Tangential boundary condition at obstacle surfaces (see :data:`SlipMode`).
    fade:
        Per-step multiplicative decay applied to dye (``1.0`` = no fade). A small
        amount of fade keeps the screen from saturating during long sessions.
    boundary:
        Per-edge :class:`BoundaryConditions`. Defaults to closed walls on all
        sides (the interactive app); scenes override this for inflow/outflow etc.
    buoyancy:
        Strength of a dye-driven buoyancy force. ``0`` disables it; positive
        values make dye rise (used by the smoke-plume scene).
    """

    n: int = 128
    dt: float = 1.0 / 60.0
    viscosity: float = 1.0e-6
    diffusion: float = 1.0e-6
    iterations: int = 20
    backend: Backend = "auto"
    obstacle_slip: SlipMode = "free"
    fade: float = 0.999
    boundary: BoundaryConditions = field(default_factory=BoundaryConditions)
    buoyancy: float = 0.0

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError(f"grid size n must be >= 4, got {self.n}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.viscosity < 0.0:
            raise ValueError(f"viscosity must be >= 0, got {self.viscosity}")
        if self.diffusion < 0.0:
            raise ValueError(f"diffusion must be >= 0, got {self.diffusion}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.backend not in ("numpy", "numba", "auto"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.obstacle_slip not in ("free", "no"):
            raise ValueError(f"unknown obstacle_slip {self.obstacle_slip!r}")
        if not 0.0 < self.fade <= 1.0:
            raise ValueError(f"fade must be in (0, 1], got {self.fade}")
        if not isinstance(self.boundary, BoundaryConditions):
            raise ValueError("boundary must be a BoundaryConditions instance")


# A small, pleasant default dye palette (RGB, 0-255). Cycled with Tab / [ / ].
DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (235, 64, 52),    # red
    (52, 165, 235),   # blue
    (95, 235, 52),    # green
    (235, 196, 52),   # amber
    (197, 52, 235),   # magenta
    (255, 255, 255),  # white
)


@dataclass(frozen=True, slots=True)
class BrushConfig:
    """How the mouse injects dye and force into the fluid.

    Both strengths are *gains*: the solver multiplies the staged source by ``dt``
    each step (so behaviour is frame-rate independent). The per-step dye added to
    a channel is therefore ``dt · dye_amount · colour``, and the per-step velocity
    is ``dt · force_scale · mouse_motion``.
    """

    radius: int = 4            # brush footprint in grid cells
    dye_amount: float = 8.0    # dye deposit gain (build-up speed under the cursor)
    force_scale: float = 90.0  # velocity gain per unit of mouse motion
    palette: tuple[tuple[int, int, int], ...] = DEFAULT_PALETTE

    def __post_init__(self) -> None:
        if self.radius < 1:
            raise ValueError(f"brush radius must be >= 1, got {self.radius}")
        if not self.palette:
            raise ValueError("palette must contain at least one colour")


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """How fields are turned into pixels and shown on screen."""

    window_size: tuple[int, int] = (768, 768)
    show_overlay: bool = False                 # velocity-field overlay on/off
    overlay_stride: int = 8                     # sample every Nth cell for arrows
    overlay_color: tuple[int, int, int] = (255, 255, 255)
    obstacle_color: tuple[int, int, int] = (70, 70, 78)
    background: tuple[int, int, int] = (0, 0, 0)
    title: str = "fluidsim — Stable Fluids"

    def __post_init__(self) -> None:
        w, h = self.window_size
        if w < 64 or h < 64:
            raise ValueError(f"window must be at least 64x64, got {self.window_size}")
        if self.overlay_stride < 1:
            raise ValueError(f"overlay_stride must be >= 1, got {self.overlay_stride}")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration: the three sub-configs bundled together."""

    sim: SimConfig = field(default_factory=SimConfig)
    brush: BrushConfig = field(default_factory=BrushConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    target_fps: int = 60

    def __post_init__(self) -> None:
        if self.target_fps < 1:
            raise ValueError(f"target_fps must be >= 1, got {self.target_fps}")
