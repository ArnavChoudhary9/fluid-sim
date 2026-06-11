"""Built-in, ready-to-run experiments.

Each factory returns a fully-configured :class:`Scene`. They double as worked
examples of the engine's features: configurable boundary conditions, obstacles,
buoyancy, periodicity, and initial conditions. Tweak the parameters or copy a
factory as the starting point for your own experiment.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Backend, BCType, BoundaryConditions, SimConfig
from .scene import Scene
from .shapes import Circle, Rectangle, Shape
from .sources import DyeSource

# A few pleasant dye colours (RGB, 0-255).
_RED = (235.0, 70.0, 50.0)
_BLUE = (60.0, 160.0, 235.0)
_AMBER = (245.0, 170.0, 40.0)
_TEAL = (40.0, 220.0, 200.0)


def wind_tunnel(
    n: int = 256,
    speed: float = 2.4,
    obstacle: Shape | None = None,
    backend: Backend = "auto",
) -> Scene:
    """Uniform inflow past an obstacle in a channel — the Kármán vortex street.

    Inflow on the left, outflow on the right, solid walls top and bottom, and a
    bluff body in the stream. Coloured dye streaklines released at the inlet make
    the wake and shed vortices visible.
    """
    bc = BoundaryConditions(
        left=BCType.INFLOW,
        right=BCType.OUTFLOW,
        top=BCType.WALL,
        bottom=BCType.WALL,
        inflow_velocity=(speed, 0.0),
    )
    # sim = SimConfig(n=n, backend=backend, boundary=bc, viscosity=2e-6, iterations=24)
    sim = SimConfig(n=n, backend=backend, boundary=bc, viscosity=1e-3, iterations=32)
    body = obstacle if obstacle is not None else Circle(0.22, 0.5, 0.06)

    # Release thin dye ribbons across the inlet, alternating colour, as tracers.
    colors = (_RED, _AMBER, _BLUE, _TEAL)
    ribbons = tuple(
        DyeSource(Rectangle(0.0, y - 0.006, 0.02, y + 0.006), colors[i % len(colors)], rate=40.0)
        for i, y in enumerate(np.linspace(0.12, 0.88, 9))
    )

    def initial(grid: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full((grid, grid), speed, np.float32), np.zeros((grid, grid), np.float32)

    return Scene(
        name="wind_tunnel",
        sim=sim,
        obstacles=(body,),
        dye_sources=ribbons,
        initial_velocity=initial,
        default_view="vorticity",
        description="Uniform flow past a bluff body; vortex shedding in the wake.",
    )


def lid_driven_cavity(n: int = 160, lid_speed: float = 1.5, backend: Backend = "auto") -> Scene:
    """The classic CFD benchmark: a closed box whose top wall slides sideways.

    The moving lid drags the fluid into a single large recirculating vortex.
    Visualise with vorticity or the seeded dye stripes.
    """
    bc = BoundaryConditions(
        left=BCType.WALL,
        right=BCType.WALL,
        top=BCType.WALL,
        bottom=BCType.WALL,
        wall_velocity=(0.0, 0.0, lid_speed, 0.0),  # top wall moves in +x
    )
    sim = SimConfig(n=n, backend=backend, boundary=bc, viscosity=1e-4, iterations=30, fade=1.0)

    def initial_dye(grid: int) -> np.ndarray:
        dye = np.zeros((grid, grid, 3), np.float32)
        rows = (np.arange(grid) // max(1, grid // 10)) % 2 == 0
        stripe = np.broadcast_to(rows[:, None], (grid, grid))
        dye[stripe] = _TEAL
        dye[~stripe] = _RED
        return dye * 0.6

    return Scene(
        name="lid_driven_cavity",
        sim=sim,
        initial_dye=initial_dye,
        default_view="vorticity",
        description="Lid-driven cavity; the canonical steady recirculation benchmark.",
    )


def smoke_plume(n: int = 176, buoyancy: float = 6.0, backend: Backend = "auto") -> Scene:
    """A buoyant smoke column rising and billowing from a hot source.

    Walls on the sides and floor, open at the top so smoke escapes. A continuous
    dye source at the base feeds the plume; the buoyancy force lifts it.
    """
    bc = BoundaryConditions(
        left=BCType.WALL,
        right=BCType.WALL,
        top=BCType.OUTFLOW,
        bottom=BCType.WALL,
    )
    sim = SimConfig(
        n=n, backend=backend, boundary=bc, buoyancy=buoyancy, viscosity=8e-7, iterations=20
    )
    emitter = DyeSource(Circle(0.5, 0.9, 0.05), _AMBER, rate=60.0)

    return Scene(
        name="smoke_plume",
        sim=sim,
        dye_sources=(emitter,),
        default_view="dye",
        description="Buoyant plume rising from a heat source; open top boundary.",
    )


def shear_layer(n: int = 200, speed: float = 1.4, backend: Backend = "auto") -> Scene:
    """A spatially-developing mixing layer that rolls up into KH vortices.

    Two streams enter side-by-side at different speeds (fast on top, slow on the
    bottom); the shear between them is unstable and rolls up into a train of
    Kelvin–Helmholtz vortices that grow downstream before leaving the outflow.

    This *spatial* (continuously-forced) setup is far more robust than a purely
    temporal one: a basic semi-Lagrangian solver is too diffusive to sustain the
    decaying perturbation of a periodic shear layer, but a sustained inlet shear
    keeps feeding the instability. The inlet streams are maintained each step by a
    driver, with a gentle oscillation at the interface to seed the roll-up.
    """
    fast, slow = speed * 1.6, speed * 0.4
    bc = BoundaryConditions(
        left=BCType.INFLOW,
        right=BCType.OUTFLOW,
        top=BCType.WALL,
        bottom=BCType.WALL,
        inflow_velocity=((fast + slow) / 2, 0.0),
    )
    sim = SimConfig(n=n, backend=backend, boundary=bc, viscosity=6e-7, iterations=22, fade=0.999)

    def initial_velocity(grid: int) -> tuple[np.ndarray, np.ndarray]:
        ys = (np.arange(grid)[:, None] + 0.5) / grid
        u = np.where(ys < 0.5, fast, slow) * np.ones((1, grid))
        return u.astype(np.float32), np.zeros((grid, grid), np.float32)

    def drive_inlet(solver, time: float, dt: float) -> None:
        ys = (np.arange(n)[:, None] + 0.5) / n
        strip = np.zeros((n, n), dtype=bool)
        strip[:, : max(2, n // 18)] = True          # thin inlet column
        top = np.broadcast_to(ys < 0.5, (n, n))
        seed = np.broadcast_to(np.abs(ys - 0.5) < 0.05, (n, n)) & strip
        # Sustain the two streams, then inject a strong oscillating vertical
        # velocity right at the interface (per-stream, so u stays fast/slow). The
        # kick must be vigorous to roll the layer up before diffusion smooths it.
        v_seed = 0.25 * speed * math.sin(time * 6.0)
        solver.set_velocity_region(strip & top, fast, 0.0)
        solver.set_velocity_region(strip & ~top, slow, 0.0)
        solver.set_velocity_region(seed & top, fast, v_seed)
        solver.set_velocity_region(seed & ~top, slow, v_seed)

    # Thin tracer ribbons either side of the interface so the roll-up is visible.
    ribbons = (
        DyeSource(Rectangle(0.0, 0.45, 0.03, 0.49), _BLUE, rate=18.0),
        DyeSource(Rectangle(0.0, 0.51, 0.03, 0.55), _AMBER, rate=18.0),
    )

    return Scene(
        name="shear_layer",
        sim=sim,
        dye_sources=ribbons,
        initial_velocity=initial_velocity,
        drivers=(drive_inlet,),
        default_view="vorticity",
        description="Spatial mixing layer; Kelvin–Helmholtz vortices roll up downstream.",
    )


# Registry so the CLI / tests can look scenes up by name.
SCENES = {
    "wind_tunnel": wind_tunnel,
    "lid_driven_cavity": lid_driven_cavity,
    "smoke_plume": smoke_plume,
    "shear_layer": shear_layer,
}
