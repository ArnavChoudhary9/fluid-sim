"""A *scene* is a complete, reproducible description of an experiment.

It bundles the physics config, the obstacles, the continuous sources, and the
initial conditions. A :class:`Simulation` turns a scene into a running solver you
can step — headlessly, or feed to the video renderer. This is the programmatic
front door to the engine.

Like every consumer of the core, this module imports the solver only through its
public interface; it never touches pygame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..core import make_solver
from ..core.solver_base import BaseSolver
from .shapes import Shape, rasterize
from .sources import DyeSource, ForceSource

# Callables that build initial fields for a given grid size ``n``.
VelocityInit = Callable[[int], tuple[np.ndarray, np.ndarray]]  # -> (u, v), each (n, n)
DyeInit = Callable[[int], np.ndarray]                          # -> (n, n, 3)
# A per-step hook for bespoke behaviour: ``driver(solver, time, dt)``.
Driver = Callable[[BaseSolver, float, float], None]


@dataclass(frozen=True, slots=True)
class Scene:
    """A self-contained, reproducible simulation setup."""

    name: str
    sim: SimConfig
    obstacles: tuple[Shape, ...] = ()
    dye_sources: tuple[DyeSource, ...] = ()
    force_sources: tuple[ForceSource, ...] = ()
    initial_velocity: VelocityInit | None = None
    initial_dye: DyeInit | None = None
    drivers: tuple[Driver, ...] = ()
    # Suggested visualisation field for this scene (consumed by the recorder).
    default_view: str = "dye"
    description: str = ""


class Simulation:
    """A running scene: owns the solver, applies sources, and advances time."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.dt = scene.sim.dt
        self.time = 0.0
        self.solver = make_solver(scene.sim)
        self._apply_initial_conditions()

    def _apply_initial_conditions(self) -> None:
        n = self.scene.sim.n
        if self.scene.obstacles:
            self.solver.set_obstacle_mask(rasterize(self.scene.obstacles, n))
        if self.scene.initial_velocity is not None:
            u, v = self.scene.initial_velocity(n)
            self.solver.set_velocity_field(u.astype(np.float32), v.astype(np.float32))
        if self.scene.initial_dye is not None:
            self.solver.set_dye_field(self.scene.initial_dye(n).astype(np.float32))

    def step(self) -> None:
        """Apply all sources and drivers, then advance the solver one timestep."""
        n = self.scene.sim.n
        for dye in self.scene.dye_sources:
            dye.apply(self.solver, n)
        for force in self.scene.force_sources:
            force.apply(self.solver, n)
        for driver in self.scene.drivers:
            driver(self.solver, self.time, self.dt)
        self.solver.step(self.dt)
        self.time += self.dt

    def run(self, steps: int) -> None:
        """Advance ``steps`` timesteps (useful for warm-up before recording)."""
        for _ in range(steps):
            self.step()

    @property
    def state(self):
        """Read-only access to the live :class:`FluidState` (for visualisation)."""
        return self.solver.state
