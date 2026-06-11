"""Continuous emitters that drive a simulation each step.

A source injects something — dye or momentum — into a region of the domain every
timestep. They are the programmatic equivalent of holding the mouse down in one
spot. Sources only ever call the solver's public injection API, so they hold no
special privilege over the simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.solver_base import BaseSolver
from .shapes import Shape

Color = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DyeSource:
    """Continuously deposit dye of ``color`` into ``region``.

    ``rate`` is a deposit gain (like the brush's ``dye_amount``): the per-step
    dye added is ``dt · rate · color``.
    """

    region: Shape
    color: Color
    rate: float = 1.0

    def apply(self, solver: BaseSolver, n: int) -> None:
        scaled = (self.color[0] * self.rate, self.color[1] * self.rate, self.color[2] * self.rate)
        solver.add_dye_region(self.region.mask(n), scaled)


@dataclass(frozen=True, slots=True)
class ForceSource:
    """Continuously apply a velocity impulse ``(fx, fy)`` into ``region``."""

    region: Shape
    fx: float
    fy: float

    def apply(self, solver: BaseSolver, n: int) -> None:
        solver.add_velocity_region(self.region.mask(n), self.fx, self.fy)
