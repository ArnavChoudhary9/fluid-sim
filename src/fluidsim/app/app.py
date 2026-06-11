"""Composition root: build every part from config and run the loop.

This is the one place that knows about *all* the subsystems at once. It performs
dependency injection — constructing the solver, renderer, brush, and pygame
backend from an :class:`~fluidsim.config.AppConfig` and handing them to the
:class:`~fluidsim.app.loop.SimulationLoop` — and guarantees the window is torn
down even if the loop raises.
"""

from __future__ import annotations

from ..config import AppConfig
from ..core.factory import make_solver
from ..interaction.brush import Brush
from ..render.renderer import Renderer
from .loop import SimulationLoop
from .pygame_backend import PygameBackend


def run(config: AppConfig | None = None) -> None:
    """Launch the interactive simulation. Blocks until the user quits."""
    config = config or AppConfig()

    solver = make_solver(config.sim)
    renderer = Renderer(config.render)
    brush = Brush(config.brush)
    backend = PygameBackend(config.render, config.sim.n)

    loop = SimulationLoop(config, solver, renderer, brush, backend)
    try:
        loop.run()
    finally:
        backend.quit()
