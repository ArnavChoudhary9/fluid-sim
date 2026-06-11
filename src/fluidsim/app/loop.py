"""The main loop — the mediator that wires solver, renderer, and input together.

The loop is the only actor that holds references to all three subsystems, and it
is deliberately thin: it gathers input, *interprets commands* (calling the
solver's source-injection API — interaction never touches the solver itself),
integrates the physics on a **fixed-timestep accumulator**, renders, and presents.

Fixed-``dt`` integration (rather than stepping by the variable frame time) keeps
the simulation deterministic and independent of frame rate; a substep cap stops
the "spiral of death" if a frame ever takes far too long.
"""

from __future__ import annotations

from ..config import AppConfig
from ..core.solver_base import BaseSolver
from ..interaction.brush import Brush
from ..interaction.commands import (
    AdjustBrush,
    ApplyForce,
    Clear,
    CycleColor,
    InjectDye,
    Quit,
    ToggleObstacle,
    ToggleOverlay,
    TogglePause,
)
from ..interaction.input_map import translate
from ..render.overlay import velocity_segments
from ..render.renderer import Renderer
from .pygame_backend import PygameBackend

# Never run more than this many physics substeps per rendered frame.
MAX_SUBSTEPS = 5


class SimulationLoop:
    """Owns per-run state (paused, overlay) and drives the simulation."""

    def __init__(
        self,
        config: AppConfig,
        solver: BaseSolver,
        renderer: Renderer,
        brush: Brush,
        backend: PygameBackend,
    ) -> None:
        self._config = config
        self._solver = solver
        self._renderer = renderer
        self._brush = brush
        self._backend = backend

        self._paused = False
        self._show_overlay = config.render.show_overlay
        self._accumulator = 0.0

    def run(self) -> None:
        """Block until the user quits."""
        dt = self._config.sim.dt
        running = True
        while running:
            elapsed = self._backend.tick(self._config.target_fps)
            frame = self._backend.poll()
            running = self._apply(translate(frame, self._brush))

            if not self._paused:
                self._integrate(elapsed, dt)

            self._present()

    # -- Internals -----------------------------------------------------------

    def _integrate(self, elapsed: float, dt: float) -> None:
        self._accumulator += elapsed
        steps = 0
        while self._accumulator >= dt and steps < MAX_SUBSTEPS:
            self._solver.step(dt)
            self._accumulator -= dt
            steps += 1
        if steps == MAX_SUBSTEPS:
            self._accumulator = 0.0  # we are behind; drop the backlog

    def _present(self) -> None:
        image = self._renderer.render(self._solver.state)
        segments = []
        if self._show_overlay:
            segments = velocity_segments(
                self._solver.state,
                stride=self._config.render.overlay_stride,
                scale=self._config.render.overlay_stride * 1.5,
            )
        self._backend.present(image, segments)
        self._backend.set_title(self._status())

    def _status(self) -> str:
        state = "paused" if self._paused else "running"
        overlay = "on" if self._show_overlay else "off"
        return (
            f"{self._config.render.title}  |  {state}  "
            f"|  brush {self._brush.radius}  |  overlay {overlay}"
        )

    def _apply(self, commands) -> bool:
        """Interpret commands; return ``False`` once a Quit is seen."""
        solver, brush = self._solver, self._brush
        for cmd in commands:
            if isinstance(cmd, InjectDye):
                solver.add_dye(cmd.cx, cmd.cy, cmd.color, cmd.radius)
            elif isinstance(cmd, ApplyForce):
                solver.add_velocity(cmd.cx, cmd.cy, cmd.fx, cmd.fy, cmd.radius)
            elif isinstance(cmd, ToggleObstacle):
                solver.set_obstacle(cmd.cx, cmd.cy, cmd.radius, cmd.solid)
            elif isinstance(cmd, Clear):
                solver.clear()
            elif isinstance(cmd, TogglePause):
                self._paused = not self._paused
            elif isinstance(cmd, ToggleOverlay):
                self._show_overlay = not self._show_overlay
            elif isinstance(cmd, CycleColor):
                brush.cycle_color(cmd.step)
            elif isinstance(cmd, AdjustBrush):
                brush.adjust_radius(cmd.delta)
            elif isinstance(cmd, Quit):
                return False
        return True
