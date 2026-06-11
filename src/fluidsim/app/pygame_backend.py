"""The pygame edge of the application — the only module that imports pygame.

It does three things and nothing else:

1. Owns the window, the clock, and the display surface.
2. Pumps raw pygame events and converts them into the toolkit-neutral
   :class:`~fluidsim.interaction.input_map.FrameInput` (this is where the
   physical key → :class:`KeyAction` binding lives).
3. Presents a rendered ``(n, n, 3)`` image plus velocity-overlay segments,
   handling the ``surfarray`` axis transpose and the scale-up to the window.

Confining pygame here is what lets the solver, renderer, and interaction layers
stay testable without a display.
"""

from __future__ import annotations

import pygame

from ..config import RenderConfig
from ..interaction.input_map import FrameInput, KeyAction, PointerState
from ..render.overlay import Segment

# Physical key → semantic action. Changing a binding is a one-line edit here and
# touches nothing else in the app.
_KEY_BINDINGS: dict[int, KeyAction] = {
    pygame.K_c: KeyAction.CLEAR,
    pygame.K_SPACE: KeyAction.PAUSE,
    pygame.K_v: KeyAction.OVERLAY,
    pygame.K_TAB: KeyAction.COLOR_NEXT,
    pygame.K_RIGHTBRACKET: KeyAction.COLOR_NEXT,
    pygame.K_LEFTBRACKET: KeyAction.COLOR_PREV,
    pygame.K_EQUALS: KeyAction.BRUSH_UP,
    pygame.K_PLUS: KeyAction.BRUSH_UP,
    pygame.K_KP_PLUS: KeyAction.BRUSH_UP,
    pygame.K_MINUS: KeyAction.BRUSH_DOWN,
    pygame.K_KP_MINUS: KeyAction.BRUSH_DOWN,
    pygame.K_ESCAPE: KeyAction.QUIT,
}


class PygameBackend:
    """Window, input pump, and presentation for the simulation."""

    def __init__(self, config: RenderConfig, grid_n: int) -> None:
        self._config = config
        self._n = grid_n
        self._overlay_color = config.overlay_color

        pygame.init()
        self._width, self._height = config.window_size
        self._screen = pygame.display.set_mode(config.window_size)
        pygame.display.set_caption(config.title)
        self._clock = pygame.time.Clock()

        # Small surface at grid resolution; scaled up each frame for crisp speed.
        self._grid_surface = pygame.Surface((grid_n, grid_n))
        self._prev_pointer: tuple[float, float] | None = None

    # -- Timing --------------------------------------------------------------

    def tick(self, fps: int) -> float:
        """Advance the clock; return real seconds elapsed since the last call."""
        return self._clock.tick(fps) / 1000.0

    def set_title(self, text: str) -> None:
        pygame.display.set_caption(text)

    # -- Input ---------------------------------------------------------------

    def poll(self) -> FrameInput:
        """Drain the event queue and snapshot the pointer into a FrameInput."""
        actions: list[KeyAction] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(KeyAction.QUIT)
            elif event.type == pygame.KEYDOWN:
                action = _KEY_BINDINGS.get(event.key)
                if action is not None:
                    actions.append(action)

        pointer = self._pointer_state()
        return FrameInput(pointer=pointer, keys=actions)

    def _pointer_state(self) -> PointerState:
        px, py = pygame.mouse.get_pos()
        left, middle, right = pygame.mouse.get_pressed(num_buttons=3)

        inside = 0 <= px < self._width and 0 <= py < self._height
        if not inside:
            self._prev_pointer = None
            return PointerState(None, None, 0.0, 0.0, bool(left), bool(right), bool(middle))

        # Continuous grid coordinates (cell centres at integer indices 1..n).
        cgx = px / self._width * self._n + 0.5
        cgy = py / self._height * self._n + 0.5
        if self._prev_pointer is None:
            dx = dy = 0.0
        else:
            dx = cgx - self._prev_pointer[0]
            dy = cgy - self._prev_pointer[1]
        self._prev_pointer = (cgx, cgy)

        gx = min(self._n, max(1, int(cgx)))
        gy = min(self._n, max(1, int(cgy)))
        return PointerState(gx, gy, dx, dy, bool(left), bool(right), bool(middle))

    # -- Presentation --------------------------------------------------------

    def present(self, image, segments: list[Segment]) -> None:
        """Blit a grid-resolution image (scaled up) plus overlay segments."""
        # surfarray expects [x, y, c]; our image is [y, x, c]. Transpose once.
        pygame.surfarray.blit_array(self._grid_surface, image.transpose(1, 0, 2))
        scaled = pygame.transform.scale(self._grid_surface, (self._width, self._height))
        self._screen.blit(scaled, (0, 0))

        for (x0, y0), (x1, y1) in segments:
            pygame.draw.line(
                self._screen,
                self._overlay_color,
                self._to_pixel(x0, y0),
                self._to_pixel(x1, y1),
            )

        pygame.display.flip()

    def _to_pixel(self, gx: float, gy: float) -> tuple[int, int]:
        px = (gx - 0.5) * self._width / self._n
        py = (gy - 0.5) * self._height / self._n
        return int(px), int(py)

    def quit(self) -> None:
        pygame.quit()
