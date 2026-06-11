"""Mutable runtime brush state.

Configuration (:class:`~fluidsim.config.BrushConfig`) is immutable; the *current*
brush — its radius and the active dye colour — genuinely changes at runtime in
response to input, so it lives here rather than in the frozen config. The brush
owns no I/O and no solver reference: it is pure state plus a couple of bounded
mutators.
"""

from __future__ import annotations

from ..config import BrushConfig
from ..render.colormap import Color, Palette

# Brush radius is clamped to this range so the user cannot shrink it to nothing
# or grow it past anything useful.
MIN_RADIUS = 1
MAX_RADIUS = 40


class Brush:
    """The live brush: radius, dye colour, and the force/dye strengths."""

    def __init__(self, config: BrushConfig) -> None:
        self._radius = config.radius
        self._palette = Palette(config.palette)
        self.dye_amount = config.dye_amount
        self.force_scale = config.force_scale

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def color(self) -> Color:
        """The colour the brush injects right now (RGB, 0-255)."""
        return self._palette.current

    def adjust_radius(self, delta: int) -> int:
        """Grow or shrink the brush, clamped to ``[MIN_RADIUS, MAX_RADIUS]``."""
        self._radius = max(MIN_RADIUS, min(MAX_RADIUS, self._radius + delta))
        return self._radius

    def cycle_color(self, step: int = 1) -> Color:
        """Move to the next/previous palette colour."""
        return self._palette.cycle(step)
