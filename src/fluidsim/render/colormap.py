"""Dye palette and colour cycling.

The dye field already carries real RGB colour (three advected channels), so the
renderer does not need a colormap to *tint* a scalar. What it needs is a small
palette to pick the colour the brush currently injects, and a way to cycle
through it. That tiny, UI-free responsibility lives here so neither the solver
nor the pygame layer owns colour state.
"""

from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(slots=True)
class Palette:
    """An ordered set of dye colours with a movable cursor."""

    colors: tuple[Color, ...]
    index: int = 0

    def __post_init__(self) -> None:
        if not self.colors:
            raise ValueError("palette needs at least one colour")
        self.index %= len(self.colors)

    @property
    def current(self) -> Color:
        """The colour the brush injects right now (RGB, 0-255)."""
        return self.colors[self.index]

    def cycle(self, step: int = 1) -> Color:
        """Advance the cursor by ``step`` (wraps around) and return the colour."""
        self.index = (self.index + step) % len(self.colors)
        return self.current
