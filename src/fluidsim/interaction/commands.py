"""Typed commands — the vocabulary between input and the rest of the app.

The interaction layer never touches the solver, the renderer, or loop state
directly. Instead it emits these small, immutable command objects, and the main
loop interprets them (calling solver source-injection methods, toggling flags,
etc.). This indirection is what lets input be unit-tested with no solver and no
display, and keeps every actor holding only the capability it needs.

Commands fall into two groups:

* *Simulation* commands carry grid-space coordinates and are applied to the
  solver (:class:`InjectDye`, :class:`ApplyForce`, :class:`ToggleObstacle`,
  :class:`Clear`).
* *Application* commands change view/loop state (:class:`TogglePause`,
  :class:`ToggleOverlay`, :class:`CycleColor`, :class:`AdjustBrush`,
  :class:`Quit`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InjectDye:
    cx: int
    cy: int
    color: tuple[float, float, float]  # hue already scaled by the deposit gain
    radius: int


@dataclass(frozen=True, slots=True)
class ApplyForce:
    cx: int
    cy: int
    fx: float
    fy: float
    radius: int


@dataclass(frozen=True, slots=True)
class ToggleObstacle:
    cx: int
    cy: int
    radius: int
    solid: bool


@dataclass(frozen=True, slots=True)
class Clear:
    """Reset velocity and dye (obstacles preserved)."""


@dataclass(frozen=True, slots=True)
class TogglePause:
    """Pause or resume the physics integration."""


@dataclass(frozen=True, slots=True)
class ToggleOverlay:
    """Show or hide the velocity-field overlay."""


@dataclass(frozen=True, slots=True)
class CycleColor:
    step: int = 1


@dataclass(frozen=True, slots=True)
class AdjustBrush:
    delta: int


@dataclass(frozen=True, slots=True)
class Quit:
    """Request application shutdown."""


# A convenient union for type hints / dispatch.
Command = (
    InjectDye
    | ApplyForce
    | ToggleObstacle
    | Clear
    | TogglePause
    | ToggleOverlay
    | CycleColor
    | AdjustBrush
    | Quit
)
