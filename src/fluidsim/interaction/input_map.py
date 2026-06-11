"""Translate normalised input into :mod:`commands` — with no pygame types.

The pygame backend converts raw device events into the toolkit-neutral structures
defined here (:class:`KeyAction`, :class:`PointerState`, :class:`FrameInput`) and
hands them to :func:`translate`. That function holds the *interaction logic* —
"left-drag paints dye and pushes the fluid", "right-drag adds an obstacle",
"middle-drag erases one" — expressed purely in grid coordinates. Because it
imports nothing from pygame, it is fully unit-testable without a display.

The physical key → :class:`KeyAction` binding lives in the backend (it owns the
device); here we only act on the semantic actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .brush import Brush
from .commands import (
    AdjustBrush,
    ApplyForce,
    Clear,
    Command,
    CycleColor,
    InjectDye,
    Quit,
    ToggleObstacle,
    ToggleOverlay,
    TogglePause,
)


class KeyAction(Enum):
    """Semantic, device-independent keyboard actions for one frame."""

    CLEAR = auto()
    PAUSE = auto()
    OVERLAY = auto()
    COLOR_NEXT = auto()
    COLOR_PREV = auto()
    BRUSH_UP = auto()
    BRUSH_DOWN = auto()
    QUIT = auto()


@dataclass(frozen=True, slots=True)
class PointerState:
    """The mouse this frame, in interior-grid coordinates.

    ``x``/``y`` are ``None`` when the pointer is outside the simulation area.
    ``dx``/``dy`` are the motion since the previous frame, in grid cells.
    """

    x: int | None
    y: int | None
    dx: float
    dy: float
    left: bool
    right: bool
    middle: bool


@dataclass(frozen=True, slots=True)
class FrameInput:
    """All input gathered for a single frame."""

    pointer: PointerState
    keys: list[KeyAction] = field(default_factory=list)


# Each key action maps to a zero-argument command factory.
_KEY_COMMANDS = {
    KeyAction.CLEAR: Clear,
    KeyAction.PAUSE: TogglePause,
    KeyAction.OVERLAY: ToggleOverlay,
    KeyAction.COLOR_NEXT: lambda: CycleColor(+1),
    KeyAction.COLOR_PREV: lambda: CycleColor(-1),
    KeyAction.BRUSH_UP: lambda: AdjustBrush(+1),
    KeyAction.BRUSH_DOWN: lambda: AdjustBrush(-1),
    KeyAction.QUIT: Quit,
}


def translate(frame: FrameInput, brush: Brush) -> list[Command]:
    """Map one frame of normalised input to a list of commands."""
    commands: list[Command] = [_KEY_COMMANDS[action]() for action in frame.keys]

    p = frame.pointer
    if p.x is None or p.y is None:
        return commands

    if p.left:
        # Scale the hue by the deposit gain; the solver then dt-scales it.
        dye = tuple(c * brush.dye_amount for c in brush.color)
        commands.append(InjectDye(p.x, p.y, dye, brush.radius))
        commands.append(
            ApplyForce(
                p.x,
                p.y,
                p.dx * brush.force_scale,
                p.dy * brush.force_scale,
                brush.radius,
            )
        )
    if p.right:
        commands.append(ToggleObstacle(p.x, p.y, brush.radius, solid=True))
    if p.middle:
        commands.append(ToggleObstacle(p.x, p.y, brush.radius, solid=False))

    return commands
