"""Input mapping is pure and pygame-free, so we can test it directly."""

from __future__ import annotations

from fluidsim.config import BrushConfig
from fluidsim.interaction.brush import MAX_RADIUS, MIN_RADIUS, Brush
from fluidsim.interaction.commands import (
    AdjustBrush,
    ApplyForce,
    Clear,
    InjectDye,
    Quit,
    ToggleObstacle,
)
from fluidsim.interaction.input_map import (
    FrameInput,
    KeyAction,
    PointerState,
    translate,
)


def _pointer(**kw) -> PointerState:
    base = dict(x=10, y=10, dx=0.0, dy=0.0, left=False, right=False, middle=False)
    base.update(kw)
    return PointerState(**base)


def test_left_drag_injects_dye_and_force() -> None:
    brush = Brush(BrushConfig())
    frame = FrameInput(pointer=_pointer(left=True, dx=2.0, dy=-1.0))
    cmds = translate(frame, brush)
    assert any(isinstance(c, InjectDye) for c in cmds)
    force = next(c for c in cmds if isinstance(c, ApplyForce))
    assert force.fx > 0 and force.fy < 0


def test_right_and_middle_toggle_obstacles() -> None:
    brush = Brush(BrushConfig())
    add = translate(FrameInput(pointer=_pointer(right=True)), brush)
    erase = translate(FrameInput(pointer=_pointer(middle=True)), brush)
    assert any(isinstance(c, ToggleObstacle) and c.solid for c in add)
    assert any(isinstance(c, ToggleObstacle) and not c.solid for c in erase)


def test_pointer_outside_emits_no_simulation_commands() -> None:
    brush = Brush(BrushConfig())
    frame = FrameInput(pointer=_pointer(x=None, y=None, left=True), keys=[KeyAction.CLEAR])
    cmds = translate(frame, brush)
    assert [type(c) for c in cmds] == [Clear]  # key still works, no dye/force


def test_keys_map_to_commands() -> None:
    brush = Brush(BrushConfig())
    frame = FrameInput(pointer=_pointer(), keys=[KeyAction.QUIT, KeyAction.BRUSH_UP])
    cmds = translate(frame, brush)
    assert any(isinstance(c, Quit) for c in cmds)
    assert any(isinstance(c, AdjustBrush) and c.delta == 1 for c in cmds)


def test_brush_radius_is_clamped() -> None:
    brush = Brush(BrushConfig(radius=1))
    for _ in range(100):
        brush.adjust_radius(-1)
    assert brush.radius == MIN_RADIUS
    for _ in range(1000):
        brush.adjust_radius(1)
    assert brush.radius == MAX_RADIUS
