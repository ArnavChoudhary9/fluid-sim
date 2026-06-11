"""The renderer is a pure function of state and needs no display."""

from __future__ import annotations

import numpy as np

from fluidsim.config import RenderConfig, SimConfig
from fluidsim.core import make_solver
from fluidsim.render.colormap import Palette
from fluidsim.render.overlay import velocity_segments
from fluidsim.render.renderer import Renderer


def test_render_shape_and_dtype() -> None:
    n = 16
    solver = make_solver(SimConfig(n=n))
    image = Renderer(RenderConfig()).render(solver.state)
    assert image.shape == (n, n, 3)
    assert image.dtype == np.uint8


def test_obstacles_render_as_their_colour() -> None:
    n = 16
    solver = make_solver(SimConfig(n=n))
    solver.state.obstacle[3:6, 3:6] = True
    cfg = RenderConfig(obstacle_color=(10, 20, 30))
    image = Renderer(cfg).render(solver.state)
    # Interior obstacle cell (grid 3..5) maps to image index 2..4.
    assert tuple(image[3, 3]) == (10, 20, 30)


def test_dye_is_clamped_to_255() -> None:
    n = 16
    solver = make_solver(SimConfig(n=n))
    solver.state.density[5, 5, :] = 9999.0
    image = Renderer(RenderConfig(background=(0, 0, 0))).render(solver.state)
    assert image[4, 4, 0] == 255


def test_palette_cycles() -> None:
    pal = Palette(((1, 1, 1), (2, 2, 2), (3, 3, 3)))
    assert pal.current == (1, 1, 1)
    assert pal.cycle() == (2, 2, 2)
    assert pal.cycle(-2) == (3, 3, 3)  # wraps around


def test_overlay_segments_are_data() -> None:
    n = 16
    solver = make_solver(SimConfig(n=n))
    solver.state.u[1:-1, 1:-1] = 1.0
    segs = velocity_segments(solver.state, stride=4, scale=2.0)
    assert segs and all(len(s) == 2 for s in segs)
