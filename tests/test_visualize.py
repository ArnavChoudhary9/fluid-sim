"""Derived fields, colormaps, and the offline Visualizer."""

from __future__ import annotations

import numpy as np
import pytest

from fluidsim.config import SimConfig
from fluidsim.core import make_solver
from fluidsim.render.colormaps import apply_colormap, available, get_colormap
from fluidsim.render.visualize import (
    VIEWS,
    VisualConfig,
    Visualizer,
    pressure,
    speed,
    vorticity,
)


def _stirred_state(n: int = 32):
    solver = make_solver(SimConfig(n=n))
    solver.add_velocity(n // 2, n // 2, 40.0, 15.0, radius=4)
    solver.add_dye(n // 2, n // 2, (200.0, 50.0, 10.0), radius=4)
    solver.step(1 / 60)
    return solver.state


def test_derived_field_shapes() -> None:
    state = _stirred_state(32)
    assert vorticity(state).shape == (32, 32)
    assert speed(state).shape == (32, 32)
    assert pressure(state).shape == (32, 32)


def test_derived_fields_do_not_mutate_state() -> None:
    state = _stirred_state(32)
    before = (state.u.copy(), state.v.copy(), state.density.copy())
    vorticity(state)
    speed(state)
    pressure(state)
    assert np.array_equal(state.u, before[0])
    assert np.array_equal(state.v, before[1])
    assert np.array_equal(state.density, before[2])


def test_colormap_lut_is_well_formed() -> None:
    for name in available():
        lut = get_colormap(name)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8


def test_apply_colormap_maps_to_rgb() -> None:
    values = np.linspace(0.0, 1.0, 50)
    rgb = apply_colormap(values, "inferno")
    assert rgb.shape == (50, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize("view", VIEWS)
def test_visualizer_renders_each_view(view: str) -> None:
    state = _stirred_state(32)
    img = Visualizer(VisualConfig(view=view, output_size=(80, 60))).render(state)
    assert img.shape == (60, 80, 3)   # (height, width, channels)
    assert img.dtype == np.uint8


@pytest.mark.parametrize("overlay", ["arrows", "streamlines"])
def test_visualizer_overlays_render(overlay: str) -> None:
    state = _stirred_state(32)
    cfg = VisualConfig(view="speed", overlay=overlay, output_size=(64, 64))
    assert Visualizer(cfg).render(state).shape == (64, 64, 3)


def test_visualizer_draws_obstacles() -> None:
    solver = make_solver(SimConfig(n=32))
    solver.state.obstacle[10:20, 10:20] = True
    cfg = VisualConfig(view="speed", output_size=(64, 64), obstacle_color=(11, 22, 33))
    img = Visualizer(cfg).render(solver.state)
    assert ((img == (11, 22, 33)).all(axis=2)).any()


def test_invalid_visual_config_rejected() -> None:
    with pytest.raises(ValueError):
        VisualConfig(view="nope")
    with pytest.raises(ValueError):
        VisualConfig(overlay="nope")
    with pytest.raises(ValueError):
        get_colormap("not-a-map")
