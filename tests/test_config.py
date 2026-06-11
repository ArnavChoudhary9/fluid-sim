"""Configuration validation: bad values must fail loudly at construction."""

from __future__ import annotations

import dataclasses

import pytest

from fluidsim.config import (
    AppConfig,
    BCType,
    BoundaryConditions,
    BrushConfig,
    RenderConfig,
    SimConfig,
)


def test_defaults_are_valid() -> None:
    cfg = AppConfig()
    assert cfg.sim.n == 128
    assert cfg.brush.radius >= 1
    assert cfg.render.window_size == (768, 768)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},                 # too small
        {"dt": 0.0},              # non-positive timestep
        {"viscosity": -1.0},      # negative diffusion coefficient
        {"diffusion": -1.0},
        {"iterations": 0},        # need at least one sweep
        {"backend": "cuda"},      # unknown backend
        {"obstacle_slip": "half"},
        {"fade": 1.5},            # outside (0, 1]
    ],
)
def test_sim_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_brush_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        BrushConfig(radius=0)
    with pytest.raises(ValueError):
        BrushConfig(palette=())


def test_render_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RenderConfig(window_size=(10, 10))
    with pytest.raises(ValueError):
        RenderConfig(overlay_stride=0)


def test_configs_are_immutable() -> None:
    cfg = SimConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n = 64  # type: ignore[misc]


def test_boundary_conditions_require_paired_periodic() -> None:
    with pytest.raises(ValueError):
        BoundaryConditions(left=BCType.PERIODIC)            # right not periodic
    with pytest.raises(ValueError):
        BoundaryConditions(top=BCType.PERIODIC)             # bottom not periodic
    # Paired is fine.
    BoundaryConditions(left=BCType.PERIODIC, right=BCType.PERIODIC)


def test_sim_config_rejects_bad_boundary() -> None:
    with pytest.raises(ValueError):
        SimConfig(boundary="walls")  # type: ignore[arg-type]
