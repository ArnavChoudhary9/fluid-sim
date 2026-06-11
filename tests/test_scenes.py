"""Built-in scenes build, run, and stay numerically sane."""

from __future__ import annotations

import numpy as np
import pytest

from fluidsim.scenes import Simulation
from fluidsim.scenes.library import SCENES, smoke_plume, wind_tunnel


@pytest.mark.parametrize("name", list(SCENES))
def test_scene_runs_and_stays_finite(name: str) -> None:
    sim = Simulation(SCENES[name](n=48, backend="numpy"))
    sim.run(40)
    s = sim.state
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert np.isfinite(s.density).all()


def test_wind_tunnel_stamps_obstacle_and_flows() -> None:
    sim = Simulation(wind_tunnel(n=64, speed=1.5, backend="numpy"))
    assert sim.state.obstacle.any()             # the cylinder was rasterised
    sim.run(80)
    u = sim.state.u
    assert u[1:-1, 40:-1].mean() > 0.3          # downstream flow develops
    assert np.allclose(sim.state.u[sim.state.obstacle], 0.0)


def test_smoke_plume_rises() -> None:
    sim = Simulation(smoke_plume(n=64, backend="numpy"))
    sim.run(120)
    column = sim.state.density[1:-1, 1:-1].sum(axis=2).sum(axis=1)
    occupied = np.where(column > 1.0)[0]
    # Dye should have climbed above the emitter near the floor (large row indices).
    assert occupied.min() < 48


def test_scene_is_reproducible() -> None:
    a = Simulation(wind_tunnel(n=48, backend="numpy"))
    b = Simulation(wind_tunnel(n=48, backend="numpy"))
    a.run(30)
    b.run(30)
    assert np.allclose(a.state.density, b.state.density)
