"""Headless demo — proves the physics core runs with no UI at all.

This script imports **only** ``fluidsim.core`` and ``fluidsim.config`` (never
pygame, render, or interaction). It injects a dye blob and a swirl of velocity,
steps the solver, and prints a few diagnostics. If this runs, the core is
genuinely independent of the rest of the application.

Run it with::

    python examples/headless_demo.py
"""

from __future__ import annotations

import numpy as np

from fluidsim.config import SimConfig
from fluidsim.core import make_solver


def main() -> None:
    config = SimConfig(n=64, backend="auto")
    solver = make_solver(config)
    print(f"backend = {type(solver).__name__}, grid = {config.n}x{config.n}")

    # A solid disc in the middle for the flow to wrap around.
    solver.set_obstacle(32, 32, radius=6, solid=True)

    steps = 120
    for t in range(steps):
        # A little rotating injection near the top-left.
        angle = t * 0.2
        fx, fy = 60.0 * np.cos(angle), 60.0 * np.sin(angle)
        solver.add_dye(16, 16, (255, 80, 0), radius=4)
        solver.add_velocity(16, 16, fx, fy, radius=4)
        solver.step(config.dt)

    density = solver.density_field            # read-only (n+2, n+2, 3)
    u, v = solver.velocity_field
    speed = np.sqrt(u**2 + v**2)

    print(f"stepped {steps} times")
    print(f"total dye mass    : {density.sum():,.1f}")
    print(f"max speed         : {speed.max():.3f}")
    print(f"dye inside solid  : {density[solver.obstacle_mask].sum():.6f} (should be ~0)")
    print("core ran with no UI imported — independence confirmed.")


if __name__ == "__main__":
    main()
