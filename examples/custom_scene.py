"""Build a *custom* CFD experiment from scratch and render it.

This shows the full programmatic API: choose boundary conditions, place arbitrary
obstacles, add continuous dye sources, set initial conditions, and run — all
without the interactive UI. Here we put an angled flat plate (a crude airfoil) in
a wind tunnel and watch the flow separate off its trailing edge.

    python examples/custom_scene.py        # writes airfoil.mp4 (needs the video extra)
"""

from __future__ import annotations

import numpy as np

from fluidsim.config import BCType, BoundaryConditions, SimConfig
from fluidsim.recording import render_scene
from fluidsim.render.visualize import VisualConfig
from fluidsim.scenes import DyeSource, Plate, Rectangle, Scene


def build_scene() -> Scene:
    speed = 1.8
    bc = BoundaryConditions(
        left=BCType.INFLOW,
        right=BCType.OUTFLOW,
        top=BCType.WALL,
        bottom=BCType.WALL,
        inflow_velocity=(speed, 0.0),
    )
    sim = SimConfig(n=220, backend="auto", boundary=bc, viscosity=1.5e-6, iterations=26)

    # An angled flat plate as a simple airfoil obstacle.
    airfoil = Plate(cx=0.32, cy=0.5, length=0.22, thickness=0.02, angle=18.0)

    # Dye ribbons across the inlet to trace the streaklines.
    ribbons = tuple(
        DyeSource(Rectangle(0.0, y - 0.005, 0.02, y + 0.005), color=(240, 120, 40), rate=45.0)
        for y in np.linspace(0.2, 0.8, 7)
    )

    def initial(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full((n, n), speed, np.float32), np.zeros((n, n), np.float32)

    return Scene(
        name="airfoil",
        sim=sim,
        obstacles=(airfoil,),
        dye_sources=ribbons,
        initial_velocity=initial,
        default_view="vorticity",
    )


def main() -> None:
    render_scene(
        build_scene(),
        "airfoil.mp4",
        frames=400,
        fps=60,
        warmup=40,
        visual=VisualConfig(view="vorticity", output_size=(960, 960), overlay="none"),
    )


if __name__ == "__main__":
    main()
