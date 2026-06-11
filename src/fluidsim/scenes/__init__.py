"""Programmatic scenes: describe an experiment, then run it headlessly.

A :class:`~fluidsim.scenes.scene.Scene` bundles physics config, obstacles,
sources, and initial conditions; a :class:`~fluidsim.scenes.scene.Simulation`
runs it. The :mod:`~fluidsim.scenes.library` provides ready-made experiments
(wind tunnel, lid-driven cavity, smoke plume, shear layer).

This subpackage is a pure consumer of the core — no pygame, no rendering deps::

    from fluidsim.scenes import Simulation
    from fluidsim.scenes.library import wind_tunnel

    sim = Simulation(wind_tunnel(n=200))
    sim.run(300)
    field = sim.state.density   # inspect or visualise
"""

from __future__ import annotations

from .library import SCENES, lid_driven_cavity, shear_layer, smoke_plume, wind_tunnel
from .scene import Scene, Simulation
from .shapes import Circle, Ellipse, Plate, Rectangle, Shape, rasterize
from .sources import DyeSource, ForceSource

__all__ = [
    "Scene",
    "Simulation",
    "Shape",
    "Circle",
    "Ellipse",
    "Rectangle",
    "Plate",
    "rasterize",
    "DyeSource",
    "ForceSource",
    "SCENES",
    "wind_tunnel",
    "lid_driven_cavity",
    "smoke_plume",
    "shear_layer",
]
