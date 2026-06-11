"""Rendering: read-only consumers that turn fields into pixels and overlay data.

Nothing in this subpackage imports pygame or mutates the simulation. The
:class:`~fluidsim.render.renderer.Renderer` produces an RGB image;
:mod:`~fluidsim.render.overlay` produces velocity line segments as plain data;
:class:`~fluidsim.render.colormap.Palette` holds the dye colour cursor.
"""

from __future__ import annotations

from .colormap import Palette
from .colormaps import apply_colormap, available, get_colormap
from .overlay import velocity_segments
from .renderer import Renderer
from .visualize import VisualConfig, Visualizer, pressure, speed, vorticity

__all__ = [
    "Renderer",
    "Palette",
    "velocity_segments",
    "Visualizer",
    "VisualConfig",
    "vorticity",
    "speed",
    "pressure",
    "get_colormap",
    "apply_colormap",
    "available",
]
