"""Rendering: read-only consumers that turn fields into pixels and overlay data.

Nothing in this subpackage imports pygame or mutates the simulation. The
:class:`~fluidsim.render.renderer.Renderer` produces an RGB image;
:mod:`~fluidsim.render.overlay` produces velocity line segments as plain data;
:class:`~fluidsim.render.colormap.Palette` holds the dye colour cursor.
"""

from __future__ import annotations

from .colormap import Palette
from .overlay import velocity_segments
from .renderer import Renderer

__all__ = ["Renderer", "Palette", "velocity_segments"]
