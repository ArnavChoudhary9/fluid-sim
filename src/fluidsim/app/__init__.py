"""Application layer: the composition root and the pygame backend.

This is the **only** subpackage permitted to import pygame. It wires the pure
core, the renderer, and the interaction layer together and runs the main loop.
"""

from __future__ import annotations

from .app import run

__all__ = ["run"]
