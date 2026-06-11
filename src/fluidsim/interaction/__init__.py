"""Interaction: normalised input → command objects.

This subpackage imports nothing from pygame. The backend produces
:class:`~fluidsim.interaction.input_map.FrameInput` (device-neutral), and
:func:`~fluidsim.interaction.input_map.translate` turns it into the
:mod:`~fluidsim.interaction.commands` the main loop applies.
"""

from __future__ import annotations

from .brush import Brush
from .commands import Command
from .input_map import FrameInput, KeyAction, PointerState, translate

__all__ = ["Brush", "Command", "FrameInput", "KeyAction", "PointerState", "translate"]
