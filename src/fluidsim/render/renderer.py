"""Turn a :class:`FluidState` into an RGB image.

:class:`Renderer.render` is a **pure function of state**: given the fields, it
returns a fresh ``uint8`` array of shape ``(n, n, 3)`` in ``[row=y, col=x]``
order. It never mutates the state and never imports pygame — converting that
array to a window is the backend's job (including the ``surfarray`` axis
transpose). This keeps the visual mapping testable without a display.
"""

from __future__ import annotations

import numpy as np

from ..config import RenderConfig
from ..core.fields import FluidState


class Renderer:
    """Composites the dye field and obstacles into a displayable image."""

    def __init__(self, config: RenderConfig) -> None:
        self._config = config
        self._background = np.array(config.background, dtype=np.float32)
        self._obstacle_color = np.array(config.obstacle_color, dtype=np.uint8)

    def render(self, state: FluidState) -> np.ndarray:
        """Return an ``(n, n, 3)`` ``uint8`` image of the interior cells."""
        dye = state.density[1:-1, 1:-1, :]          # (n, n, 3) float view
        image = np.clip(dye + self._background, 0.0, 255.0).astype(np.uint8)

        # Obstacles are opaque and drawn last so they always read as walls.
        solid = state.obstacle[1:-1, 1:-1]
        if solid.any():
            image[solid] = self._obstacle_color
        return image
