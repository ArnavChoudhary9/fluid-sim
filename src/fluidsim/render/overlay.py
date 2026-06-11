"""Velocity-field overlay generation (data only, no drawing).

To keep the renderer free of any UI toolkit, the overlay is produced as plain
*data*: a list of line segments in grid coordinates. The pygame backend rasterises
them. This separation means the overlay can be unit-tested without a display and
swapped for a different drawing backend without change here.
"""

from __future__ import annotations

from ..core.fields import FluidState

# A line segment in interior-grid coordinates: ((x0, y0), (x1, y1)).
Segment = tuple[tuple[float, float], tuple[float, float]]


def velocity_segments(
    state: FluidState,
    stride: int,
    scale: float,
) -> list[Segment]:
    """Sample the velocity field on a coarse lattice and return arrow segments.

    Parameters
    ----------
    state:
        The fluid state to read (velocity only; never mutated).
    stride:
        Sample every ``stride`` cells.
    scale:
        Multiplier converting velocity magnitude to segment length in cells.
    """
    n = state.grid.n
    u, v = state.u, state.v
    segments: list[Segment] = []
    for y in range(1 + stride // 2, n + 1, stride):
        for x in range(1 + stride // 2, n + 1, stride):
            if state.obstacle[y, x]:
                continue
            dx = float(u[y, x]) * scale
            dy = float(v[y, x]) * scale
            if dx * dx + dy * dy < 1e-6:
                continue
            segments.append(((x, y), (x + dx, y + dy)))
    return segments
