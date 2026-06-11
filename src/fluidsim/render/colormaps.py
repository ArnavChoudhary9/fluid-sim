"""Perceptual colormaps for scientific field visualisation.

Rather than depend on matplotlib, we embed a handful of well-known colormaps as a
short list of anchor colours and linearly interpolate them into a 256-entry
lookup table. This is dependency-free and visually indistinguishable from the
originals at video resolution.

Sequential maps (``viridis``, ``inferno``, ``magma``, ``turbo``) suit unsigned
fields like speed; the diverging ``coolwarm`` suits signed fields like vorticity
and pressure, where zero should read as neutral.
"""

from __future__ import annotations

import numpy as np

# Each colormap is a list of (position in [0, 1], (r, g, b)) anchors.
_ANCHORS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "viridis": [
        (0.0, (68, 1, 84)), (0.14, (72, 40, 120)), (0.29, (62, 74, 137)),
        (0.43, (49, 104, 142)), (0.57, (38, 130, 142)), (0.71, (31, 158, 137)),
        (0.86, (53, 183, 121)), (1.0, (253, 231, 37)),
    ],
    "inferno": [
        (0.0, (0, 0, 4)), (0.14, (40, 11, 84)), (0.29, (101, 21, 110)),
        (0.43, (159, 42, 99)), (0.57, (212, 72, 66)), (0.71, (245, 125, 21)),
        (0.86, (250, 193, 39)), (1.0, (252, 255, 164)),
    ],
    "magma": [
        (0.0, (0, 0, 4)), (0.14, (28, 16, 68)), (0.29, (79, 18, 123)),
        (0.43, (129, 37, 129)), (0.57, (181, 54, 122)), (0.71, (229, 80, 100)),
        (0.86, (251, 135, 97)), (1.0, (252, 253, 191)),
    ],
    "turbo": [
        (0.0, (48, 18, 59)), (0.14, (70, 107, 227)), (0.29, (40, 174, 236)),
        (0.43, (45, 225, 167)), (0.57, (145, 247, 79)), (0.71, (228, 219, 52)),
        (0.86, (252, 131, 44)), (1.0, (165, 14, 2)),
    ],
    "coolwarm": [
        (0.0, (59, 76, 192)), (0.25, (122, 146, 233)), (0.5, (221, 221, 221)),
        (0.75, (229, 146, 124)), (1.0, (180, 4, 38)),
    ],
    "gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
}

# Maps where the data is signed and should be centred on zero.
DIVERGING = frozenset({"coolwarm"})

_LUT_CACHE: dict[str, np.ndarray] = {}


def _build_lut(name: str) -> np.ndarray:
    anchors = _ANCHORS[name]
    positions = np.array([p for p, _ in anchors])
    colors = np.array([c for _, c in anchors], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(ramp, positions, colors[:, channel]).round()
    return lut


def get_colormap(name: str) -> np.ndarray:
    """Return the ``(256, 3)`` ``uint8`` lookup table for ``name`` (cached)."""
    if name not in _ANCHORS:
        raise ValueError(f"unknown colormap {name!r}; choose from {sorted(_ANCHORS)}")
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _build_lut(name)
    return _LUT_CACHE[name]


def apply_colormap(values: np.ndarray, name: str) -> np.ndarray:
    """Map an array of values in ``[0, 1]`` through a colormap to RGB ``uint8``."""
    lut = get_colormap(name)
    idx = np.clip(values * 255.0, 0.0, 255.0).astype(np.intp)
    return lut[idx]


def available() -> list[str]:
    """Names of all built-in colormaps."""
    return sorted(_ANCHORS)
