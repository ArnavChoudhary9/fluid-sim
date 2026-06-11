"""High-quality field visualisation for offline (video) rendering.

Where :mod:`fluidsim.render.renderer` does the fast, real-time dye blit for the
pygame window, this module produces *publication-quality* frames: it can colour
the dye directly or map a derived scalar field — **vorticity**, **speed**, or
**pressure** — through a perceptual colormap, supersample it smoothly to any
resolution, composite obstacles, and overlay velocity arrows or streamlines.

Everything here is a pure, read-only function of a :class:`FluidState`. No pygame,
no file I/O — the recorder feeds the frames it returns to a video encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.fields import FluidState
from .colormaps import DIVERGING, apply_colormap

# Which scalar a view maps, and a sensible default colormap for each.
_DEFAULT_CMAP = {
    "vorticity": "coolwarm",
    "speed": "viridis",
    "pressure": "coolwarm",
}
VIEWS = ("dye", "vorticity", "speed", "pressure")
OVERLAYS = ("none", "arrows", "streamlines")


# -- Derived fields (read-only; each returns an (n, n) interior array) --------

def vorticity(state: FluidState) -> np.ndarray:
    """Curl of the velocity field, ``∂v/∂x − ∂u/∂y`` (signed)."""
    u, v = state.u, state.v
    return 0.5 * (
        (v[1:-1, 2:] - v[1:-1, :-2]) - (u[2:, 1:-1] - u[:-2, 1:-1])
    )


def speed(state: FluidState) -> np.ndarray:
    """Velocity magnitude ``|u|`` (unsigned)."""
    u = state.u[1:-1, 1:-1]
    v = state.v[1:-1, 1:-1]
    return np.sqrt(u * u + v * v)


def pressure(state: FluidState) -> np.ndarray:
    """The pressure field from the most recent projection (signed)."""
    return state.pressure[1:-1, 1:-1].copy()


def dye_rgb(state: FluidState) -> np.ndarray:
    """The raw RGB dye field, interior only ``(n, n, 3)``."""
    return state.density[1:-1, 1:-1]


_FIELD_FUNCS = {"vorticity": vorticity, "speed": speed, "pressure": pressure}


@dataclass(frozen=True, slots=True)
class VisualConfig:
    """Settings for a :class:`Visualizer`."""

    view: str = "dye"                              # one of VIEWS
    colormap: str | None = None                    # default chosen per view
    output_size: tuple[int, int] = (1080, 1080)    # (width, height) in pixels
    value_range: tuple[float, float] | None = None  # fixed colour range; auto if None
    auto_percentile: float = 99.0                  # percentile for auto range
    overlay: str = "none"                          # one of OVERLAYS
    overlay_color: tuple[int, int, int] = (255, 255, 255)
    overlay_density: int = 26                      # arrows/streamlines per axis
    obstacle_color: tuple[int, int, int] = (38, 40, 48)
    background: tuple[int, int, int] = (6, 6, 10)
    edge_softness: float = 1.0                     # obstacle edge anti-aliasing (0 = hard)

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"unknown view {self.view!r}; choose from {VIEWS}")
        if self.overlay not in OVERLAYS:
            raise ValueError(f"unknown overlay {self.overlay!r}; choose from {OVERLAYS}")
        w, h = self.output_size
        if w < 16 or h < 16:
            raise ValueError(f"output_size too small: {self.output_size}")
        if self.edge_softness < 0:
            raise ValueError(f"edge_softness must be >= 0, got {self.edge_softness}")


class Visualizer:
    """Turns a :class:`FluidState` into a high-resolution RGB frame."""

    def __init__(self, config: VisualConfig | None = None) -> None:
        self.config = config or VisualConfig()

    def render(self, state: FluidState) -> np.ndarray:
        cfg = self.config
        width, height = cfg.output_size

        if cfg.view == "dye":
            base = np.clip(dye_rgb(state), 0.0, 255.0)
            image = _resize_bilinear(base, height, width)
            image = np.clip(image + np.asarray(cfg.background, np.float64), 0, 255)
            image = image.astype(np.uint8)
        else:
            field = _FIELD_FUNCS[cfg.view](state)
            norm = self._normalise(field)
            norm_hi = _resize_bilinear(norm, height, width)
            cmap = cfg.colormap or _DEFAULT_CMAP[cfg.view]
            image = apply_colormap(np.clip(norm_hi, 0.0, 1.0), cmap)

        # Composite obstacles on top with anti-aliased coverage: upscale the
        # boolean mask as a float, smooth the staircase, and alpha-blend the
        # obstacle colour. Soft edges read as a smooth surface rather than blocks.
        solid = state.obstacle[1:-1, 1:-1]
        if solid.any():
            # Soften over ~half a grid cell (in output pixels), scaled by config.
            cell_px = height / state.grid.n
            radius = int(round(cfg.edge_softness * 0.5 * cell_px))
            coverage = _obstacle_coverage(solid, height, width, radius)
            alpha = coverage[..., None]
            obstacle = np.asarray(cfg.obstacle_color, np.float64)
            image = (image * (1.0 - alpha) + obstacle * alpha).round().astype(np.uint8)

        if cfg.overlay == "arrows":
            self._draw_arrows(image, state)
        elif cfg.overlay == "streamlines":
            self._draw_streamlines(image, state)
        return image

    # -- normalisation -------------------------------------------------------

    def _normalise(self, field: np.ndarray) -> np.ndarray:
        cfg = self.config
        cmap = cfg.colormap or _DEFAULT_CMAP[cfg.view]
        diverging = cmap in DIVERGING
        if cfg.value_range is not None:
            lo, hi = cfg.value_range
        elif diverging:  # symmetric range centred on zero
            a = np.percentile(np.abs(field), cfg.auto_percentile)
            a = float(a) if a > 1e-9 else 1.0
            lo, hi = -a, a
        else:  # sequential range from zero (or min) to a high percentile
            hi = float(np.percentile(field, cfg.auto_percentile))
            lo = float(min(0.0, field.min()))
            if hi - lo < 1e-9:
                hi = lo + 1.0
        return (field - lo) / (hi - lo)

    # -- overlays ------------------------------------------------------------

    def _lattice(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Interior grid-coordinate seed lattice for overlays."""
        count = max(2, min(self.config.overlay_density, n))
        coords = np.linspace(1.5, n - 0.5, count)
        gx, gy = np.meshgrid(coords, coords)
        return gx.ravel(), gy.ravel()

    def _to_pixels(self, gx: np.ndarray, gy: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.config.output_size
        return (gx - 0.5) / n * width, (gy - 0.5) / n * height

    def _draw_arrows(self, image: np.ndarray, state: FluidState) -> None:
        n = state.grid.n
        gx, gy = self._lattice(n)
        u = _sample(state.u, gx, gy)
        v = _sample(state.v, gx, gy)
        mag = np.sqrt(u * u + v * v)
        ref = np.percentile(mag, 90) or 1.0
        spacing = (n - 2) / max(2, min(self.config.overlay_density, n))
        scale = spacing / (ref + 1e-9)
        x0, y0 = self._to_pixels(gx, gy, n)
        x1, y1 = self._to_pixels(gx + scale * u, gy + scale * v, n)
        for i in range(len(x0)):
            if mag[i] < 1e-4:
                continue
            _draw_line(image, x0[i], y0[i], x1[i], y1[i], self.config.overlay_color, 2)

    def _draw_streamlines(self, image: np.ndarray, state: FluidState) -> None:
        n = state.grid.n
        gx, gy = self._lattice(n)
        steps, step_len = 26, 0.6
        for direction in (1.0, -1.0):
            px = gx.copy()
            py = gy.copy()
            pts_x = [px.copy()]
            pts_y = [py.copy()]
            for _ in range(steps):
                u = _sample(state.u, px, py)
                v = _sample(state.v, px, py)
                speed_ = np.sqrt(u * u + v * v) + 1e-6
                px = np.clip(px + direction * step_len * u / speed_, 1.0, n)
                py = np.clip(py + direction * step_len * v / speed_, 1.0, n)
                pts_x.append(px.copy())
                pts_y.append(py.copy())
            xs = np.array(pts_x)  # (steps+1, seeds)
            ys = np.array(pts_y)
            for s in range(xs.shape[1]):
                gxx, gyy = self._to_pixels(xs[:, s], ys[:, s], n)
                for k in range(len(gxx) - 1):
                    _draw_line(image, gxx[k], gyy[k], gxx[k + 1], gyy[k + 1],
                               self.config.overlay_color, 1)


# -- low-level helpers --------------------------------------------------------

def _resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinearly resample a 2-D or 3-D array to ``(out_h, out_w[, c])`` (float)."""
    in_h, in_w = arr.shape[:2]
    ys = np.clip((np.arange(out_h) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)
    wx = (xs - x0)
    src = arr.astype(np.float64)
    # Gather the four corners; row-index first, then column-index.
    a = src[y0][:, x0]
    b = src[y0][:, x1]
    c = src[y1][:, x0]
    d = src[y1][:, x1]
    if src.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return top * (1 - wy) + bot * wy


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur of a 2-D array (edge-padded), O(H·W) via cumulative sums."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    # Blur along rows (axis 1).
    pad = np.pad(a, ((0, 0), (radius, radius)), mode="edge")
    cs = np.concatenate([np.zeros((pad.shape[0], 1)), np.cumsum(pad, axis=1)], axis=1)
    a = (cs[:, k:] - cs[:, :-k]) / k
    # Blur along columns (axis 0).
    pad = np.pad(a, ((radius, radius), (0, 0)), mode="edge")
    cs = np.concatenate([np.zeros((1, pad.shape[1])), np.cumsum(pad, axis=0)], axis=0)
    return (cs[k:, :] - cs[:-k, :]) / k


def _obstacle_coverage(solid: np.ndarray, out_h: int, out_w: int, radius: int) -> np.ndarray:
    """Anti-aliased obstacle coverage in ``[0, 1]`` at output resolution.

    Bilinearly upscaling the boolean mask already turns the per-cell staircase
    into a ramp; an optional box blur widens that into a smooth rim. Interior
    cells (all-ones neighbourhoods) stay fully opaque.
    """
    coverage = _resize_bilinear(solid.astype(np.float64), out_h, out_w)
    coverage = _box_blur(coverage, radius)
    return np.clip(coverage, 0.0, 1.0)


def _sample(field: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinearly sample a padded field at grid coordinates ``(gx, gy)``."""
    x0 = np.floor(gx).astype(np.intp)
    y0 = np.floor(gy).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    sx = gx - x0
    sy = gy - y0
    return (
        (1 - sy) * ((1 - sx) * field[y0, x0] + sx * field[y0, x1])
        + sy * ((1 - sx) * field[y1, x0] + sx * field[y1, x1])
    )


def _draw_line(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float,
    color: tuple[int, int, int], thickness: int,
) -> None:
    """Draw a straight line into an RGB image (simple, anti-alias-free)."""
    h, w = image.shape[:2]
    length = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, length).round().astype(int)
    ys = np.linspace(y0, y1, length).round().astype(int)
    r = thickness // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            px = np.clip(xs + dx, 0, w - 1)
            py = np.clip(ys + dy, 0, h - 1)
            image[py, px] = color
