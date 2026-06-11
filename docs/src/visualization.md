# Field Visualisation

The interactive window uses a fast dye blit. For **offline, publication-quality**
frames, `fluidsim.render.visualize` offers a richer `Visualizer`: it can colour
the dye directly or map a **derived field** through a perceptual colormap,
supersample to any resolution, overlay flow lines, and composite obstacles with
anti-aliased edges. Everything in it is a pure, read-only function of a
`FluidState`.

```python
from fluidsim.render.visualize import Visualizer, VisualConfig

vis = Visualizer(VisualConfig(view="vorticity", output_size=(1080, 1080)))
frame = vis.render(sim.state)        # (height, width, 3) uint8
```

## Views (derived fields)

| `view` | Field | Default colormap | Notes |
|---|---|---|---|
| `dye` | the RGB dye itself | — (direct colour) | what the window shows |
| `vorticity` | curl \\(\partial v/\partial x - \partial u/\partial y\\) | `coolwarm` (diverging) | reveals vortices — the standard CFD view |
| `speed` | \\(\lvert\mathbf{u}\rvert\\) | `viridis` (sequential) | flow magnitude |
| `pressure` | the projection's pressure field | `coolwarm` (diverging) | high/low pressure regions |

`vorticity`, `speed`, and `pressure` are also importable as standalone functions
returning the interior `(n, n)` field.

## Colormaps

`fluidsim.render.colormaps` embeds several perceptual maps (no matplotlib
dependency) as anchor colours interpolated into a 256-entry lookup table:
`viridis`, `inferno`, `magma`, `turbo` (sequential), and `coolwarm` (diverging),
plus `gray`. Signed fields use a diverging map centred on zero; unsigned fields
use a sequential map. Override per view with `VisualConfig(colormap=...)`.

The colour range auto-scales from a percentile of the data (`auto_percentile`,
default 99) — robust to outliers — or you can pin it with `value_range=(lo, hi)`
for a consistent scale across a whole video.

## Supersampling

The chosen field is computed at grid resolution and then **bilinearly upsampled**
to `output_size` before (for scalar views) colour-mapping, giving smooth gradients
at any resolution. The simulation grid and the output resolution are independent:
render a modest 192² grid to a crisp 1080² frame.

> **Tip — keep the aspect square.** The domain is square (`n × n`). A non-square
> `output_size` stretches it (a circle becomes an ellipse). Use a square
> `output_size` unless you intend the stretch.

## Overlays

`VisualConfig(overlay=...)` draws the velocity field over the frame:

- `"arrows"` — a quiver lattice, auto-scaled so arrows read clearly.
- `"streamlines"` — short streaklines integrated forward and backward through the
  velocity field, conveying flow direction continuously.
- `"none"` — clean field only (default).

`overlay_density` controls how many seeds per axis; `overlay_color` their colour.

## Smooth obstacles

Obstacles are composited with **anti-aliased coverage**, not a hard mask: the
boolean solid mask is upscaled as a *float* coverage field and the obstacle colour
is alpha-blended, so edges read as a smooth surface rather than blocks. The
`edge_softness` knob (default `1.0`) scales the soft rim to roughly half a grid
cell; set it to `0` for a hard edge. This works for any obstacle, including ones
drawn by hand in the interactive app.

## Putting it together

```python
from fluidsim.scenes import Simulation
from fluidsim.scenes.library import wind_tunnel
from fluidsim.render.visualize import Visualizer, VisualConfig
import imageio.v2 as imageio

sim = Simulation(wind_tunnel(n=200))
sim.run(260)                                   # develop the wake
vis = Visualizer(VisualConfig(view="vorticity", output_size=(900, 900)))
imageio.imwrite("vortices.png", vis.render(sim.state))
```

To turn a whole run into a video, see [Rendering to Video](video.md).
