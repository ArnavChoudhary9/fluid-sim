# Programmatic CFD: Overview

Beyond the interactive window, fluidsim is a **programmable CFD engine**. You can
describe an experiment in code — its boundary conditions, obstacles, forces, and
initial state — run it headlessly, and render the result to a high-quality video.
This section covers that workflow.

## The two ways to use the engine

| | Interactive | Programmatic |
|---|---|---|
| Entry point | `fluidsim` (pygame window) | `fluidsim.scenes` + `fluidsim.recording` |
| Driven by | your mouse and keyboard | a [`Scene`](scenes.md) description |
| Output | live window | MP4 / PNG frames |
| Boundaries | closed walls | any [boundary conditions](cfd-boundary-conditions.md) |

Both sit on top of the **same physics core**. The CFD features are additions to
the solver, exposed through new configuration and a thin scene/visualisation
layer — none of which the interactive app needs to know about.

## What the extension adds

1. **Configurable boundary conditions** — inflow, outflow, periodic, and moving
   walls, set per edge. This is what makes a *wind tunnel* (inflow one side,
   outflow the other, an obstacle between) possible. See
   [Boundary Conditions](cfd-boundary-conditions.md).
2. **A buoyancy force** — dye-driven lift for smoke-plume simulations.
3. **A scene/experiment system** — [`Scene`](scenes.md) bundles config,
   obstacles ([shapes](scenes.md#shapes)), continuous [sources](scenes.md#sources),
   and initial conditions; `Simulation` runs it. A [library](scenes.md#built-in-scenes)
   of ready-made experiments ships with it.
4. **Field visualisation** — derived fields ([vorticity, speed, pressure](visualization.md))
   through perceptual colormaps, supersampled to any resolution, with
   streamline/arrow overlays and anti-aliased obstacles.
5. **Offline video rendering** — encode frames to MP4 (or a PNG sequence) with
   [`fluidsim.recording`](video.md) or the `fluidsim-render` CLI.

## A 30-second tour

```python
from fluidsim.scenes.library import wind_tunnel
from fluidsim.render.visualize import VisualConfig
from fluidsim.recording import render_scene

render_scene(
    wind_tunnel(n=200),                       # inflow past a cylinder
    "tunnel.mp4",
    frames=400,
    visual=VisualConfig(view="vorticity"),    # colour-map the curl
)
```

That produces a Kármán vortex street — the classic CFD demonstration — as an MP4.

## Architecture: still least-privilege

The new pieces respect the same layering as the rest of the project (see
[Least-Privilege & Layering](architecture-layering.md)):

```
        recording.py   (encodes frames → video; imports imageio lazily)
              │
   ┌──────────┼─────────────┐
   ▼          ▼             ▼
 scenes/    render/      (config)
   │      visualize.py
   │  (derived fields, colormaps — read-only)
   ▼
 core/   ← pure physics; gains boundary conditions + buoyancy, knows nothing above
```

`scenes/` and `render/visualize.py` are pure consumers of the core; `recording.py`
is the only new module that touches an optional dependency (imageio), and it does
so lazily. The [layering test](architecture-layering.md) now also forbids `core`
from importing `scenes` or `recording`.
