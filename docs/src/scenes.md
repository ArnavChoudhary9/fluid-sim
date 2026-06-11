# Scenes & Experiments

A **scene** is a complete, reproducible description of an experiment: its physics
config, obstacles, continuous sources, and initial conditions. A `Simulation`
turns a scene into a running solver you can step headlessly or feed to the
[video renderer](video.md).

```python
from fluidsim.scenes import Simulation
from fluidsim.scenes.library import wind_tunnel

sim = Simulation(wind_tunnel(n=200))
sim.run(300)                 # warm up
field = sim.state.density    # read fields for visualisation / analysis
```

`Simulation` builds the solver via `make_solver`, stamps the obstacles, applies
the initial conditions, and on each `step()` runs the sources/drivers before
advancing the physics. Two identical scenes produce identical results — the
engine is deterministic.

## Built-in scenes

`fluidsim.scenes.library` ships four ready-made experiments (each a function
returning a `Scene`, and registered in `SCENES`):

| Scene | What it shows | Features exercised |
|---|---|---|
| `wind_tunnel` | Flow past a bluff body → **Kármán vortex street** | inflow/outflow, obstacle |
| `lid_driven_cavity` | The classic steady recirculating vortex | moving wall |
| `smoke_plume` | A buoyant column rising and billowing | buoyancy, outflow top |
| `shear_layer` | **Kelvin–Helmholtz** roll-up of two streams | periodic BCs, initial shear |

Each accepts `n` (grid size) and `backend`, plus scene-specific knobs (e.g.
`wind_tunnel(speed=...)`, `lid_driven_cavity(lid_speed=...)`). Each also suggests
a `default_view` for the renderer.

```python
from fluidsim.scenes.library import smoke_plume
sim = Simulation(smoke_plume(n=180, buoyancy=6.0))
```

## Shapes

Obstacles (and source regions) are described by **shape primitives** in
`fluidsim.scenes.shapes`, using **normalised** coordinates in `[0, 1]` so a scene
works at any grid resolution. Each rasterises to an interior boolean mask.

| Shape | Parameters |
|---|---|
| `Circle(cx, cy, r)` | centre, radius |
| `Ellipse(cx, cy, rx, ry)` | centre, semi-axes |
| `Rectangle(x0, y0, x1, y1)` | opposite corners |
| `Plate(cx, cy, length, thickness, angle)` | a rotated bar (airfoil-like) |

```python
from fluidsim.scenes import Circle, Plate
cylinder = Circle(0.25, 0.5, 0.06)
airfoil = Plate(0.32, 0.5, length=0.22, thickness=0.02, angle=18.0)
```

`rasterize(shapes, n)` unions several shapes into one mask. The solver exposes
`set_obstacle_mask(mask)` to install it.

## Sources

**Sources** inject something every step — the programmatic equivalent of holding
the mouse. They live in `fluidsim.scenes.sources` and only use the solver's
public injection API.

- `DyeSource(region, color, rate)` — deposit dye (per-step amount `dt · rate · color`).
- `ForceSource(region, fx, fy)` — apply a continuous velocity impulse.

```python
from fluidsim.scenes import DyeSource, Rectangle
inlet_dye = DyeSource(Rectangle(0.0, 0.45, 0.02, 0.55), color=(240, 120, 40), rate=45.0)
```

## Building a custom scene

A `Scene` ties it together. Beyond obstacles and sources you can supply
`initial_velocity(n) -> (u, v)` and `initial_dye(n) -> (n, n, 3)` callables for
initial conditions, and `drivers` for bespoke per-step behaviour.

```python
import numpy as np
from fluidsim.config import SimConfig, BoundaryConditions, BCType
from fluidsim.scenes import Scene, Circle, DyeSource, Rectangle, Simulation

bc = BoundaryConditions(left=BCType.INFLOW, right=BCType.OUTFLOW,
                        top=BCType.WALL, bottom=BCType.WALL, inflow_velocity=(1.8, 0.0))

scene = Scene(
    name="my_tunnel",
    sim=SimConfig(n=200, boundary=bc, viscosity=1.5e-6),
    obstacles=(Circle(0.25, 0.5, 0.07),),
    dye_sources=(DyeSource(Rectangle(0, 0.48, 0.02, 0.52), (240, 120, 40), rate=45.0),),
    initial_velocity=lambda n: (np.full((n, n), 1.8, np.float32), np.zeros((n, n), np.float32)),
    default_view="vorticity",
)

sim = Simulation(scene)
sim.run(400)
```

See `examples/custom_scene.py` for a complete angled-airfoil tunnel, and
[Rendering to Video](video.md) to turn any scene into an MP4.

## The solver API scenes rely on

Scenes drive the solver through a few bulk-setup methods added to the
`BaseSolver` interface (all take `(n, n)` interior arrays, no ghost border):

- `set_obstacle_mask(mask)` — install solids.
- `set_velocity_field(u, v)` / `set_dye_field(dye)` — initial conditions.
- `add_dye_region(mask, color)` / `add_velocity_region(mask, fx, fy)` — masked
  injection (used by sources).
- `set_velocity_region(mask, fx, fy)` — Dirichlet-pin a region's velocity.

These keep the [least-privilege contract](architecture-backends.md): scenes still
only *inject* and *set up*; they never reach into raw solver internals.
