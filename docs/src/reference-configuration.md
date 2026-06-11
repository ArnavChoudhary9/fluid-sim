# Configuration Reference

All configuration is held in **immutable** (`frozen=True`) dataclasses in
`fluidsim/config.py`. Immutability is a least-privilege choice: a consumer cannot
reach in and mutate shared settings — to change behaviour you construct a new
config. Invalid values are rejected at construction time (in `__post_init__`), so
mistakes fail loudly instead of producing silent `NaN`s deep in the solver.

## `SimConfig` — the physics

| Field | Default | Meaning |
|---|---|---|
| `n` | `128` | grid resolution per axis (interior cells); arrays are `(n+2)²` |
| `dt` | `1/60` | fixed simulation timestep in seconds |
| `viscosity` | `1e-6` | kinematic viscosity \\(\nu\\) — momentum diffusion |
| `diffusion` | `1e-6` | dye diffusion coefficient \\(\kappa\\) |
| `iterations` | `20` | Gauss–Seidel sweeps per diffuse/project solve |
| `backend` | `"auto"` | `"numpy"`, `"numba"`, or `"auto"` |
| `obstacle_slip` | `"free"` | `"free"` (slide) or `"no"` (stick) at obstacles |
| `fade` | `0.999` | per-step dye decay in `(0, 1]`; `1.0` = no fade |
| `boundary` | all walls | per-edge [`BoundaryConditions`](cfd-boundary-conditions.md) |
| `buoyancy` | `0.0` | dye-driven lift; `>0` makes smoke rise |

`BoundaryConditions` holds the four edges (`left`/`right`/`top`/`bottom`, each a
`BCType` of `WALL`/`INFLOW`/`OUTFLOW`/`PERIODIC`), plus `inflow_velocity` and
`wall_velocity` (tangential speeds for moving walls). See
[Boundary Conditions](cfd-boundary-conditions.md).

## `BrushConfig` — mouse injection

| Field | Default | Meaning |
|---|---|---|
| `radius` | `4` | brush footprint in cells (also obstacle brush size) |
| `dye_amount` | `8.0` | dye deposit gain; per-step dye is `dt · dye_amount · colour` |
| `force_scale` | `90.0` | velocity gain; per-step force is `dt · force_scale · motion` |
| `palette` | 6 colours | cycle-able dye colours (RGB 0–255) |

Both strengths are *gains* multiplied by `dt` in the solver, so injection is
frame-rate independent.

## `RenderConfig` — appearance

| Field | Default | Meaning |
|---|---|---|
| `window_size` | `(768, 768)` | window pixels (min 64×64) |
| `show_overlay` | `False` | start with the velocity overlay on |
| `overlay_stride` | `8` | sample every Nth cell for overlay arrows |
| `overlay_color` | white | overlay line colour |
| `obstacle_color` | grey | how solids are drawn |
| `background` | black | canvas colour (dye is composited additively over it) |
| `title` | … | window title prefix |

## `AppConfig` — the bundle

Wraps `sim`, `brush`, and `render` plus `target_fps` (default `60`). This is what
`fluidsim.app.run` takes.

## Building a config

```python
from fluidsim.config import AppConfig, SimConfig, BrushConfig, RenderConfig

config = AppConfig(
    sim=SimConfig(n=256, backend="numba", viscosity=5e-6, iterations=30),
    brush=BrushConfig(radius=6, dye_amount=10.0),
    render=RenderConfig(window_size=(900, 900), show_overlay=True),
    target_fps=60,
)
```

The CLI (`fluidsim --help`) builds the same object from flags — see
[Running the Simulation](getting-started-running.md).

## Validation examples

```python
SimConfig(n=2)              # ValueError: grid size n must be >= 4
SimConfig(viscosity=-1)     # ValueError: viscosity must be >= 0
SimConfig(backend="cuda")   # ValueError: unknown backend 'cuda'
RenderConfig(window_size=(10, 10))  # ValueError: window must be at least 64x64
```

These rules are exercised by `tests/test_config.py`.
