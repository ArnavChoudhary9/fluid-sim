# Boundary Conditions

The interactive app runs in a closed box. A real CFD engine needs more: fluid
that *enters*, *leaves*, *wraps around*, or slides along a *moving* wall. fluidsim
sets each of the four domain edges independently via
[`BoundaryConditions`](reference-configuration.md).

This chapter explains the four conditions, the maths of how each fills the ghost
cells (building on [Boundary Conditions & Sign Conventions](math-boundary-conditions.md)),
and how to configure them.

## The four conditions

```python
from fluidsim.config import BoundaryConditions, BCType, SimConfig

bc = BoundaryConditions(
    left=BCType.INFLOW,
    right=BCType.OUTFLOW,
    top=BCType.WALL,
    bottom=BCType.WALL,
    inflow_velocity=(1.8, 0.0),
)
sim = SimConfig(n=200, boundary=bc)
```

### `WALL` — closed (the default)

No flow through the edge. The normal velocity is reflected (negated), scalars
copy their neighbour (Neumann), and the tangential velocity is copied (free-slip).
This is exactly the closed-box behaviour of the interactive app, and when *all*
edges are static walls the engine takes a fast path identical to the original
solver.

### `INFLOW` — prescribed velocity (Dirichlet)

Fluid enters at a fixed velocity `inflow_velocity`. The edge's velocity ghost
*and* its first interior cell are pinned each step:

\\[ u_{\text{edge}} = u_{\text{inflow}} \\]

This is the upstream side of a wind tunnel.

### `OUTFLOW` — open (zero-gradient)

Fluid leaves freely. Every field copies its interior neighbour (homogeneous
Neumann, \\(\partial/\partial n = 0\\)), and — crucially — the **pressure is
pinned to zero** at an outflow edge (Dirichlet) so the projection allows a net
mass flux out:

\\[ p_{\text{edge}} = 0 \\]

This is the downstream side of a wind tunnel. Without the pressure condition the
domain could not conserve mass against a steady inflow.

### `PERIODIC` — wrap-around

The edge wraps to the opposite edge: what leaves the right re-enters the left.
Must be set on **both** edges of an axis (the config rejects a half-periodic
setup). Periodic left/right with walls top/bottom is the natural setting for a
[shear layer](scenes.md#built-in-scenes).

### Moving walls

A `WALL` edge can be given a non-zero tangential speed via `wall_velocity`
(`(left, right, top, bottom)`), turning it into a moving wall that drags the
fluid. The ghost cell enforces the wall's surface velocity \\(U_w\\):

\\[ u_{\text{ghost}} = 2 U_w - u_{\text{inner}} \\]

so that the average at the wall equals \\(U_w\\). This drives the
[lid-driven cavity](scenes.md#built-in-scenes) benchmark.

## How it integrates with the solver

Two BC-aware functions in `core/boundary.py` replace the plain `set_bnd` calls:

- `apply_boundary(b, field, n, bc)` fills the ghost ring per-edge for velocity
  and scalar fields. It falls back to `set_bnd` when every edge is a static wall,
  so the default is byte-for-byte unchanged (and as fast).
- `apply_pressure_boundary(p, n, bc)` applies the matching **pressure** condition:
  Neumann at walls/inflow, Dirichlet \\(p=0\\) at outflow, wrap if periodic.

The solver threads `self.config.boundary` through every boundary pass — diffusion
sweeps, the projection's pressure solve, and the post-step velocity fix — so the
conditions hold throughout the [Stable-Fluids step](math-stable-fluids.md).

> **Pitfall — pressure condition must match the velocity condition.** An inflow
> (prescribed velocity) needs a *Neumann* pressure (the projection must not fight
> the imposed velocity), while an outflow (open) needs a *Dirichlet* pressure so
> the flow can actually leave. Mixing these up gives a tunnel that either stalls
> or leaks. `apply_pressure_boundary` encodes the correct pairing.

## Buoyancy

Set `SimConfig.buoyancy` to a positive value to add a dye-driven lift force, used
for smoke. The force is proportional to the *normalised* smoke concentration, so
the knob is independent of the dye colour scale:

\\[ \mathbf{f}_y = -\,\text{buoyancy}\cdot \tilde{d}, \qquad
\tilde{d} = \frac{1}{3\cdot 255}\sum_{\text{channels}} d \\]

(negative \\(f_y\\) is "up", since y increases downward). See the
[smoke plume scene](scenes.md#built-in-scenes).

## Validation against the wind tunnel

`tests/test_boundary_conditions.py` checks each condition directly (inflow pins
the velocity, outflow is zero-gradient, periodic wraps, the moving wall drags),
plus an end-to-end test that inflow→outflow past a cylinder develops a downstream
flow and never leaks into the solid.
