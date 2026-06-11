# Stable Fluids: The Algorithm

Jos Stam's *Stable Fluids* (SIGGRAPH 1999) and *Real-Time Fluid Dynamics for
Games* (GDC 2003) take the [operator splitting](math-navier-stokes.md) idea and
choose each sub-solver so that the whole scheme is **unconditionally stable** —
it never blows up, no matter how large the timestep or velocity. That stability
is what makes interactive mouse-driven fluid possible.

## One timestep

Each step advances the velocity field, then the dye field. In the code this is
`BaseSolver.step`, which calls `_vel_step` then `_dens_step`.

### Velocity step (`_vel_step`)

\\[
\textbf{add force} \\;\to\\; \textbf{diffuse} \\;\to\\; \textbf{project}
\\;\to\\; \textbf{advect} \\;\to\\; \textbf{project}
\\]

```python
u += dt * u_src;  v += dt * v_src     # 1. inject mouse force
diffuse(u); diffuse(v)                # 2. viscosity  (implicit solve)
project(u, v)                         # 3. make divergence-free
advect(u); advect(v)                  # 4. self-advection (semi-Lagrangian)
project(u, v)                         # 5. make divergence-free again
```

### Density step (`_dens_step`)

\\[
\textbf{add dye} \\;\to\\; \textbf{diffuse} \\;\to\\; \textbf{advect}
\\]

```python
density += dt * density_src           # 1. inject mouse dye
diffuse(density)                      # 2. dye diffusion
advect(density)                       # 3. transport dye by the velocity field
```

The dye step needs no projection — dye does not have to be divergence-free, it
just rides along.

## Why project *twice*

> **Pitfall — projecting only once breaks incompressibility.** Notice projection
> appears both *before* and *after* advection in the velocity step. This is not
> redundant:
>
> - **Before advect (step 3):** semi-Lagrangian advection traces velocity
>   *backwards along the velocity field itself*. For that trace to be physically
>   meaningful, the field it traces through must already be mass-conserving
>   (divergence-free).
> - **After advect (step 5):** advection re-introduces a little divergence of its
>   own. The second projection removes it so the field handed to the next step is
>   clean again.
>
> Omit either projection and you get visible compressibility artifacts — dye
> bunching, sources/sinks appearing from nowhere, and unstable swirl. The code
> calls `_project` on both sides for exactly this reason.

## Why this is stable

Each sub-step is individually unconditionally stable:

- **Diffusion** is solved *implicitly* (backward Euler), which is stable for any
  timestep and any viscosity. [Diffusion](math-diffusion.md)
- **Advection** is *semi-Lagrangian*: it asks "what value was here a moment ago?"
  and interpolates, which can never produce a value outside the existing range —
  so it cannot blow up. [Advection](math-advection.md)
- **Projection** solves a Poisson equation, a stable elliptic solve.
  [Projection](math-projection.md)

The price of guaranteed stability is **numerical dissipation** — the dye and
swirl gently fade over time. That trade-off, and ways to soften it, are discussed
in [Dissipation](math-dissipation.md).

## The discrete grid

All of this happens on a regular grid. The simulation is `n × n`, stored in
`(n+2) × (n+2)` arrays with a one-cell ghost border for boundaries (see
[Boundaries](math-boundary-conditions.md)). We index `field[y, x]` with `y`
increasing downward to match screen coordinates, and the grid spacing is taken as
\\(h = 1\\) (it cancels out of the projection, so we drop it for clarity).

The next chapters derive each of the three primitives — advect, diffuse, project
— down to the exact array operations in `numpy_solver.py`.
