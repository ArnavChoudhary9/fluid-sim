# Design Overview & Decisions

This chapter records *why* the project is built the way it is. Each decision
traces back to one of the three guiding principles: **modularity**,
**least privilege**, and **clean code**.

## The one rule everything follows

> **The physics core knows nothing about the UI.**

The solver depends only on NumPy. It has no idea that pygame, a renderer, or a
mouse exist. Everything else is a *consumer* of the solver's small public
interface. This single rule is what makes "modular and independent" concrete
rather than aspirational, and it drives the whole layout.

Dependencies flow in exactly one direction:

```
        ┌─────────────┐
        │    app/     │  composition root — the ONLY place pygame lives
        └──────┬──────┘
               │ uses
   ┌───────────┼────────────┐
   ▼           ▼            ▼
┌──────┐  ┌──────────┐  ┌──────────────┐
│render│  │interaction│  │   config     │
└──┬───┘  └────┬─────┘  └──────┬───────┘
   │ reads     │ injects via   │ parameterises
   └───────────┴────► ┌────────▼────────┐
                      │      core/      │  pure physics, NumPy(+Numba) only
                      └─────────────────┘
```

`core` never imports `render`, `interaction`, `app`, or pygame. A
[layering test](architecture-layering.md) enforces this automatically.

## Key decisions and their rationale

### Method: Stam's *Stable Fluids* (grid-based)

We simulate smoke/dye on an Eulerian grid with Jos Stam's semi-Lagrangian method.
It is **unconditionally stable** (no timestep-driven blow-ups), beautifully
suited to interactive mouse painting, and the best-documented fluid method there
is. The alternative — SPH particles — gives splashier liquids but is harder to
keep stable and to render cleanly in 2D. For "paint and push smoke with the
mouse", Stable Fluids is the right tool. Full derivation in
[Navier–Stokes](math-navier-stokes.md).

### Grid: padded ghost cells

An `n × n` simulation lives inside `(n+2) × (n+2)` arrays. The extra one-cell
border (*ghost cells*) holds boundary values, so boundary handling becomes clean
index arithmetic instead of special-casing edges. See
[Boundary Conditions](math-boundary-conditions.md).

### Dye: three advected channels (RGB)

The dye field carries real colour — three channels, each advected by the shared
velocity field — so colours transport and mix correctly and "cycle dye colour"
shows true colour rather than a tinted intensity. The cost (3× the cheap scalar
advection) is negligible next to the pressure solve.

### Numbers: `float32`

Fields are single precision for memory-bandwidth and `pygame.surfarray`
friendliness. The iterative solves are perfectly tolerant of it at these grid
sizes.

### Two backends behind one interface

A pure-NumPy reference solver and an optional Numba-JIT solver implement the
**same** abstract interface. `import fluidsim` never requires numba; it is touched
only if actually requested, and missing/incompatible numba falls back to NumPy
with a warning. See [Backends](architecture-backends.md).

### Integration: fixed-timestep accumulator

The app steps physics at a fixed `dt` regardless of frame rate, for deterministic
behaviour. A substep cap prevents a "spiral of death" on a slow frame. See
[Module Reference](architecture-modules.md).

### Configuration: frozen dataclasses

All configuration is immutable (`frozen=True`). A consumer literally *cannot*
mutate shared settings — least privilege at the data level. Genuinely mutable
runtime state (brush size, paused flag, active colour) lives in small dedicated
holders, not in config. See [Configuration Reference](reference-configuration.md).

## Pitfalls this design deliberately avoids

These are the classic ways a Stable-Fluids solver goes wrong. Each has a home in
the math chapters:

1. **Forgetting to project twice** → compressible artifacts. [Projection](math-projection.md)
2. **Wrong boundary sign** for velocity vs scalar → leaks or sticking. [Boundaries](math-boundary-conditions.md)
3. **Obstacles that leak** because projection treats solids as fluid. [Obstacles](math-obstacles.md)
4. **Numerical dissipation** smearing the dye. [Dissipation](math-dissipation.md)
5. **`surfarray` axis order** (width, height) vs our (row, col). [Modules](architecture-modules.md)
6. **Frame-coupled `dt`** making behaviour machine-dependent. [Modules](architecture-modules.md)
7. **`import fluidsim` requiring numba.** [Backends](architecture-backends.md)
