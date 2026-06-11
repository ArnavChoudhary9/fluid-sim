# Module Reference

Every module has a single responsibility. This chapter is the map.

```
src/fluidsim/
├── __init__.py            public API (make_solver, configs); NumPy-only imports
├── __main__.py            `python -m fluidsim` → cli.main()
├── config.py              frozen dataclasses: SimConfig, BrushConfig, RenderConfig, AppConfig
├── cli.py                 argparse → AppConfig; console entry point
├── core/                  PHYSICS — pure, NumPy(+Numba) only
│   ├── grid.py            padded ghost-cell geometry & coordinate clamping
│   ├── fields.py          FluidState: u, v, RGB density, obstacle mask + scratch
│   ├── solver_base.py     abstract Solver interface (the least-privilege contract)
│   ├── boundary.py        set_bnd + obstacle boundary enforcement
│   ├── numpy_solver.py    reference solver (vectorised, red-black Gauss–Seidel)
│   ├── numba_solver.py    JIT backend; overrides only the hot loops
│   ├── _numba_kernels.py  @njit kernels (the ONLY module importing numba)
│   └── factory.py         make_solver(config): backend select + fallback
├── render/                READ-ONLY consumers → pixels (no pygame)
│   ├── colormap.py        Palette: dye colour cursor
│   ├── renderer.py        Renderer.render(state) → (n, n, 3) uint8
│   └── overlay.py         velocity overlay → line-segment data
├── interaction/           INPUT → command objects (no pygame)
│   ├── commands.py        InjectDye / ApplyForce / ToggleObstacle / … dataclasses
│   ├── brush.py           mutable runtime brush state (radius, colour cursor)
│   └── input_map.py       normalised input → commands
└── app/                   COMPOSITION ROOT — the only place pygame lives
    ├── pygame_backend.py  window, surfarray blit, event pump → neutral input
    ├── loop.py            fixed-timestep loop; commands → solver
    └── app.py             run(): build everything from config, run loop
```

## core/

| Module | Responsibility |
|---|---|
| `grid.py` | The `(n+2)×(n+2)` ghost-cell convention, the interior slice, and back-trace coordinate clamping — all in one tiny, stateless place. |
| `fields.py` | `FluidState`: owns every array (velocity `u`,`v`; RGB `density`; `obstacle` mask; reusable scratch buffers). Provides **read-only views** for consumers. |
| `solver_base.py` | `BaseSolver`: the abstract contract — *inject* (`add_dye`, `add_velocity`, `set_obstacle`), *step*, *read* (`density_field`, …). Implements all the input/output plumbing once. |
| `boundary.py` | `set_bnd` (wall sign conventions) and `apply_obstacle_bnd` (internal solids). The highest-risk code; see [Boundaries](math-boundary-conditions.md). |
| `numpy_solver.py` | The reference physics: `add source → diffuse → project → advect → project`, all vectorised. |
| `numba_solver.py` / `_numba_kernels.py` | Optional acceleration; see [Backends](architecture-backends.md). |
| `factory.py` | The single place that maps a backend name to a concrete solver. |

## render/

Pure functions of state — they read fields and produce pixels or data, never
mutating anything and never importing pygame.

- `renderer.py` — `Renderer.render(state)` returns an `(n, n, 3)` `uint8` image in
  `[row=y, col=x]` order. Dye is clamped to `[0, 255]`; obstacles are composited
  last so they always read as walls.
- `overlay.py` — `velocity_segments(state, …)` returns a list of line segments in
  grid coordinates. **Data, not drawing** — the backend rasterises them.
- `colormap.py` — `Palette`, the dye colour cursor.

## interaction/

- `commands.py` — small frozen command objects, the vocabulary between input and
  the loop.
- `brush.py` — `Brush`, the live brush (radius + colour cursor), with bounded
  mutators.
- `input_map.py` — `translate(frame_input, brush)` turns toolkit-neutral input
  (`FrameInput`, `PointerState`, `KeyAction`) into commands. No pygame types
  appear here, so it is fully unit-testable.

## scenes/ and recording.py (the CFD engine)

Two further consumers of the core power the programmatic CFD engine. `scenes/`
(shapes, sources, `Scene` + `Simulation`, and a library of experiments) and
`recording.py` (offline video) are pure consumers — no pygame — and `core` is
forbidden from importing them by the [layering test](architecture-layering.md).
They are documented in their own section, starting at
[Programmatic CFD: Overview](cfd-engine.md). The HQ field visualiser lives in
`render/visualize.py` with colormaps in `render/colormaps.py`.

## app/

The composition root, and the **only** subpackage allowed to import pygame.

- `pygame_backend.py` — owns the window, clock, and surface. Converts raw pygame
  events into neutral `FrameInput`, and presents a rendered image plus overlay.
- `loop.py` — `SimulationLoop`: the mediator. Gathers input, *interprets
  commands* (calling the solver's injection API — interaction never touches the
  solver itself), integrates physics on a fixed-`dt` accumulator, renders, and
  presents.
- `app.py` — `run(config)`: builds solver, renderer, brush, and backend from an
  `AppConfig` and runs the loop, guaranteeing the window is torn down on exit.

## Two implementation gotchas that live in `app/`

> **`surfarray` axis order.** `pygame.surfarray` expects `[x, y]` (width, height),
> but our fields are `[y, x]` (row, col). `pygame_backend.present` transposes
> exactly once, at the blit edge. Doing it anywhere else would silently flip the
> image.

> **Fixed-timestep accumulator.** `loop._integrate` adds the real elapsed time to
> an accumulator and runs whole `dt` steps out of it, capped at `MAX_SUBSTEPS`.
> Stepping by the variable frame time instead would make viscosity and advection
> behave differently on fast vs slow machines.
