# Extending the Simulation

The layering that keeps the project clean also makes it easy to extend: most
additions touch exactly one module. This chapter sketches the common ones.

## Use the solver as a library (no UI)

The core is a standalone fluid library. Import only `fluidsim.core` and drive it
yourself — this is exactly what `examples/headless_demo.py` does:

```python
from fluidsim.config import SimConfig
from fluidsim.core import make_solver

solver = make_solver(SimConfig(n=128, backend="auto"))
solver.set_obstacle(64, 64, radius=8, solid=True)

for t in range(300):
    solver.add_dye(20, 64, (255, 80, 0), radius=4)
    solver.add_velocity(20, 64, 60.0, 0.0, radius=4)
    solver.step(1 / 60)

image = solver.density_field        # read-only (n+2, n+2, 3) array
```

Great for batch runs, headless rendering to video, or driving the fluid from
another program.

## Add a new dye palette or colour

Edit `DEFAULT_PALETTE` in `config.py`, or pass a custom `palette` to
`BrushConfig`. Colour cycling and the brush pick it up automatically via
`render/colormap.py::Palette`.

## Rebind keys or change controls

The physical-key → action map is the dict `_KEY_BINDINGS` in
`app/pygame_backend.py`. Change a binding there and nothing else moves. To add a
*new* action: add a `KeyAction` enum value (`interaction/input_map.py`), map a key
to it in the backend, emit a command for it in `translate`, and handle that
command in `loop._apply`. Four small, localised edits.

## Add a new interaction (a new command)

1. Define a frozen command in `interaction/commands.py`.
2. Emit it from `interaction/input_map.translate` based on the input.
3. Interpret it in `app/loop.SimulationLoop._apply` (the only place commands meet
   the solver).

Input still never touches the solver directly — the command indirection is
preserved.

## Reduce dissipation (sharper dye)

Swap the advection scheme for a **MacCormack/BFECC** variant, or add **vorticity
confinement**, as discussed in [Dissipation](math-dissipation.md). Both live
entirely inside the solver:

- MacCormack: replace `_advect` with a forward-then-backward error-corrected
  version. Implement it once on the NumPy backend; the Numba backend can mirror it
  later.
- Vorticity confinement: add a force term inside `_vel_step`, after diffusion,
  before the first projection.

Keep the [parity test](reference-testing.md) green by changing both backends, or
gate the feature to the NumPy backend while you prototype.

## Add a whole new solver backend

Subclass `BaseSolver` (or `NumpySolver`) and implement `_vel_step` / `_dens_step`
— for example an SPH particle solver or a GPU/`moderngl` one. Register it in
`core/factory.make_solver` behind a new `backend` name. Because consumers only see
the `BaseSolver` interface, the renderer, input, and loop need **no** changes.

## Swap the UI toolkit

Rewrite the single file `app/pygame_backend.py` to target another toolkit
(`pyglet`, `moderngl`, a web canvas via `pygbag`, …). It must produce a
`FrameInput` from input and accept an `(n, n, 3)` image plus overlay segments.
Nothing outside `app/` is affected — the [layering test](architecture-layering.md)
guarantees the rest of the code never depended on pygame in the first place.

## 3D, someday

The grid, boundary, and primitive structure generalise to 3D (add a `z` axis and
a `w` velocity component, extend the stencils and the back-trace). Rendering would
need a volumetric approach rather than a flat blit — a much larger change, but the
solver design does not stand in the way.
