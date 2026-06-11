# fluid-sim

An interactive, real-time **2D fluid simulation** in Python. Paint glowing dye
with the mouse, push the fluid around, and drop solid obstacles for the flow to
swirl past — all driven by a physically based solver of the incompressible
Navier–Stokes equations (Jos Stam's *Stable Fluids*).

> Built to be **modular and independent**, to follow the **principle of least
> privilege**, and to read as **clean, understandable code**. The physics core
> has zero UI dependencies; the renderer can only read; input can only inject.

## Features

- **Stable Fluids** grid solver — unconditionally stable incompressible flow.
- **Coloured dye** (3 advected channels) that mixes and transports realistically.
- **Live interaction** — inject dye + force, build/erase obstacles, toggle a
  velocity overlay, cycle colours, resize the brush.
- **Programmable CFD engine** — configurable boundary conditions (inflow,
  outflow, periodic, moving walls), buoyancy, and a scene/experiment system for
  wind tunnels and more.
- **High-quality offline rendering** — vorticity / speed / pressure views through
  perceptual colormaps, supersampled, with streamline overlays, encoded to **MP4**.
- **Two backends** behind one interface: pure **NumPy**, plus an optional
  **Numba**-JIT backend with graceful fallback.
- **Documented** — an mdBook covering every design decision, the full math, and
  usage (see [`docs/`](docs/src/SUMMARY.md)).
- **Tested** — physics invariants (divergence-free, no-leak obstacles), backend
  parity, and a layering test that *enforces* the least-privilege rule.

## Quick start

```bash
python -m venv .venv
# Windows:        .venv\Scripts\Activate.ps1
# macOS / Linux:  source .venv/bin/activate

pip install -e .            # runtime deps: numpy, pygame-ce
fluidsim                    # launch the window  (or: python -m fluidsim)
```

Optional extras:

```bash
pip install -e ".[numba]"   # faster backend for larger grids
pip install -e ".[dev]"     # pytest, ruff, mypy
```

Run the **UI-free** core (proves the physics is independent):

```bash
python examples/headless_demo.py
```

## Controls

| Input | Action |
|---|---|
| **Left-drag** | inject dye + push the fluid |
| **Right-drag** | place solid obstacles |
| **Middle-drag** | erase obstacles |
| `Space` | pause / resume |
| `C` | clear dye & velocity |
| `V` | toggle velocity overlay |
| `Tab` / `[` `]` | cycle dye colour |
| `+` / `-` | brush size |
| `Esc` | quit |

Tweak it: `fluidsim --backend numba --grid 256 --viscosity 1e-5 --overlay`
(`fluidsim --help` for all flags).

## Programmatic CFD & video

Beyond the interactive toy, fluidsim is a scriptable CFD engine. Describe an
experiment as a **scene** (boundary conditions, obstacles, sources) and render it
to a high-quality video:

```bash
pip install -e ".[video]"
# Flow past a cylinder → Kármán vortex street, vorticity view → MP4
fluidsim-render wind_tunnel tunnel.mp4 --view vorticity --frames 600
```

```python
from fluidsim.recording import render_scene
from fluidsim.render.visualize import VisualConfig
from fluidsim.scenes.library import wind_tunnel

render_scene(wind_tunnel(n=200), "tunnel.mp4", frames=400,
            visual=VisualConfig(view="vorticity", output_size=(960, 960)))
```

Built-in scenes: `wind_tunnel`, `lid_driven_cavity`, `smoke_plume`, `shear_layer`
— or compose your own (see `examples/custom_scene.py`). Views: `dye`, `vorticity`,
`speed`, `pressure`, with streamline/arrow overlays. Full guide in the
[CFD Engine docs](docs/src/cfd-engine.md).

## Project layout

```text
src/fluidsim/
  core/         pure physics (NumPy + optional Numba) — no UI; boundary conditions, buoyancy
  render/       fields → pixels (read-only): fast blit + HQ visualize.py + colormaps
  interaction/  input → command objects (no pygame)
  app/          composition root + pygame backend (the only pygame importer)
  scenes/       programmatic experiments: shapes, sources, Scene + Simulation, library
  recording.py  render a scene to MP4 / PNG frames (imageio, optional)
tests/          physics invariants, backend parity, layering enforcement
examples/       headless_demo.py, render_video.py, custom_scene.py
docs/           mdBook: decisions, mathematics, CFD engine, and usage
```

## Documentation

The full book lives in [`docs/`](docs/src/SUMMARY.md). Highlights:

- [Design Overview & Decisions](docs/src/architecture-overview.md)
- The Mathematics — from [Navier–Stokes](docs/src/math-navier-stokes.md) down to
  [projection](docs/src/math-projection.md) and
  [obstacles](docs/src/math-obstacles.md)
- [Configuration Reference](docs/src/reference-configuration.md) ·
  [Extending the Simulation](docs/src/extending.md)

Build it with [mdBook](https://rust-lang.github.io/mdBook/): `mdbook serve docs --open`.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

[GNU AGPL-3.0-or-later](LICENSE). Method © the work of **Jos Stam** (*Stable
Fluids*, SIGGRAPH 1999; *Real-Time Fluid Dynamics for Games*, GDC 2003).
